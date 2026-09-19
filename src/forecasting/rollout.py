"""
Rollout Engine for NetWorld-AI.
Simulates future K-step state trajectories, forecasts threat risk probabilities,
computes autoencoder confidence bounds, maps predicted states into MITRE ATT&CK stages,
and supports multi-trajectory stochastic future branching via MC Dropout.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import torch

from src.models import LSTMWorldModel, Autoencoder
from src.forecasting.stage_mapper import StageMapper
from src.utils import load_config, setup_logger

logger = setup_logger("rollout_engine")


def cluster_trajectories(trajectory_list: List[List[str]]) -> List[Dict[str, Any]]:
    """
    Groups stage sequences by path string representation and computes probability distribution.

    Args:
        trajectory_list: List of stage sequences, e.g. [["Reconnaissance", "Lateral Movement"], ...]

    Returns:
        List[Dict[str, Any]]: Sorted list of dicts descending by probability:
            [{"path": "Reconnaissance -> Lateral Movement", "count": 12, "probability": 0.60}, ...]
    """
    if not trajectory_list:
        return []

    N = len(trajectory_list)
    counts: Dict[str, int] = {}

    for seq in trajectory_list:
        path_str = " -> ".join(seq)
        counts[path_str] = counts.get(path_str, 0) + 1

    results = [
        {
            "path": path_str,
            "count": count,
            "probability": float(count / N)
        }
        for path_str, count in counts.items()
    ]

    # Sort descending by probability
    results.sort(key=lambda x: x["probability"], reverse=True)
    return results


class RolloutEngine:
    """
    Simulates K-step future threat dynamics using LSTMWorldModel rollout.
    """

    def __init__(
        self,
        model: LSTMWorldModel,
        scaler: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        autoencoder: Optional[Autoencoder] = None,
        stage_mapper: Optional[StageMapper] = None
    ) -> None:
        """
        Initializes RolloutEngine.

        Args:
            model: Trained LSTMWorldModel instance.
            scaler: Fitted StandardScaler for inverse feature transformation.
            config: Loaded configuration dictionary.
            autoencoder: Optional Autoencoder for uncertainty confidence bands.
            stage_mapper: Optional StageMapper for MITRE tactic assignment.
        """
        self.model = model
        self.scaler = scaler
        self.config = config or load_config("config/config.yaml") if load_config else {}
        self.autoencoder = autoencoder
        self.stage_mapper = stage_mapper or StageMapper()
        self.feature_names: List[str] = self.config.get("features", [])

    def simulate(
        self,
        initial_sequence: np.ndarray,
        k_steps: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Executes K-step autoregressive rollout simulation from a seed sequence.

        Args:
            initial_sequence: 2D array of shape (sequence_length, num_features) or 3D (1, sequence_length, num_features).
            k_steps: Number of future time windows to forecast (default: 5).

        Returns:
            List[Dict[str, Any]]: Forecast results per future time window:
                - window: Step index (1..k_steps)
                - risk: Predicted risk probability score (0.0 to 1.0)
                - stage: MITRE ATT&CK stage name
                - confidence: Stage classification confidence
                - evidence: Supporting evidence text
                - features: Unscaled/scaled feature values for the predicted state
                - confidence_lower: Lower bound risk confidence interval
                - confidence_upper: Upper bound risk confidence interval
        """
        device = next(self.model.parameters()).device

        # Standardize input shape to (1, sequence_length, num_features)
        if isinstance(initial_sequence, np.ndarray):
            if initial_sequence.ndim == 2:
                seq_tensor = torch.tensor(initial_sequence, dtype=torch.float32).unsqueeze(0).to(device)
            else:
                seq_tensor = torch.tensor(initial_sequence, dtype=torch.float32).to(device)
        else:
            seq_tensor = initial_sequence.to(device)
            if seq_tensor.dim() == 2:
                seq_tensor = seq_tensor.unsqueeze(0)

        # Execute recursive model rollout
        forecasted_states, forecasted_risks, forecasted_stages = self.model.rollout(seq_tensor, k_steps=k_steps)

        # Convert outputs to numpy
        states_np = forecasted_states.squeeze(0).cpu().numpy() # (k_steps, num_features)
        risks_np = forecasted_risks.squeeze(0).cpu().numpy().squeeze(-1) # (k_steps,)
        stages_np = forecasted_stages.squeeze(0).cpu().numpy() # (k_steps, 7)

        results = []

        for step in range(k_steps):
            state_vec = states_np[step]
            risk_val = float(np.clip(risks_np[step], 0.0, 1.0))
            pred_stage_idx = int(np.argmax(stages_np[step]))

            # Unscale feature values if scaler is available
            if self.scaler is not None and hasattr(self.scaler, "inverse_transform"):
                try:
                    unscaled_state = self.scaler.inverse_transform(state_vec.reshape(1, -1))[0]
                except Exception:
                    unscaled_state = state_vec
            else:
                unscaled_state = state_vec

            # Map predicted state vector to MITRE ATT&CK tactic
            stage, stage_conf, evidence = self.stage_mapper.map_stage(unscaled_state, self.feature_names, predicted_stage_idx=pred_stage_idx)

            # Autoencoder confidence interval calculation
            if self.autoencoder is not None:
                step_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0).to(device)
                _, lower_b, upper_b = self.autoencoder.get_confidence_interval(step_tensor, n_samples=10)
                # Compute reconstruction uncertainty variance
                unc = float(np.mean(upper_b - lower_b))
                conf_lower = float(max(0.0, risk_val - (0.5 * unc)))
                conf_upper = float(min(1.0, risk_val + (0.5 * unc)))
            else:
                conf_lower = float(max(0.0, risk_val - 0.05))
                conf_upper = float(min(1.0, risk_val + 0.05))

            step_record = {
                "window": step + 1,
                "risk": risk_val,
                "stage": stage,
                "confidence": stage_conf,
                "evidence": evidence,
                "features": unscaled_state.tolist(),
                "confidence_lower": conf_lower,
                "confidence_upper": conf_upper
            }
            results.append(step_record)

        logger.info(f"Successfully simulated {k_steps}-step rollout trajectory.")
        return results

    def simulate_multiple_futures(
        self,
        initial_sequence: np.ndarray,
        k_steps: int = 5,
        n_samples: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Simulates N=20 stochastic forward rollout trajectories using MC Dropout to capture branching future attack paths.

        Args:
            initial_sequence: Seed 2D sequence matrix of shape (seq_len, num_features).
            k_steps: Number of future time steps to simulate (default: 5).
            n_samples: Number of MC Dropout stochastic forward paths (default: 20).

        Returns:
            List[Dict[str, Any]]: Sorted list of trajectory path probabilities:
                [{"path": "Reconnaissance -> Lateral Movement -> Exfiltration", "count": 12, "probability": 0.60}, ...]
        """
        device = next(self.model.parameters()).device

        if isinstance(initial_sequence, np.ndarray):
            if initial_sequence.ndim == 2:
                seed_tensor = torch.tensor(initial_sequence, dtype=torch.float32).unsqueeze(0).to(device)
            else:
                seed_tensor = torch.tensor(initial_sequence, dtype=torch.float32).to(device)
        else:
            seed_tensor = initial_sequence.to(device)
            if seed_tensor.dim() == 2:
                seed_tensor = seed_tensor.unsqueeze(0)

        # Enable dropout layers for stochastic sampling while keeping other layers in eval mode
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()

        trajectory_list = []

        for _ in range(n_samples):
            current_seq = seed_tensor.clone()
            path_stages = []

            with torch.no_grad():
                for _ in range(k_steps):
                    next_state, risk_logits, stage_logits, _ = self.model(current_seq)
                    pred_stage_idx = int(torch.argmax(stage_logits, dim=-1).item())

                    # Convert feature vector and predicted stage index to MITRE stage label
                    state_vec = next_state.squeeze(0).cpu().numpy()
                    if self.scaler is not None and hasattr(self.scaler, "inverse_transform"):
                        try:
                            unscaled_state = self.scaler.inverse_transform(state_vec.reshape(1, -1))[0]
                        except Exception:
                            unscaled_state = state_vec
                    else:
                        unscaled_state = state_vec

                    stage_name, _, _ = self.stage_mapper.map_stage(unscaled_state, self.feature_names, predicted_stage_idx=pred_stage_idx)
                    path_stages.append(stage_name)

                    # Slide window forward
                    next_state_seq = next_state.unsqueeze(1)
                    current_seq = torch.cat([current_seq[:, 1:, :], next_state_seq], dim=1)

            trajectory_list.append(path_stages)

        # Reset model to standard eval mode
        self.model.eval()

        # Cluster trajectories and compute probabilities
        grouped_futures = cluster_trajectories(trajectory_list)
        logger.info(f"Simulated {n_samples} stochastic trajectories across {k_steps} steps. Found {len(grouped_futures)} unique attack paths.")
        return grouped_futures
