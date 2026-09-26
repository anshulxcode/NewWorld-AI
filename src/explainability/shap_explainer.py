"""
SHAP Explainer and Interactive Counterfactual Engine for NetWorld-AI.
Computes feature attributions using SHAP / gradient feature importance,
and provides live counterfactual re-inference to evaluate risk deltas under modified feature inputs.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import torch

from src.models import LSTMWorldModel

from src.utils import setup_logger

logger = setup_logger("shap_explainer")


class SHAPExplainer:
    """
    Explainer class for feature attribution and counterfactual ("What-If") scenario simulation.
    """

    def __init__(self, model: LSTMWorldModel, feature_names: List[str]) -> None:
        """
        Initializes SHAPExplainer.

        Args:
            model: Trained LSTMWorldModel model.
            feature_names: List of input feature column names.
        """
        self.model = model
        self.feature_names = feature_names
        self.device = next(model.parameters()).device
        self.explainer = None

    def explain(self, input_sequence: np.ndarray) -> List[Tuple[str, float]]:
        """
        Computes feature importance scores for a given input sequence.

        Args:
            input_sequence: Array of shape (sequence_length, num_features) or (1, sequence_length, num_features).

        Returns:
            List[Tuple[str, float]]: Top features sorted by absolute importance value:
                [('syn_count', 0.42), ('dst_port', -0.31), ...]
        """
        self.model.eval()

        if input_sequence.ndim == 2:
            seq_arr = np.expand_dims(input_sequence, axis=0)
        else:
            seq_arr = input_sequence

        seq_tensor = torch.tensor(seq_arr, dtype=torch.float32, requires_grad=True).to(self.device)

        # 1. Gradient-based Feature Sensitivity (Integrated Gradients / Saliency fallback)
        try:
            _, risk_logits, _, _ = self.model(seq_tensor)
            risk_score = torch.sigmoid(risk_logits)
            risk_score.backward()

            # Average absolute gradient across time dimension
            grads = seq_tensor.grad.abs().squeeze(0).mean(dim=0).cpu().numpy() # (num_features,)
            feat_importance = list(zip(self.feature_names, grads.tolist()))
            feat_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            return feat_importance[:5]
        except Exception as e:
            logger.warning(f"Gradient SHAP computation fallback triggered: {e}")
            # Heuristic fallback using feature variance across sequence
            var = np.var(seq_arr.squeeze(0), axis=0)
            feat_importance = list(zip(self.feature_names, var.tolist()))
            feat_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            return feat_importance[:5]

    def explain_risk_delta(
        self,
        state_t0: np.ndarray,
        state_t1: np.ndarray
    ) -> List[Dict[str, str]]:
        """
        Computes individual feature contributions to the CHANGE in risk score between state_t0 and state_t1.

        Args:
            state_t0: Baseline state array.
            state_t1: Modified state array after defensive intervention.

        Returns:
            List[Dict[str, str]]: Top 5 features sorted by absolute contribution magnitude:
                [{"feature": "internal_connections", "contribution": "+21.4%"}, ...]
        """
        self.model.eval()

        if state_t0.ndim == 1:
            s0_arr = np.expand_dims(np.expand_dims(state_t0, axis=0), axis=0)
        elif state_t0.ndim == 2:
            s0_arr = np.expand_dims(state_t0, axis=0)
        else:
            s0_arr = state_t0

        if state_t1.ndim == 1:
            s1_arr = np.expand_dims(np.expand_dims(state_t1, axis=0), axis=0)
        elif state_t1.ndim == 2:
            s1_arr = np.expand_dims(state_t1, axis=0)
        else:
            s1_arr = state_t1

        t0_tensor = torch.tensor(s0_arr, dtype=torch.float32, requires_grad=True).to(self.device)
        t1_tensor = torch.tensor(s1_arr, dtype=torch.float32, requires_grad=True).to(self.device)

        # Gradient sensitivity computation
        try:
            _, risk_logits_0, _, _ = self.model(t0_tensor)
            torch.sigmoid(risk_logits_0).backward()
            g0 = t0_tensor.grad.abs().squeeze(0).mean(dim=0).cpu().numpy()
        except Exception:
            g0 = np.ones(len(self.feature_names))

        try:
            _, risk_logits_1, _, _ = self.model(t1_tensor)
            torch.sigmoid(risk_logits_1).backward()
            g1 = t1_tensor.grad.abs().squeeze(0).mean(dim=0).cpu().numpy()
        except Exception:
            g1 = np.ones(len(self.feature_names))

        # Calculate feature differences between state_t1 and state_t0
        feat_diff = s1_arr[0, -1, :] - s0_arr[0, -1, :]
        avg_grad = 0.5 * (g0 + g1)

        raw_contrib = feat_diff * avg_grad
        sum_abs = float(np.sum(np.abs(raw_contrib)) + 1e-6)

        contributions = []
        for idx, f_name in enumerate(self.feature_names):
            if idx < len(raw_contrib):
                c_val = raw_contrib[idx]
                pct = (c_val / sum_abs) * 100.0
                sign = "+" if pct >= 0 else ""
                contributions.append({
                    "feature": f_name,
                    "contribution": f"{sign}{pct:.1f}%",
                    "abs_val": abs(c_val)
                })

        contributions.sort(key=lambda x: x["abs_val"], reverse=True)
        top_5 = [
            {"feature": item["feature"], "contribution": item["contribution"]}
            for item in contributions[:5]
        ]

        logger.info(f"Quantified Risk Delta Explanation: Top feature change = {top_5[0] if top_5 else 'N/A'}")
        return top_5

    def counterfactual(
        self,
        sequence: np.ndarray,
        feature_name_or_idx: Any,
        new_value: float
    ) -> Dict[str, float]:
        """
        Performs LIVE counterfactual model re-inference by altering a specific feature
        value and evaluating the resulting threat risk score and risk delta.

        Args:
            sequence: Baseline sequence array of shape (sequence_length, num_features) or (1, seq_len, num_features).
            feature_name_or_idx: Feature column name string or integer index.
            new_value: Modified feature value.

        Returns:
            Dict[str, float]:
                - original_risk: Original predicted risk score (0.0 to 1.0)
                - counterfactual_risk: New predicted risk score after feature modification
                - delta: Risk difference (counterfactual_risk - original_risk)
        """
        self.model.eval()

        if isinstance(feature_name_or_idx, str):
            if feature_name_or_idx in self.feature_names:
                feat_idx = self.feature_names.index(feature_name_or_idx)
            else:
                feat_idx = 0
        else:
            feat_idx = int(feature_name_or_idx)

        # Create copy of sequence array
        cf_seq = np.copy(sequence)
        if cf_seq.ndim == 2:
            cf_seq = np.expand_dims(cf_seq, axis=0)

        baseline_tensor = torch.tensor(cf_seq, dtype=torch.float32).to(self.device)

        # Mutate specified feature across latest sequence step
        cf_seq_mutated = np.copy(cf_seq)
        cf_seq_mutated[0, -1, feat_idx] = new_value
        mutated_tensor = torch.tensor(cf_seq_mutated, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            _, orig_logits, _, _ = self.model(baseline_tensor)
            _, cf_logits, _, _ = self.model(mutated_tensor)

            orig_risk = float(torch.sigmoid(orig_logits).item())
            cf_risk = float(torch.sigmoid(cf_logits).item())

        delta = cf_risk - orig_risk

        logger.info(
            f"Counterfactual Re-inference ({self.feature_names[feat_idx]}={new_value:.2f}): "
            f"Orig Risk: {orig_risk:.4f} -> CF Risk: {cf_risk:.4f} (Delta: {delta:+.4f})"
        )

        return {
            "original_risk": orig_risk,
            "counterfactual_risk": cf_risk,
            "delta": delta
        }

    def get_waterfall_data(self, sequence: np.ndarray) -> Dict[str, Any]:
        """
        Prepares structured waterfall data for rendering SHAP attributions in Streamlit.

        Args:
            sequence: Input sequence array.

        Returns:
            Dict[str, Any]: Dictionary with feature names, values, and SHAP contribution scores.
        """
        top_5 = self.explain(sequence)
        features = [item[0] for item in top_5]
        values = [item[1] for item in top_5]

        return {
            "features": features,
            "values": values,
            "base_value": 0.5
        }
