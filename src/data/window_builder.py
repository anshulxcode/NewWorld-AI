"""
Sliding Window and Sequence Generator for NetWorld-AI.
Transforms tabular network flows into temporal sequence vectors for LSTM modeling,
and executes grouped time-based dataset splits to prevent attack episode data leakage.
"""

import os
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.utils import load_config, setup_logger

logger = setup_logger("window_builder")


class WindowBuilder:
    """
    Constructs sliding time windows, formats 3D temporal sequence tensors for PyTorch LSTM models,
    and performs attack-episode grouped train/test splitting.
    """

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        window_size: int = 100,
        sequence_length: int = 10,
        config_path: str = "config/config.yaml"
    ) -> None:
        """
        Initializes WindowBuilder with parameters.

        Args:
            df: Input cleaned & normalized DataFrame.
            window_size: Number of raw flows per aggregated state window.
            sequence_length: Number of consecutive state windows per LSTM input sequence.
            config_path: Path to YAML configuration file.
        """
        self.config = load_config(config_path) if os.path.exists(config_path) else {}
        self.window_size = self.config.get("hyperparameters", {}).get("window_size", window_size)
        self.sequence_length = self.config.get("hyperparameters", {}).get("sequence_length", sequence_length)
        self.target_features = self.config.get("features", [])
        self.df = df

    def build_windows(
        self,
        df: Optional[pd.DataFrame] = None,
        feature_cols: Optional[List[str]] = None,
        label_col: str = "Binary_Label"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Groups sequential flows into fixed aggregated state vectors.

        Args:
            df: Input DataFrame (uses self.df if None).
            feature_cols: List of numerical feature names.
            label_col: Column name containing binary or multi-class label (default: Binary_Label).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - X_windows: Array of shape (num_windows, num_features)
                - y_windows: Array of shape (num_windows,) containing max/majority label per window
                - episode_ids: Array of shape (num_windows,) containing episode identifier
        """
        target_df = df if df is not None else self.df
        if target_df is None or target_df.empty:
            raise ValueError("Input DataFrame is empty or None.")

        cols = feature_cols or self.target_features
        # Ensure all feature columns exist in target_df
        available_cols = [c for c in cols if c in target_df.columns]
        if not available_cols:
            raise ValueError("None of the specified feature columns are present in the DataFrame.")

        # Determine target label
        if label_col not in target_df.columns:
            if "Binary_Label" in target_df.columns:
                actual_label_col = "Binary_Label"
            elif "Label" in target_df.columns:
                actual_label_col = "Label"
            else:
                target_df["Binary_Label"] = 0
                actual_label_col = "Binary_Label"
        else:
            actual_label_col = label_col

        # Check for episode column
        ep_col = None
        for col in ["Episode_ID", "episode_id", "Episode"]:
            if col in target_df.columns:
                ep_col = col
                break

        num_rows = len(target_df)
        window_size = max(1, self.window_size)
        num_windows = num_rows // window_size

        if num_windows == 0:
            # If dataset smaller than window_size, use entire dataset as single window
            logger.warning(f"Dataset length ({num_rows}) < window_size ({window_size}). Using 1 adaptive window.")
            X_mean = target_df[available_cols].mean(numeric_only=True).values
            y_val = target_df[actual_label_col].max() if pd.api.types.is_numeric_dtype(target_df[actual_label_col]) else 1
            ep_val = target_df[ep_col].iloc[0] if ep_col else 0
            return np.expand_dims(X_mean, 0), np.array([y_val]), np.array([ep_val])

        # Map string labels to multi-class MITRE stage integers
        # 0=Recon, 1=Initial Access, 2=Discovery, 3=Lateral, 4=C2, 5=Exfil, 6=BENIGN
        def map_label_to_stage_id(label_str: str) -> int:
            lbl = str(label_str).strip().upper()
            if "BENIGN" in lbl or lbl == "0":
                return 6
            elif "PORTSCAN" in lbl or "PORT SCAN" in lbl:
                return 0 # Reconnaissance
            elif "BRUTE" in lbl or "WEB ATTACK" in lbl or "DDOS" in lbl or "DOS" in lbl or "PATATOR" in lbl:
                return 1 # Initial Access
            elif "DISCOVERY" in lbl or "ENUMERATION" in lbl:
                return 2 # Discovery
            elif "INFILTRATION" in lbl or "LATERAL" in lbl:
                return 3 # Lateral Movement
            elif "BOT" in lbl or "C2" in lbl or "BEACON" in lbl:
                return 4 # Command & Control
            elif "EXFIL" in lbl:
                return 5 # Exfiltration
            return 0 # Default attack fallback to Reconnaissance

        X_windows = []
        y_windows = []
        episode_ids = []

        for i in range(num_windows):
            start_idx = i * window_size
            end_idx = start_idx + window_size
            window_sub = target_df.iloc[start_idx:end_idx]

            # Aggregate window features: mean metric across flow block
            feat_mean = window_sub[available_cols].mean(numeric_only=True).values
            X_windows.append(feat_mean)

            # Multi-class MITRE stage label assignment: pick non-BENIGN stage if present, or majority stage
            lbl_series = window_sub[actual_label_col]
            stage_ids = [map_label_to_stage_id(val) for val in lbl_series]
            non_benign = [s for s in stage_ids if s != 6]
            if non_benign:
                # Assign predominant attack stage
                lbl_val = int(pd.Series(non_benign).mode().iloc[0])
            else:
                lbl_val = 6 # BENIGN
            y_windows.append(lbl_val)

            # Episode ID tracking
            if ep_col:
                ep_val = window_sub[ep_col].iloc[0]
            else:
                # Synthetic episode ID grouping: cluster consecutive windows into attack episode IDs
                ep_val = i // 10  # 10 windows per episode block
            episode_ids.append(ep_val)

        X_arr = np.array(X_windows, dtype=np.float32)
        y_arr = np.array(y_windows, dtype=np.int64)
        ep_arr = np.array(episode_ids, dtype=np.int64)

        logger.info(f"Built {len(X_arr)} state windows of aggregated size {window_size}.")
        return X_arr, y_arr, ep_arr

    def create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Constructs sliding 3D sequence tensors for LSTM training.

        Args:
            X: Aggregated window array of shape (num_windows, num_features).
            y: Target labels array of shape (num_windows,).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - X_seq: 3D tensor of shape (num_sequences, sequence_length, num_features)
                - y_seq: Target label of sequence end state of shape (num_sequences,)
        """
        seq_len = self.sequence_length
        num_windows = len(X)

        if num_windows < seq_len:
            logger.warning(f"Window count ({num_windows}) < sequence_length ({seq_len}). Padding sequences.")
            # Zero pad along time dimension to match sequence length
            pad_len = seq_len - num_windows
            X_padded = np.pad(X, ((pad_len, 0), (0, 0)), mode="edge")
            return np.expand_dims(X_padded, 0), np.array([y[-1] if len(y) > 0 else 0])

        X_seqList = []
        y_seqList = []

        for i in range(num_windows - seq_len + 1):
            X_seqList.append(X[i : i + seq_len])
            # Target is the label at the sequence endpoint
            y_seqList.append(y[i + seq_len - 1])

        X_seq = np.array(X_seqList, dtype=np.float32)
        y_seq = np.array(y_seqList, dtype=np.int64)

        logger.info(f"Generated 3D sequence tensor with shape {X_seq.shape}.")
        return X_seq, y_seq

    def grouped_time_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        episode_ids: np.ndarray,
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits sequence dataset into train and test sets using group-aware episode splitting.
        Prevents data leakage by ensuring entire attack episodes remain in either train or test set.

        Args:
            X: Input sequence array.
            y: Target labels array.
            episode_ids: Array of episode group identifiers corresponding to each sequence.
            test_size: Fraction of episode groups to reserve for testing (default: 0.2).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                X_train, X_test, y_train, y_test
        """
        if len(X) != len(episode_ids):
            # Align episode_ids if sequence creation reduced length
            diff = len(episode_ids) - len(X)
            if diff > 0:
                episode_ids = episode_ids[diff:]

        unique_groups = np.unique(episode_ids)
        if len(unique_groups) < 2:
            logger.warning("Fewer than 2 episode groups found. Performing sequential time split.")
            split_idx = int(len(X) * (1 - test_size))
            return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=episode_ids))

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        logger.info(
            f"Grouped time split completed without leakage. "
            f"Train shape: {X_train.shape}, Test shape: {X_test.shape} across {len(unique_groups)} episode groups."
        )
        return X_train, X_test, y_train, y_test

    def save_processed(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        output_dir: str = "data/processed"
    ) -> str:
        """
        Saves preprocessed train and test sequence tensors into a compressed NumPy .npz file.

        Args:
            X_train: Training sequence features.
            X_test: Testing sequence features.
            y_train: Training target labels.
            y_test: Testing target labels.
            output_dir: Output directory path.

        Returns:
            str: Path to saved .npz artifact.
        """
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "processed_data.npz")

        np.savez_compressed(
            save_path,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test
        )
        logger.info(f"Processed dataset successfully saved to {save_path}")
        return save_path