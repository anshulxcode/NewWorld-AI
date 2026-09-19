"""
CTU-13 NetFlow (Argus Format) Loader & Feature Extractor for NetWorld-AI.
Extracts the overlapping feature subset from CTU-13 Argus netflow records,
maps them to the exact 37-feature schema order of CIC-IDS2018, and applies
the pre-fitted StandardScaler (models/scaler.joblib) for fair cross-dataset generalization testing.
"""

import os
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
import joblib
from scipy.stats import entropy

from src.utils import load_config, setup_logger

logger = setup_logger("ctu13_loader")


class CTU13Loader:
    """
    Loader for CTU-13 dataset in Argus NetFlow CSV format.
    Extracts overlapping features and formats sequences for cross-dataset evaluation.
    """

    def __init__(
        self,
        filepath: str = "data/sample/ctu13_sample.csv",
        scaler_path: str = "models/scaler.joblib",
        config_path: str = "config/config.yaml"
    ) -> None:
        """
        Initializes CTU13Loader.

        Args:
            filepath: Path to CTU-13 Argus CSV dataset.
            scaler_path: Path to pre-fitted StandardScaler from CIC-IDS2018 training.
            config_path: Path to YAML configuration file.
        """
        self.filepath = filepath
        self.scaler_path = scaler_path
        self.config = load_config(config_path) if os.path.exists(config_path) else {}
        self.target_features: List[str] = self.config.get("features", [])

    def load_raw(self) -> pd.DataFrame:
        """
        Loads raw CTU-13 CSV file into a pandas DataFrame.

        Returns:
            pd.DataFrame: Raw CTU-13 NetFlow DataFrame.
        """
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"CTU-13 dataset file not found at {self.filepath}")

        df = pd.read_csv(self.filepath, low_memory=False)
        logger.info(f"Loaded CTU-13 raw dataset from {self.filepath} with shape {df.shape}")
        return df

    def extract_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Extracts overlapping features from CTU-13 Argus NetFlow fields and maps them into
        the exact 37-feature schema order expected by NetWorld-AI models.

        Args:
            df: Raw CTU-13 NetFlow DataFrame.

        Returns:
            Tuple[pd.DataFrame, List[str]]: Extracted feature DataFrame and target feature list.
        """
        processed_df = pd.DataFrame(index=df.index)

        # Helper function to find matching column case-insensitively
        def get_col(possible_names: List[str], default_val: float = 0.0) -> pd.Series:
            for name in possible_names:
                for col in df.columns:
                    if col.strip().lower() == name.strip().lower():
                        return pd.to_numeric(df[col], errors="coerce").fillna(default_val)
            return pd.Series(default_val, index=df.index)

        # 1. Flow-Level Metrics
        # Protocol: tcp -> 6, udp -> 17, icmp -> 1
        proto_series = df["Proto"].astype(str).str.lower() if "Proto" in df.columns else pd.Series("tcp", index=df.index)
        proto_mapped = proto_series.map({"tcp": 6, "udp": 17, "icmp": 1}).fillna(6).astype(float)

        dur = get_col(["Dur", "Duration"], default_val=0.001)
        tot_pkts = get_col(["TotPkts", "TotalPkts", "Pkts"], default_val=1.0)
        tot_bytes = get_col(["TotBytes", "TotalBytes", "Bytes"], default_val=64.0)
        src_bytes = get_col(["SrcBytes", "SrcByts"], default_val=32.0)
        bwd_bytes = np.maximum(0.0, tot_bytes - src_bytes)

        # Estimate forward and backward packet split based on byte ratio
        byte_ratio = np.clip(src_bytes / np.maximum(tot_bytes, 1.0), 0.0, 1.0)
        fwd_pkts = np.maximum(1.0, np.round(tot_pkts * byte_ratio))
        bwd_pkts = np.maximum(0.0, tot_pkts - fwd_pkts)

        processed_df["src_port"] = get_col(["Sport", "SrcPort"], default_val=1024.0)
        processed_df["dst_port"] = get_col(["Dport", "DstPort"], default_val=80.0)
        processed_df["protocol"] = proto_mapped
        processed_df["duration"] = dur
        processed_df["total_packets"] = tot_pkts
        processed_df["total_bytes"] = tot_bytes
        processed_df["packet_rate"] = tot_pkts / np.maximum(dur, 1e-4)
        processed_df["bytes_rate"] = tot_bytes / np.maximum(dur, 1e-4)

        # 2. TCP State Flags parsing from Argus 'State' column
        state_str = df["State"].astype(str).str.upper() if "State" in df.columns else pd.Series("", index=df.index)
        processed_df["syn_count"] = state_str.str.contains("S").astype(float)
        processed_df["ack_count"] = state_str.str.contains("A").astype(float)
        processed_df["fin_count"] = state_str.str.contains("F").astype(float)
        processed_df["rst_count"] = state_str.str.contains("R").astype(float)
        processed_df["psh_count"] = state_str.str.contains("P").astype(float)
        processed_df["syn_ack_ratio"] = processed_df["syn_count"] / np.maximum(processed_df["ack_count"], 1.0)

        # 3. Inter-Arrival Time (IAT) Approximations
        processed_df["iat_mean"] = dur / np.maximum(tot_pkts, 1.0)
        processed_df["iat_std"] = 0.0   # Not in Argus NetFlow summary
        processed_df["iat_max"] = dur
        processed_df["iat_min"] = 0.0

        # 4. Directional Split & Sizing
        processed_df["fwd_packets"] = fwd_pkts
        processed_df["bwd_packets"] = bwd_pkts
        processed_df["fwd_bytes"] = src_bytes
        processed_df["bwd_bytes"] = bwd_bytes
        processed_df["fwd_bwd_ratio"] = src_bytes / np.maximum(bwd_bytes, 1.0)

        processed_df["avg_packet_size"] = tot_bytes / np.maximum(tot_pkts, 1.0)
        processed_df["payload_std"] = 0.0
        processed_df["fwd_pkt_len_mean"] = src_bytes / np.maximum(fwd_pkts, 1.0)
        processed_df["bwd_pkt_len_mean"] = bwd_bytes / np.maximum(bwd_pkts, 1.0)
        processed_df["pkt_len_max"] = processed_df["avg_packet_size"] * 1.5
        processed_df["pkt_len_min"] = processed_df["avg_packet_size"] * 0.5

        # 5. Reconnaissance & Host Statistics
        src_ip_col = "SrcAddr" if "SrcAddr" in df.columns else "SrcIP"
        dst_port_col = "Dport" if "Dport" in df.columns else "DstPort"

        if src_ip_col in df.columns and dst_port_col in df.columns:
            unique_dst = df.groupby(src_ip_col)[dst_port_col].transform("nunique")
            processed_df["unique_dst_ports_per_src"] = unique_dst.fillna(1.0)

            def calc_port_entropy(series: pd.Series) -> float:
                counts = series.value_counts(normalize=True)
                return float(entropy(counts, base=2)) if len(counts) > 0 else 0.0

            port_entropy_map = df.groupby(src_ip_col)[dst_port_col].apply(calc_port_entropy).to_dict()
            processed_df["port_sequence_entropy"] = df[src_ip_col].map(port_entropy_map).fillna(0.0)
        else:
            processed_df["unique_dst_ports_per_src"] = 1.0
            processed_df["port_sequence_entropy"] = 0.0

        recon_norm = np.clip(processed_df["unique_dst_ports_per_src"] / 100.0, 0, 1.0)
        syn_ratio = np.clip(processed_df["syn_count"] / np.maximum(tot_pkts, 1.0), 0, 1.0)
        processed_df["recon_score"] = (0.6 * recon_norm) + (0.4 * syn_ratio)
        processed_df["baseline_deviation_score"] = 0.0

        # 6. Zero-fill unavailable NetFlow fields for 37-feature schema compatibility
        processed_df["ttl_mean"] = 64.0
        processed_df["ttl_std"] = 0.0
        processed_df["tcp_window_mean"] = 8192.0
        processed_df["fragment_flag_count"] = 0.0
        processed_df["retransmission_count"] = 0.0

        # Standardize CTU-13 Ground Truth Label string (e.g. Botnet -> ATTACK, Normal/Background -> BENIGN)
        if "Label" in df.columns:
            raw_labels = df["Label"].astype(str).str.upper()
            processed_df["Label"] = np.where(
                raw_labels.str.contains("BOTNET") | raw_labels.str.contains("ATTACK") | raw_labels.str.contains("CC") | raw_labels.str.contains("SCAN"),
                "ATTACK",
                "BENIGN"
            )
            processed_df["Binary_Label"] = np.where(processed_df["Label"] == "BENIGN", 0, 1)
        else:
            processed_df["Label"] = "BENIGN"
            processed_df["Binary_Label"] = 0

        # Reorder columns to match the exact 37-feature target list order
        ordered_cols = [f for f in self.target_features if f in processed_df.columns]
        for f in self.target_features:
            if f not in ordered_cols:
                processed_df[f] = 0.0
                ordered_cols.append(f)

        output_df = processed_df[self.target_features].copy()
        output_df["Label"] = processed_df["Label"]
        output_df["Binary_Label"] = processed_df["Binary_Label"]

        logger.info(f"Extracted {len(self.target_features)} overlapping features from CTU-13 NetFlow.")
        return output_df, self.target_features

    def normalize_with_cic_scaler(
        self,
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> pd.DataFrame:
        """
        Applies the pre-fitted StandardScaler from CIC-IDS2018 training to CTU-13 features.
        Reuses the existing scaler to ensure fair out-of-distribution evaluation.

        Args:
            df: DataFrame containing extracted CTU-13 features.
            feature_cols: List of 37 feature names.

        Returns:
            pd.DataFrame: Scaled feature DataFrame copy.
        """
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"CIC-IDS2018 StandardScaler not found at {self.scaler_path}. Train model first.")

        scaler: StandardScaler = joblib.load(self.scaler_path)
        scaled_df = df.copy()

        # Ensure all feature_cols exist in df
        for col in feature_cols:
            if col not in scaled_df.columns:
                scaled_df[col] = 0.0

        scaled_df[feature_cols] = scaler.transform(scaled_df[feature_cols])
        logger.info(f"Transformed CTU-13 features using CIC-IDS2018 StandardScaler from {self.scaler_path}")
        return scaled_df
