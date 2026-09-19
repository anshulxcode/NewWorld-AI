"""
CSV Data Loader and Feature Extraction Module for NetWorld-AI.
Handles raw network traffic CSVs (including CIC-IDS2018 and generic formats),
cleans data, extracts 25-40 temporal/flow features, and normalizes feature vectors.
"""

import os
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import pandas as pd
import joblib
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler

from src.utils import load_config, setup_logger

logger = setup_logger("csv_loader")


class CSVLoader:
    """
    Data loader and preprocessor for network traffic flow datasets.
    Supports official CIC-IDS2018 CSV schema and generic network flow CSV formats.
    """

    def __init__(self, filepath: str, config_path: str = "config/config.yaml") -> None:
        """
        Initializes CSVLoader with target dataset path and configuration.

        Args:
            filepath: Path to the raw CSV file.
            config_path: Path to YAML configuration file.
        """
        self.filepath = filepath
        self.config = load_config(config_path)
        self.target_features: List[str] = self.config.get("features", [])
        self.scaler_path: str = self.config.get("paths", {}).get("scaler_path", "models/scaler.joblib")

    def load_raw(self) -> pd.DataFrame:
        """
        Loads raw CSV file into a pandas DataFrame and standardizes column name whitespace.

        Returns:
            pd.DataFrame: Loaded raw dataset.

        Raises:
            FileNotFoundError: If filepath does not exist.
            ValueError: If the file is empty.
        """
        if not os.path.exists(self.filepath):
            logger.error(f"File not found: {self.filepath}")
            raise FileNotFoundError(f"Raw data file not found at: {self.filepath}")

        logger.info(f"Loading raw dataset from {self.filepath}")
        try:
            df = pd.read_csv(self.filepath, low_memory=False)
        except pd.errors.EmptyDataError:
            logger.error(f"CSV file is empty: {self.filepath}")
            raise ValueError(f"CSV file is empty: {self.filepath}")

        if df.empty:
            logger.warning(f"Loaded DataFrame is empty from {self.filepath}")
            return df

        # Strip whitespace from column names commonly found in CIC datasets
        df.columns = df.columns.str.strip()
        logger.info(f"Loaded dataset with shape {df.shape}")
        return df

    def extract_features(self, df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Maps CIC-IDS2018 and generic column schemas into standardized feature metrics.

        Features include:
          - Flow: src_port, dst_port, protocol, duration, total_packets, total_bytes, packet_rate, bytes_rate
          - TCP Flags: syn_count, ack_count, fin_count, rst_count, psh_count, syn_ack_ratio
          - Timing: iat_mean, iat_std, iat_max, iat_min
          - Direction: fwd_packets, bwd_packets, fwd_bwd_ratio, fwd_bytes, bwd_bytes
          - Packet: avg_packet_size, payload_std, fwd_pkt_len_mean, bwd_pkt_len_mean, pkt_len_max, pkt_len_min
          - Recon: unique_dst_ports_per_src, recon_score

        Args:
            df: Optional DataFrame input. If None, self.load_raw() will be called.

        Returns:
            Tuple[pd.DataFrame, List[str]]: Processed DataFrame containing feature columns and the feature list.
        """
        if df is None:
            df = self.load_raw()

        if df.empty:
            return df, self.target_features

        processed_df = pd.DataFrame(index=df.index)

        # Helper extraction lambda with column fallback mapping
        def get_col(col_options: List[str], default_val: float = 0.0) -> pd.Series:
            for col in col_options:
                if col in df.columns:
                    return pd.to_numeric(df[col], errors="coerce").fillna(default_val)
            return pd.Series(default_val, index=df.index)

        # 1. Flow Level Metrics
        processed_df["src_port"] = get_col(["Src Port", "Source Port", "src_port"], default_val=0)
        processed_df["dst_port"] = get_col(["Dst Port", "Destination Port", "dst_port"], default_val=80)
        processed_df["protocol"] = get_col(["Protocol", "protocol"], default_val=6)
        processed_df["duration"] = get_col(["Flow Duration", "duration", "Flow_Duration"], default_val=1.0)

        fwd_pkts = get_col(["Tot Fwd Pkts", "Total Fwd Packets", "fwd_packets", "Tot_Fwd_Pkts"], default_val=1.0)
        bwd_pkts = get_col(["Tot Bwd Pkts", "Total Backward Packets", "bwd_packets", "Tot_Bwd_Pkts"], default_val=0.0)
        processed_df["fwd_packets"] = fwd_pkts
        processed_df["bwd_packets"] = bwd_pkts
        processed_df["total_packets"] = fwd_pkts + bwd_pkts

        fwd_bytes = get_col(["TotLen Fwd Pkts", "Total Length of Fwd Packets", "fwd_bytes", "TotLen_Fwd_Pkts"], default_val=0.0)
        bwd_bytes = get_col(["TotLen Bwd Pkts", "Total Length of Bwd Packets", "bwd_bytes", "TotLen_Bwd_Pkts"], default_val=0.0)
        processed_df["fwd_bytes"] = fwd_bytes
        processed_df["bwd_bytes"] = bwd_bytes
        processed_df["total_bytes"] = fwd_bytes + bwd_bytes

        processed_df["packet_rate"] = get_col(["Flow Pkts/s", "Flow Packets/s", "packet_rate"], default_val=0.0)
        processed_df["bytes_rate"] = get_col(["Flow Byts/s", "Flow Bytes/s", "bytes_rate"], default_val=0.0)

        # Fallback calculation for rates if 0 or missing
        duration_sec = np.maximum(processed_df["duration"] / 1e6, 1e-6) # Microseconds to seconds
        rate_mask = processed_df["packet_rate"] == 0
        processed_df.loc[rate_mask, "packet_rate"] = processed_df.loc[rate_mask, "total_packets"] / duration_sec.loc[rate_mask]
        
        bytes_mask = processed_df["bytes_rate"] == 0
        processed_df.loc[bytes_mask, "bytes_rate"] = processed_df.loc[bytes_mask, "total_bytes"] / duration_sec.loc[bytes_mask]

        # 2. TCP Flag Counts & Ratios
        processed_df["syn_count"] = get_col(["SYN Flag Cnt", "SYN Flag Count", "syn_count"], default_val=0)
        processed_df["ack_count"] = get_col(["ACK Flag Cnt", "ACK Flag Count", "ack_count"], default_val=0)
        processed_df["fin_count"] = get_col(["FIN Flag Cnt", "FIN Flag Count", "fin_count"], default_val=0)
        processed_df["rst_count"] = get_col(["RST Flag Cnt", "RST Flag Count", "rst_count"], default_val=0)
        processed_df["psh_count"] = get_col(["PSH Flag Cnt", "PSH Flag Count", "psh_count"], default_val=0)

        ack_denom = np.maximum(processed_df["ack_count"], 1.0)
        processed_df["syn_ack_ratio"] = processed_df["syn_count"] / ack_denom

        # 3. Timing / IAT Metrics
        processed_df["iat_mean"] = get_col(["Flow IAT Mean", "iat_mean"], default_val=0.0)
        processed_df["iat_std"] = get_col(["Flow IAT Std", "iat_std"], default_val=0.0)
        processed_df["iat_max"] = get_col(["Flow IAT Max", "iat_max"], default_val=0.0)
        processed_df["iat_min"] = get_col(["Flow IAT Min", "iat_min"], default_val=0.0)

        # 4. Directional Ratios
        bwd_denom = np.maximum(processed_df["bwd_packets"], 1.0)
        processed_df["fwd_bwd_ratio"] = processed_df["fwd_packets"] / bwd_denom

        # 5. Packet Sizing Statistics
        processed_df["avg_packet_size"] = get_col(["Pkt Size Avg", "Pkt Len Mean", "avg_packet_size"], default_val=0.0)
        processed_df["payload_std"] = get_col(["Pkt Len Std", "payload_std"], default_val=0.0)
        processed_df["fwd_pkt_len_mean"] = get_col(["Fwd Pkt Len Mean", "fwd_pkt_len_mean"], default_val=0.0)
        processed_df["bwd_pkt_len_mean"] = get_col(["Bwd Pkt Len Mean", "bwd_pkt_len_mean"], default_val=0.0)
        processed_df["pkt_len_max"] = get_col(["Pkt Len Max", "pkt_len_max"], default_val=0.0)
        processed_df["pkt_len_min"] = get_col(["Pkt Len Min", "pkt_len_min"], default_val=0.0)

        # 6. Reconnaissance Metrics
        src_ip_col = None
        for col in ["Src IP", "Source IP", "src_ip"]:
            if col in df.columns:
                src_ip_col = col
                break

        if src_ip_col is not None:
            # Group by src_ip to calculate unique dst_ports visited per source IP
            dst_port_col_name = "Dst Port" if "Dst Port" in df.columns else "dst_port"
            dst_ports_per_src = df.groupby(src_ip_col)[dst_port_col_name].transform("nunique")
            processed_df["unique_dst_ports_per_src"] = dst_ports_per_src.fillna(1.0)

            # Shannon entropy calculation of destination port sequence distribution per source IP
            def calc_port_entropy(series: pd.Series) -> float:
                counts = series.value_counts(normalize=True)
                return float(entropy(counts, base=2)) if len(counts) > 0 else 0.0

            port_entropy_map = df.groupby(src_ip_col)[dst_port_col_name].apply(calc_port_entropy).to_dict()
            processed_df["port_sequence_entropy"] = df[src_ip_col].map(port_entropy_map).fillna(0.0)
        else:
            # Fallback estimation based on SYN flag density & port variation
            processed_df["unique_dst_ports_per_src"] = np.where(processed_df["syn_count"] > 0, processed_df["syn_count"], 1.0)
            processed_df["port_sequence_entropy"] = np.where(processed_df["unique_dst_ports_per_src"] > 1, np.log2(np.maximum(processed_df["unique_dst_ports_per_src"], 1)), 0.0)

        # Recon score: combining unique dst ports, high SYN ratio, and low packet count per flow
        recon_norm = np.clip(processed_df["unique_dst_ports_per_src"] / 100.0, 0, 1.0)
        syn_ratio = np.clip(processed_df["syn_count"] / np.maximum(processed_df["total_packets"], 1.0), 0, 1.0)
        processed_df["recon_score"] = (0.6 * recon_norm) + (0.4 * syn_ratio)

        # 7. Packet-Level Header Details (TTL, Window, Fragmentation, Retransmissions)
        processed_df["ttl_mean"] = get_col(["ttl_mean", "TTL Mean", "Init Fwd Win Byts"], default_val=64.0)
        processed_df["ttl_std"] = get_col(["ttl_std", "TTL Std"], default_val=0.0)
        processed_df["tcp_window_mean"] = get_col(["tcp_window_mean", "Init Fwd Win Byts", "Init_Win_bytes_forward"], default_val=8192.0)
        processed_df["fragment_flag_count"] = get_col(["fragment_flag_count", "Fwd URG Flags"], default_val=0.0)
        processed_df["retransmission_count"] = get_col(["retransmission_count"], default_val=0.0)

        # Preserve Label if present in raw DataFrame
        label_col = None
        for col in ["Label", "label", "Class", "class"]:
            if col in df.columns:
                label_col = col
                break

        if label_col is not None:
            processed_df["Label"] = df[label_col].astype(str)
        else:
            processed_df["Label"] = "BENIGN"

        # Preserve timestamp or Episode_ID if available
        for col in ["Timestamp", "timestamp", "Episode_ID", "episode_id"]:
            if col in df.columns:
                processed_df[col] = df[col]

        logger.info(f"Extracted {len(self.target_features)} target features successfully.")
        return processed_df, self.target_features

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans data by replacing infinite values, handling NaNs, removing duplicates,
        and mapping label strings to standard format.

        Args:
            df: Feature DataFrame to clean.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        if df.empty:
            return df

        initial_len = len(df)

        # Separate Label and metadata columns before numerical cleaning
        metadata_cols = [col for col in ["Label", "Timestamp", "timestamp", "Episode_ID", "episode_id"] if col in df.columns]
        num_cols = [col for col in df.columns if col not in metadata_cols]

        # 1. Replace infinite values with NaN and then fillna with column median/0
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
        for col in num_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                fill_val = median_val if pd.notnull(median_val) else 0.0
                df[col] = df[col].fillna(fill_val)

        # 2. Drop duplicate rows
        df = df.drop_duplicates()
        dropped = initial_len - len(df)
        if dropped > 0:
            logger.info(f"Dropped {dropped} duplicate rows.")

        # 3. Clean and standardize Label values
        if "Label" in df.columns:
            df["Label"] = df["Label"].str.strip().str.upper()
            # Binary flag: 0 for BENIGN, 1 for ATTACK
            df["Binary_Label"] = np.where(df["Label"] == "BENIGN", 0, 1)

        return df

    def normalize_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        scaler_path: Optional[str] = None,
        is_train: bool = True
    ) -> Tuple[pd.DataFrame, StandardScaler]:
        """
        Scales feature vectors using StandardScaler and saves fitted scaler with joblib.

        Args:
            df: DataFrame containing features.
            feature_cols: List of numeric feature column names to scale.
            scaler_path: Optional path to save/load joblib scaler file.
            is_train: If True, fits new scaler. If False, loads existing scaler.

        Returns:
            Tuple[pd.DataFrame, StandardScaler]: Scaled DataFrame copy and fitted StandardScaler.
        """
        target_path = scaler_path or self.scaler_path
        scaled_df = df.copy()

        # Ensure all feature_cols exist in df, filling missing ones with 0.0
        for col in feature_cols:
            if col not in scaled_df.columns:
                scaled_df[col] = 0.0

        if is_train:
            scaler = StandardScaler()
            scaled_df[feature_cols] = scaler.fit_transform(scaled_df[feature_cols])
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            joblib.dump(scaler, target_path)
            logger.info(f"Fitted StandardScaler saved to {target_path}")
        else:
            if not os.path.exists(target_path):
                raise FileNotFoundError(f"Scaler file not found for inference at {target_path}")
            scaler = joblib.load(target_path)
            scaled_df[feature_cols] = scaler.transform(scaled_df[feature_cols])
            logger.info(f"Loaded StandardScaler from {target_path}")

        return scaled_df, scaler