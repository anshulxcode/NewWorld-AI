"""
Synthetic Sample Dataset Generator for NetWorld-AI (Phase 1).
Generates a realistic 10,000-row network flow CSV dataset mimicking official CIC-IDS2018 columns
with 80% BENIGN traffic and 20% multi-stage attack scenarios across episode groups.
"""

import os
from typing import Tuple
import numpy as np
import pandas as pd

from src.utils import set_seed, setup_logger

logger = setup_logger("generate_sample_data")


def generate_sample_cic_ids(
    num_rows: int = 10000,
    output_path: str = "data/sample/sample_cic_ids.csv",
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates synthetic network traffic flows mimicking the CIC-IDS2018 dataset schema.

    Args:
        num_rows: Total number of flow rows to generate (default: 10,000).
        output_path: Destination filepath for the CSV.
        seed: Random seed for deterministic generation.

    Returns:
        pd.DataFrame: Generated dataset.
    """
    set_seed(seed)
    logger.info(f"Generating synthetic sample dataset with {num_rows} rows...")

    # Class distribution: 80% BENIGN, 20% Attacks
    num_benign = int(num_rows * 0.8)
    num_attacks = num_rows - num_benign

    attack_types = ["DDoS", "PortScan", "Bot", "Infiltration", "Brute Force"]
    labels = ["BENIGN"] * num_benign + list(np.random.choice(attack_types, size=num_attacks, p=[0.4, 0.3, 0.15, 0.1, 0.05]))
    
    # Shuffle labels across rows while grouping into 50 episode blocks
    episode_ids = np.repeat(np.arange(50), num_rows // 50)
    if len(episode_ids) < num_rows:
        episode_ids = np.pad(episode_ids, (0, num_rows - len(episode_ids)), mode="edge")

    # Generate synthetic features based on realistic distributions
    dst_ports = np.random.choice([80, 443, 22, 53, 8080, 21, 3389], size=num_rows, p=[0.4, 0.35, 0.05, 0.1, 0.05, 0.02, 0.03])
    
    # Inject PortScan specific pattern
    for i in range(num_rows):
        if labels[i] == "PortScan":
            dst_ports[i] = np.random.randint(1024, 65535)

    protocols = np.random.choice([6, 17, 1], size=num_rows, p=[0.8, 0.18, 0.02]) # TCP, UDP, ICMP
    flow_duration = np.random.exponential(scale=50000, size=num_rows) + 10 # in us

    tot_fwd_pkts = np.random.negative_binomial(n=5, p=0.3, size=num_rows) + 1
    tot_bwd_pkts = np.random.negative_binomial(n=4, p=0.4, size=num_rows)

    totlen_fwd_pkts = tot_fwd_pkts * np.random.uniform(40, 1460, size=num_rows)
    totlen_bwd_pkts = tot_bwd_pkts * np.random.uniform(40, 1460, size=num_rows)

    # TCP Flags
    syn_flag_cnt = np.where(np.array(labels) == "PortScan", np.random.randint(1, 5, size=num_rows), np.random.choice([0, 1], size=num_rows, p=[0.7, 0.3]))
    ack_flag_cnt = np.where(np.array(labels) == "PortScan", 0, np.random.choice([0, 1], size=num_rows, p=[0.2, 0.8]))
    fin_flag_cnt = np.random.choice([0, 1], size=num_rows, p=[0.85, 0.15])
    rst_flag_cnt = np.where(np.array(labels) == "DDoS", 1, np.random.choice([0, 1], size=num_rows, p=[0.95, 0.05]))
    psh_flag_cnt = np.random.choice([0, 1], size=num_rows, p=[0.6, 0.4])

    flow_byts_s = (totlen_fwd_pkts + totlen_bwd_pkts) / (flow_duration / 1e6)
    flow_pkts_s = (tot_fwd_pkts + tot_bwd_pkts) / (flow_duration / 1e6)

    flow_iat_mean = flow_duration / (tot_fwd_pkts + tot_bwd_pkts)
    flow_iat_std = flow_iat_mean * np.random.uniform(0.1, 0.5, size=num_rows)
    flow_iat_max = flow_iat_mean * np.random.uniform(1.2, 2.5, size=num_rows)
    flow_iat_min = np.maximum(0, flow_iat_mean * np.random.uniform(0.1, 0.8, size=num_rows))

    pkt_len_mean = (totlen_fwd_pkts + totlen_bwd_pkts) / (tot_fwd_pkts + tot_bwd_pkts)
    pkt_len_std = pkt_len_mean * np.random.uniform(0.1, 0.4, size=num_rows)
    pkt_len_max = pkt_len_mean * np.random.uniform(1.5, 2.0, size=num_rows)
    pkt_len_min = np.maximum(0, pkt_len_mean * np.random.uniform(0.1, 0.5, size=num_rows))

    # Construct DataFrame with official CIC-IDS2018 column headers + extended packet features
    df = pd.DataFrame({
        "Dst Port": dst_ports,
        "Protocol": protocols,
        "Timestamp": pd.date_range(start="2026-09-12 00:00:00", periods=num_rows, freq="100ms").astype(str),
        "Flow Duration": flow_duration,
        "Tot Fwd Pkts": tot_fwd_pkts,
        "Tot Bwd Pkts": tot_bwd_pkts,
        "TotLen Fwd Pkts": totlen_fwd_pkts,
        "TotLen Bwd Pkts": totlen_bwd_pkts,
        "Fwd Pkt Len Max": totlen_fwd_pkts / tot_fwd_pkts * 1.2,
        "Fwd Pkt Len Min": totlen_fwd_pkts / tot_fwd_pkts * 0.8,
        "Fwd Pkt Len Mean": totlen_fwd_pkts / tot_fwd_pkts,
        "Fwd Pkt Len Std": (totlen_fwd_pkts / tot_fwd_pkts) * 0.2,
        "Bwd Pkt Len Max": np.where(tot_bwd_pkts > 0, (totlen_bwd_pkts / np.maximum(tot_bwd_pkts, 1)) * 1.2, 0),
        "Bwd Pkt Len Min": np.where(tot_bwd_pkts > 0, (totlen_bwd_pkts / np.maximum(tot_bwd_pkts, 1)) * 0.8, 0),
        "Bwd Pkt Len Mean": np.where(tot_bwd_pkts > 0, totlen_bwd_pkts / np.maximum(tot_bwd_pkts, 1), 0),
        "Bwd Pkt Len Std": np.where(tot_bwd_pkts > 0, (totlen_bwd_pkts / np.maximum(tot_bwd_pkts, 1)) * 0.2, 0),
        "Flow Byts/s": flow_byts_s,
        "Flow Pkts/s": flow_pkts_s,
        "Flow IAT Mean": flow_iat_mean,
        "Flow IAT Std": flow_iat_std,
        "Flow IAT Max": flow_iat_max,
        "Flow IAT Min": flow_iat_min,
        "Fwd IAT Tot": flow_duration * 0.9,
        "Fwd IAT Mean": flow_iat_mean,
        "Fwd IAT Std": flow_iat_std,
        "Fwd IAT Max": flow_iat_max,
        "Fwd IAT Min": flow_iat_min,
        "Bwd IAT Tot": flow_duration * 0.8,
        "Bwd IAT Mean": flow_iat_mean,
        "Bwd IAT Std": flow_iat_std,
        "Bwd IAT Max": flow_iat_max,
        "Bwd IAT Min": flow_iat_min,
        "Fwd PSH Flags": psh_flag_cnt,
        "Bwd PSH Flags": 0,
        "Fwd URG Flags": 0,
        "Bwd URG Flags": 0,
        "Fwd Header Len": tot_fwd_pkts * 20,
        "Bwd Header Len": tot_bwd_pkts * 20,
        "Fwd Pkts/s": flow_pkts_s * 0.6,
        "Bwd Pkts/s": flow_pkts_s * 0.4,
        "Pkt Len Min": pkt_len_min,
        "Pkt Len Max": pkt_len_max,
        "Pkt Len Mean": pkt_len_mean,
        "Pkt Len Std": pkt_len_std,
        "Pkt Len Var": pkt_len_std ** 2,
        "FIN Flag Cnt": fin_flag_cnt,
        "SYN Flag Cnt": syn_flag_cnt,
        "RST Flag Cnt": rst_flag_cnt,
        "PSH Flag Cnt": psh_flag_cnt,
        "ACK Flag Cnt": ack_flag_cnt,
        "URG Flag Cnt": 0,
        "CWE Flag Count": 0,
        "ECE Flag Cnt": 0,
        "Down/Up Ratio": tot_bwd_pkts / tot_fwd_pkts,
        "Pkt Size Avg": pkt_len_mean,
        "Fwd Seg Size Avg": totlen_fwd_pkts / tot_fwd_pkts,
        "Bwd Seg Size Avg": np.where(tot_bwd_pkts > 0, totlen_bwd_pkts / np.maximum(tot_bwd_pkts, 1), 0),
        "Fwd Byts/b Avg": 0,
        "Fwd Pkts/b Avg": 0,
        "Fwd Blk Rate Avg": 0,
        "Bwd Byts/b Avg": 0,
        "Bwd Pkts/b Avg": 0,
        "Bwd Blk Rate Avg": 0,
        "Subflow Fwd Pkts": tot_fwd_pkts,
        "Subflow Fwd Byts": totlen_fwd_pkts,
        "Subflow Bwd Pkts": tot_bwd_pkts,
        "Subflow Bwd Byts": totlen_bwd_pkts,
        "Init Fwd Win Byts": np.random.randint(1024, 65535, size=num_rows),
        "Init Bwd Win Byts": np.random.randint(1024, 65535, size=num_rows),
        "Fwd Act Data Pkts": tot_fwd_pkts,
        "Fwd Seg Size Min": 20,
        "Active Mean": 0,
        "Active Std": 0,
        "Active Max": 0,
        "Active Min": 0,
        "Idle Mean": 0,
        "Idle Std": 0,
        "Idle Max": 0,
        "Idle Min": 0,
        "Src IP": [f"192.168.1.{(i % 50) + 1}" for i in range(num_rows)],
        "Src Port": np.random.randint(1024, 65535, size=num_rows),
        "port_sequence_entropy": np.random.uniform(0.1, 3.5, size=num_rows),
        "ttl_mean": np.random.choice([64, 128, 255], size=num_rows),
        "ttl_std": np.random.uniform(0.0, 5.0, size=num_rows),
        "tcp_window_mean": np.random.uniform(1024, 65535, size=num_rows),
        "fragment_flag_count": np.random.choice([0, 1], size=num_rows, p=[0.95, 0.05]),
        "retransmission_count": np.random.choice([0, 1, 2], size=num_rows, p=[0.9, 0.08, 0.02]),
        "Episode_ID": episode_ids,
        "Label": labels
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Successfully generated synthetic dataset at: {output_path}")

    # Display label distribution summary
    logger.info("Label breakdown:")
    for label, count in df["Label"].value_counts().items():
        logger.info(f"  - {label}: {count} ({count / num_rows * 100:.1f}%)")

    return df


if __name__ == "__main__":
    df_sample = generate_sample_cic_ids(num_rows=10000)
    print("\n--- Synthetic Dataset Generation Complete ---")
    print(f"File saved to: data/sample/sample_cic_ids.csv")
    print(f"Shape: {df_sample.shape}")
    print("\nSample Class Counts:")
    print(df_sample["Label"].value_counts())
