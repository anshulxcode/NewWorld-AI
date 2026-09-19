"""
Synthetic CTU-13 NetFlow (Argus Format) Sample Dataset Generator for NetWorld-AI.
Generates a realistic 2,000-row CTU-13 Argus NetFlow CSV mimicking official CTU-13 schema
with Botnet attack flows and normal background traffic.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import set_seed, setup_logger

logger = setup_logger("generate_ctu13_sample")


def generate_ctu13_sample_csv(
    num_rows: int = 2000,
    output_path: str = "data/sample/ctu13_sample.csv",
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates synthetic CTU-13 Argus NetFlow CSV dataset.

    Args:
        num_rows: Number of netflow records (default: 2000).
        output_path: Target CSV path.
        seed: Random seed.

    Returns:
        pd.DataFrame: Generated dataset.
    """
    set_seed(seed)
    logger.info(f"Generating synthetic CTU-13 Argus dataset with {num_rows} rows...")

    num_normal = int(num_rows * 0.75)
    num_botnet = num_rows - num_normal

    labels = ["Flow"] * num_normal + ["Botnet"] * num_botnet

    durations = np.random.exponential(scale=15.0, size=num_rows) + 0.001
    protocols = np.random.choice(["tcp", "udp", "icmp"], size=num_rows, p=[0.75, 0.2, 0.05])

    src_ports = np.random.randint(1024, 65535, size=num_rows)
    dst_ports = np.random.choice([80, 443, 22, 53, 8080, 6667], size=num_rows, p=[0.35, 0.3, 0.1, 0.1, 0.05, 0.1])

    # Botnet IRC C2 signature on port 6667
    for i in range(num_rows):
        if labels[i] == "Botnet":
            dst_ports[i] = np.random.choice([6667, 80, 443, 22], p=[0.4, 0.3, 0.2, 0.1])

    tot_pkts = np.random.negative_binomial(n=4, p=0.3, size=num_rows) + 1
    tot_bytes = tot_pkts * np.random.uniform(64, 1460, size=num_rows)
    src_bytes = tot_bytes * np.random.uniform(0.3, 0.8, size=num_rows)

    states = np.random.choice(["FSPA_SRPA", "PA_", "S_", "FA_", "SR_"], size=num_rows, p=[0.5, 0.2, 0.15, 0.1, 0.05])
    for i in range(num_rows):
        if labels[i] == "Botnet" and dst_ports[i] in [6667, 22]:
            states[i] = "S_"  # High SYN scan rate

    df = pd.DataFrame({
        "StartTime": pd.date_range(start="2011-08-10 09:46:53", periods=num_rows, freq="200ms").astype(str),
        "Dur": durations,
        "Proto": protocols,
        "SrcAddr": [f"147.32.84.{(i % 20) + 100}" for i in range(num_rows)],
        "Sport": src_ports,
        "Dir": "->",
        "DstAddr": [f"147.32.84.{(i % 5) + 1}" for i in range(num_rows)],
        "Dport": dst_ports,
        "State": states,
        "sTos": 0,
        "dTos": 0,
        "TotPkts": tot_pkts,
        "TotBytes": tot_bytes,
        "SrcBytes": src_bytes,
        "Label": labels
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Generated synthetic CTU-13 dataset at: {output_path}")
    return df


if __name__ == "__main__":
    generate_ctu13_sample_csv()
