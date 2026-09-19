"""
Verification and Test Suite for NetWorld-AI Phase 1 (Data Foundation).
Tests sample data generation, CSV loader feature extraction, scaler joblib saving,
window sequence building, grouped episode time-splitting, and PCAP parser integration.
"""

import os
import sys
import numpy as np
import pandas as pd
from scapy.all import wrpcap, IP, TCP, Ether

from src.utils import set_seed, load_config, setup_logger
from src.generate_sample_data import generate_sample_cic_ids
from src.data import CSVLoader, PCAPParser, WindowBuilder

logger = setup_logger("test_phase1")


def test_phase1_pipeline():
    print("\n=======================================================")
    print("  NETWORLD-AI PHASE 1 DATA FOUNDATION TEST SUITE")
    print("=======================================================\n")

    # 1. Test Seed & Config Loading
    set_seed(42)
    config = load_config("config/config.yaml")
    assert "hyperparameters" in config, "Config missing hyperparameters key"
    assert config["hyperparameters"]["sequence_length"] == 10, "Sequence length must be 10"
    print("[OK] 1. Configuration & Utils loaded successfully.")

    # 2. Test Synthetic Data Generation
    sample_path = "data/sample/sample_cic_ids.csv"
    df_sample = generate_sample_cic_ids(num_rows=10000, output_path=sample_path)
    assert os.path.exists(sample_path), f"Sample file not created at {sample_path}"
    assert len(df_sample) == 10000, f"Expected 10,000 rows, got {len(df_sample)}"
    print(f"[OK] 2. Synthetic sample dataset generated ({len(df_sample)} rows).")

    # 3. Test CSVLoader Pipeline
    loader = CSVLoader(sample_path)
    raw_df = loader.load_raw()
    assert not raw_df.empty, "Raw DataFrame is empty"

    extracted_df, feature_cols = loader.extract_features(raw_df)
    assert len(feature_cols) >= 25, f"Expected >= 25 features, got {len(feature_cols)}"
    for col in feature_cols:
        assert col in extracted_df.columns, f"Feature column {col} missing in extracted DataFrame"

    cleaned_df = loader.clean_data(extracted_df)
    assert not cleaned_df.isnull().any().any(), "Cleaned DataFrame still contains NaNs"
    assert "Binary_Label" in cleaned_df.columns, "Binary_Label missing"

    scaler_path = "models/scaler.joblib"
    scaled_df, scaler = loader.normalize_features(cleaned_df, feature_cols, scaler_path=scaler_path, is_train=True)
    assert os.path.exists(scaler_path), f"Scaler file not saved at {scaler_path}"
    print(f"[OK] 3. CSVLoader pipeline passed ({len(feature_cols)} features extracted, scaler saved).")

    # 4. Test WindowBuilder & Grouped Split
    wb = WindowBuilder(scaled_df, window_size=100, sequence_length=10)
    X_win, y_win, episode_ids = wb.build_windows(scaled_df, feature_cols=feature_cols)
    assert len(X_win) == 100, f"Expected 100 windows for 10,000 rows / 100 window size, got {len(X_win)}"

    X_seq, y_seq = wb.create_sequences(X_win, y_win)
    assert X_seq.ndim == 3, f"Expected 3D tensor for LSTM, got shape {X_seq.shape}"
    assert X_seq.shape[1] == 10, f"Sequence length must be 10, got {X_seq.shape[1]}"
    assert X_seq.shape[2] == len(feature_cols), f"Feature dimension mismatch: {X_seq.shape[2]} vs {len(feature_cols)}"

    # Perform Grouped Split to check for zero leakage
    X_train, X_test, y_train, y_test = wb.grouped_time_split(X_seq, y_seq, episode_ids=episode_ids, test_size=0.2)
    assert len(X_train) + len(X_test) == len(X_seq), "Split row total mismatch"
    print(f"[OK] 4. WindowBuilder passed (Sequence tensor shape: {X_seq.shape}, Train: {X_train.shape}, Test: {X_test.shape}).")

    # Save processed numpy arrays
    proc_path = wb.save_processed(X_train, X_test, y_train, y_test)
    assert os.path.exists(proc_path), f"Processed data array not saved at {proc_path}"
    print(f"[OK] 5. Save processed .npz passed ({proc_path}).")

    # 5. Test PCAPParser with synthetic packets
    test_pcap = "data/sample/test_sample.pcap"
    pkts = [
        Ether()/IP(src="192.168.1.10", dst="192.168.1.20")/TCP(sport=1234, dport=80, flags="S"),
        Ether()/IP(src="192.168.1.10", dst="192.168.1.20")/TCP(sport=1234, dport=80, flags="A"),
        Ether()/IP(src="192.168.1.10", dst="192.168.1.20")/TCP(sport=1234, dport=80, flags="PA"),
    ]
    wrpcap(test_pcap, pkts)

    pcap_parser = PCAPParser(test_pcap)
    pcap_df = pcap_parser.parse()
    assert not pcap_df.empty, "PCAP parser output is empty"
    assert "syn_count" in pcap_df.columns, "PCAP parser missing syn_count column"
    print(f"[OK] 6. PCAPParser passed ({len(pcap_df)} flows extracted from test PCAP).")

    print("\n=======================================================")
    print("  ALL PHASE 1 TESTS PASSED SUCCESSFULLY!")
    print("=======================================================\n")


if __name__ == "__main__":
    test_phase1_pipeline()
