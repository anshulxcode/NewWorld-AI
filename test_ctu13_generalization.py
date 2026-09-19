"""
Unit Test Suite for CTU-13 Cross-Dataset Generalization (NetWorld-AI).
Tests CTU13Loader feature extraction, StandardScaler reuse, and cross_dataset_eval execution.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.utils import set_seed, load_config
from src.data.ctu13_loader import CTU13Loader
from src.evaluate import cross_dataset_eval


def test_ctu13_generalization():
    print("\n=======================================================")
    print("  CTU-13 CROSS-DATASET GENERALIZATION TEST SUITE")
    print("=======================================================\n")

    set_seed(42)
    ctu13_path = "data/sample/ctu13_sample.csv"
    scaler_path = "models/scaler.joblib"

    assert os.path.exists(ctu13_path), f"CTU-13 sample file missing at {ctu13_path}"
    assert os.path.exists(scaler_path), f"Pre-fitted scaler missing at {scaler_path}"

    # 1. Test CTU13Loader Feature Extraction
    loader = CTU13Loader(filepath=ctu13_path, scaler_path=scaler_path)
    raw_df = loader.load_raw()
    extracted_df, feat_names = loader.extract_features(raw_df)

    assert len(feat_names) == 37, f"Expected 37 features in schema, got {len(feat_names)}"
    assert "protocol" in extracted_df.columns, "Extracted features missing protocol column"
    print("[OK] 1. CTU13Loader Argus NetFlow feature extraction passed.")

    # 2. Test Normalization with Reused CIC-IDS2018 StandardScaler
    scaled_df = loader.normalize_with_cic_scaler(extracted_df, feat_names)
    assert not scaled_df[feat_names].isnull().any().any(), "Scaled features contain NaNs"
    print("[OK] 2. CIC-IDS2018 StandardScaler reuse passed.")

    # 3. Test cross_dataset_eval() Execution
    res = cross_dataset_eval(ctu13_csv_path=ctu13_path)
    assert "ctu13_generalization_metrics" in res, "Missing ctu13_generalization_metrics key"
    assert "cic_ids_metrics" in res, "Missing cic_ids_metrics key"
    print("[OK] 3. cross_dataset_eval() side-by-side comparison passed.")

    print("\n=======================================================")
    print("  ALL CTU-13 GENERALIZATION TESTS PASSED SUCCESSFULLY!")
    print("=======================================================\n")


if __name__ == "__main__":
    test_ctu13_generalization()
