"""
Unit Test Suite for What-If Interventions, Quantified Risk Delta Explainability,
and Host EWMA Baseline Tracking (NetWorld-AI).
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.utils import set_seed, load_config
from src.models import LSTMWorldModel, Autoencoder
from src.forecasting import RolloutEngine, StageMapper
from src.explainability import SHAPExplainer
from src.features.host_baseline import HostBaselineManager


def test_new_capabilities():
    print("\n=======================================================")
    print("  NETWORLD-AI ADVANCED CAPABILITIES UNIT TEST SUITE")
    print("=======================================================\n")

    set_seed(42)
    config = load_config("config/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feat_names = config.get("features", ["dst_port", "packet_rate", "bytes_rate", "syn_count", "recon_score"])
    model = LSTMWorldModel(input_size=len(feat_names), hidden_size=64).to(device)

    # -------------------------------------------------------------------------
    # TEST 1: Task 3 - Host Baseline EWMA & Deviation Z-Score Test
    # -------------------------------------------------------------------------
    db_path = "scratch/test_host_baselines.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    baseline_mgr = HostBaselineManager(db_path=db_path, alpha=0.1)
    test_ip = "192.168.1.105"

    stats_normal = {
        "connections_per_minute": 10.0,
        "unique_dst_ports_touched": 2.0,
        "internal_vs_external_ratio": 1.0
    }
    # Initial observation
    dev_0 = baseline_mgr.get_baseline_deviation(test_ip, stats_normal)
    assert dev_0 == 0.0, f"Initial deviation score should be 0.0, got {dev_0}"

    # Update with several normal observations to build baseline
    for _ in range(5):
        baseline_mgr.get_baseline_deviation(test_ip, stats_normal)

    # Anomaly observation with high traffic spike
    stats_anomaly = {
        "connections_per_minute": 500.0,
        "unique_dst_ports_touched": 150.0,
        "internal_vs_external_ratio": 10.0
    }
    dev_anom = baseline_mgr.get_baseline_deviation(test_ip, stats_anomaly)
    assert dev_anom > 2.0, f"Anomaly deviation score should be high (> 2.0), got {dev_anom:.2f}"
    print(f"[OK] 1. HostBaselineManager EWMA & Z-Score Deviation passed (Anomaly Deviation Score = {dev_anom:.2f}).")

    # -------------------------------------------------------------------------
    # TEST 2: Task 2 - Quantified Counterfactual Explainability (explain_risk_delta)
    # -------------------------------------------------------------------------
    explainer = SHAPExplainer(model, feat_names)
    s0 = np.random.randn(10, len(feat_names))
    s1 = np.copy(s0)
    s1[-1, 0] += 25.0 # Spike feature 0

    delta_explanation = explainer.explain_risk_delta(s0, s1)
    assert isinstance(delta_explanation, list), "Expected list output from explain_risk_delta"
    assert len(delta_explanation) <= 5, f"Expected top 5 features max, got {len(delta_explanation)}"
    assert "feature" in delta_explanation[0] and "contribution" in delta_explanation[0], "Explanation dict missing required keys"
    assert delta_explanation[0]["contribution"].endswith("%"), "Contribution score format must end with %"
    print(f"[OK] 2. SHAPExplainer.explain_risk_delta() passed (Top Feature Delta: {delta_explanation[0]}).")

    # -------------------------------------------------------------------------
    # TEST 3: Task 1 - What-If Counterfactual Interventions & Multi-Trajectory
    # -------------------------------------------------------------------------
    engine = RolloutEngine(model, config=config)
    seed_seq = np.random.randn(10, len(feat_names))

    # Without action
    orig_futures = engine.simulate_multiple_futures(seed_seq, k_steps=5, n_samples=20)
    assert len(orig_futures) > 0, "Original trajectory futures cannot be empty"

    # With action (Isolate Host)
    isolated_seq = np.zeros_like(seed_seq)
    mod_futures = engine.simulate_multiple_futures(isolated_seq, k_steps=5, n_samples=20)
    assert len(mod_futures) > 0, "Modified trajectory futures cannot be empty"
    print(f"[OK] 3. What-If Counterfactual Interventions & Multi-Trajectory passed ({len(orig_futures)} baseline vs {len(mod_futures)} intervened paths).")

    print("\n=======================================================")
    print("  ALL NEW CAPABILITIES UNIT TESTS PASSED SUCCESSFULLY!")
    print("=======================================================\n")


if __name__ == "__main__":
    test_new_capabilities()
