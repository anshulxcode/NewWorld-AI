"""
Test Suite for NetWorld-AI Phase 3 (Application Layer).
Tests RolloutEngine, StageMapper, SHAPExplainer, MitigationEngine, and Streamlit app compilation.
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.utils import set_seed, load_config
from src.models import LSTMWorldModel, Autoencoder
from src.forecasting import RolloutEngine, StageMapper
from src.explainability import SHAPExplainer, MitigationEngine


def test_phase3():
    print("\n=======================================================")
    print("  NETWORLD-AI PHASE 3 APPLICATION LAYER TEST SUITE")
    print("=======================================================\n")

    set_seed(42)
    config = load_config("config/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feat_names = config.get("features", [])
    model = LSTMWorldModel(input_size=len(feat_names), hidden_size=64).to(device)
    ae = Autoencoder(input_dim=len(feat_names)).to(device)

    # 1. Test StageMapper
    mapper = StageMapper()
    sample_vec = np.zeros(len(feat_names))
    stage, conf, ev = mapper.map_stage(sample_vec, feat_names)
    assert stage == "BENIGN / Normal Traffic", "Stage mapping baseline failed"
    print("[OK] 1. StageMapper test passed.")

    # 2. Test RolloutEngine & Multi-Trajectory Forecasting
    engine = RolloutEngine(model, config=config, autoencoder=ae, stage_mapper=mapper)
    seed_seq = np.random.randn(10, len(feat_names))
    results = engine.simulate(seed_seq, k_steps=5)
    assert len(results) == 5, f"Expected 5 rollout steps, got {len(results)}"
    assert "stage" in results[0], "Rollout item missing stage key"

    multi_futures = engine.simulate_multiple_futures(seed_seq, k_steps=5, n_samples=20)
    assert len(multi_futures) > 0, "Multi-trajectory futures cannot be empty"
    prob_sum = sum(f["probability"] for f in multi_futures)
    assert abs(prob_sum - 1.0) < 1e-5, f"Trajectory probabilities must sum to 1.0, got {prob_sum}"
    print("[OK] 2. RolloutEngine & Multi-Trajectory Forecasting (MC Dropout N=20) test passed.")

    # 3. Test SHAPExplainer & Counterfactual Re-inference
    explainer = SHAPExplainer(model, feat_names)
    top_5 = explainer.explain(seed_seq)
    assert len(top_5) == 5, f"Expected top 5 features, got {len(top_5)}"

    cf_res = explainer.counterfactual(seed_seq, feature_name_or_idx=0, new_value=10.0)
    assert "delta" in cf_res, "Counterfactual result missing delta key"
    print("[OK] 3. SHAPExplainer & Counterfactual Re-inference test passed.")

    # 4. Test MitigationEngine
    mit_engine = MitigationEngine()
    mit_rec = mit_engine.get_recommendation("Lateral Movement", top_5, predicted_risk=0.85)
    assert "iptables_rule" in mit_rec, "Mitigation recommendation missing iptables_rule key"
    assert "iptables" in mit_rec["iptables_rule"], "Invalid iptables command generated"
    print("[OK] 4. MitigationEngine test passed.")

    # 5. Check Streamlit App Compilation
    import py_compile
    py_compile.compile("app/streamlit_app.py")
    print("[OK] 5. Streamlit App (app/streamlit_app.py) syntax & compilation passed.")

    print("\n=======================================================")
    print("  ALL PHASE 3 TESTS PASSED SUCCESSFULLY!")
    print("=======================================================\n")


if __name__ == "__main__":
    test_phase3()
