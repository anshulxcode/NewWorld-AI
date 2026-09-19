"""
FastAPI REST API for NetWorld-AI (Phase 3).
Provides programmatic SIEM / Firewall integration endpoints for real-time flow risk scoring,
K-step threat rollout forecasting, and automated iptables mitigation generation.
"""

import os
import sys
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import load_config
from src.models import LSTMWorldModel, Autoencoder
from src.forecasting import RolloutEngine, StageMapper
from src.explainability import SHAPExplainer, MitigationEngine

app = FastAPI(
    title="NetWorld-AI Threat Forecasting API",
    description="SIH 2026 NTRO PS 26153 — REST API for Temporal Dynamics Learning & K-Step Attack Forecasting",
    version="1.0.0"
)

# Global model state
config = load_config("config/config.yaml") if os.path.exists("config/config.yaml") else {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lstm_model = None
autoencoder = None
rollout_engine = None
explainer = None
mitigation_engine = MitigationEngine()
feat_names = config.get("features", [])


@app.on_event("startup")
def load_artifacts():
    global lstm_model, autoencoder, rollout_engine, explainer
    models_dir = config.get("paths", {}).get("model_path", "models")
    lstm_path = os.path.join(models_dir, "best_lstm.pt")

    if os.path.exists(lstm_path):
        ckpt = torch.load(lstm_path, map_location=device)
        input_size = ckpt.get("input_size", len(feat_names))
        hidden_size = ckpt.get("hidden_size", 128)
        lstm_model = LSTMWorldModel(input_size=input_size, hidden_size=hidden_size).to(device)
        lstm_model.load_state_dict(ckpt["model_state_dict"])
        lstm_model.eval()

        ae_path = os.path.join(models_dir, "autoencoder.pt")
        if os.path.exists(ae_path):
            ae_ckpt = torch.load(ae_path, map_location=device)
            autoencoder = Autoencoder(input_dim=input_size).to(device)
            autoencoder.load_state_dict(ae_ckpt["model_state_dict"])
            autoencoder.eval()

        rollout_engine = RolloutEngine(lstm_model, config=config, autoencoder=autoencoder)
        explainer = SHAPExplainer(lstm_model, feat_names)


class SequenceInput(BaseModel):
    sequence: List[List[float]] # (10, num_features)
    k_steps: Optional[int] = 5


@app.get("/")
def root():
    return {
        "system": "NetWorld-AI Threat Forecasting API",
        "status": "ONLINE",
        "model_loaded": lstm_model is not None,
        "features_count": len(feat_names)
    }


@app.post("/api/v1/predict")
def predict_threat(payload: SequenceInput):
    """
    Evaluates threat risk and forecasts K-step future rollout from input sequence.
    """
    if lstm_model is None:
        raise HTTPException(status_code=503, detail="LSTM model checkpoint not loaded.")

    seq_arr = np.array(payload.sequence, dtype=np.float32)
    if seq_arr.ndim != 2:
        raise HTTPException(status_code=400, detail="Expected 2D sequence matrix of shape (seq_len, num_features).")

    rollout_results = rollout_engine.simulate(seq_arr, k_steps=payload.k_steps)
    top_features = explainer.explain(seq_arr)
    mitigation = mitigation_engine.get_recommendation(rollout_results[0]["stage"], top_features, rollout_results[0]["risk"])

    return {
        "current_risk": rollout_results[0]["risk"],
        "current_stage": rollout_results[0]["stage"],
        "forecast_rollout": rollout_results,
        "top_shap_features": top_features,
        "mitigation": mitigation
    }
