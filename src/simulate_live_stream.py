"""
Live Traffic Stream Simulator for NetWorld-AI (SIH 2026).
Simulates real-time streaming network flow window sequence submissions to the FastAPI REST API
or directly through the Rollout Engine to demonstrate live threat progression forecasting.
"""

import time
import requests
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import load_config, setup_logger

logger = setup_logger("StreamSimulator")

API_URL = "http://localhost:8000/api/v1/predict"


def generate_synthetic_window(seq_len=10, num_features=37, attack_intensity=0.0):
    """
    Generates a synthetic sequence matrix representing benign or progressing attack traffic.
    """
    base_flow = np.random.normal(loc=0.0, scale=1.0, size=(seq_len, num_features))
    if attack_intensity > 0:
        # Inject attack signals into specific features (e.g. flow duration, packet rate, entropy)
        base_flow[:, :5] += attack_intensity * np.random.uniform(1.5, 3.0, size=(seq_len, 5))
        base_flow[:, 10:15] += attack_intensity * np.random.uniform(2.0, 4.0, size=(seq_len, 5))
    return base_flow.tolist()


def run_live_simulation(steps=10, delay=1.5):
    """
    Simulates a live multi-stage attack progression over time.
    """
    print("\n" + "=" * 65)
    print(" 🚀 NETWORLD-AI: LIVE ATTACK TELEMETRY STREAM SIMULATOR")
    print("=" * 65)

    stages = [
        ("BENIGN", 0.0),
        ("RECONNAISSANCE (Port Scan)", 0.2),
        ("INITIAL ACCESS (Brute Force)", 0.5),
        ("LATERAL MOVEMENT (RDP/SMB)", 0.8),
        ("EXFILTRATION / C2", 1.0)
    ]

    for step, (stage_desc, intensity) in enumerate(stages, 1):
        print(f"\n[TIME STEP t={step}] Ingesting Flow Sequence | Target State: {stage_desc}")
        seq = generate_synthetic_window(seq_len=10, num_features=37, attack_intensity=intensity)

        try:
            response = requests.post(API_URL, json={"sequence": seq, "k_steps": 5}, timeout=5)
            if response.status_code == 200:
                res = response.json()
                print(f"  └─ Current Risk Score: {res['current_risk'] * 100:.1f}%")
                print(f"  └─ Forecast MITRE Stage: {res['current_stage']}")
                print(f"  └─ Top Explaining Feature: {res['top_shap_features'][0]['feature'] if res['top_shap_features'] else 'N/A'}")
                print(f"  └─ Recommended Action: {res['mitigation']['action']}")
            else:
                print(f"  └─ API Error ({response.status_code}): {response.text}")
        except Exception as e:
            print(f"  └─ Connection failed to API ({API_URL}). Ensure `python src/api.py` or uvicorn is running. Error: {e}")

        time.sleep(delay)

    print("\n" + "=" * 65)
    print(" ✅ Simulation Complete.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_live_simulation()
