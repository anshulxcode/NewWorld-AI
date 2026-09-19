# NetWorld-AI: Temporal Network Traffic Dynamics & Future Attack Forecasting
**Smart India Hackathon (SIH) 2026 — NTRO Problem Statement 26153**

NetWorld-AI is an advanced artificial intelligence system designed for the National Technical Research Organisation (NTRO). It learns temporal dynamics of multi-vector network traffic and autoregressively forecasts future cyber attack states $K$-steps into the future before exploitation occurs.

---

## Technical Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                           NETWORLD-AI SYSTEM PIPELINE                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
      ┌────────────────────────────────┴────────────────────────────────┐
      ▼                                                                 ▼
[ Raw Network CSV ]                                             [ Raw PCAP Packet Trace ]
(CIC-IDS2018 Schema)                                             (Scapy Header Parser)
      │                                                                 │
      └────────────────────────────────┬────────────────────────────────┘
                                       │
                                       ▼
                       [ CSVLoader / PCAPParser Module ]
                Extracts 37 Standardized Network Metrics (Packet TTL, Window, Entropy)
                Cleans NaNs, Infs & Scaled via StandardScaler
                                       │
                                       ▼
                          [ WindowBuilder Module ]
                Aggregates Flows into Fixed State Windows
                Constructs 3D Sequence Tensors (N, 10, 37)
                Grouped Episode Split (Zero Data Leakage)
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
 [ Attention-LSTM World Model ]  [ Autoencoder Model ]   [ Baseline LR Model ]
 Multi-Task State Forecast,     Zero-Day Anomaly &     Logistic Regression
 7-Class MITRE Stage & Risk     Novelty Detection      Classifier Benchmark
            │                          │                          │
            └──────────────────────────┼──────────────────────────┘
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
 [ Streamlit Web Dashboard ]                             [ FastAPI REST API ]
   Interactive 5-Tab Interface                             Headless SIEM / Firewall
   (Risk Gauge, SHAP, Slider)                              Integration (/api/v1/predict)
```

---

## Key Features & Unique Selling Propositions (USPs)

1. **Autoregressive $K$-Step Threat Forecasting:** Predicts future state transitions ($K=1..5$ steps ahead) using an Attention-LSTM World Model.
2. **Multi-Class MITRE ATT&CK Stage Mapping:** Classifies 7 operational attack stages (Reconnaissance, Initial Access, Discovery, Lateral Movement, C2, Exfiltration, Benign).
3. **Zero-Day Novelty Detection:** Unsupervised Autoencoder trained strictly on `BENIGN` traffic computes reconstruction errors and Monte Carlo (MC) Dropout confidence bands.
4. **Live Counterfactual "What-If" Simulator (NTRO USP):** Interactive UI sliders perform real-time forward passes through the model to calculate instant risk deltas ($\Delta \text{Risk}$).
5. **Actionable Automated Mitigation:** Translates predicted MITRE ATT&CK stages into severity levels, remediation playbooks, copyable Linux `iptables` CLI rules, and Snort/Suricata signature templates.
6. **Headless SIEM & Firewall REST API:** Enterprise FastAPI endpoints for real-time traffic sequence evaluation and automated defensive execution.

---

## Directory Structure

```text
networld-ai/
├── app/
│   └── streamlit_app.py         # Main Interactive Streamlit Dashboard (5 Tabs)
├── src/
│   ├── data/
│   │   ├── csv_loader.py         # CSV Loader & Feature Extraction (37 features + entropy)
│   │   ├── pcap_parser.py        # Scapy PCAP Header Parser (TTL, Window, Flags)
│   │   └── window_builder.py     # 3D Sequence Tensor Builder & Multi-Class Stage Mapping
│   ├── models/
│   │   ├── lstm_world.py         # Attention-LSTM World Model (Next-State, Stage & Risk Heads)
│   │   ├── autoencoder.py        # Novelty Autoencoder with LayerNorm & MC Dropout
│   │   └── baseline_lr.py        # Baseline Logistic Regression Classifier
│   ├── forecasting/
│   │   ├── rollout.py            # Autoregressive K-Step Rollout Engine
│   │   └── stage_mapper.py       # 7-Class MITRE ATT&CK Stage Mapper
│   ├── explainability/
│   │   ├── shap_explainer.py     # SHAP Attributions & Live Counterfactual Engine
│   │   └── mitigation_engine.py  # iptables & Suricata Mitigation Rule Generator
│   ├── api.py                    # FastAPI REST API Engine for SIEM Integration
│   ├── simulate_live_stream.py   # Live Telemetry Attack Stream Simulator
│   ├── utils.py                  # Logger & Config Loader
│   ├── generate_sample_data.py   # Synthetic Dataset Generator
│   ├── train.py                  # Multi-task Training Pipeline
│   └── evaluate.py               # Model Benchmarking & Metric Exporter
├── config/
│   └── config.yaml               # 37 Features, Hyperparameters & Paths
├── models/                       # Trained Models (.pt, .joblib, comparison.csv)
├── data/                         # Raw, Sample & Processed Datasets
├── report_of_project.md          # Comprehensive Project Report
├── Dockerfile                    # Containerization Build Specification
├── docker-compose.yml            # Multi-service Deployment Orchestrator
├── run.bat                       # One-click Windows launcher
├── run.sh                        # One-click Linux/macOS launcher
└── requirements.txt              # Dependencies
```

---

## Deployment & Quick Start Guide

### Option 1: Docker (Recommended for Hackathon Demo)
```bash
docker-compose up --build
```
- **Streamlit Dashboard:** `http://localhost:8501`
- **FastAPI REST Docs:** `http://localhost:8000/docs`

### Option 2: Local Python Environment
```bash
# 1. Setup Virtual Environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/macOS

# 2. Install Dependencies
pip install -r requirements.txt

# 3. Train & Evaluate
python src/train.py
python src/evaluate.py

# 4. Launch Streamlit UI
streamlit run app/streamlit_app.py

# 5. Launch REST API (Optional)
uvicorn src/api.py:app --reload --port 8000

# 6. Run Live Attack Telemetry Simulator (Optional Demo)
python src/simulate_live_stream.py
```

---

## Datasets
NetWorld-AI is benchmarked on standard cyber security datasets:
1. **CIC-IDS2018 Dataset:** UNB CIC Dataset Download
2. **CTU-13 Dataset:** CTU University Botnet & Traffic Traces
