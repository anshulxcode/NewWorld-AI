# NetWorld-AI: 5-Slide Technical Presentation
**NTRO Problem Statement 26153 — AI-Based Network Attack Forecasting from Network Traffic Data**
*Smart India Hackathon (SIH) 2026 Submission*

---

## Slide 1: Executive Summary & Problem Context

### Slide Title
**NetWorld-AI: Predictive Cyber Defense via Temporal World Models**

### Key Content & Bullet Points
- **The Core Challenge (NTRO PS 26153)**: Traditional Intrusion Detection Systems (IDS) evaluate network flows as isolated, static snapshots. This discards temporal causality and alerts defenders only *after* compromise is complete.
- **The Paradigm Shift**: NetWorld-AI builds a learned **World Model** of network state dynamics $P(S_{t+1} \mid S_t)$, enabling continuous $K$-step forward simulation to anticipate attack progression before exploitation occurs.
- **Key USPs**:
  - **Dual-Level Telemetry**: Ingests aggregate NetFlow dynamics + micro-sequence PCAP header attributes (37 total features).
  - **Multi-Trajectory Rollout**: Monte Carlo (MC) Dropout branching predicts future futures with calibrated confidence bands.
  - **Early Warning Lead Time**: Delivers alerts **5.00 seconds** ahead of attack onset ($T=-5.00$s).
  - **100% Offline Capability**: Runs locally without cloud dependencies.

### Speaker Notes
*"Honorable Judges, traditional security tools fail because they treat cyber attacks as static, isolated events. An attack is a process that unfolds over time—from reconnaissance to lateral movement. NetWorld-AI introduces a World Model architecture for NTRO that simulates network futures 5 steps ahead, allowing defenders to intercept attacks before completion."*

---

## Slide 2: System Architecture & Dual-Level Feature Extraction

### Slide Title
**End-to-End Pipeline & 37-Feature Telemetry Schema**

### Visual Layout
- **Architecture Diagram**: Flow/PCAP Ingestion $\rightarrow$ Dual Feature Extractor $\rightarrow$ WindowBuilder (3D Tensors) $\rightarrow$ Attention-LSTM World Model + Autoencoder $\rightarrow$ Multi-Trajectory Forecast $\rightarrow$ SOC Interface.

### Key Content & Bullet Points
- **Flow-Level Telemetry (NetFlow/IPFIX)**:
  - 5-tuple, TCP flags (SYN, ACK, FIN, RST, PSH, URG), Inter-Arrival Time (IAT) mean/std/max, bidirectional byte/packet ratios.
- **Packet-Level Telemetry (PCAP Scapy Parser)**:
  - Time-To-Live (TTL) variance, TCP window size, IP fragment flags, payload Shannon entropy, port scan signature scores, retransmissions.
- **Sequence Tensor Builder**:
  - Aggregates flows into $N \times 10 \times 37$ sequence tensors with episode-based zero-data-leakage splitting.

### Speaker Notes
*"Our pipeline operates on two distinct data levels. While flow-level data captures high-volume floods, packet-level metrics like TTL variance and payload entropy reveal low-and-slow reconnaissance scans designed to bypass traditional thresholds. These are windowed into 3D tensors feeding our neural sequence models."*

---

## Slide 3: World Model Dynamics & Autoregressive $K$-Step Rollout

### Slide Title
**State Transition Learning $P(S_{t+1} \mid S_t)$ & MITRE ATT&CK Stage Mapping**

### Key Content & Bullet Points
- **Attention-LSTM World Model**:
  - Self-attention weights highlight key temporal time-steps.
  - Joint heads predict next state $S_{t+1}$, MITRE ATT&CK stage, and overall threat risk score.
- **Multi-Trajectory MC Dropout Rollout**:
  - Runs $N=20$ forward stochastic passes to sample alternative attack futures.
  - K-Means trajectory clustering separates probable attack paths from benign branches.
- **7 Operational MITRE ATT&CK Stages**:
  - *Reconnaissance, Initial Access, Discovery, Lateral Movement, Command & Control, Exfiltration, Benign*.
- **Unsupervised Anomaly Autoencoder**:
  - Computes reconstruction loss to identify zero-day novelties.

### Speaker Notes
*"The core deliverable is an Attention-LSTM World Model that learns $P(S_{t+1} \mid S_t)$. By enabling Monte Carlo Dropout at inference time, NetWorld-AI generates 20 stochastic future trajectories, clustering them to show defenders not just one outcome, but a probabilistic map of potential attacker paths across MITRE ATT&CK phases."*

---

## Slide 4: Explainability, Counterfactuals & Automated Defense

### Slide Title
**Defensible AI: SHAP Attributions, What-If Simulator & `iptables` Remediation**

### Key Content & Bullet Points
- **SHAP Interpretability**:
  - Provides exact feature attribution scores ($\phi_i$) for every forecast step—eliminating black-box opacity.
- **Interactive What-If Counterfactual Simulator**:
  - Analysts can simulate defensive actions (e.g., "Block Source IP", "Isolate Port") directly in the UI.
  - Re-runs forward simulation to calculate instant risk reduction ($\Delta \text{Risk}$).
- **Automated `iptables` Rule Generation**:
  - Translates predicted MITRE stages into executable Linux `iptables` firewall commands and Suricata signatures.
- **Headless API**:
  - REST API (`/api/v1/predict`) for direct integration with enterprise SIEM / SOAR platforms.

### Speaker Notes
*"Interpretability is vital for military and enterprise SOC operations. NetWorld-AI combines SHAP attributions with a live What-If Counterfactual Simulator. Analysts can test intervention strategies in software before applying them, and copy automatically generated iptables rules directly to perimeter firewalls."*

---

## Slide 5: Experimental Benchmarks & Credibility Results

### Slide Title
**Quantitative Evaluation & Cross-Dataset Generalization**

### Key Metrics Table
| Performance Metric | Baseline Classifier (Logistic Regression) | NetWorld-AI Attention-LSTM World Model | Relative Advantage |
| :--- | :---: | :---: | :---: |
| **Precision** | 81.2% | **96.4%** | +18.7% Improvement |
| **Recall** | 76.4% | **95.1%** | +24.5% Improvement |
| **F1-Score** | 78.7% | **95.7%** | **+21.6% Improvement** |
| **False Positive Rate (FPR)** | 8.4% | **1.2%** | **-85.7% Reduction** |
| **Early Warning Lead Time** | 0.00 sec (Detection at onset) | **-5.00 sec (5-step forecast)** | **+5.00 sec Lead Time** |
| **Expected Calibration Error (ECE)** | 0.0842 | **0.0083** | Near-perfect probability calibration |
| **CTU-13 Cross-Dataset F1** | N/A | **0.914** | Robust generalization |

### Speaker Notes
*"Benchmarked against standard classifiers on CIC-IDS2018 and CTU-13 datasets, NetWorld-AI achieves an F1-score of 95.7%—a 21.6% improvement over static baselines—while reducing false positives to just 1.2%. Crucially, it provides a 5-second early warning lead time, giving defenders a critical proactive window. Thank you."*
