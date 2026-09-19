# NetWorld-AI: 2-Minute Demo Video Script
**NTRO Problem Statement 26153 — AI-Based Network Attack Forecasting**

---

## Video Specifications
- **Total Duration:** 2 Minutes (120 Seconds)
- **Target Audience:** NTRO Technical Evaluation Committee & SIH 2026 Jury
- **Audio:** Clear voiceover with subtle background ambient tech audio
- **Visual:** High-definition 1080p screen recording of the dark Streamlit SOC Dashboard (`http://localhost:8501`)

---

## Second-by-Second Breakdown Script

### [0:00 - 0:20] Segment 1: The Problem & Solution Overview
* **Visual:** Fade in on the NetWorld-AI SOC Dashboard header: `NETWORLD-AI // SECURITY OPERATIONS CENTER`. Show the dense 3-card metric panel (`CURRENT RISK SCORE: 0.892 (HIGH)`, `ANOMALY RECONSTRUCTION LOSS: 0.0412`, `MAX FORECAST PROB: 0.941`).
* **Voiceover:**
  > "Traditional intrusion detection systems evaluate network traffic in isolation, firing alerts only *after* compromise occurs. For NTRO Problem Statement 26153, we present NetWorld-AI: an Attention-LSTM World Model that learns temporal traffic state transition dynamics $P(S_{t+1} \mid S_t)$ to forecast cyber attacks up to 5 steps into the future before the kill chain is completed."

---

### [0:20 - 0:45] Segment 2: Dual-Level Feature Extraction & Temporal Rollout
* **Visual:** Navigate to **Tab 1: Network State** and **Tab 2: K-Step Threat Rollout**. Cursor hovers over the 37-feature NetFlow + PCAP stream matrix (TTL variance, TCP window size, port scan scores). Transition to the Plotly trajectory forecast timeline showing current time $T=0$ and forward rollout steps $T=+1$ to $T=+5$.
* **Voiceover:**
  > "NetWorld-AI ingests dual-level traffic telemetry—combining NetFlow aggregate dynamics with packet-level PCAP metrics like TTL variance and payload entropy. Using autoregressive forward simulation and Monte Carlo Dropout, our model generates multi-trajectory future branches, predicting an upcoming infiltration 5 seconds ahead of time with zero data leakage."

---

### [0:45 - 1:10] Segment 3: MITRE ATT&CK Mapping & Host Baselining
* **Visual:** Highlight the **MITRE ATT&CK Stage Timeline** transitioning from *Reconnaissance* $\rightarrow$ *Initial Access* $\rightarrow$ *Lateral Movement*. Show the **Host Baseline Deviation Monitor** displaying real-time EWMA $Z$-scores (`192.168.10.15` | $Z=4.21$ | `HIGH RISK`).
* **Voiceover:**
  > "As network state transitions evolve, NetWorld-AI automatically maps trajectory states to seven recognized MITRE ATT&CK tactics. Simultaneously, our per-host EWMA baseline engine calculates statistical $Z$-score anomalies to catch zero-day lateral pivots."

---

### [1:10 - 1:35] Segment 4: Explainability & What-If Counterfactual Simulator
* **Visual:** Switch to **Tab 3: Explainability & Counterfactual Simulator**. Point out the SHAP feature attribution bar chart showing top driving features (`syn_count`, `port_scan_recon_score`). Click the **Block Source IP** counterfactual button. Watch the live trajectory risk score drop instantly from `0.892` down to `0.114`.
* **Voiceover:**
  > "To ensure full interpretability for defenders, SHAP attributions highlight the exact packet flags and port ratios driving the prediction. Security analysts can test active interventions using our live What-If Counterfactual Simulator—seeing instant risk reductions when isolating a host or dropping port traffic."

---

### [1:35 - 2:00] Segment 5: Automated Defense, API Integration & Benchmarking
* **Visual:** Show **Tab 4: Automated Defense** with copyable Linux `iptables` CLI commands and Suricata rules. Quickly flip to **Tab 5: Model Credibility** showing the F1 score comparison table (World Model `0.957` vs Baseline `0.787`), ECE calibration curve (`0.0083`), and CTU-13 cross-dataset test.
* **Voiceover:**
  > "For proactive defense, NetWorld-AI automatically outputs execution-ready `iptables` firewall rules and headless REST API endpoints. Benchmarked against standard classifiers, our World Model delivers a 21.6% higher F1 score, an 85% lower false positive rate, and a 5-second early warning lead time. NetWorld-AI transforms cyber defense from reactive logging to predictive protection. Thank you."
