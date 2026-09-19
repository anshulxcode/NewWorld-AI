# Detailed Project Report & Feature Analysis — NetWorld-AI
**SIH 2026 NTRO Problem Statement 26153: Temporal Network Traffic Dynamics & Future Attack Forecasting**

---

## 1. Executive Summary & Problem Context

In modern cyber defense operations for national infrastructure (NTRO), traditional Intrusion Detection Systems (IDS) fail because they rely on **static signature matching** or **isolated per-flow classification**. Modern cyber attacks (such as multi-stage APTs, ransomware pivots, and zero-day exploits) unfold over extended time horizons across multiple temporal states.

**NetWorld-AI** addresses this core challenge by building an **AI World Model** that:
1. Learns the continuous temporal dynamics of network traffic.
2. Autoregressively forecasts future threat states $K$-steps into the future before exploitation occurs.
3. Detects zero-day anomalies using unsupervised Autoencoder reconstruction errors and Monte Carlo Dropout confidence bands.
4. Provides live interactive counterfactual ("What-If") model re-inference.
5. Generates copyable, actionable Linux `iptables` CLI mitigation commands mapped to MITRE ATT&CK tactics.

---

## 2. Comprehensive 31-Feature Inventory & Deep-Dive Analysis

NetWorld-AI extracts **31 network, temporal, directional, and reconnaissance features**. Below is the exhaustive feature-by-feature analysis explaining **what it is**, **why we use it**, and **how it makes the system unique**.

### Category A: Flow Level Metrics (Basic Session Dynamics)

#### 1. `src_port` (Source Port Number)
- **Use Case:** Identifies the client-side ephemeral port originating the network request.
- **Why we use it:** Attack tools often use fixed or non-standard source ports (e.g. source port 0 or specific botnet signatures).
- **Uniqueness Value:** Helps sequence windows group multi-connection sessions initiated by the same threat actor.

#### 2. `dst_port` (Destination Port Number)
- **Use Case:** Identifies the target service being accessed (e.g., 80/443 for Web, 22 for SSH, 445 for SMB, 3389 for RDP).
- **Why we use it:** Critical for mapping MITRE ATT&CK tactics (e.g., Port 445 spikes indicate Lateral Movement; Port 22/80 SYN floods indicate Initial Access).
- **Uniqueness Value:** Combined with time-series state rollout to predict when an attacker will pivot from web ports (80) to internal admin ports (445).

#### 3. `protocol` (Transport Protocol)
- **Use Case:** Encodes transport layer protocol (6 = TCP, 17 = UDP, 1 = ICMP).
- **Why we use it:** Attack vectors vary drastically by protocol (TCP SYN floods vs UDP DNS amplification vs ICMP ping sweeps).
- **Uniqueness Value:** Allows the World Model to detect cross-protocol transition shifts during multi-vector attacks.

#### 4. `duration` (Flow Duration)
- **Use Case:** Total active time of a network flow in microseconds.
- **Why we use it:** Short durations (<100us) indicate rapid port probes or RST drops; long durations indicate persistent C2 sessions or heavy data exfiltration.
- **Uniqueness Value:** Enables temporal sequence models to differentiate transient scanning noise from persistent APT footholds.

#### 5. `total_packets` (Total Packet Count)
- **Use Case:** Combined sum of forward and backward packets in the flow.
- **Why we use it:** High packet volume signals volumetric DDoS or continuous brute-force streams.
- **Uniqueness Value:** Acts as a primary state parameter in $K$-step state rollout forecasting.

#### 6. `total_bytes` (Total Transferred Bytes)
- **Use Case:** Aggregated payload byte volume across both communication directions.
- **Why we use it:** Essential for detecting exfiltration attacks and buffer overflow attempts.
- **Uniqueness Value:** Directly informs the Exfiltration rule engine when outbound volume exceeds historical baselines.

#### 7. `packet_rate` (Packets Transferred Per Second)
- **Use Case:** Speed of packet transmission ($\text{total\_packets} / \text{duration\_sec}$).
- **Why we use it:** High packet rates ($>10,000\text{ pkts/s}$) immediately isolate automated flooding tools (LOIC, hping3) from human browsing.
- **Uniqueness Value:** Provides rate-of-change dynamics necessary for derivative threat forecasting.

#### 8. `bytes_rate` (Bytes Transferred Per Second)
- **Use Case:** Data throughput rate ($\text{total\_bytes} / \text{duration\_sec}$).
- **Why we use it:** Detects high-bandwidth data exfiltration bursts.
- **Uniqueness Value:** Used by the Mitigation Engine to automatically build rate-limiting `iptables` rules.

---

### Category B: TCP Flag Counts & Ratios (Protocol Behavior)

#### 9. `syn_count` (SYN Flag Count)
- **Use Case:** Number of TCP Synchronize (SYN) handshake initiation requests.
- **Why we use it:** High SYN count without corresponding ACKs is the textbook signature of SYN Flood DDoS and stealthy port scans.
- **Uniqueness Value:** Serves as a primary input parameter for the interactive Counterfactual What-If Slider in the dashboard.

#### 10. `ack_count` (ACK Flag Count)
- **Use Case:** Number of Acknowledgment (ACK) packets confirming data receipt.
- **Why we use it:** Verifies whether handshakes successfully established legitimate bidirectional TCP sessions.
- **Uniqueness Value:** Used to compute handshake completion ratios.

#### 11. `fin_count` (FIN Flag Count)
- **Use Case:** Number of Finish (FIN) connection termination packets.
- **Why we use it:** Differentiates graceful session termination from abrupt connection resets or orphaned sessions.
- **Uniqueness Value:** Identifies stealthy FIN port scanning techniques (FIN scans bypass simple packet filters).

#### 12. `rst_count` (RST Flag Count)
- **Use Case:** Number of Connection Reset (RST) packets.
- **Why we use it:** High RST counts occur when an attacker scans closed ports or when firewalls block connection attempts.
- **Uniqueness Value:** Helps the Autoencoder detect abnormal network rejection rates.

#### 13. `psh_count` (PSH Flag Count)
- **Use Case:** Number of Push (PSH) packets requesting immediate data buffer flushing to application layer.
- **Why we use it:** Command & Control (C2) beaconing and interactive shell sessions heavily use PSH flags for real-time keystroke transmission.
- **Uniqueness Value:** High PSH frequency combined with low packet sizes strongly signals interactive C2 remote access trojans (RATs).

#### 14. `syn_ack_ratio` (SYN-to-ACK Ratio)
- **Use Case:** Calculated ratio $\text{syn\_count} / \max(\text{ack\_count}, 1)$.
- **Why we use it:** Normal traffic has a balanced ratio ($\approx 1.0$). A ratio $> 5.0$ indicates incomplete handshakes (scanning/flooding).
- **Uniqueness Value:** Normalized metric that remains invariant to overall traffic scale, ensuring robust generalization.

---

### Category C: Inter-Arrival Time (IAT) Metrics (Temporal Rhythm)

#### 15. `iat_mean` (Mean Inter-Arrival Time)
- **Use Case:** Average time interval between consecutive packets in microseconds.
- **Why we use it:** Measures traffic pace and communication frequency.
- **Uniqueness Value:** Establishes the baseline temporal tempo of network sessions.

#### 16. `iat_std` (Standard Deviation of IAT)
- **Use Case:** Variance and volatility of packet arrival timing intervals.
- **Why we use it:** Automated C2 botnet beacons transmit at fixed time intervals ($\text{iat\_std} \approx 0$). Human web activity is random ($\text{iat\_std} \gg 0$).
- **Uniqueness Value:** **Critical C2 Detection Metric:** Low IAT variance is a primary indicator for mapping the MITRE Command & Control tactic.

#### 17. `iat_max` (Maximum Inter-Arrival Time)
- **Use Case:** Longest recorded pause between packets in a flow.
- **Why we use it:** Detects sleep cycles in advanced malware and long-polling HTTP C2 channels.
- **Uniqueness Value:** Captures slow-and-low evasive threat behavior.

#### 18. `iat_min` (Minimum Inter-Arrival Time)
- **Use Case:** Shortest time gap between consecutive packets.
- **Why we use it:** Identifies automated burst packet transmission.
- **Uniqueness Value:** Distinguishes high-speed hardware-generated traffic from software-queued flows.

---

### Category D: Directional Split Metrics (Traffic Asymmetry)

#### 19. `fwd_packets` (Forward Packet Count)
- **Use Case:** Packets originating from source IP toward destination.
- **Why we use it:** Measures outbound request volume.
- **Uniqueness Value:** Forms numerator for directional asymmetry calculation.

#### 20. `bwd_packets` (Backward Packet Count)
- **Use Case:** Response packets returned from destination back to source IP.
- **Why we use it:** Measures inbound response volume.
- **Uniqueness Value:** Absence of backward packets ($0$) confirms non-responsive port scanning or spoofed DDoS.

#### 21. `fwd_bytes` (Forward Byte Volume)
- **Use Case:** Total payload bytes sent in forward direction.
- **Why we use it:** Detects outbound data push.
- **Uniqueness Value:** Used to detect outbound tunnel creation.

#### 22. `bwd_bytes` (Backward Byte Volume)
- **Use Case:** Total payload bytes returned in backward direction.
- **Why we use it:** Detects large data downloads or response amplification.
- **Uniqueness Value:** Helps classify drive-by download attacks versus data exfiltration.

#### 23. `fwd_bwd_ratio` (Forward-to-Backward Packet Ratio)
- **Use Case:** Calculated ratio $\text{fwd\_packets} / \max(\text{bwd\_packets}, 1)$.
- **Why we use it:** Normal interactive web traffic has balanced bidirectional flow. Highly asymmetric ratios ($> 3.0$) indicate unidirectional scanning or exfiltration.
- **Uniqueness Value:** Key feature for distinguishing benign web browsing from malicious lateral probing.

---

### Category E: Packet Sizing Statistics (Payload Characteristics)

#### 24. `avg_packet_size` (Average Packet Size)
- **Use Case:** Mean byte size of packets across the entire flow.
- **Why we use it:** Small average packet sizes ($\approx 40\text{B}$) signal header-only TCP SYN scans. Large packet sizes ($\approx 1460\text{B}$) signal file transfers or video streams.
- **Uniqueness Value:** Provides instant structural classification of traffic type without inspecting encrypted payload contents.

#### 25. `payload_std` (Standard Deviation of Packet Lengths)
- **Use Case:** Variance in packet sizes within a flow.
- **Why we use it:** Fixed packet size variance ($\text{std} = 0$) indicates automated botnet heartbeat packets.
- **Uniqueness Value:** Complements `iat_std` to provide 2D fingerprinting of automated malware channels.

#### 26. `fwd_pkt_len_mean` (Mean Forward Packet Length)
- **Use Case:** Average size of request packets.
- **Why we use it:** Identifies oversized request headers used in HTTP flood DDoS or buffer overflow exploits.
- **Uniqueness Value:** Focuses specifically on attacker-controlled payload sizes.

#### 27. `bwd_pkt_len_mean` (Mean Backward Packet Length)
- **Use Case:** Average size of response packets.
- **Why we use it:** Identifies server error responses (small) vs data retrieval (large).
- **Uniqueness Value:** Measures target server response behavior.

#### 28. `pkt_len_max` (Maximum Packet Length)
- **Use Case:** Largest single packet observed in the flow.
- **Why we use it:** Detects Maximum Transmission Unit (MTU) size saturations ($\approx 1500\text{B}$).
- **Uniqueness Value:** Identifies jumbo frames and data encapsulation overhead.

#### 29. `pkt_len_min` (Minimum Packet Length)
- **Use Case:** Smallest single packet observed in the flow.
- **Why we use it:** Detects empty TCP ACK keep-alives and zero-payload probes.
- **Uniqueness Value:** Establishes lower boundary for packet sizing distribution.

---

### Category F: Reconnaissance & Advanced Anomaly Metrics

#### 30. `unique_dst_ports_per_src` (Target Port Diversity per Source IP)
- **Use Case:** Tracks how many distinct destination ports a single source IP touches within a temporal window.
- **Why we use it:** **Direct Port Scan Metric:** A normal client touches 1–3 ports (80, 443, 53). A port scanner touches dozens or hundreds of ports.
- **Uniqueness Value:** Instantly flags horizontal and vertical network sweeps regardless of packet rate.

#### 31. `recon_score` (Composite Reconnaissance Score)
- **Use Case:** Custom engineered composite metric: $0.6 \times \text{norm}(\text{unique\_dst\_ports}) + 0.4 \times \text{syn\_ratio}$.
- **Why we use it:** Fuses port diversity with TCP handshake behavior into a unified 0.0–1.0 score.
- **Uniqueness Value:** **Proprietary Feature:** Built specifically for NetWorld-AI to provide robust early warning before an attacker transitions from scanning to exploitation.

---

## 3. Why NetWorld-AI is Unique (USPs Breakdown)

| Feature / USP | Traditional Systems (Snort, Suricata, Basic ML) | NetWorld-AI (Our Solution) | Why It Matters for NTRO / SIH 2026 |
| :--- | :--- | :--- | :--- |
| **Temporal Dynamic Learning** | Evaluates isolated individual packets or static flow records. | Learns continuous state transitions using a 3D Attention-LSTM sequence model. | Captures multi-stage APT attack progression across time. |
| **Future Attack Forecasting** | Reactive: Alerts ONLY after attack has occurred. | Proactive: Autoregressively forecasts threat states $K$-steps ahead ($K=1..5$). | Gives security analysts time to block threats BEFORE exploitation. |
| **Zero-Day Novelty Detection** | Requires signature updates or labeled attack data. | Unsupervised Autoencoder with Monte Carlo Dropout 95% confidence bands. | Detects unknown zero-day attacks without needing prior signatures. |
| **Interactive Counterfactual Engine** | Static report output; no user interactivity. | Live model re-inference on slider input ($\Delta \text{Risk}$). | Allows analysts to test "What-If" security scenarios in real time. |
| **Automated Actionable Mitigation** | Generic textual alerts (e.g. "Suspicious traffic detected"). | Maps stages to MITRE ATT&CK and outputs copyable Linux `iptables` CLI commands. | Enables automated, zero-delay firewall response. |
| **Data Leakage Prevention** | Random cross-validation splits attack episodes into train & test. | Grouped Episode Time Split (`grouped_time_split`) keeps episodes intact. | Guarantees genuine, un-cheated evaluation metrics (100% realistic). |

---

## 4. Summary Table of Files Generated

1. [`config/config.yaml`](file:///d:/SIH%20project/networld-ai/config/config.yaml) — Hyperparameters and 31-feature schema.
2. [`src/utils.py`](file:///d:/SIH%20project/networld-ai/src/utils.py) — Seed reproducibility, YAML config loader, logging.
3. [`src/data/csv_loader.py`](file:///d:/SIH%20project/networld-ai/src/data/csv_loader.py) — 31-feature extraction, CIC-IDS2018 mapping, StandardScaler export.
4. [`src/data/pcap_parser.py`](file:///d:/SIH%20project/networld-ai/src/data/pcap_parser.py) — Scapy packet header parser and 4-tuple flow aggregator.
5. [`src/data/window_builder.py`](file:///d:/SIH%20project/networld-ai/src/data/window_builder.py) — 3D sequence tensor builder & grouped episode time splitter.
6. [`src/models/lstm_world.py`](file:///d:/SIH%20project/networld-ai/src/models/lstm_world.py) — Attention-LSTM World Model with $K$-step rollout.
7. [`src/models/autoencoder.py`](file:///d:/SIH%20project/networld-ai/src/models/autoencoder.py) — Novelty Autoencoder with LayerNorm & MC Dropout.
8. [`src/models/baseline_lr.py`](file:///d:/SIH%20project/networld-ai/src/models/baseline_lr.py) — Logistic Regression baseline classifier.
9. [`src/train.py`](file:///d:/SIH%20project/networld-ai/src/train.py) — Main multi-task model training pipeline.
10. [`src/evaluate.py`](file:///d:/SIH%20project/networld-ai/src/evaluate.py) — Model benchmarking and comparison metrics.
11. [`src/forecasting/rollout.py`](file:///d:/SIH%20project/networld-ai/src/forecasting/rollout.py) — Autoregressive forecast rollout simulator.
12. [`src/forecasting/stage_mapper.py`](file:///d:/SIH%20project/networld-ai/src/forecasting/stage_mapper.py) — MITRE ATT&CK tactic mapper.
13. [`src/explainability/shap_explainer.py`](file:///d:/SIH%20project/networld-ai/src/explainability/shap_explainer.py) — SHAP feature attributions & live counterfactual engine.
14. [`src/explainability/mitigation_engine.py`](file:///d:/SIH%20project/networld-ai/src/explainability/mitigation_engine.py) — Actionable `iptables` mitigation rule generator.
15. [`app/streamlit_app.py`](file:///d:/SIH%20project/networld-ai/app/streamlit_app.py) — 5-Tab interactive Streamlit dashboard.
16. [`docs/architecture.md`](file:///d:/SIH%20project/networld-ai/docs/architecture.md) — Technical specifications and Mermaid DFDs.
17. [`run.bat`](file:///d:/SIH%20project/networld-ai/run.bat) & [`run.sh`](file:///d:/SIH%20project/networld-ai/run.sh) — One-click execution scripts.
