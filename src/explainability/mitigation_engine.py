"""
Actionable Automated Mitigation Engine for NetWorld-AI (NTRO USP).
Translates forecasted MITRE ATT&CK stages, risk probabilities, and top SHAP features
into automated Linux iptables commands, mitigation actions, severity levels, and defensive rationale.
"""

from typing import List, Dict, Any, Tuple


class MitigationEngine:
    """
    Translates threat predictions into actionable network defense playbooks and iptables commands.
    """

    def __init__(self) -> None:
        """Initializes MitigationEngine rules."""
        pass

    def get_recommendation(
        self,
        stage: str,
        top_features: List[Tuple[str, float]],
        predicted_risk: float
    ) -> Dict[str, Any]:
        """
        Generates automated mitigation recommendation, iptables command, and defensive rationale.

        Args:
            stage: Current/predicted MITRE ATT&CK tactic string.
            top_features: List of top SHAP feature tuples [(feature_name, importance_score), ...].
            predicted_risk: Model predicted threat risk probability (0.0 to 1.0).

        Returns:
            Dict[str, Any]:
                - stage: Tactic name
                - severity: LOW / MEDIUM / HIGH / CRITICAL
                - action: Prescribed remediation action summary
                - iptables_rule: Copyable Linux iptables CLI command
                - rationale: Detailed defensive rationale text
        """
        # Determine Severity
        if predicted_risk < 0.3:
            severity = "LOW"
        elif predicted_risk < 0.6:
            severity = "MEDIUM"
        elif predicted_risk < 0.85:
            severity = "HIGH"
        else:
            severity = "CRITICAL"

        # Safe default for benign traffic
        if stage == "BENIGN / Normal Traffic" or predicted_risk < 0.25:
            return {
                "stage": "BENIGN / Normal Traffic",
                "severity": "NORMAL",
                "action": "No immediate block required. Maintain standard perimeter monitoring.",
                "iptables_rule": "# Traffic baseline normal - No active iptables rule required",
                "rationale": "Forecasted network metrics remain within standard baseline operational ranges."
            }

        top_feat_names = [f[0] for f in top_features] if top_features else []

        # 1. Reconnaissance Defense
        if "Reconnaissance" in stage or "syn_count" in top_feat_names:
            return {
                "stage": "Reconnaissance",
                "severity": severity,
                "action": "Throttle port scanning and block aggressive SYN packet floods.",
                "iptables_rule": "iptables -A INPUT -p tcp --syn -m limit --limit 1/s --limit-burst 3 -j ACCEPT",
                "rationale": f"Predicted reconnaissance scan (Risk: {predicted_risk*100:.1f}%) driven by anomalous SYN packet frequency."
            }

        # 2. Initial Access / Brute Force Defense
        if "Initial Access" in stage or "dst_port" in top_feat_names:
            return {
                "stage": "Initial Access",
                "severity": severity,
                "action": "Rate-limit SSH/RDP/Auth authentication attempts and block suspicious source IP.",
                "iptables_rule": "iptables -A INPUT -p tcp --dport 22 -m state --state NEW -m recent --set --name SSH\niptables -A INPUT -p tcp --dport 22 -m state --state NEW -m recent --update --seconds 60 --hitcount 4 -j DROP",
                "rationale": f"Forecasted authentication brute-force attempt (Risk: {predicted_risk*100:.1f}%) targeting administrative interfaces."
            }

        # 3. Lateral Movement Defense
        if "Lateral Movement" in stage:
            return {
                "stage": "Lateral Movement",
                "severity": severity,
                "action": "Isolate internal subnet segment and drop unauthorized SMB (Port 445) and RDP (Port 3389) traffic.",
                "iptables_rule": "iptables -A FORWARD -p tcp -m multiport --dports 445,135,139,3389 -j DROP",
                "rationale": f"Predicted lateral movement pivot attempt (Risk: {predicted_risk*100:.1f}%) targeting internal network services."
            }

        # 4. Command & Control Defense
        if "Command & Control" in stage or "iat_std" in top_feat_names:
            return {
                "stage": "Command & Control",
                "severity": severity,
                "action": "Terminate active beaconing session and block C2 channel IP.",
                "iptables_rule": "iptables -A OUTPUT -p tcp -m state --state ESTABLISHED -m string --algo bm --string 'c2_beacon' -j REJECT",
                "rationale": f"Periodic C2 beaconing pattern detected with {predicted_risk*100:.1f}% risk score confidence."
            }

        # 5. Exfiltration Defense
        if "Exfiltration" in stage or "bytes_rate" in top_feat_names or "total_bytes" in top_feat_names:
            return {
                "stage": "Exfiltration",
                "severity": severity,
                "action": "Drop high-volume outbound data transfers exceeding perimeter bandwidth policy.",
                "iptables_rule": "iptables -A FORWARD -p tcp --dport 80,443 -m connbytes --connbytes 50000000: --connbytes-dir original --connbytes-mode bytes -j DROP",
                "rationale": f"Abnormal high-speed outbound data exfiltration stream detected ({predicted_risk*100:.1f}% risk)."
            }

        # Generic High Threat Fallback
        return {
            "stage": stage,
            "severity": severity,
            "action": "Enable defensive rate-limiting and initiate automated subnet containment.",
            "iptables_rule": f"iptables -A INPUT -p tcp -m limit --limit 10/s -j ACCEPT",
            "rationale": f"Multi-vector threat anomaly detected with forecasted risk of {predicted_risk*100:.1f}%."
        }
