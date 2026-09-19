"""
MITRE ATT&CK Stage Mapping Engine for NetWorld-AI.
Maps temporal feature metrics and threat signals into explicit MITRE ATT&CK tactics, confidence levels, and evidence metrics.
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np


class StageMapper:
    """
    Maps feature state vectors to MITRE ATT&CK tactics based on domain heuristic rules.
    """

    def __init__(self) -> None:
        """Initializes StageMapper with tactic definitions."""
        self.tactics = [
            "BENIGN / Normal Traffic",
            "Reconnaissance",
            "Initial Access",
            "Discovery",
            "Lateral Movement",
            "Command & Control",
            "Exfiltration"
        ]

    def map_stage(
        self,
        state_vector: np.ndarray,
        feature_names: List[str],
        predicted_stage_idx: Optional[int] = None
    ) -> Tuple[str, float, str]:
        """
        Maps a single window feature vector to a MITRE ATT&CK stage using model predictions or rule fallbacks.

        Args:
            state_vector: 1D array of feature values for a state window.
            feature_names: List of feature names corresponding to state_vector indices.
            predicted_stage_idx: Optional integer class predicted by LSTMWorldModel (0..6).

        Returns:
            Tuple[str, float, str]:
                - stage: MITRE ATT&CK tactic name
                - confidence: Confidence score (0.0 to 1.0)
                - evidence: Concise explanatory evidence string
        """
        stage_names = [
            "Reconnaissance",
            "Initial Access",
            "Discovery",
            "Lateral Movement",
            "Command & Control",
            "Exfiltration",
            "BENIGN / Normal Traffic"
        ]

        if predicted_stage_idx is not None and 0 <= predicted_stage_idx < len(stage_names):
            mapped_stage = stage_names[predicted_stage_idx]
            if mapped_stage == "BENIGN / Normal Traffic":
                return mapped_stage, 0.95, "Model classified traffic baseline as BENIGN."
            else:
                return mapped_stage, 0.92, f"Neural World Model classified state as {mapped_stage}."

        feat_map = dict(zip(feature_names, state_vector))

        # Extract key metrics
        dst_port = feat_map.get("dst_port", 80)
        syn_count = feat_map.get("syn_count", 0)
        bytes_rate = feat_map.get("bytes_rate", 0)
        total_bytes = feat_map.get("total_bytes", 0)
        packet_rate = feat_map.get("packet_rate", 0)
        unique_dst_ports = feat_map.get("unique_dst_ports_per_src", 1)
        recon_score = feat_map.get("recon_score", 0)
        fwd_bwd_ratio = feat_map.get("fwd_bwd_ratio", 1)
        iat_std = feat_map.get("iat_std", 1.0)
        iat_mean = feat_map.get("iat_mean", 1.0)

        # 1. Reconnaissance Rule: High unique dst ports or high recon score
        if recon_score > 0.4 or unique_dst_ports > 5 or (syn_count > 3 and fwd_bwd_ratio > 2):
            confidence = min(0.95, 0.5 + (recon_score * 0.5))
            evidence = f"High port scan density (unique_dst_ports={int(unique_dst_ports)}, recon_score={recon_score:.2f})"
            return "Reconnaissance", float(confidence), evidence

        # 2. Initial Access Rule: High SYN count on authentication ports (22, 80, 443) or brute force
        if syn_count > 5 or (dst_port in [22, 21, 3389, 80, 443] and packet_rate > 50):
            confidence = min(0.92, 0.6 + (syn_count / 20.0))
            evidence = f"High SYN connection rate ({int(syn_count)} SYN packets) on auth port {int(dst_port)}"
            return "Initial Access", float(confidence), evidence

        # 3. Lateral Movement Rule: High traffic spikes targeting internal administration ports (445, 3389, 22)
        if dst_port in [445, 3389, 135, 139] or (dst_port == 22 and fwd_bwd_ratio > 3):
            confidence = 0.88
            evidence = f"Lateral pivot signature detected targeting admin service port {int(dst_port)}"
            return "Lateral Movement", float(confidence), evidence

        # 4. Command & Control Rule: Low IAT variance (periodic beaconing pattern) with sustained packets
        if iat_std < (0.2 * max(iat_mean, 1e-3)) and packet_rate > 5:
            confidence = 0.85
            evidence = f"Periodic C2 beaconing interval detected (IAT std={iat_std:.2f}us, mean={iat_mean:.2f}us)"
            return "Command & Control", float(confidence), evidence

        # 5. Exfiltration Rule: High outbound byte rates or massive total byte count
        if bytes_rate > 1000000 or total_bytes > 5000000:
            confidence = min(0.98, 0.7 + (bytes_rate / 5000000.0))
            evidence = f"Massive outbound data transfer (bytes_rate={bytes_rate/1e6:.2f} MB/s, total_bytes={total_bytes/1e6:.2f} MB)"
            return "Exfiltration", float(confidence), evidence

        # 6. Discovery Rule: Broad sweep with moderate packet rate
        if unique_dst_ports > 2 or packet_rate > 30:
            confidence = 0.75
            evidence = f"Internal network enumeration scan (packet_rate={packet_rate:.1f} pkts/s)"
            return "Discovery", float(confidence), evidence

        # Default BENIGN
        return "BENIGN / Normal Traffic", 0.95, "Traffic patterns within normal baseline parameters."
