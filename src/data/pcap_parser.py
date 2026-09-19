"""
PCAP Packet Parser and Flow Aggregator using Scapy for NetWorld-AI.
Extracts raw packet headers from PCAP/PCAPNG files, aggregates packets into 4-tuple flow sessions,
and builds standardized feature DataFrames compatible with CSVLoader.
"""

import os
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import pandas as pd
from scapy.all import PcapReader, IP, IPv6, TCP, UDP, Packet

from src.utils import load_config, setup_logger

logger = setup_logger("pcap_parser")


class PCAPParser:
    """
    Parses PCAP network trace files and aggregates packets into statistical flows.
    """

    def __init__(self, pcap_path: str, config_path: str = "config/config.yaml") -> None:
        """
        Initializes PCAPParser with target PCAP file path and configuration.

        Args:
            pcap_path: Path to the raw PCAP or PCAPNG file.
            config_path: Path to YAML configuration file.
        """
        self.pcap_path = pcap_path
        self.config = load_config(config_path)
        self.target_features: List[str] = self.config.get("features", [])

    def parse(self) -> pd.DataFrame:
        """
        Reads PCAP file, extracts packet metadata, aggregates by flow 4-tuple,
        and computes feature vectors adhering to NetWorld-AI schema.

        Returns:
            pd.DataFrame: Aggregated flow metrics DataFrame matching CSVLoader output schema.

        Raises:
            FileNotFoundError: If pcap_path does not exist.
        """
        if not os.path.exists(self.pcap_path):
            logger.error(f"PCAP file not found: {self.pcap_path}")
            raise FileNotFoundError(f"PCAP file not found at: {self.pcap_path}")

        logger.info(f"Parsing PCAP file: {self.pcap_path}")

        packets_data = []

        try:
            with PcapReader(self.pcap_path) as pcap_reader:
                for pkt in pcap_reader:
                    parsed_pkt = self._extract_packet_info(pkt)
                    if parsed_pkt:
                        packets_data.append(parsed_pkt)
        except Exception as e:
            logger.error(f"Error parsing PCAP with Scapy: {e}")
            raise RuntimeError(f"Failed to parse PCAP file {self.pcap_path}: {e}")

        if not packets_data:
            logger.warning(f"No valid IP/TCP/UDP packets found in {self.pcap_path}")
            return self._create_empty_flow_df()

        raw_df = pd.DataFrame(packets_data)
        logger.info(f"Extracted header metadata from {len(raw_df)} packets.")

        # Aggregate packets into 4-tuple flow sessions
        flow_df = self._aggregate_flows(raw_df)
        return flow_df

    def _extract_packet_info(self, pkt: Packet) -> Optional[Dict[str, Any]]:
        """
        Extracts relevant fields from an individual Scapy packet object.

        Args:
            pkt: Scapy Packet instance.

        Returns:
            Optional[Dict[str, Any]]: Extracted fields dictionary, or None if non-IP packet.
        """
        if not (pkt.haslayer(IP) or pkt.haslayer(IPv6)):
            return None

        ip_layer = pkt[IP] if pkt.haslayer(IP) else pkt[IPv6]
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        proto = ip_layer.proto if hasattr(ip_layer, "proto") else 0

        src_port = 0
        dst_port = 0
        tcp_flags = ""
        tcp_window = 0
        tcp_seq = 0
        length = len(pkt)
        timestamp = float(pkt.time)

        ttl = getattr(ip_layer, "ttl", 64) if hasattr(ip_layer, "ttl") else getattr(ip_layer, "hlim", 64)
        flags = getattr(ip_layer, "flags", 0)
        # Check for MF (More Fragments) or DF (Don't Fragment) flag
        fragment_flag = 1 if (hasattr(flags, "MF") and flags.MF) or (hasattr(flags, "DF") and flags.DF) else 0

        if pkt.haslayer(TCP):
            tcp_layer = pkt[TCP]
            src_port = int(tcp_layer.sport)
            dst_port = int(tcp_layer.dport)
            tcp_flags = str(tcp_layer.flags)
            tcp_window = int(getattr(tcp_layer, "window", 0))
            tcp_seq = int(getattr(tcp_layer, "seq", 0))
        elif pkt.haslayer(UDP):
            udp_layer = pkt[UDP]
            src_port = int(udp_layer.sport)
            dst_port = int(udp_layer.dport)

        return {
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": src_port,
            "dst_port": dst_port,
            "protocol": proto,
            "length": length,
            "tcp_flags": tcp_flags,
            "timestamp": timestamp,
            "ttl": ttl,
            "tcp_window": tcp_window,
            "fragment_flag": fragment_flag,
            "tcp_seq": tcp_seq
        }

    def _aggregate_flows(self, pkt_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates raw packet rows into flow sessions identified by 4-tuple (src_ip, dst_ip, src_port, dst_port).

        Args:
            pkt_df: DataFrame containing raw packet records.

        Returns:
            pd.DataFrame: Aggregated flow features DataFrame.
        """
        # Create canonical flow identifier
        pkt_df["flow_key"] = (
            pkt_df["src_ip"] + ":" +
            pkt_df["src_port"].astype(str) + "->" +
            pkt_df["dst_ip"] + ":" +
            pkt_df["dst_port"].astype(str)
        )

        flows = []
        unique_dst_ports_map = pkt_df.groupby("src_ip")["dst_port"].nunique().to_dict()

        for flow_key, group in pkt_df.groupby("flow_key"):
            group = group.sort_values("timestamp")
            timestamps = group["timestamp"].values
            lengths = group["length"].values
            flags_list = group["tcp_flags"].values

            src_ip = group["src_ip"].iloc[0]
            dst_ip = group["dst_ip"].iloc[0]
            src_port = group["src_port"].iloc[0]
            dst_port = group["dst_port"].iloc[0]
            protocol = group["protocol"].iloc[0]

            duration_sec = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.001
            duration_us = duration_sec * 1e6 # microsecond scale

            total_packets = len(group)
            total_bytes = float(lengths.sum())

            packet_rate = total_packets / max(duration_sec, 1e-6)
            bytes_rate = total_bytes / max(duration_sec, 1e-6)

            # TCP Flags breakdown
            syn_count = sum(1 for f in flags_list if "S" in f)
            ack_count = sum(1 for f in flags_list if "A" in f)
            fin_count = sum(1 for f in flags_list if "F" in f)
            rst_count = sum(1 for f in flags_list if "R" in f)
            psh_count = sum(1 for f in flags_list if "P" in f)
            syn_ack_ratio = syn_count / max(ack_count, 1.0)

            # Inter-Arrival Times (IAT)
            if len(timestamps) > 1:
                iats = np.diff(timestamps) * 1e6 # in us
                iat_mean = float(np.mean(iats))
                iat_std = float(np.std(iats))
                iat_max = float(np.max(iats))
                iat_min = float(np.min(iats))
            else:
                iat_mean = iat_std = iat_max = iat_min = 0.0

            # Directional estimations (assuming primary flow initiator is forward)
            fwd_packets = total_packets
            bwd_packets = 0.0
            fwd_bytes = total_bytes
            bwd_bytes = 0.0
            fwd_bwd_ratio = fwd_packets / 1.0

            # Sizing statistics
            avg_packet_size = float(np.mean(lengths))
            payload_std = float(np.std(lengths)) if len(lengths) > 1 else 0.0
            fwd_pkt_len_mean = avg_packet_size
            bwd_pkt_len_mean = 0.0
            pkt_len_max = float(np.max(lengths))
            pkt_len_min = float(np.min(lengths))

            # Packet-Level Features (TTL, Window, Fragmentation, Retransmissions)
            ttls = group["ttl"].values if "ttl" in group.columns else np.array([64])
            ttl_mean = float(np.mean(ttls))
            ttl_std = float(np.std(ttls)) if len(ttls) > 1 else 0.0

            windows = group["tcp_window"].values if "tcp_window" in group.columns else np.array([0])
            tcp_window_mean = float(np.mean(windows))

            frag_flags = group["fragment_flag"].values if "fragment_flag" in group.columns else np.array([0])
            fragment_flag_count = int(np.sum(frag_flags))

            # TCP sequence number retransmission tracking
            seqs = group["tcp_seq"].values if "tcp_seq" in group.columns else np.array([])
            non_zero_seqs = [s for s in seqs if s > 0]
            retransmission_count = int(len(non_zero_seqs) - len(set(non_zero_seqs))) if len(non_zero_seqs) > 1 else 0

            # Reconnaissance score
            unique_dst_ports = float(unique_dst_ports_map.get(src_ip, 1))
            recon_norm = np.clip(unique_dst_ports / 100.0, 0, 1.0)
            syn_ratio = np.clip(syn_count / max(total_packets, 1.0), 0, 1.0)
            recon_score = (0.6 * recon_norm) + (0.4 * syn_ratio)

            flow_record = {
                "src_port": src_port,
                "dst_port": dst_port,
                "protocol": protocol,
                "duration": duration_us,
                "total_packets": total_packets,
                "total_bytes": total_bytes,
                "packet_rate": packet_rate,
                "bytes_rate": bytes_rate,
                "syn_count": syn_count,
                "ack_count": ack_count,
                "fin_count": fin_count,
                "rst_count": rst_count,
                "psh_count": psh_count,
                "syn_ack_ratio": syn_ack_ratio,
                "iat_mean": iat_mean,
                "iat_std": iat_std,
                "iat_max": iat_max,
                "iat_min": iat_min,
                "fwd_packets": fwd_packets,
                "bwd_packets": bwd_packets,
                "fwd_bytes": fwd_bytes,
                "bwd_bytes": bwd_bytes,
                "fwd_bwd_ratio": fwd_bwd_ratio,
                "avg_packet_size": avg_packet_size,
                "payload_std": payload_std,
                "fwd_pkt_len_mean": fwd_pkt_len_mean,
                "bwd_pkt_len_mean": bwd_pkt_len_mean,
                "pkt_len_max": pkt_len_max,
                "pkt_len_min": pkt_len_min,
                "unique_dst_ports_per_src": unique_dst_ports,
                "recon_score": recon_score,
                "ttl_mean": ttl_mean,
                "ttl_std": ttl_std,
                "tcp_window_mean": tcp_window_mean,
                "fragment_flag_count": fragment_flag_count,
                "retransmission_count": retransmission_count,
                "Label": "BENIGN",
                "Binary_Label": 0
            }
            flows.append(flow_record)

        flow_df = pd.DataFrame(flows)
        logger.info(f"Aggregated {len(pkt_df)} packets into {len(flow_df)} network flow sessions.")
        return flow_df

    def _create_empty_flow_df(self) -> pd.DataFrame:
        """
        Creates an empty DataFrame with target feature columns.

        Returns:
            pd.DataFrame: Empty DataFrame with correct columns.
        """
        cols = self.target_features + ["Label", "Binary_Label"]
        return pd.DataFrame(columns=cols)
