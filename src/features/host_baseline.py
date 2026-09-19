"""
Per-Host Baseline Tracking and EWMA Anomaly Deviation Module for NetWorld-AI.
Maintains an Exponentially Weighted Moving Average (EWMA, alpha=0.1) per source IP for:
1. connections_per_minute (flow packet rate)
2. unique_dst_ports_touched
3. internal_vs_external_ratio

Provides z-score deviation scoring (get_baseline_deviation) and persistent SQLite storage.
Adds 32nd/38th feature 'baseline_deviation_score'.
"""

import os
import sqlite3
from typing import Dict, Any, Optional
import numpy as np

from src.utils import setup_logger

logger = setup_logger("host_baseline")


class HostBaselineManager:
    """
    Manages per-host learned normal behavior profiles using EWMA and SQLite persistence.
    """

    def __init__(self, db_path: str = "models/host_baselines.db", alpha: float = 0.1) -> None:
        """
        Initializes HostBaselineManager.

        Args:
            db_path: Filepath to SQLite database for persisting host baselines.
            alpha: Smoothing factor for EWMA calculation (default: 0.1).
        """
        self.db_path = db_path
        self.alpha = alpha
        self.in_memory_hosts: Dict[str, Dict[str, float]] = {}

        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_sqlite()

    def _init_sqlite(self) -> None:
        """Initializes SQLite table schema for host baselines."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS host_baselines (
                    ip TEXT PRIMARY KEY,
                    mean_conn REAL, var_conn REAL,
                    mean_ports REAL, var_ports REAL,
                    mean_ratio REAL, var_ratio REAL,
                    update_count INTEGER
                )
            """)
            conn.commit()
            
            # Load existing baselines into memory cache
            cursor.execute("SELECT ip, mean_conn, var_conn, mean_ports, var_ports, mean_ratio, var_ratio, update_count FROM host_baselines")
            rows = cursor.fetchall()
            for row in rows:
                self.in_memory_hosts[row[0]] = {
                    "mean_conn": row[1], "var_conn": row[2],
                    "mean_ports": row[3], "var_ports": row[4],
                    "mean_ratio": row[5], "var_ratio": row[6],
                    "update_count": row[7]
                }
            conn.close()
            logger.info(f"Initialized SQLite host baseline database with {len(rows)} learned profiles.")
        except Exception as e:
            logger.warning(f"SQLite DB initialization fallback to in-memory: {e}")

    def get_baseline_deviation(self, ip: str, current_stats: Dict[str, float]) -> float:
        """
        Computes z-score-like deviation from that host's learned normal behavior.
        Formula: Mean of per-metric z-scores |x - mean| / (std + 1e-4).

        Args:
            ip: Source IP address string.
            current_stats: Dict containing metrics for current window:
                - connections_per_minute
                - unique_dst_ports_touched
                - internal_vs_external_ratio

        Returns:
            float: Documented 32nd feature 'baseline_deviation_score' (>= 0.0).
        """
        conn_val = float(current_stats.get("connections_per_minute", current_stats.get("packet_rate", 10.0)))
        ports_val = float(current_stats.get("unique_dst_ports_touched", current_stats.get("unique_dst_ports_per_src", 1.0)))
        ratio_val = float(current_stats.get("internal_vs_external_ratio", current_stats.get("fwd_bwd_ratio", 1.0)))

        profile = self.in_memory_hosts.get(ip)

        if not profile:
            # First observation: return baseline 0.0 deviation and update profile
            self.update_baseline(ip, current_stats)
            return 0.0

        # Calculate per-metric z-score deviations
        z_conn = abs(conn_val - profile["mean_conn"]) / (np.sqrt(max(profile["var_conn"], 1e-4)) + 1e-4)
        z_ports = abs(ports_val - profile["mean_ports"]) / (np.sqrt(max(profile["var_ports"], 1e-4)) + 1e-4)
        z_ratio = abs(ratio_val - profile["mean_ratio"]) / (np.sqrt(max(profile["var_ratio"], 1e-4)) + 1e-4)

        dev_score = float((z_conn + z_ports + z_ratio) / 3.0)

        # Update host baseline with new observation
        self.update_baseline(ip, current_stats)
        return dev_score

    def update_baseline(self, ip: str, current_stats: Dict[str, float]) -> None:
        """
        Updates EWMA (alpha=0.1) mean and variance statistics for a source IP.

        Args:
            ip: Source IP address string.
            current_stats: Dict containing metric values.
        """
        conn_val = float(current_stats.get("connections_per_minute", current_stats.get("packet_rate", 10.0)))
        ports_val = float(current_stats.get("unique_dst_ports_touched", current_stats.get("unique_dst_ports_per_src", 1.0)))
        ratio_val = float(current_stats.get("internal_vs_external_ratio", current_stats.get("fwd_bwd_ratio", 1.0)))

        a = self.alpha

        if ip not in self.in_memory_hosts:
            profile = {
                "mean_conn": conn_val, "var_conn": 1.0,
                "mean_ports": ports_val, "var_ports": 1.0,
                "mean_ratio": ratio_val, "var_ratio": 1.0,
                "update_count": 1
            }
        else:
            profile = self.in_memory_hosts[ip]
            old_mc = profile["mean_conn"]
            old_mp = profile["mean_ports"]
            old_mr = profile["mean_ratio"]

            profile["mean_conn"] = (a * conn_val) + ((1.0 - a) * old_mc)
            profile["var_conn"] = (a * ((conn_val - old_mc) ** 2)) + ((1.0 - a) * profile["var_conn"])

            profile["mean_ports"] = (a * ports_val) + ((1.0 - a) * old_mp)
            profile["var_ports"] = (a * ((ports_val - old_mp) ** 2)) + ((1.0 - a) * profile["var_ports"])

            profile["mean_ratio"] = (a * ratio_val) + ((1.0 - a) * old_mr)
            profile["var_ratio"] = (a * ((ratio_val - old_mr) ** 2)) + ((1.0 - a) * profile["var_ratio"])

            profile["update_count"] += 1

        self.in_memory_hosts[ip] = profile

        # Persist to SQLite
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO host_baselines 
                (ip, mean_conn, var_conn, mean_ports, var_ports, mean_ratio, var_ratio, update_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ip, profile["mean_conn"], profile["var_conn"],
                profile["mean_ports"], profile["var_ports"],
                profile["mean_ratio"], profile["var_ratio"],
                profile["update_count"]
            ))
            conn.commit()
            conn.close()
        except Exception:
            pass
