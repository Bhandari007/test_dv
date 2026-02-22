"""
CSI feature extraction from API/CSV records (record-based path only).
Must match training feature engineering; sequence length from inference_config.
"""
import json
from collections import deque
from typing import Dict, List, Optional

import numpy as np

from inference_config import SEQUENCE_LENGTH


class CSIFeatureExtractor:
    """Extract features from CSI data in API/CSV record form. Same 34 features as training."""

    def __init__(self):
        self.packet_history = deque(maxlen=SEQUENCE_LENGTH)

    def parse_csi_from_record(self, csi_data) -> Optional[np.ndarray]:
        """
        Parse CSI from API/CSV record. csi_data can be list of [I,Q] or JSON string.
        Returns: Array of I/Q pairs shape (subcarriers, 2) or None.
        """
        if csi_data is None:
            return None
        try:
            if isinstance(csi_data, str):
                csi_data = json.loads(csi_data)
            arr = np.array(csi_data, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != 2:
                return None
            if len(arr) == 0:
                return None
            return arr
        except Exception:
            return None

    def count_radar_targets_from_record(self, radar_targets) -> int:
        """
        Count non-zero radar detections from API/CSV record.
        radar_targets: list of sensor dicts, e.g. [{"0": [{x_mm, y_mm, ...}, ...]}, {"1": [...]}]
        """
        if radar_targets is None:
            return 0
        try:
            if isinstance(radar_targets, str):
                radar_targets = json.loads(radar_targets)
            count = 0
            for sensor in radar_targets:
                if not isinstance(sensor, dict):
                    continue
                for _, targets in sensor.items():
                    for t in targets if isinstance(targets, list) else []:
                        x = t.get("x_mm", t.get("xMm", 0))
                        y = t.get("y_mm", t.get("yMm", 0))
                        if x != 0 or y != 0:
                            count += 1
            return count
        except Exception:
            return 0

    def extract_packet_features_from_record(self, record: Dict) -> Optional[Dict]:
        """
        Extract features from a single API/CSV record (same 33 features as training).
        record: dict with csi_data, rssi, timestamp_ms, radar_targets (snake_case or camelCase).
        """
        csi_data = record.get("csi_data") or record.get("csiData")
        csi_array = self.parse_csi_from_record(csi_data)
        if csi_array is None or len(csi_array) == 0:
            return None

        rssi = record.get("rssi", -70)
        if isinstance(rssi, str):
            rssi = int(rssi)
        timestamp_ms = record.get("timestamp_ms") or record.get("timestampMs")
        if timestamp_ms is None:
            return None
        if isinstance(timestamp_ms, str):
            timestamp_ms = int(timestamp_ms)

        radar_targets = record.get("radar_targets") or record.get("radarTargets")
        target_count = self.count_radar_targets_from_record(radar_targets)

        try:
            basic_features = self._compute_csi_features(csi_array, rssi)
        except Exception:
            return None

        temporal_features = self._compute_single_packet_temporal(basic_features)
        full_features = np.concatenate([basic_features, temporal_features])

        packet_data = {
            "timestamp_ms": timestamp_ms,
            "features": full_features,
            "target_count": target_count,
            "actual_occupied": target_count > 0,
        }
        self.packet_history.append(packet_data)
        return packet_data

    def _compute_csi_features(self, csi_data: np.ndarray, rssi: int) -> np.ndarray:
        """Compute 9 basic CSI features (amplitude, power, I/Q variance, subcarrier count, RSSI)."""
        I = csi_data[:, 0]
        Q = csi_data[:, 1]
        amplitude = np.sqrt(I**2 + Q**2)
        power = I**2 + Q**2
        features = [
            np.mean(amplitude),
            np.std(amplitude),
            np.var(amplitude),
            np.mean(power),
            np.std(power),
            np.var(power),
            np.var(I) + np.var(Q),
            float(len(csi_data)),
            float(rssi),
        ]
        return np.array(features, dtype=np.float32)

    def get_sequence_features(self) -> Optional[Dict]:
        """
        Get features for a sequence (SEQUENCE_LENGTH timesteps).
        Returns: Dict with 'features' (30, 34), 'actual_occupied', 'timestamp_start', 'timestamp_end'
        or None if insufficient data.
        """
        if len(self.packet_history) < SEQUENCE_LENGTH:
            return None

        recent_packets = list(self.packet_history)[-SEQUENCE_LENGTH:]
        features_list = []
        for idx, p in enumerate(recent_packets):
            packet_features = p["features"]
            sequence_position = idx / SEQUENCE_LENGTH
            full_features = np.concatenate([packet_features, [sequence_position]])
            features_list.append(full_features)

        sequence_features = np.array(features_list)
        occupied_count = sum(1 for p in recent_packets if p.get("actual_occupied", False))
        actual_occupied = occupied_count > (SEQUENCE_LENGTH / 2)
        timestamp_start = recent_packets[0]["timestamp_ms"]
        timestamp_end = recent_packets[-1]["timestamp_ms"]

        return {
            "features": sequence_features,
            "actual_occupied": actual_occupied,
            "timestamp_start": timestamp_start,
            "timestamp_end": timestamp_end,
        }

    def _compute_single_packet_temporal(self, basic_features: np.ndarray) -> np.ndarray:
        """Compute 24 temporal features (6 key features x 4 each: rolling_std, diff, lag1, lag2)."""
        ROLLING_WINDOW = 10
        key_indices = [0, 3, 1, 4, 6, 8]
        current_key_features = basic_features[key_indices]

        if len(self.packet_history) > 0:
            history_basic = np.array([p["features"][:9] for p in self.packet_history])
            history_key_features = history_basic[:, key_indices]
        else:
            history_key_features = np.array([]).reshape(0, 6)

        temporal_features = []
        for j in range(6):
            if len(history_key_features) > 0:
                feature_series = np.concatenate([history_key_features[:, j], [current_key_features[j]]])
            else:
                feature_series = np.array([current_key_features[j]])

            temporal_features.append(
                np.std(feature_series[-ROLLING_WINDOW:]) if len(feature_series) >= ROLLING_WINDOW
                else (np.std(feature_series) if len(feature_series) > 1 else 0.0)
            )
            temporal_features.append(feature_series[-1] - feature_series[-2] if len(feature_series) > 1 else 0.0)
            for lag in [1, 2]:
                temporal_features.append(feature_series[-lag - 1] if len(feature_series) > lag else 0.0)

        return np.array(temporal_features, dtype=np.float32)
