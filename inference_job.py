"""
Fetch radar data, group by location, build sequences, run inference, format and optionally POST results.
All feature engineering is delegated to inference_code.real_time_inference (record-based path).
"""
import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import torch

from config import (
    FETCH_LIMIT,
    INFERENCE_RESULTS_DIR,
    INFERENCE_RESULTS_URL,
    RADAR_DATA_URL,
    REPO_ROOT,
)

logger = logging.getLogger(__name__)

# Lazy import after path is set in main
_inference_engine = None
_Config = None
_CSIFeatureExtractor = None


def _ensure_imports():
    global _inference_engine, _Config, _CSIFeatureExtractor
    if _Config is not None:
        return
    from inference_code.real_time_inference import (
        Config as _C,
        CSIFeatureExtractor as _E,
        InferenceEngine as _IE,
    )
    _Config = _C
    _CSIFeatureExtractor = _E
    _inference_engine = _IE


def _model_path_for_engine(model_path: str) -> str:
    """
    Return a path to a .pt file in the format InferenceEngine expects (dict with 'model_state_dict').
    If the checkpoint is already in that format, return the resolved path; otherwise write a temp file.
    """
    p = Path(model_path)
    if not p.is_absolute():
        p = REPO_ROOT / model_path
    resolved = str(p.resolve())
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return resolved
    # Raw state_dict: write normalized checkpoint to a temp file (not deleted; engine keeps model in memory)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        torch.save({"model_state_dict": checkpoint}, f.name)
        return f.name


def get_engine(model_path: str, scaler_path: str):
    """Load and return InferenceEngine (call once at startup)."""
    _ensure_imports()
    p = Path(model_path)
    if not p.is_absolute():
        p = REPO_ROOT / model_path
    s = Path(scaler_path)
    if not s.is_absolute():
        s = REPO_ROOT / scaler_path
    path_to_use = _model_path_for_engine(str(p))
    return _inference_engine(path_to_use, str(s), receiver_name="api", device="cpu")


def _normalize_record(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize API record to snake_case and required types."""
    try:
        ts = raw.get("timestamp_ms") or raw.get("timestampMs")
        if ts is None:
            return None
        if isinstance(ts, str):
            ts = int(ts)
        rssi = raw.get("rssi", -70)
        if isinstance(rssi, str):
            rssi = int(rssi)
        return {
            "rx_mac": raw.get("rx_mac") or raw.get("rxMac") or "",
            "room_id": raw.get("room_id") if raw.get("room_id") is not None else raw.get("roomId"),
            "building_id": raw.get("building_id") if raw.get("building_id") is not None else raw.get("buildingId"),
            "timestamp_ms": ts,
            "rssi": rssi,
            "csi_data": raw.get("csi_data") or raw.get("csiData"),
            "radar_targets": raw.get("radar_targets") or raw.get("radarTargets"),
        }
    except (TypeError, ValueError):
        return None


def fetch_radar_data(
    client: httpx.Client,
    *,
    rx_mac: Optional[str] = None,
    room_id: Optional[str] = None,
    building_id: Optional[str] = None,
    limit: int = FETCH_LIMIT,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """GET radar data and return list of normalized records."""
    params = {"limit": limit, "offset": offset}
    if rx_mac:
        params["rx_mac"] = rx_mac
    if room_id is not None:
        params["room_id"] = room_id
    if building_id is not None:
        params["building_id"] = building_id
    try:
        r = client.get(RADAR_DATA_URL, params=params, timeout=30.0)
        r.raise_for_status()
        data = r.json()
        if not data.get("success") or "data" not in data:
            logger.warning("Radar API returned success=false or no data")
            return []
        normalized = []
        for row in data["data"]:
            rec = _normalize_record(row)
            if rec and rec.get("csi_data") is not None:
                normalized.append(rec)
        return normalized
    except httpx.HTTPError as e:
        logger.exception("Failed to fetch radar data: %s", e)
        raise
    except Exception as e:
        logger.exception("Unexpected error fetching radar data: %s", e)
        raise


def _group_by_location(records: List[Dict[str, Any]]) -> Dict[tuple, List[Dict[str, Any]]]:
    """Group by (room_id, building_id, rx_mac). Each group sorted by timestamp_ms ascending."""
    groups = {}
    for r in records:
        key = (r["room_id"], r["building_id"], r["rx_mac"])
        groups.setdefault(key, []).append(r)
    for key in groups:
        groups[key] = sorted(groups[key], key=lambda x: x["timestamp_ms"])
    return groups


def run_inference_for_records(
    records: List[Dict[str, Any]],
    engine,
    sequence_length: int = 30,
) -> List[Dict[str, Any]]:
    """
    Group records by (room_id, building_id, rx_mac). For each group with >= sequence_length
    packets, build sequence, predict, and append one result. Uses record-based feature
    extraction from real_time_inference.
    """
    _ensure_imports()
    groups = _group_by_location(records)
    results = []
    for (room_id, building_id, rx_mac), group_records in groups.items():
        if len(group_records) < sequence_length:
            logger.debug(
                "Skipping group (%s, %s, %s): only %d packets (need %d)",
                room_id, building_id, rx_mac, len(group_records), sequence_length,
            )
            continue
        # Use last `sequence_length` packets in time order
        window = group_records[-sequence_length:]
        extractor = _CSIFeatureExtractor()
        for rec in window:
            extractor.extract_packet_features_from_record(rec)
        seq = extractor.get_sequence_features()
        if seq is None:
            logger.debug("No sequence for (%s, %s, %s)", room_id, building_id, rx_mac)
            continue
        try:
            occupied, prob = engine.predict(seq["features"])
        except Exception as e:
            logger.exception("Prediction failed for (%s, %s, %s): %s", room_id, building_id, rx_mac, e)
            continue
        ts_start_ms = seq["timestamp_start"]
        ts_end_ms = seq["timestamp_end"]
        ts_start = datetime.fromtimestamp(ts_start_ms / 1000.0, tz=timezone.utc)
        ts_end = datetime.fromtimestamp(ts_end_ms / 1000.0, tz=timezone.utc)
        results.append({
            "timestamp_start": ts_start.isoformat(),
            "timestamp_end": ts_end.isoformat(),
            "room_id": room_id,
            "building_id": building_id,
            "occupied": occupied,
            "occupied_probability": round(float(prob), 6),
            "rx_mac": rx_mac,
        })
    return results


def write_results_to_local_dir(results: List[Dict[str, Any]]) -> None:
    """Write inference results to INFERENCE_RESULTS_DIR as a timestamped JSON file."""
    import os
    out_dir_str = os.getenv("INFERENCE_RESULTS_DIR", "") or INFERENCE_RESULTS_DIR
    if not out_dir_str or not results:
        return
    try:
        out_dir = Path(out_dir_str)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        path = out_dir / f"inference_{ts}.json"
        path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info("Wrote %d inference result(s) to %s", len(results), path)
    except Exception as e:
        logger.exception("Failed to write inference results to %s: %s", out_dir_str, e)


def post_results(client: httpx.Client, results: List[Dict[str, Any]]) -> None:
    """POST inference results to INFERENCE_RESULTS_URL. Logs and continues on failure."""
    if not INFERENCE_RESULTS_URL or not results:
        return
    try:
        r = client.post(INFERENCE_RESULTS_URL, json=results, timeout=30.0)
        r.raise_for_status()
        logger.info("Posted %d inference result(s) to %s", len(results), INFERENCE_RESULTS_URL)
    except httpx.HTTPError as e:
        logger.exception("Failed to POST inference results to %s: %s", INFERENCE_RESULTS_URL, e)
    except Exception as e:
        logger.exception("Unexpected error posting results: %s", e)


def run_once(
    client: httpx.Client,
    engine,
    *,
    rx_mac: Optional[str] = None,
    room_id: Optional[str] = None,
    building_id: Optional[str] = None,
    limit: int = FETCH_LIMIT,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Fetch radar data, run inference, optionally POST results. Returns list of result dicts."""
    records = fetch_radar_data(
        client, rx_mac=rx_mac, room_id=room_id, building_id=building_id, limit=limit, offset=offset,
    )
    _ensure_imports()
    seq_len = getattr(_Config, "SEQUENCE_LENGTH", 30)
    results = run_inference_for_records(records, engine, sequence_length=seq_len)
    write_results_to_local_dir(results)
    post_results(client, results)
    return results
