"""
Fetch radar data, group by location, build sequences, run inference, format and optionally POST results.
Feature extraction and model inference are self-contained in this service (no inference_code dependency).
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
    DATABASE_URL,
    FETCH_LIMIT,
    INFERENCE_RESULTS_DIR,
    INFERENCE_RESULTS_URL,
    RADAR_DATA_URL,
    REPO_ROOT,
    INFERENCE_WATERMARK_FILE,
)
from feature_extractor import CSIFeatureExtractor
from inference_config import SEQUENCE_LENGTH
from model_engine import InferenceEngine

logger = logging.getLogger(__name__)


def _timestamp_to_datetime(ts: Any) -> datetime:
    """
    Convert a numeric timestamp (seconds or milliseconds since epoch) to an aware UTC datetime.

    Heuristic:
    - If value looks like Unix seconds (1e9–2e9), treat it as seconds.
    - Otherwise treat it as milliseconds.
    """
    try:
        ts_float = float(ts)
    except (TypeError, ValueError):
        # Fallback to epoch start for clearly invalid values
        return datetime.fromtimestamp(0, tz=timezone.utc)
    if 1e9 <= ts_float <= 2e9:
        seconds = ts_float
    else:
        seconds = ts_float / 1000.0
    return datetime.fromtimestamp(seconds, tz=timezone.utc)


def _location_key(room_id: Any, building_id: Any, rx_mac: str) -> str:
    """Build a stable key for watermark per location."""
    return f"{room_id}|{building_id}|{rx_mac or ''}"


def _load_watermark(path: str | None = None) -> Dict[str, int]:
    """
    Load watermark from JSON file.

    Returns mapping: location_key -> last_timestamp_end_ms (int).
    """
    import os

    watermark_path = Path(path or INFERENCE_WATERMARK_FILE)
    if not watermark_path.exists():
        return {}
    try:
        text = watermark_path.read_text(encoding="utf-8")
        data = json.loads(text or "{}")
        if not isinstance(data, dict):
            return {}
        result: Dict[str, int] = {}
        for key, value in data.items():
            try:
                result[str(key)] = int(value)
            except (TypeError, ValueError):
                continue
        return result
    except Exception as e:
        logger.warning("Failed to load watermark file %s: %s", watermark_path, e)
        return {}


def _save_watermark(path: str | None, watermark: Dict[str, int]) -> None:
    """Persist watermark mapping to JSON file (best-effort, atomic where possible)."""
    watermark_path = Path(path or INFERENCE_WATERMARK_FILE)
    try:
        watermark_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = watermark_path.with_suffix(watermark_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(watermark, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(watermark_path)
        logger.info("Updated watermark file: %s", watermark_path)
    except Exception as e:
        logger.warning("Failed to write watermark file %s: %s", watermark_path, e)


def _iso_timestamp_to_ms(ts_iso: str) -> int:
    """Convert ISO 8601 timestamp to milliseconds since epoch."""
    try:
        dt = datetime.fromisoformat(ts_iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        return 0


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
    p = Path(model_path)
    if not p.is_absolute():
        p = REPO_ROOT / model_path
    s = Path(scaler_path)
    if not s.is_absolute():
        s = REPO_ROOT / scaler_path
    path_to_use = _model_path_for_engine(str(p))
    return InferenceEngine(path_to_use, str(s), receiver_name="api", device="cpu")


def _normalize_db_row(row: Any) -> Optional[Dict[str, Any]]:
    """Map a DB row (radar_readings) to the same normalized record shape as _normalize_record."""
    try:
        # row can be a dict (from cursor with row_factory) or a sequence
        if hasattr(row, "keys"):
            raw = dict(row)
        else:
            # Assume row is tuple-like; we need column names - use cursor.description
            raise TypeError("DB row must be a dict-like mapping")
        ts = raw.get("timestamp_ms")
        if ts is None:
            return None
        if isinstance(ts, str):
            ts = int(ts)
        rssi = raw.get("rssi", -70)
        if isinstance(rssi, str):
            rssi = int(rssi)
        # radar_targets: feature extractor expects list of sensor dicts; DB has ld2450_targets, rd03d_targets (JSON)
        radar_targets = raw.get("radar_targets") or raw.get("ld2450_targets") or raw.get("rd03d_targets")
        return {
            "rx_mac": raw.get("rx_mac") or "",
            "room_id": raw.get("room_id"),
            "building_id": raw.get("building_id"),
            "timestamp_ms": ts,
            "rssi": rssi,
            "csi_data": raw.get("csi_data"),
            "radar_targets": radar_targets,
        }
    except (TypeError, ValueError, KeyError):
        return None


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


MAX_PAGES_FETCH_ALL = 500


def fetch_all_radar_data(
    client: httpx.Client,
    *,
    rx_mac: Optional[str] = None,
    room_id: Optional[str] = None,
    building_id: Optional[str] = None,
    page_size: int = FETCH_LIMIT,
    max_pages: int = MAX_PAGES_FETCH_ALL,
) -> List[Dict[str, Any]]:
    """GET all radar data by paginating until hasMore is false. Returns list of normalized records."""
    all_records: List[Dict[str, Any]] = []
    offset = 0
    for page in range(max_pages):
        params = {"limit": page_size, "offset": offset}
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
        except httpx.HTTPError as e:
            logger.exception("Failed to fetch radar data (page %d): %s", page + 1, e)
            raise
        except Exception as e:
            logger.exception("Unexpected error fetching radar data (page %d): %s", page + 1, e)
            raise
        if not data.get("success") or "data" not in data:
            logger.warning("Radar API returned success=false or no data on page %d", page + 1)
            break
        rows = data.get("data") or []
        for row in rows:
            rec = _normalize_record(row)
            if rec and rec.get("csi_data") is not None:
                all_records.append(rec)
        pagination = data.get("pagination") or {}
        has_more = pagination.get("hasMore", False)
        if not has_more or len(rows) == 0:
            break
        offset += page_size
    if page + 1 >= max_pages and (data.get("pagination") or {}).get("hasMore"):
        logger.warning("Stopped after %d pages (max_pages=%d); more data may remain", max_pages, max_pages)
    logger.info("Fetched %d total records in %d page(s)", len(all_records), page + 1)
    return all_records


def fetch_radar_data_from_db(
    conn: Any,
    *,
    rx_mac: Optional[str] = None,
    room_id: Optional[str] = None,
    building_id: Optional[str] = None,
    limit: int = FETCH_LIMIT,
    offset: int = 0,
    min_timestamp_ms: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Fetch latest radar data from PostgreSQL radar_readings (ORDER BY timestamp_ms DESC). Returns normalized records."""
    import psycopg
    from psycopg.rows import dict_row

    logger.info(
        "Fetching radar_readings from DB (limit=%s, offset=%s, filters: rx_mac=%s, room_id=%s, building_id=%s)",
        limit, offset, rx_mac, room_id, building_id,
    )
    conditions = ["1=1"]
    params: List[Any] = []
    if rx_mac:
        conditions.append("rx_mac = %s")
        params.append(rx_mac)
    if room_id is not None:
        conditions.append("room_id = %s")
        params.append(room_id)
    if building_id is not None:
        conditions.append("building_id = %s")
        params.append(building_id)
    if min_timestamp_ms is not None:
        conditions.append("timestamp_ms > %s")
        params.append(min_timestamp_ms)
    params.extend([limit, offset])
    sql = f"""
        SELECT rx_mac, room_id, building_id, timestamp_ms, rssi, channel, csi_data
        FROM radar_readings
        WHERE {" AND ".join(conditions)}
        ORDER BY timestamp_ms DESC
        LIMIT %s OFFSET %s
    """
    normalized = []
    try:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            raw_count = len(rows)
            for row in rows:
                rec = _normalize_db_row(row)
                if rec and rec.get("csi_data") is not None:
                    normalized.append(rec)
        logger.info(
            "Fetched %d record(s) from radar_readings (%d with valid csi_data)",
            raw_count, len(normalized),
        )
    except Exception as e:
        logger.exception("Failed to fetch radar_readings: %s", e)
        raise
    return normalized


def insert_inference_results(conn: Any, results: List[Dict[str, Any]]) -> None:
    """Insert inference result dicts into the inference_data table."""
    if not results:
        return
    import psycopg

    logger.debug("Inserting %d result(s) into inference_data", len(results))
    sql = """
        INSERT INTO inference_data (timestamp_start, timestamp_end, room_id, building_id, occupied, occupied_probability, rx_mac)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    try:
        with conn.cursor() as cur:
            for r in results:
                cur.execute(
                    sql,
                    (
                        r["timestamp_start"],
                        r["timestamp_end"],
                        r["room_id"],
                        r["building_id"],
                        r["occupied"],
                        r["occupied_probability"],
                        r["rx_mac"],
                    ),
                )
        logger.info("Inserted %d inference result(s) into inference_data", len(results))
    except Exception as e:
        logger.exception("Failed to insert into inference_data: %s", e)
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
    sequence_length: int = None,
) -> List[Dict[str, Any]]:
    """
    Group records by (room_id, building_id, rx_mac).

    - For each group with >= sequence_length packets, build sequence, predict, and append one result.
    - For groups with insufficient packets, append a result with occupied/occupied_probability set to None
      so that downstream auditing can see that no prediction was made.
    """
    if sequence_length is None:
        sequence_length = SEQUENCE_LENGTH
    groups = _group_by_location(records)
    results = []
    for (room_id, building_id, rx_mac), group_records in groups.items():
        if not group_records:
            continue
        # Compute timestamps for the group from raw packet timestamps
        first_ts_ms = group_records[0]["timestamp_ms"]
        last_ts_ms = group_records[-1]["timestamp_ms"]
        ts_start = _timestamp_to_datetime(first_ts_ms)
        ts_end = _timestamp_to_datetime(last_ts_ms)

        if len(group_records) < sequence_length:
            logger.info(
                "Insufficient packets for (%s, %s, %s): only %d packets (need %d); skipped inference",
                room_id,
                building_id,
                rx_mac,
                len(group_records),
                sequence_length,
            )
            continue

        # Use last `sequence_length` packets in time order for inference
        window = group_records[-sequence_length:]
        extractor = CSIFeatureExtractor()
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
        ts_start = _timestamp_to_datetime(ts_start_ms)
        ts_end = _timestamp_to_datetime(ts_end_ms)
        results.append(
            {
                "timestamp_start": ts_start.isoformat(),
                "timestamp_end": ts_end.isoformat(),
                "room_id": room_id,
                "building_id": building_id,
                "occupied": occupied,
                "occupied_probability": round(float(prob), 6),
                "rx_mac": rx_mac,
            }
        )
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
    results = run_inference_for_records(records, engine, sequence_length=SEQUENCE_LENGTH)
    write_results_to_local_dir(results)
    post_results(client, results)
    return results


def run_once_all_data(
    client: httpx.Client,
    engine,
    *,
    rx_mac: Optional[str] = None,
    room_id: Optional[str] = None,
    building_id: Optional[str] = None,
    page_size: int = FETCH_LIMIT,
) -> List[Dict[str, Any]]:
    """Fetch all radar data (paginate), run inference once, save and POST results. Returns list of result dicts."""
    records = fetch_all_radar_data(
        client,
        rx_mac=rx_mac,
        room_id=room_id,
        building_id=building_id,
        page_size=page_size,
    )
    results = run_inference_for_records(records, engine, sequence_length=SEQUENCE_LENGTH)
    write_results_to_local_dir(results)
    post_results(client, results)
    return results


def run_once_db(
    conn: Any,
    engine: Any,
    *,
    rx_mac: Optional[str] = None,
    room_id: Optional[str] = None,
    building_id: Optional[str] = None,
    limit: int = FETCH_LIMIT,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Fetch latest radar data from DB, run inference, insert into inference_data. Returns list of result dicts."""
    logger.info(
        "run_once_db started (limit=%s, offset=%s, rx_mac=%s, room_id=%s, building_id=%s)",
        limit, offset, rx_mac, room_id, building_id,
    )
    records = fetch_radar_data_from_db(
        conn,
        rx_mac=rx_mac,
        room_id=room_id,
        building_id=building_id,
        limit=limit,
        offset=offset,
        min_timestamp_ms=None,
    )
    logger.info("run_once_db: got %d records, running inference", len(records))
    results = run_inference_for_records(records, engine, sequence_length=SEQUENCE_LENGTH)
    logger.info("run_once_db: inference produced %d result(s)", len(results))
    insert_inference_results(conn, results)
    write_results_to_local_dir(results)
    return results


def _filter_results_with_watermark(
    results: List[Dict[str, Any]],
    watermark: Dict[str, int],
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Filter out results that are not newer than the watermark for their location.

    Returns (filtered_results, updated_watermark).
    """
    updated = dict(watermark)
    filtered: List[Dict[str, Any]] = []
    for r in results:
        key = _location_key(r["room_id"], r["building_id"], r["rx_mac"])
        ts_end_ms = _iso_timestamp_to_ms(r["timestamp_end"])
        last_ms = updated.get(key)
        if last_ms is not None and ts_end_ms <= last_ms:
            logger.info(
                "Skipping result for (%s): timestamp_end %s (ms=%s) not newer than watermark %s",
                key,
                r["timestamp_end"],
                ts_end_ms,
                last_ms,
            )
            continue
        filtered.append(r)
        updated[key] = ts_end_ms
    return filtered, updated


def run_once_db_incremental(
    engine: Any,
    *,
    rx_mac: Optional[str] = None,
    room_id: Optional[str] = None,
    building_id: Optional[str] = None,
    limit: int = FETCH_LIMIT,
    offset: int = 0,
    watermark_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Incremental DB-based inference for the scheduler.

    - Loads per-location watermark (last timestamp_end in ms) from file.
    - Optionally fetches only radar_readings newer than the global minimum watermark.
    - Runs inference (including NULL-occupied rows for insufficient packets).
    - Filters out results that are not newer than each location's watermark.
    - Inserts only new results into inference_data and updates the watermark file.
    """
    from db import get_connection

    if not DATABASE_URL:
        raise ValueError("DATABASE_URL is not set; run_once_db_incremental requires a DB")

    watermark_file = watermark_path or INFERENCE_WATERMARK_FILE
    watermark = _load_watermark(watermark_file)
    min_timestamp_ms = min(watermark.values()) if watermark else None

    logger.info(
        "run_once_db_incremental started (limit=%s, offset=%s, rx_mac=%s, room_id=%s, building_id=%s, min_timestamp_ms=%s)",
        limit,
        offset,
        rx_mac,
        room_id,
        building_id,
        min_timestamp_ms,
    )

    with get_connection() as conn:
        records = fetch_radar_data_from_db(
            conn,
            rx_mac=rx_mac,
            room_id=room_id,
            building_id=building_id,
            limit=limit,
            offset=offset,
            min_timestamp_ms=min_timestamp_ms,
        )
        logger.info("run_once_db_incremental: got %d records, running inference", len(records))
        results = run_inference_for_records(records, engine, sequence_length=SEQUENCE_LENGTH)
        if not results:
            logger.info("run_once_db_incremental: no results produced; nothing to insert into inference_data")
            return []

        filtered, updated_watermark = _filter_results_with_watermark(results, watermark)
        if not filtered:
            logger.info(
                "run_once_db_incremental: no new inference windows after watermark; nothing inserted into inference_data",
            )
            # Still persist updated watermark in case structure changed
            _save_watermark(watermark_file, updated_watermark)
            return []

        insert_inference_results(conn, filtered)

    _save_watermark(watermark_file, updated_watermark)
    return filtered


def run_once_all_data_db(
    conn: Any,
    engine: Any,
    *,
    rx_mac: Optional[str] = None,
    room_id: Optional[str] = None,
    building_id: Optional[str] = None,
    page_size: int = FETCH_LIMIT,
    max_pages: int = MAX_PAGES_FETCH_ALL,
) -> List[Dict[str, Any]]:
    """Fetch all radar data from DB (paginate), run inference once, insert into inference_data. Returns result dicts."""
    logger.info(
        "run_once_all_data_db started (filters: rx_mac=%s, room_id=%s, building_id=%s)",
        rx_mac, room_id, building_id,
    )
    all_records: List[Dict[str, Any]] = []
    offset = 0
    for _ in range(max_pages):
        page = fetch_radar_data_from_db(
            conn,
            rx_mac=rx_mac,
            room_id=room_id,
            building_id=building_id,
            limit=page_size,
            offset=offset,
        )
        if not page:
            break
        all_records.extend(page)
        offset += page_size
    logger.info("Fetched %d total records from DB in %d page(s)", len(all_records), (offset // page_size) or 1)
    results = run_inference_for_records(all_records, engine, sequence_length=SEQUENCE_LENGTH)
    logger.info("run_once_all_data_db: inference produced %d result(s)", len(results))
    insert_inference_results(conn, results)
    write_results_to_local_dir(results)
    return results
