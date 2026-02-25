"""
FastAPI app for occupancy inference: fetch radar data, run BiLSTM inference every 15s, return/post results.
Run from occupancy-service: uv run uvicorn main:app --host 0.0.0.0 --port 8000
"""
import logging
import os
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import httpx
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

import config
from db import get_connection
from inference_job import (
    get_engine,
    run_once,
    run_once_all_data,
    run_once_all_data_db,
    run_once_db,
    run_once_db_incremental,
)

_log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=_log_format,
)
logger = logging.getLogger(__name__)

if config.LOG_FILE:
    log_path = Path(config.LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        config.LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(_log_format))
    logging.getLogger().addHandler(file_handler)
    logger.info("Logging to file: %s", config.LOG_FILE)

# Global engine (loaded at startup if MODEL_PATH/SCALER_PATH set)
_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _scheduler
    _scheduler = None
    model_path = os.getenv("MODEL_PATH", "") or getattr(config, "MODEL_PATH", "")
    scaler_path = os.getenv("SCALER_PATH", "") or getattr(config, "SCALER_PATH", "")
    if model_path and scaler_path:
        try:
            _engine = get_engine(model_path, scaler_path)
            logger.info("Model and scaler loaded")
            if config.RUN_SCHEDULER:
                _scheduler = BackgroundScheduler()
                _scheduler.add_job(
                    _scheduled_job,
                    "interval",
                    seconds=config.INFERENCE_INTERVAL_SECONDS,
                    id="inference",
                )
                _scheduler.start()
                logger.info("Scheduler started: inference every %ds", config.INFERENCE_INTERVAL_SECONDS)
        except Exception as e:
            logger.exception("Failed to load model/scaler: %s", e)
    else:
        logger.warning("MODEL_PATH or SCALER_PATH not set; inference endpoints will fail")
    yield
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
    _engine = None


app = FastAPI(title="Occupancy Inference Service", lifespan=lifespan)


@app.get("/health")
async def health():
    """Check service, optional radar API reachability, and DB connectivity when DATABASE_URL is set."""
    ok = _engine is not None
    radar_ok = False
    db_ok = False
    if config.DATABASE_URL:
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            db_ok = True
        except Exception as e:
            db_ok = False
            logger.info("Health check: database unreachable: %s", e)
    else:
        try:
            r = httpx.get(config.RADAR_DATA_URL, params={"limit": 1}, timeout=5.0)
            radar_ok = r.status_code == 200 and r.json().get("success") is True
        except Exception:
            radar_ok = False
    return {
        "status": "ok" if ok else "degraded",
        "model_loaded": ok,
        "database_ok": db_ok if config.DATABASE_URL else None,
        "radar_api_reachable": radar_ok if not config.DATABASE_URL else None,
    }


@app.post("/run-inference")
async def run_inference(
    rx_mac: str | None = Query(None),
    room_id: str | None = Query(None),
    building_id: str | None = Query(None),
    limit: int = Query(config.FETCH_LIMIT, ge=1, le=2000),
    offset: int = Query(0, ge=0),
):
    """
    Fetch radar data (with optional location filters), run inference per (room_id, building_id, rx_mac),
    return results. When DATABASE_URL is set, reads from PostgreSQL and writes to inference_data.
    """
    if _engine is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "Model not loaded; set MODEL_PATH and SCALER_PATH"},
        )
    logger.info(
        "run_inference request (limit=%s, offset=%s, rx_mac=%s, room_id=%s, building_id=%s)",
        limit, offset, rx_mac, room_id, building_id,
    )
    try:
        if config.DATABASE_URL:
            logger.info("Using DB")
            with get_connection() as conn:
                results = run_once_db(
                    conn,
                    _engine,
                    rx_mac=rx_mac,
                    room_id=room_id,
                    building_id=building_id,
                    limit=limit,
                    offset=offset,
                )
        else:
            logger.info("Using radar API")
            with httpx.Client() as client:
                results = run_once(
                    client,
                    _engine,
                    rx_mac=rx_mac,
                    room_id=room_id,
                    building_id=building_id,
                    limit=limit,
                    offset=offset,
                )
        return {"results": results}
    except Exception as e:
        logger.exception("run_inference failed: %s", e)
        return JSONResponse(status_code=500, content={"detail": str(e)})


@app.post("/run-inference-all")
async def run_inference_all(
    rx_mac: str | None = Query(None),
    room_id: str | None = Query(None),
    building_id: str | None = Query(None),
):
    """
    Fetch all radar data (paginate), run inference once, return results.
    When DATABASE_URL is set, reads from PostgreSQL and writes to inference_data.
    """
    if _engine is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "Model not loaded; set MODEL_PATH and SCALER_PATH"},
        )
    logger.info(
        "run_inference_all request (rx_mac=%s, room_id=%s, building_id=%s)",
        rx_mac, room_id, building_id,
    )
    try:
        if config.DATABASE_URL:
            logger.info("Using DB")
            with get_connection() as conn:
                results = run_once_all_data_db(
                    conn,
                    _engine,
                    rx_mac=rx_mac,
                    room_id=room_id,
                    building_id=building_id,
                )
        else:
            logger.info("Using radar API")
            with httpx.Client() as client:
                results = run_once_all_data(
                    client,
                    _engine,
                    rx_mac=rx_mac,
                    room_id=room_id,
                    building_id=building_id,
                )
        return {"results": results}
    except Exception as e:
        logger.exception("run_inference_all failed: %s", e)
        return JSONResponse(status_code=500, content={"detail": str(e)})


def _scheduled_job():
    """Called every INFERENCE_INTERVAL_SECONDS by the scheduler."""
    if _engine is None:
        return
    logger.info("Scheduled inference job starting")
    try:
        if config.DATABASE_URL:
            logger.info("Using DB (incremental)")
            run_once_db_incremental(_engine)
        else:
            logger.info("Using radar API")
            with httpx.Client() as client:
                run_once(client, _engine)
    except Exception as e:
        logger.exception("Scheduled inference job failed: %s", e)


_scheduler = None


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
