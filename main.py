"""
FastAPI app for occupancy inference: fetch radar data, run BiLSTM inference every 15s, return/post results.
Run from repo root: uv run --project occupancy-service uvicorn main:app --app-dir occupancy-service
Or from occupancy-service: PYTHONPATH=.. uvicorn main:app
"""
import logging
import os
import sys
from pathlib import Path

# Add repo root (inference_code) and this app dir (config, inference_job)
_app_dir = Path(__file__).resolve().parent
_repo_root = _app_dir.parent
for p in (_app_dir, _repo_root):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

import config
from inference_job import get_engine, run_once

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

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
                from apscheduler.schedulers.background import BackgroundScheduler
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
    """Check service and optional radar API reachability."""
    ok = _engine is not None
    try:
        import httpx
        r = httpx.get(config.RADAR_DATA_URL, params={"limit": 1}, timeout=5.0)
        radar_ok = r.status_code == 200 and r.json().get("success") is True
    except Exception:
        radar_ok = False
    return {
        "status": "ok" if ok else "degraded",
        "model_loaded": ok,
        "radar_api_reachable": radar_ok,
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
    return and optionally POST results.
    """
    if _engine is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "Model not loaded; set MODEL_PATH and SCALER_PATH"},
        )
    import httpx
    try:
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


def _scheduled_job():
    """Called every INFERENCE_INTERVAL_SECONDS by the scheduler."""
    if _engine is None:
        return
    import httpx
    try:
        with httpx.Client() as client:
            run_once(client, _engine)
    except Exception as e:
        logger.exception("Scheduled inference job failed: %s", e)


_scheduler = None


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
