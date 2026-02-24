"""Configuration from environment."""
import os
from pathlib import Path

# Repo root (Darwin) for model/scaler paths and inference_code imports
REPO_ROOT = Path(__file__).resolve().parent.parent

DATABASE_URL = os.getenv("DATABASE_URL", "")
RADAR_DATA_URL = os.getenv("RADAR_DATA_URL", "http://localhost:3000/api/radar-data")
INFERENCE_RESULTS_URL = os.getenv("INFERENCE_RESULTS_URL", "")  # Optional; when set, POST results here
INFERENCE_RESULTS_DIR = os.getenv("INFERENCE_RESULTS_DIR", "")  # Optional; when set, write results to this dir as JSON
MODEL_PATH = os.getenv("MODEL_PATH", "")
SCALER_PATH = os.getenv("SCALER_PATH", "")
FETCH_LIMIT = int(os.getenv("FETCH_LIMIT", "500"))
RUN_SCHEDULER = os.getenv("RUN_SCHEDULER", "true").lower() in ("1", "true", "yes")
INFERENCE_INTERVAL_SECONDS = int(os.getenv("INFERENCE_INTERVAL_SECONDS", "15"))
LOG_FILE = os.getenv("LOG_FILE", "")  # When set, logs are also written to this file (with rotation)
