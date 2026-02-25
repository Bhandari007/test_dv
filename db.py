"""PostgreSQL connection helper for occupancy inference (read radar_readings, write inference_data)."""
import logging
from contextlib import contextmanager
from typing import Any, Generator
from urllib.parse import urlparse

import psycopg

import config

logger = logging.getLogger(__name__)


def _sanitized_db_info() -> str:
    """Return host, port, db name for logging (no password)."""
    url = config.DATABASE_URL or ""
    if not url.strip():
        return "no URL"
    if not url.startswith(("postgresql://", "postgres://")):
        url = "postgresql://" + url.split("://", 1)[-1]
    try:
        p = urlparse(url)
        host = p.hostname or "localhost"
        port = p.port or 5432
        db = (p.path or "/").lstrip("/") or "postgres"
        return f"host={host} port={port} db={db}"
    except Exception:
        return "unknown"


@contextmanager
def get_connection() -> Generator[Any, None, None]:
    """Yield a psycopg connection from DATABASE_URL. Closes on exit."""
    if not config.DATABASE_URL:
        raise ValueError("DATABASE_URL is not set")
    db_info = _sanitized_db_info()
    try:
        conn = psycopg.connect(config.DATABASE_URL)
    except Exception as e:
        logger.exception("DB connection failed (%s): %s", db_info, e)
        raise
    logger.info("DB connection opened (%s)", db_info)
    try:
        yield conn
        conn.commit()
        logger.debug("DB connection committed and closed.")
    except Exception as e:
        conn.rollback()
        logger.warning("DB connection rollback: %s", e)
        raise
    finally:
        conn.close()
