from __future__ import annotations

import asyncpg
import logging
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse, urlunparse

import pyarrow as pa
from prefect.logging import get_run_logger
from asyncpg.exceptions import DuplicateDatabaseError

__all__ = ["SensorRepository", "ensure_database_exists"]


def _logger() -> logging.Logger:
    try:
        return get_run_logger()
    except Exception:
        return logging.getLogger("sensor_repository")


class SensorRepository:
    """Async context-managed Timescale/Postgres connection pool.

    Usage::

        async with SensorRepository(db_url) as repo:
            arrow_tbl = await repo.get_sensor_data(start, end)
    """

    def __init__(self, db_url: str, *, min_size: int = 1, max_size: int = 5):
        self._db_url = db_url
        self._pool: Optional[asyncpg.Pool] = None
        self._min_size = min_size
        self._max_size = max_size

    # ------------------------------------------------------------------
    # Async context manager helpers
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "SensorRepository":
        log = _logger()
        log.info("Creating asyncpg pool …")
        self._pool = await asyncpg.create_pool(
            self._db_url, min_size=self._min_size, max_size=self._max_size
        )
        # quick health-check
        async with self._pool.acquire() as conn:
            await conn.execute("SELECT 1")
        log.info("Database connection ready.")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._pool is not None:
            await self._pool.close()
            _logger().info("Connection pool closed.")

    # ------------------------------------------------------------------
    # Data extraction
    # ------------------------------------------------------------------

    async def get_sensor_data(self, start: datetime, end: datetime) -> pa.Table | None:
        """Fetch sensor rows between *start* (inclusive) and *end* (exclusive) from sensor_data_merged."""
        if self._pool is None:
            raise RuntimeError("Pool not initialised – use as async context manager")

        SQL = (
            'SELECT * FROM public.sensor_data_merged '
            'WHERE "time" >= $1 AND "time" < $2 ORDER BY "time";'
        )
        log = _logger()
        log.debug("Fetching sensor rows from public.sensor_data_merged from %s to %s", start, end)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(SQL, start, end)
        if not rows:
            log.warning("No sensor rows returned for interval from public.sensor_data_merged.")
            return None

        cols = {k: [r[k] for r in rows] for k in rows[0].keys()}
        return pa.table(cols)


# --- Database Utility ---

async def ensure_database_exists(db_url: str, db_name: str):
    """Connects to the default 'postgres' database and creates the target database if it doesn't exist."""
    log = _logger()
    conn = None
    try:
        parsed_url = urlparse(db_url)
        default_db_path = "/postgres"
        default_db_url_parts = parsed_url._replace(path=default_db_path)
        default_db_url = urlunparse(default_db_url_parts)

        log.debug(f"Connecting to default DB URL '{default_db_url}' to check/create '{db_name}'")
        conn = await asyncpg.connect(default_db_url)

        exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", db_name)

        if exists:
            log.info(f"Database '{db_name}' already exists.")
        else:
            log.info(f"Database '{db_name}' does not exist. Attempting to create...")
            await conn.execute(f'CREATE DATABASE "{db_name}"')
            log.info(f"Database '{db_name}' created successfully.")

    except DuplicateDatabaseError:
        log.warning(f"Database '{db_name}' already exists (caught DuplicateDatabaseError).")
    except Exception as e:
        log.error(f"Failed to ensure database '{db_name}' exists: {e}", exc_info=True)
        raise
    finally:
        if conn:
            await conn.close() 