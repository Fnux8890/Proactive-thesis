from __future__ import annotations

"""Prefect flow orchestrator – thin wrapper delegating to dao, transforms,
validation, and loading modules.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import polars as pl
from prefect import flow
from prefect.exceptions import MissingContextError
from prefect.logging.loggers import get_run_logger as prefect_get_run_logger

from .config import load_config
from dao.sensor_repository import SensorRepository
from transforms.core import transform_features
from validation.ge_runner import validate_with_ge
from loading.feast_loader import persist_features


# ---------------------------------------------------------------------------
# Logger helper (works inside & outside Prefect context)
# ---------------------------------------------------------------------------

def _logger() -> logging.Logger:
    try:
        return prefect_get_run_logger()
    except MissingContextError:
        return logging.getLogger("flow_main")


# ---------------------------------------------------------------------------
# Prefect flow
# ---------------------------------------------------------------------------

@flow(log_prints=True)
async def main_feature_flow(run_date_str: str = "auto") -> None:
    log = _logger()
    log.info("Main daily feature flow started …")

    # 1. determine run_date
    if run_date_str == "auto":
        run_date = (datetime.utcnow() - timedelta(days=1)).date()
    else:
        run_date = datetime.strptime(run_date_str, "%Y-%m-%d").date()
    start_ts = datetime.combine(run_date, datetime.min.time())
    end_ts = start_ts + timedelta(days=1)

    # 2. load configuration
    cfg_path = Path("/app/plant_config.json")
    cfg = load_config(cfg_path)

    # 3. extract raw data
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL not set")

    async with SensorRepository(db_url) as repo:
        arrow_tbl = await repo.get_sensor_data(start_ts, end_ts)
    if arrow_tbl is None:
        log.warning("No rows for date %s – nothing to do", run_date)
        return
    raw_df = pl.from_arrow(arrow_tbl)

    # 4. transform
    feat_df = transform_features(raw_df, cfg)

    # 5. validate
    if not validate_with_ge(feat_df, skip_validation=True):
        raise ValueError("Great-Expectations validation failed – aborting load stage")

    # 6. persist (parquet + Feast)
    persist_features(feat_df, run_date)
    log.info("Flow finished for %s", run_date)


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    date_arg = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    asyncio.run(main_feature_flow(date_arg)) 