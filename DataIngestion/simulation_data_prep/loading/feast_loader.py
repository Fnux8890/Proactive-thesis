from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import polars as pl
from prefect import task
from prefect.logging import get_run_logger
from prefect.exceptions import MissingContextError

# Conditional import for Feast
try:
    import feast
    from feast import FeatureStore
    # Import the feature view definition - adjust path if structure changes
    from feast_repo.features import daily_climate_view
    FEAST_AVAILABLE = True
except ImportError:
    FEAST_AVAILABLE = False


def _logger() -> logging.Logger:
    try:
        return get_run_logger()
    except MissingContextError:
        return logging.getLogger("feast_loader")

# Keep as Prefect task if needed by flow
@task
def persist_features(
    df: pl.DataFrame,
    run_date: date,
    output_dir: str | Path = "/app/output",
    feast_repo_path: str | Path = "/app/feast_repo",
    enable_feast_ingest: bool = True,
) -> Path:
    """Saves features to Parquet and optionally ingests into Feast."""
    logger = _logger()
    if df.is_empty():
        logger.warning("DataFrame is empty, skipping persistence step.")
        # Return an indicative path, maybe empty or None?
        # For now, returning expected path but it won't exist.
        return Path(output_dir) / Path(f"features_{run_date.strftime('%Y-%m-%d')}.parquet")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    partition_date_str = run_date.strftime("%Y-%m-%d")
    output_path = output_dir / f"features_{partition_date_str}.parquet"

    logger.info(f"Saving features to {output_path}")
    try:
        df.write_parquet(output_path, use_pyarrow=True)
        logger.info("Features saved to Parquet successfully.")
    except Exception as e:
        logger.error(f"Failed to save features to {output_path}: {e}")
        raise

    # ---------------- Feast ingestion (optional) -----------------
    if enable_feast_ingest and FEAST_AVAILABLE:
        logger.info("Attempting Feast ingestion...")
        try:
            pandas_df = df.to_pandas()

            # Map columns if missing for Feast
            if "event_timestamp" not in pandas_df.columns and "time" in pandas_df.columns:
                pandas_df["event_timestamp"] = pandas_df["time"]
            if "greenhouse_id" not in pandas_df.columns:
                pandas_df["greenhouse_id"] = 1 # Assuming single greenhouse

            store = FeatureStore(repo_path=str(feast_repo_path))
            store.apply([daily_climate_view]) # Ensure view is registered
            store.ingest(feature_view=daily_climate_view, source=pandas_df)
            logger.info("Feast ingest completed (offline + online stores updated).")
        except Exception as feast_e:
            logger.error(f"Feast ingestion failed: {feast_e}", exc_info=True)
            # Decide if this should raise or just warn
            # raise  # Uncomment to fail the task on Feast error
    elif enable_feast_ingest and not FEAST_AVAILABLE:
        logger.warning("Feast ingestion enabled but Feast library not found. Skipping.")
    else:
        logger.info("Feast ingestion disabled. Skipping.")

    return output_path 