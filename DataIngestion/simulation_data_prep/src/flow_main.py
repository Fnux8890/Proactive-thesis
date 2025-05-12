from __future__ import annotations

"""Prefect flow orchestrator for processing greenhouse sensor data segments.

Orchestrates data extraction, cleaning (outliers, imputation), feature transformation,
validation, and persistence for defined data segments.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta, date # Added date
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import polars as pl
import pandas as pd # Keep pandas for potential temporary use if cleaning functions need it
from prefect import flow, task
from prefect.exceptions import MissingContextError
from prefect.logging.loggers import get_run_logger as prefect_get_run_logger

# Core pipeline modules
from .config import load_plant_config, load_data_processing_config, DataProcessingConfig, PlantConfig
from dao.sensor_repository import SensorRepository, ensure_database_exists
from .data_cleaning import apply_outlier_treatment_pl, impute_missing_data_pl
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
        # Fallback logger for running outside Prefect context (e.g., local testing)
        logger = logging.getLogger("flow_main_local")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger


# ---------------------------------------------------------------------------
# Helper to check env var for boolean flags
# ---------------------------------------------------------------------------
def _is_truthy(value: str | None) -> bool:
    return value is not None and value.lower() in ('true', '1', 't', 'y', 'yes')


# ---------------------------------------------------------------------------
# Prefect Tasks for Pipeline Stages
# ---------------------------------------------------------------------------

@task
def load_configurations_task(plant_cfg_path_str: str, data_proc_cfg_path_str: str) -> Tuple[PlantConfig, DataProcessingConfig]:
    """Loads plant biology and data processing configurations."""
    log = _logger()
    try:
        plant_cfg = load_plant_config(Path(plant_cfg_path_str))
        log.info("Successfully loaded plant configuration.")
        data_proc_cfg = load_data_processing_config(Path(data_proc_cfg_path_str))
        log.info("Successfully loaded data processing configuration.")
        return plant_cfg, data_proc_cfg
    except FileNotFoundError as fnf_e:
        log.exception(f"Configuration file not found: {fnf_e}")
        raise
    except Exception as cfg_e:
        log.exception(f"Error loading configuration: {cfg_e}")
        raise

@task
def determine_segment_task(segment_name: str, data_proc_cfg: DataProcessingConfig) -> Tuple[date, date]:
    """Determines the start and end dates for a given segment name from the config."""
    log = _logger()
    log.info(f"Determining date range for segment: '{segment_name}'")
    if not hasattr(data_proc_cfg, 'data_segments') or not data_proc_cfg.data_segments:
        log.error("'data_segments' not found or empty in DataProcessingConfig.")
        raise ValueError("Configuration missing 'data_segments'.")

    segment_found = None
    for segment in data_proc_cfg.data_segments:
        if segment.get('name') == segment_name:
            segment_found = segment
            break

    if not segment_found:
        log.error(f"Segment '{segment_name}' not defined in data_processing_config.json -> data_segments.")
        raise ValueError(f"Segment '{segment_name}' not found in configuration.")

    try:
        start_date_str = segment_found.get('start_date')
        end_date_str = segment_found.get('end_date')
        if not start_date_str or not end_date_str:
             raise ValueError(f"Segment '{segment_name}' missing start_date or end_date.")
        
        # Parse dates assuming ISO format like "YYYY-MM-DDTHH:MM:SSZ"
        start_date = datetime.fromisoformat(start_date_str.replace('Z', '+00:00')).date()
        end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00')).date()
        
        # Convert dates to start/end timestamps for retrieval (start inclusive, end exclusive)
        start_ts = datetime.combine(start_date, datetime.min.time())
        end_ts = datetime.combine(end_date + timedelta(days=1), datetime.min.time()) # End date is inclusive in config, so add 1 day for query

        log.info(f"Segment '{segment_name}' corresponds to: {start_ts} (inclusive) to {end_ts} (exclusive)")
        return start_ts, end_ts
    except Exception as e:
        log.exception(f"Error parsing dates for segment '{segment_name}': {e}")
        raise ValueError(f"Invalid date format or missing dates for segment '{segment_name}'.")

@task
async def extract_data_task(data_proc_cfg: DataProcessingConfig, start_ts: datetime, end_ts: datetime) -> pl.DataFrame:
    """Ensures DB exists and extracts data for the specified time range using SensorRepository."""
    log = _logger()
    db_url_for_repo: Optional[str] = None
    target_db_name_for_repo: Optional[str] = None
    if data_proc_cfg and data_proc_cfg.db_connection:
        cfg_db = data_proc_cfg.db_connection
        db_url_for_repo = f"postgresql://{cfg_db.user}:{cfg_db.password}@{cfg_db.host}:{cfg_db.port}/{cfg_db.dbname}"
        target_db_name_for_repo = cfg_db.dbname
        log.info(f"Using DB URL for SensorRepository: postgresql://{cfg_db.user}:******@{cfg_db.host}:{cfg_db.port}/{cfg_db.dbname}")
    else:
        log.error("Database connection parameters not found in DataProcessingConfig.")
        raise RuntimeError("Missing DB connection configuration.")

    # Ensure DB exists
    try:
        if not target_db_name_for_repo:
             raise ValueError("Target database name is not configured.")
        log.info(f"Ensuring database '{target_db_name_for_repo}' exists...")
        await ensure_database_exists(db_url_for_repo, target_db_name_for_repo)
        log.info(f"Database '{target_db_name_for_repo}' confirmed or created.")
    except Exception as ensure_db_e:
        log.exception(f"Failed to ensure database '{target_db_name_for_repo}' exists: {ensure_db_e}")
        raise
    
    # Extract Data
    raw_arrow_tbl: pa.Table | None = None
    try:
        async with SensorRepository(db_url_for_repo) as repo:
            log.info(f"Extracting data from {start_ts} to {end_ts}")
            raw_arrow_tbl = await repo.get_sensor_data(start_ts, end_ts)
        
        if raw_arrow_tbl is None:
            log.warning("No sensor rows found for this segment.")
            # Return empty Polars DataFrame with expected schema if possible? 
            # For now, returning empty; flow might need to handle this.
            return pl.DataFrame() 
        
        raw_df = pl.from_arrow(raw_arrow_tbl)
        log.info(f"Successfully extracted {raw_df.height} raw rows via SensorRepository.")
        return raw_df
    except Exception as e_extract:
        log.exception(f"Error during data extraction: {e_extract}")
        raise

@task
def merge_timestamps_task(raw_df: pl.DataFrame) -> pl.DataFrame:
    """Merges rows with duplicate timestamps, keeping the first entry."""
    log = _logger()
    if raw_df.is_empty(): return raw_df
    try:
        n_raw = raw_df.height
        log.info(f"Checking {n_raw} rows for duplicate timestamps...")
        # Polars handles time conversion automatically if data is Arrow timestamp
        # Check if 'time' column exists
        if "time" not in raw_df.columns:
            log.error("'time' column missing, cannot merge timestamps.")
            raise ValueError("Missing 'time' column for merging.")
            
        merged_df = raw_df.group_by("time", maintain_order=True).agg(
            pl.all().exclude("time").first()
        )
        n_merged = merged_df.height
        if n_raw > n_merged:
            log.info(f"Merged {n_raw - n_merged} rows with duplicate timestamps. Proceeding with {n_merged} unique timestamp rows.")
        else:
            log.info("No duplicate timestamps found to merge.")
        return merged_df
    except Exception as e_merge:
        log.exception(f"Error during timestamp merging process: {e_merge}")
        raise

@task
def clean_data_task(df: pl.DataFrame, data_proc_cfg: DataProcessingConfig, segment_name: str) -> pl.DataFrame:
    """Applies outlier handling and imputation using Polars functions from data_cleaning module."""
    log = _logger()
    if df.is_empty(): return df

    # --- Outlier Handling (Polars) --- 
    outlier_rules = []
    if hasattr(data_proc_cfg, 'outlier_detection') and data_proc_cfg.outlier_detection:
        # Example logic: Use segment-specific rules if they exist, else global
        # This assumes segment rules might be stored like `outlier_detection.segment_rules[segment_name]` 
        # Or adjust based on your final config structure. For now, using global rules.
        if hasattr(data_proc_cfg.outlier_detection, 'rules') and data_proc_cfg.outlier_detection.rules:
            # Need to convert Pydantic OutlierRule models to simple dicts if the function expects dicts
            outlier_rules = [rule.model_dump() for rule in data_proc_cfg.outlier_detection.rules]
            log.info(f"Using global outlier rules for segment '{segment_name}'. Count: {len(outlier_rules)}")
        else:
             log.warning(f"No outlier rules found in config for segment '{segment_name}'.")
    else:
        log.warning("'outlier_detection' section not found in config.")

    df_cleaned = df # Start with the input df
    if outlier_rules:
        log.info(f"Applying outlier treatment...")
        try:
            # Call the Polars version
            df_cleaned = apply_outlier_treatment_pl(df, outlier_rules) # Assuming default time_col='time' is ok
            log.info("Outlier treatment applied.")
        except Exception as e_outlier:
            log.exception(f"Error applying Polars outlier treatment: {e_outlier}. Proceeding with data as is.")
            df_cleaned = df # Fallback to original on error
    else:
        log.info("Skipping outlier treatment as no rules were defined/found.")

    # --- Imputation (Polars) --- 
    imputation_strategies = {}
    default_strategy_dict = None
    if hasattr(data_proc_cfg, 'imputation'):
        # Similar logic for segment-specific imputation strategies if needed
        if hasattr(data_proc_cfg.imputation, 'column_specific_strategies'):
            # Convert Pydantic ImputationStrategy models to dicts
            imputation_strategies = {col: strategy.model_dump() for col, strategy in data_proc_cfg.imputation.column_specific_strategies.items()}
            log.info(f"Using general column-specific imputation strategies for segment '{segment_name}'.")
        
        if hasattr(data_proc_cfg.imputation, 'default_strategy') and data_proc_cfg.imputation.default_strategy:
             default_strategy_dict = data_proc_cfg.imputation.default_strategy.model_dump()
             log.info(f"Using default imputation strategy: {default_strategy_dict}")
    else:
        log.warning("'imputation' section not found in config. Imputation might use internal defaults.")

    log.info("Applying missing data imputation...")
    try:
        # Call the Polars version
        df_imputed = impute_missing_data_pl(
            df_cleaned, 
            imputation_strategies=imputation_strategies, 
            default_strategy=default_strategy_dict 
            # Assuming default time_col='time' is ok
        )
        log.info("Imputation applied.")
    except Exception as e_impute:
        log.exception(f"Error applying Polars imputation: {e_impute}. Proceeding with data post-outlier handling.")
        df_imputed = df_cleaned # Fallback on error

    return df_imputed

@task
def transform_features_task(cleaned_df: pl.DataFrame, plant_cfg: PlantConfig, data_proc_cfg: DataProcessingConfig, segment_name: str) -> pl.DataFrame:
    """Transforms raw data into features using domain logic and configurations."""
    log = _logger()
    if cleaned_df.is_empty(): return cleaned_df
    log.info("Starting feature transformation...")
    try:
        # Pass segment_name to transform_features if it needs to adapt logic/columns
        # This requires modifying transforms.core.transform_features signature
        # For now, assume transform_features adapts based on available columns and config
        feat_df = transform_features(cleaned_df, plant_cfg, data_proc_cfg)
        log.info("Successfully transformed features.")
        return feat_df
    except Exception as e_transform:
        log.exception(f"Error during feature transformation: {e_transform}")
        raise

@task
def validate_data_task(feat_df: pl.DataFrame, skip_validation: bool) -> None:
    """Validates the feature DataFrame using Great Expectations."""
    log = _logger()
    if feat_df.is_empty():
        log.warning("Feature DataFrame is empty, skipping validation.")
        return
    log.info("Starting data validation...")
    if not skip_validation:
        try:
            validation_passed = validate_with_ge(feat_df)
            if not validation_passed:
                log.error("Great Expectations validation failed.")
                raise ValueError("Great-Expectations validation failed.")
            else:
                log.info("Great Expectations validation passed.")
        except Exception as e_validate:
            log.exception(f"Error during data validation: {e_validate}")
            raise
    else:
        log.info("Skipping Great Expectations validation as configured.")

@task
async def persist_features_task(feat_df: pl.DataFrame, run_date: date, segment_name: str) -> None:
    """Persists the feature DataFrame to Parquet and Feast."""
    log = _logger()
    if feat_df.is_empty():
        log.warning("Feature DataFrame is empty, skipping persistence.")
        return
        
    log.info(f"Starting data persistence for segment '{segment_name}', date {run_date}...")
    try:
        # Modify persist_features to potentially use segment_name in output path/tags
        # This requires changes to loading.feast_loader.persist_features
        # For now, passing run_date as before
        persist_features(feat_df, run_date) 
        log.info(f"Successfully persisted features for segment '{segment_name}'.")
    except Exception as e_persist:
        log.exception(f"Error during data persistence for segment '{segment_name}': {e_persist}")
        raise

# ---------------------------------------------------------------------------
# Prefect Flow Definition
# ---------------------------------------------------------------------------

@flow(log_prints=True)
async def main_feature_flow(segment_name: str) -> None: # Changed parameter
    """Main Prefect flow to process sensor data for a specific segment."""
    log = _logger()
    log.info(f"Main feature flow started for segment: '{segment_name}' …")

    # --- Configuration --- 
    cfg_path_str = os.getenv("PLANT_CONFIG_PATH", "/app/plant_config.json") 
    data_proc_cfg_path_str = os.getenv("DATA_PROCESSING_CONFIG_PATH", "/app/src/data_processing_config.json") 
    skip_validation_env = os.getenv("SKIP_GE_VALIDATION")
    skip_validation = _is_truthy(skip_validation_env)
    log.info(f"Plant config path: {cfg_path_str}")
    log.info(f"Data Processing config path: {data_proc_cfg_path_str}")
    log.info(f"Skip Great Expectations validation: {skip_validation}")

    # --- Load Configs --- 
    # This is now a task
    plant_cfg, data_proc_cfg = load_configurations_task(cfg_path_str, data_proc_cfg_path_str)

    # --- Determine Segment Dates --- 
    # This is now a task
    start_ts, end_ts = determine_segment_task(segment_name, data_proc_cfg)
    run_date = start_ts.date() # Use start date of segment for run identification

    # --- Pipeline Stages as Tasks --- 
    # 1. Extract Data
    raw_df = await extract_data_task(data_proc_cfg, start_ts, end_ts)

    # 2. Merge Timestamps
    merged_df = merge_timestamps_task(raw_df)

    # 3. Clean Data (Outliers + Imputation) 
    # This task assumes data_cleaning.py operates on Polars
    cleaned_df = clean_data_task(merged_df, data_proc_cfg, segment_name)

    # 4. Transform Features
    # Pass potentially segment-aware configs
    feat_df = transform_features_task(cleaned_df, plant_cfg, data_proc_cfg, segment_name)

    # 5. Validate Data
    # Wait for feat_df before validating
    validate_data_task(feat_df, skip_validation, wait_for=[feat_df]) 

    # 6. Persist Features
    # Wait for validation to complete (implicitly via feat_df dependency if validate_data_task has side effects or raises errors)
    await persist_features_task(feat_df, run_date, segment_name, wait_for=[feat_df]) 

    log.info(f"Flow finished successfully for segment: '{segment_name}'")

# ---------------------------------------------------------------------------
# CLI helper - Updated to require a segment name
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the Prefect feature flow for a specific segment.")
    parser.add_argument("-s", "--segment", type=str, required=True, 
                        help="Name of the data segment to process (must match a name in data_processing_config -> data_segments).")
    args = parser.parse_args()
    
    asyncio.run(main_feature_flow(segment_name=args.segment)) 