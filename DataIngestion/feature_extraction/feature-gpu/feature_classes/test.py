#!/usr/bin/env -S uv run --isolated
# /// script
# dependencies = ["pandas", "cudf-cu12", "numpy", "sqlalchemy", "psycopg2-binary"] # Added sqlalchemy & driver
# ///

import cudf
import pandas as pd
import numpy as np
import logging
import os
import json
from sqlalchemy import create_engine, text
from typing import Dict, Any

# Import your feature classes
from simple import RollingStatisticalFeatures
from moderate import ModerateComplexityFeatures
# from hard import HardComplexityOrCPUFeatures

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
logger = logging.getLogger("test_feature_classes")

# --- Database Configuration (for fetching sample data) ---
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
# IMPORTANT: Adjust DB_HOST depending on where test.py is run relative to the DB
# If test.py is run in a Docker container on the same network as a DB service named 'db':
DB_HOST = os.getenv("DB_HOST", "db") 
# If test.py is run locally and DB is on localhost (or port-mapped from Docker to localhost):
# DB_HOST = os.getenv("DB_HOST", "localhost") 
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Simplified DB Connector for testing
class SimpleDBConnector:
    def __init__(self, db_url):
        self.engine = None
        try:
            self.engine = create_engine(db_url)
            with self.engine.connect() as connection:
                logger.info(f"DBConnector: Successfully connected to {db_url.replace(DB_PASSWORD, '******')}")
        except Exception as e:
            logger.error(f"DBConnector: Error creating engine for {db_url.replace(DB_PASSWORD, '******')}: {e}")
            raise

    def fetch_data_to_pandas(self, query: str, params: Dict[str, Any] = None) -> pd.DataFrame:
        if not self.engine:
            raise ConnectionError("DBConnector: Database engine not initialized.")
        try:
            with self.engine.connect() as connection:
                df = pd.read_sql_query(sql=text(query).bindparams(**params) if params else text(query), con=connection)
            return df
        except Exception as e:
            logger.error(f"DBConnector: Error fetching data to Pandas: {e}")
            raise
    
    def dispose(self):
        if self.engine:
            self.engine.dispose()
            logger.info("DBConnector: Database engine disposed.")

def fetch_sample_data_from_db(num_rows=200, era_identifier='Era1') -> cudf.DataFrame:
    """Fetches a small sample of actual data from preprocessed_features table."""
    logger.info(f"--- Fetching sample data from DB for Era: {era_identifier}, Limit: {num_rows} ---")
    connector = None
    try:
        connector = SimpleDBConnector(DB_URL)
        query = f"""
        SELECT time, era_identifier, features 
        FROM public.preprocessed_features
        WHERE era_identifier = :era_id
        ORDER BY time ASC
        LIMIT :row_limit;
        """
        params = {"era_id": era_identifier, "row_limit": num_rows}
        
        pdf_raw = connector.fetch_data_to_pandas(query, params)
        
        if pdf_raw.empty:
            logger.warning("Fetched empty DataFrame from DB for sample.")
            return cudf.DataFrame()

        logger.info(f"Successfully fetched {len(pdf_raw)} raw rows from DB for sample.")

        # Unnest features JSONB (assuming it's a list of dicts, pd.json_normalize handles Series of dicts)
        # If pdf_raw["features"] is string representation of JSON, it needs json.loads first
        # Based on extract_features_gpu.py, it should be dicts after initial pandas load
        try:
            # Attempt to load if it's stringified JSON in the sample (less likely from preprocess.py output)
            # If preprocess.py stores dicts, this json.loads isn't needed
            # For safety, check type of first element
            if not pdf_raw.empty and isinstance(pdf_raw['features'].iloc[0], str):
                logger.info("Sample features column appears to be string-encoded JSON, applying json.loads.")
                features_series = pdf_raw["features"].apply(json.loads)
            else:
                features_series = pdf_raw["features"] # Assume it's already a series of dicts
            
            feat_pdf = pd.json_normalize(features_series)
        except Exception as e:
            logger.error(f"Error normalizing 'features' JSON in sample: {e}. First element: {pdf_raw['features'].iloc[0] if not pdf_raw.empty else 'N/A'}")
            return cudf.DataFrame()
            
        full_pdf = pd.concat([pdf_raw[["time", "era_identifier"]], feat_pdf], axis=1)
        logger.info(f"Sample data - Loaded and unnested to {full_pdf.shape[0]} rows, {full_pdf.shape[1]} columns.")

        cdf = cudf.from_pandas(full_pdf)
        cdf["time"] = cudf.to_datetime(cdf["time"], utc=True) # Ensure time is datetime and UTC
        cdf = cdf.set_index('time').sort_index() # Set DatetimeIndex
        
        logger.info(f"Created sample cuDF DataFrame from DB with shape: {cdf.shape}, Columns: {cdf.columns.to_list()}")
        return cdf

    except Exception as e:
        logger.error(f"Failed to fetch or process sample data from DB: {e}")
        return cudf.DataFrame()
    finally:
        if connector:
            connector.dispose()


def main():
    logger.info("--- Starting Feature Class Test Script ---")
    sample_cdf = fetch_sample_data_from_db(num_rows=200, era_identifier='Era1')

    if sample_cdf.empty:
        logger.error("Failed to load any valid sample data from DB. Aborting tests.")
        return

    # Now that we have sample_cdf (and it's not empty), proceed to feature testing.
    # The dynamic column selection within each test section will handle finding usable columns.

    # --- Test RollingStatisticalFeatures --- 
    logger.info("\n--- Testing RollingStatisticalFeatures ---")
    stats_to_compute_custom = ['mean', 'median', 'quantile_25', 'std', 'min', 'max', 'var', 'skew', 'kurtosis']
    simple_calculator_custom = RollingStatisticalFeatures(statistics=stats_to_compute_custom)
    window_size_rows_10 = 10 

    simple_test_col_candidates = ['air_temp_c', 'radiation_w_m2', 'co2_measured_ppm', 'spot_price_dkk_mwh', 'temperature_2m', 'relative_humidity_percent']
    target_col_for_simple_test = None
    for col_candidate in simple_test_col_candidates:
        if col_candidate in sample_cdf.columns:
            target_col_for_simple_test = col_candidate
            logger.info(f"Found '{target_col_for_simple_test}' for simple features testing.")
            break

    if target_col_for_simple_test:
        logger.info(f"Calculating simple stats for '{target_col_for_simple_test}', {window_size_rows_10}-sample window...")
        # Ensure the column actually has data to process to avoid all-NA outputs if possible
        if not sample_cdf[target_col_for_simple_test].isnull().all():
            simple_feats = simple_calculator_custom.compute(sample_cdf[target_col_for_simple_test], window_size_rows_10, f"{window_size_rows_10}samples")
            if not simple_feats.empty:
                print(f"Simple Features for '{target_col_for_simple_test}' ({window_size_rows_10}-sample window) - Head:")
                print(simple_feats.head())
            else:
                print(f"No simple features generated for '{target_col_for_simple_test}'.")
        else:
            logger.warning(f"Column '{target_col_for_simple_test}' is all NaN. Skipping simple feature calculation for it.")
    else:
        logger.error(f"Could not find any suitable candidate columns {simple_test_col_candidates} for simple features testing in sample_cdf.")
        logger.info(f"Available columns for testing: {sample_cdf.columns.to_list()}")

    # --- Test ModerateComplexityFeatures --- 
    logger.info("\n--- Testing ModerateComplexityFeatures ---")
    moderate_calculator = ModerateComplexityFeatures()
    moderate_test_col_candidates = ['radiation_w_m2', 'air_temp_c', 'outside_temp_c', 'wind_speed_10m'] 
    target_col_for_moderate_test = None
    for col_candidate in moderate_test_col_candidates:
        if col_candidate in sample_cdf.columns:
            target_col_for_moderate_test = col_candidate
            logger.info(f"Found '{target_col_for_moderate_test}' for moderate features testing.")
            break

    if target_col_for_moderate_test:
        logger.info(f"Calculating moderate features (rolling_sum_abs_diff) for '{target_col_for_moderate_test}', {window_size_rows_10}-sample window...")
        if not sample_cdf[target_col_for_moderate_test].isnull().all():
            moderate_feats = moderate_calculator.compute_rolling_sum_abs_diff(sample_cdf[target_col_for_moderate_test], window_size_rows_10, f"{window_size_rows_10}samples")
            if not moderate_feats.empty:
                print(f"Moderate Features for '{target_col_for_moderate_test}' ({window_size_rows_10}-sample window) - Head:")
                print(moderate_feats.head())
            else:
                print(f"No moderate features (sum_abs_diff) generated for '{target_col_for_moderate_test}'.")
        else:
            logger.warning(f"Column '{target_col_for_moderate_test}' is all NaN. Skipping moderate feature calculation for it.")
    else:
        logger.error(f"Could not find any suitable candidate columns {moderate_test_col_candidates} for moderate features testing in sample_cdf.")
        logger.info(f"Available columns for testing: {sample_cdf.columns.to_list()}")

    logger.info("\n--- Feature Class Test Script Finished ---")

if __name__ == "__main__":
    main()
