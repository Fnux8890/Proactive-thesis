"""Feature extraction script with backend adapter support.

This script connects to the TimescaleDB instance, reads the `preprocessed_features` hypertable
 (written by the preprocessing pipeline), unfolds the JSONB `features` column into a wide
DataFrame, converts it to the long format expected by *tsfresh*, and finally extracts a rich
set of statistical features which are persisted back into the database.  The output
table is automatically promoted to a TimescaleDB hypertable so downstream steps can
query it efficiently.

Usage (inside container):

    uv run python extract_features.py

Environment variables (all have sensible defaults for docker-compose):

    DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME  –  connection parameters
    FEATURES_TABLE                                 –  destination table for the
                                                    selected features
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

# Import backend adapter components
import pandas as original_pandas
from tsfresh.feature_extraction import EfficientFCParameters

from ..backend.backend import BACKEND_TYPE, USE_GPU, DataFrame, ensure_backend, pd

# Conditional cuDF import
if USE_GPU:
    try:
        import cudf
    except ImportError:
        # Using print as logger might not be initialized at import time.
        print("WARNING: Cascade Code - USE_GPU is True, but cuDF is not available. Feature extraction will use pandas.")
        cudf = None
else:
    cudf = None
from ..backend.dtypes import ensure_datetime_index, get_numeric_columns, replace_sentinels
from ..db import chunked_query
from ..db_utils_optimized import SQLAlchemyPostgresConnector
from ..features import tsfresh_extract_features
from . import config
from .feature_utils import (
    make_hypertable_if_needed,
    select_relevant_features,
)
from .feature_utils_supervised import select_tsfresh_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------

# Use configuration constants from ``config``
DB_USER = config.DB_USER
DB_PASSWORD = config.DB_PASSWORD
DB_HOST = config.DB_HOST
DB_PORT = config.DB_PORT
DB_NAME = config.DB_NAME
FEATURES_TABLE = config.FEATURES_TABLE
FC_PARAMS_NAME = "Custom (preprocess_config.json-Guided)"
USE_GPU_FLAG = config.USE_GPU_FLAG

# Paths
era_definitions_dir_path = config.ERA_DEFINITIONS_DIR_PATH
OUTPUT_PATH = config.OUTPUT_PATH
SELECTED_OUTPUT_PATH = config.SELECTED_OUTPUT_PATH

# Supervised feature selection configuration
USE_SUPERVISED_SELECTION = config.USE_SUPERVISED_SELECTION
FDR_LEVEL = config.FDR_LEVEL
N_JOBS = config.N_JOBS
CHUNKSIZE = config.CHUNKSIZE
TARGET_COLUMN = config.TARGET_COLUMN

# Placeholder for tsfresh configuration per sensor
kind_to_fc_parameters_global: dict[str, Any] = {}


def save_dataframe_to_parquet(
    df: DataFrame,
    output_path: Path,
    df_type_name: str = "dataframe"
) -> None:
    """
    Save a DataFrame to Parquet format with proper handling for cuDF/pandas.

    Args:
        df: DataFrame to save (pandas or cuDF)
        output_path: Path where to save the parquet file
        use_gpu_flag: Whether GPU processing is enabled
        df_type_name: Descriptive name for logging (e.g., "full features", "selected features")
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get compression setting from environment with default
    compression = os.getenv("PARQUET_COMPRESSION", "snappy")

    try:
        # The backend adapter ensures we have the right DataFrame type
        # Standardize parameters for both pandas and cuDF
        df.to_parquet(output_path, compression=compression, index=False)
        logger.info(
            f"Successfully wrote {df_type_name} to {output_path} "
            f"using {BACKEND_TYPE} with compression={compression}, index=False"
        )

    except Exception as e:
        logger.error(f"Failed to write {df_type_name} to Parquet: {e}", exc_info=True)
        raise


def perform_feature_selection(features_df, consolidated_df, target_column_name):
    """Select features either using supervised or unsupervised methods based on configuration.

    Parameters
    ----------
    features_df : DataFrame
        Feature matrix with observations as rows.
    consolidated_df : DataFrame
        DataFrame containing the target column.
    target_column_name : str
        Name of the target column (typically config.TARGET_COLUMN).

    Returns
    -------
    DataFrame
        Selected features.
    """
    if USE_SUPERVISED_SELECTION and target_column_name in consolidated_df.columns:
        logger.info(
            f"Attempting supervised feature selection with target '{target_column_name}' "
            f"using global configs (FDR: {FDR_LEVEL}, N_JOBS: {N_JOBS}, CHUNKSIZE: {CHUNKSIZE})."
        )
        try:
            # Store original features_df for fallback
            original_features_df_for_fallback = features_df.copy()
            if consolidated_df[target_column_name].empty:
                logger.warning(
                    f"Target column '{target_column_name}' is empty. "
                    "Falling back to unsupervised feature selection."
                )
                return select_relevant_features(original_features_df_for_fallback)

            # Extract the target series
            y_series = consolidated_df[target_column_name]

            # Initialize aligned versions
            features_df_aligned = features_df
            y_series_aligned = y_series

            if len(y_series) < len(features_df):
                logger.warning(
                    f"Target series (len {len(y_series)}) is shorter than feature matrix (len {len(features_df)}). "
                    f"Aligning feature matrix to target series length for supervised selection."
                )
                features_df_aligned = features_df.iloc[: len(y_series)]
            elif len(y_series) > len(features_df):
                logger.info(
                    f"Target series (len {len(y_series)}) is longer than feature matrix (len {len(features_df)}). "
                    f"Aligning target series to feature matrix length."
                )
                y_series_aligned = y_series.iloc[: len(features_df)]

            # Ensure both aligned dataframes have fresh, identical 0-based RangeIndex
            features_df_aligned = features_df_aligned.reset_index(drop=True)
            y_series_aligned = y_series_aligned.reset_index(drop=True)

            # Final check for index alignment
            if not features_df_aligned.index.equals(y_series_aligned.index):
                logger.error(
                    "CRITICAL: Index mismatch detected between aligned features and target series. "
                    "Falling back to unsupervised feature selection."
                )
                return select_relevant_features(original_features_df_for_fallback)

            if y_series_aligned.empty:
                logger.warning(
                    "Aligned target series is empty. Falling back to unsupervised selection."
                )
                return select_relevant_features(original_features_df_for_fallback)

            # Perform supervised feature selection
            selected_features_df = select_tsfresh_features(
                features_df_aligned,
                y_series_aligned,
                fdr_level=FDR_LEVEL,
                n_jobs=N_JOBS,
                chunksize=CHUNKSIZE,
            )

            if selected_features_df.shape[1] == 0:
                logger.warning(
                    "Supervised feature selection resulted in 0 features. "
                    "Falling back to unsupervised selection."
                )
                return select_relevant_features(original_features_df_for_fallback)

            logger.info(
                f"Supervised feature selection reduced features from "
                f"{features_df_aligned.shape[1]} to {selected_features_df.shape[1]}."
            )
            return selected_features_df

        except Exception as e:
            logger.error(
                f"Supervised feature selection failed: {str(e)}. "
                "Falling back to unsupervised feature selection.",
                exc_info=True,
            )
            return select_relevant_features(features_df)
    else:
        logger.info("Using unsupervised feature selection.")
        return select_relevant_features(features_df)


def load_preprocessed_data() -> DataFrame | None:
    """Load data from TimescaleDB using chunked queries, ensuring pandas for intermediate processing."""
    start_date = os.getenv("FEATURE_EXTRACTION_START_DATE")
    end_date = os.getenv("FEATURE_EXTRACTION_END_DATE")

    clauses = []
    query_params = {}

    if start_date:
        clauses.append("time >= :start_date")
        query_params['start_date'] = start_date

    if end_date:
        clauses.append("time <= :end_date")
        query_params['end_date'] = end_date

    where_clause = ""
    if clauses:
        where_clause = "WHERE " + " AND ".join(clauses)
        logger.info(f"Applying date filter with params: {query_params}")

    # base_query is an f-string, so it's a string
    base_query = f"""
    WITH feature_data AS (
        SELECT
            time,
            era_identifier,
            (jsonb_each_text(features)).key as feature_name,
            (jsonb_each_text(features)).value::float as feature_value
        FROM preprocessed_features
        {where_clause}
    )
    SELECT
        time,
        era_identifier,
        jsonb_object_agg(feature_name, feature_value) as features
    FROM feature_data
    GROUP BY time, era_identifier
    ORDER BY time
    """

    sql_to_execute = base_query # This is a string
    # If query_params exist, SQLAlchemy's text() can be used for robust parameter handling
    # by some database connectors, though direct param passing to execute is often preferred.
    # For chunked_query, we pass the SQL string and params dict separately.

    logger.info("Loading data from TimescaleDB preprocessed_features hypertable...")

    chunk_size = int(os.getenv("FEATURE_EXTRACTION_CHUNK_SIZE", "10000"))

    connection_params = {
        "host": DB_HOST, "port": DB_PORT, "database": DB_NAME,
        "user": DB_USER, "password": DB_PASSWORD
    }

    pandas_chunks_list = []
    chunk_count = 0
    total_rows = 0

    for chunk in chunked_query(
        sql_to_execute,
        params=query_params if query_params else None,
        chunksize=chunk_size,
        connection_params=connection_params
    ):
        if chunk.empty:
            continue

        chunk_count += 1
        chunk_rows = len(chunk)
        total_rows += chunk_rows
        logger.info(f"Processing chunk {chunk_count} with {chunk_rows} rows (type: {type(chunk).__name__})")

        current_chunk_pandas = chunk
        if USE_GPU and cudf is not None and isinstance(chunk, cudf.DataFrame):
            logger.debug(f"Chunk {chunk_count} is cuDF, converting to pandas for json_normalize and concat.")
            current_chunk_pandas = chunk.to_pandas()
        elif not isinstance(chunk, original_pandas.DataFrame):
            # This case implies chunk is pandas if USE_GPU is False, or if cuDF import failed.
            # If it's truly some other type, it's an unexpected situation.
            if not isinstance(chunk, pd.DataFrame): # pd here is the aliased one from backend
                 logger.warning(f"Chunk {chunk_count} is unexpected type {type(chunk).__name__}. Ensure it's pandas compatible.")
            # Assuming it's pandas-like if not cuDF
            current_chunk_pandas = original_pandas.DataFrame(chunk)

        try:
            features_expanded = original_pandas.json_normalize(current_chunk_pandas['features'])
        except KeyError:
            logger.error(f"Chunk {chunk_count} ('{type(current_chunk_pandas).__name__}') does not contain 'features' column. Skipping. Columns: {current_chunk_pandas.columns}")
            continue
        except Exception as e:
            logger.error(f"Error during json_normalize on chunk {chunk_count} ('{type(current_chunk_pandas).__name__}'): {e}. Skipping chunk.")
            logger.debug(f"Problematic 'features' series sample: {current_chunk_pandas['features'].head() if 'features' in current_chunk_pandas else 'Not Available'}")
            continue

        chunk_time_era_pandas = current_chunk_pandas[['time', 'era_identifier']].reset_index(drop=True)
        features_expanded = features_expanded.reset_index(drop=True)

        processed_chunk_df_pandas = original_pandas.concat([chunk_time_era_pandas, features_expanded], axis=1)
        pandas_chunks_list.append(processed_chunk_df_pandas)

        if chunk_count % 10 == 0:
            logger.info(f"Processed {chunk_count} chunks, {total_rows} total rows")

    if not pandas_chunks_list:
        logger.warning("No data loaded after processing all chunks from preprocessed_features.")
        return None

    final_df_pandas = original_pandas.concat(pandas_chunks_list, ignore_index=True)
    logger.info(f"Concatenated all {len(pandas_chunks_list)} chunks into a single pandas DataFrame with {len(final_df_pandas)} rows.")

    all_data_df = ensure_backend(final_df_pandas)

    logger.info(f"Total rows loaded and backend ensured: {len(all_data_df)} (type: {type(all_data_df).__name__})")
    logger.info(f"Consolidated data shape: {all_data_df.shape}")

    return all_data_df


def process_consolidated_data(consolidated_df: DataFrame) -> DataFrame:
    """Process the consolidated DataFrame: datetime conversion, sorting, sentinel replacement."""

    # Convert time column to datetime
    consolidated_df = ensure_datetime_index(consolidated_df, time_column="time")

    # Load sentinel value configuration
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_file_path = os.path.join(base_dir, "pre_process", "preprocess_config.json")

    sentinel_map = {}
    if os.path.exists(config_file_path):
        logger.info(f"Loading sentinel value configuration from: {config_file_path}")
        try:
            with open(config_file_path) as f:
                config_data = json.load(f)

            sentinel_replacements = config_data.get("sentinel_replacements", {})
            if isinstance(sentinel_replacements, dict):
                for str_key, replacement in sentinel_replacements.items():
                    try:
                        numeric_key = float(str_key)
                        sentinel_map[numeric_key] = None if replacement is None else replacement
                    except ValueError:
                        logger.warning(f"Could not parse sentinel value '{str_key}' to numeric")

            if sentinel_map:
                logger.info(f"Loaded {len(sentinel_map)} sentinel replacement rules")
                consolidated_df = replace_sentinels(consolidated_df, sentinel_map)

        except Exception as e:
            logger.error(f"Error loading sentinel configuration: {e}")

    return consolidated_df


def prepare_tsfresh_data(consolidated_df: DataFrame) -> DataFrame:
    """Convert wide format to long format for tsfresh."""

    numeric_columns = get_numeric_columns(consolidated_df)
    numeric_columns = [c for c in numeric_columns if c not in ["time", "era_identifier"]]

    logger.info(f"Found {len(numeric_columns)} numeric feature columns")

    # Convert to long format
    value_vars = numeric_columns
    id_vars = ["time", "era_identifier"]

    long_df = pd.melt(
        consolidated_df,
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="kind",
        value_name="value",
    )

    # Sort for tsfresh
    long_df = long_df.sort_values(["era_identifier", "kind", "time"])

    logger.info(f"Long format shape: {long_df.shape}")
    logger.info(f"Unique 'kind' values: {long_df['kind'].nunique()}")
    logger.info(f"Unique 'era_identifier' values: {long_df['era_identifier'].nunique()}")

    return long_df


def extract_and_save_features(
    long_df: DataFrame,
    consolidated_df: DataFrame,
    connector: SQLAlchemyPostgresConnector
) -> None:
    """Extract features using tsfresh and save to database."""

    # Load feature calculation parameters
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_file_path = os.path.join(base_dir, "data_processing_config.json")

    tsfresh_config = EfficientFCParameters()  # Default

    if os.path.exists(config_file_path):
        try:
            with open(config_file_path) as f:
                config_data = json.load(f)

            feature_params = config_data.get("tsfresh_feature_calculation_params", {})
            if feature_params:
                tsfresh_config = feature_params
                logger.info("Loaded custom tsfresh parameters from config")
        except Exception as e:
            logger.warning(f"Could not load tsfresh config: {e}, using defaults")

    # Extract features using the adapter
    logger.info("Starting feature extraction...")
    features_df = tsfresh_extract_features(
        long_df,
        column_id="era_identifier",
        column_sort="time",
        column_kind="kind",
        column_value="value",
        default_fc_parameters=tsfresh_config,
        n_jobs=N_JOBS,
        show_warnings=False,
        disable_progressbar=False,
    )

    logger.info(f"Extracted features shape: {features_df.shape}")

    # Save full features to parquet
    save_dataframe_to_parquet(
        features_df,
        OUTPUT_PATH,
        use_gpu_flag=USE_GPU_FLAG,
        df_type_name="full features"
    )

    # Perform feature selection
    selected_features_df = perform_feature_selection(
        features_df,
        consolidated_df,
        TARGET_COLUMN
    )

    logger.info(f"Selected features shape: {selected_features_df.shape}")

    # Save selected features to parquet
    save_dataframe_to_parquet(
        selected_features_df,
        SELECTED_OUTPUT_PATH,
        use_gpu_flag=USE_GPU_FLAG,
        df_type_name="selected features"
    )

    # Save to database
    try:
        # Ensure we're using pandas for database operations
        from ..features.adapters import ensure_pandas
        selected_pandas = ensure_pandas(selected_features_df)

        selected_pandas.to_sql(
            FEATURES_TABLE,
            connector.engine,
            if_exists="replace",
            index=True,
            index_label="era_identifier",
        )
        logger.info(f"Features saved to database table: {FEATURES_TABLE}")

        # Make it a hypertable
        make_hypertable_if_needed(connector, FEATURES_TABLE, "era_identifier")

    except Exception as e:
        logger.error(f"Failed to save features to database: {e}", exc_info=True)
        raise


def main():
    """Main entry point for feature extraction."""
    logger.info(f"Starting feature extraction with backend: {BACKEND_TYPE}")

    # Initialize database connection
    connector = SQLAlchemyPostgresConnector(
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
    )

    try:
        # Load data from database
        consolidated_df = load_preprocessed_data()
        if consolidated_df is None:
            return

        # Process the data
        consolidated_df = process_consolidated_data(consolidated_df)

        # Convert to long format for tsfresh
        long_df = prepare_tsfresh_data(consolidated_df)

        # Extract features and save
        extract_and_save_features(long_df, consolidated_df, connector)

        logger.info("Feature extraction completed successfully!")

    except Exception as e:
        logger.error(f"Feature extraction failed: {e}", exc_info=True)
        raise
    finally:
        connector.close()


if __name__ == "__main__":
    main()
