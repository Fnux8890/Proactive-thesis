#!/usr/bin/env -S uv run --isolated
"""tsfresh feature extraction runner.

This script handles the tsfresh feature extraction phase of the pipeline,
operating on preprocessed data with proper segmentation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.utilities.dataframe_functions import impute

from core import (
    ProcessingMetrics,
    get_database_url,
    load_and_validate_config,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def melt_preprocessed_data(
    engine,
    table_name: str = "preprocessed_greenhouse_data",
    segment_col: str = "segment_id",
    time_col: str = "timestamp",
) -> pd.DataFrame:
    """Melt preprocessed data into long format for tsfresh.

    Args:
        engine: SQLAlchemy engine
        table_name: Name of preprocessed data table
        segment_col: Column containing segment IDs
        time_col: Time column name

    Returns:
        DataFrame in long format with columns: id, time, kind, value
    """
    logger.info(f"Loading data from {table_name}")

    # Query to get preprocessed data
    query = f"""
    SELECT * FROM {table_name}
    WHERE {segment_col} IS NOT NULL
    ORDER BY {segment_col}, {time_col}
    """

    df = pd.read_sql(query, engine)
    logger.info(f"Loaded {len(df)} rows from {table_name}")

    # Identify value columns (exclude id and time columns)
    id_cols = [segment_col, time_col, "entity_id"]
    value_cols = [col for col in df.columns if col not in id_cols]

    # Melt to long format
    df_long = pd.melt(
        df,
        id_vars=[segment_col, time_col],
        value_vars=value_cols,
        var_name="kind",
        value_name="value",
    )

    # Rename for tsfresh compatibility
    df_long = df_long.rename(columns={segment_col: "id", time_col: "time"})

    logger.info(
        f"Melted data to {len(df_long)} rows "
        f"({len(df[segment_col].unique())} segments, {len(value_cols)} features)"
    )

    return df_long


def extract_tsfresh_features(
    df_long: pd.DataFrame,
    feature_params: dict | None = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Extract features using tsfresh.

    Args:
        df_long: Long-format DataFrame with columns: id, time, kind, value
        feature_params: Feature extraction parameters (defaults to comprehensive)
        n_jobs: Number of parallel jobs

    Returns:
        Feature matrix with one row per segment
    """
    logger.info("Starting tsfresh feature extraction")

    if feature_params is None:
        feature_params = ComprehensiveFCParameters()

    # Extract features
    features = extract_features(
        df_long,
        column_id="id",
        column_sort="time",
        column_kind="kind",
        column_value="value",
        default_fc_parameters=feature_params,
        n_jobs=n_jobs,
        show_warnings=False,
        disable_progressbar=False,
    )

    logger.info(f"Extracted {features.shape[1]} features for {features.shape[0]} segments")

    # Impute any remaining NaN/Inf values
    features = impute(features)

    return features


def save_features_to_db(
    features: pd.DataFrame,
    engine,
    table_name: str = "features_tsfresh",
    if_exists: str = "replace",
) -> None:
    """Save extracted features to database.

    Args:
        features: Feature matrix
        engine: SQLAlchemy engine
        table_name: Target table name
        if_exists: How to handle existing table
    """
    logger.info(f"Saving features to {table_name}")

    # Reset index to make segment_id a column
    features_df = features.reset_index()
    features_df = features_df.rename(columns={"index": "segment_id"})

    # Save to database
    features_df.to_sql(
        table_name,
        engine,
        if_exists=if_exists,
        index=False,
        method="multi",
        chunksize=1000,
    )

    logger.info(f"Saved {len(features_df)} feature vectors to {table_name}")


def main() -> None:
    """Main entry point for tsfresh runner."""
    try:
        # Load configuration
        config_path = Path("preprocess_config.json")
        config = load_and_validate_config(config_path)

        # Create database connection
        db_url = get_database_url(config)
        engine = create_engine(db_url)

        # Initialize metrics
        metrics = ProcessingMetrics()

        # Step 1: Load and melt preprocessed data
        df_long = melt_preprocessed_data(
            engine,
            table_name=config.output_table,
        )
        metrics.total_rows_input = len(df_long)

        # Step 2: Extract features
        features = extract_tsfresh_features(
            df_long,
            n_jobs=4,  # Adjust based on available cores
        )
        metrics.total_rows_output = len(features)

        # Step 3: Save features
        save_features_to_db(
            features,
            engine,
            table_name="features_tsfresh",
        )

        # Log summary
        logger.info(
            f"Feature extraction complete: "
            f"{metrics.total_rows_input} input rows -> "
            f"{metrics.total_rows_output} feature vectors"
        )

    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        raise
    finally:
        if "engine" in locals():
            engine.dispose()


if __name__ == "__main__":
    main()
