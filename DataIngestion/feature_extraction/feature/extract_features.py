"""Feature extraction script.

This script connects to the TimescaleDB instance, reads the `preprocessed_features` hypertable
 (written by the preprocessing pipeline), unfolds the JSONB `features` column into a wide
 DataFrame, converts it to the long format expected by *tsfresh*, and finally extracts a rich
 set of statistical features.  The resulting feature set is written to a parquet file so that
 downstream modelling steps (e.g. LSTM/GAN training) can load it efficiently.

Usage (inside container):

    uv run python extract_features.py

Environment variables (all have sensible defaults for docker-compose):

    DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME  –  connection parameters
    OUTPUT_PATH                                    –  where the parquet file will be written
    ERA_IDENTIFIER                                 –  optional filter (process only one era)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np  # Added for np.number
import pandas as pd
from tsfresh import extract_features

# Local helper for DB access
from db_utils import SQLAlchemyPostgresConnector

# Import sqlalchemy at the top level if not already done
import sqlalchemy


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------


def _env(key: str, default: str) -> str:
    """Read *key* from the environment or fall back to *default*."""

    return os.getenv(key, default)


DB_USER = _env("DB_USER", "postgres")
DB_PASSWORD = _env("DB_PASSWORD", "postgres")
DB_HOST = _env("DB_HOST", "db")
DB_PORT = _env("DB_PORT", "5432")
DB_NAME = _env("DB_NAME", "postgres")

# Location where the parquet file will be written inside the container
DEFAULT_OUTPUT_PATH = Path("/app/data/output/tsfresh_features.parquet")
OUTPUT_PATH = Path(_env("OUTPUT_PATH", str(DEFAULT_OUTPUT_PATH)))

# Optional: only process a single era, useful during debugging
# FILTER_ERA: str | None = os.getenv("ERA_IDENTIFIER")


# -----------------------------------------------------------------------------
# Data loading & preparation helpers
# -----------------------------------------------------------------------------


def _get_connector() -> SQLAlchemyPostgresConnector:
    """Return an instance of *SQLAlchemyPostgresConnector* using env credentials."""

    return SQLAlchemyPostgresConnector(
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        db_name=DB_NAME,
    )


def get_unique_eras(connector: SQLAlchemyPostgresConnector) -> list[str]:
    """Fetch the list of unique era identifiers from the database."""
    logging.info("Fetching unique era identifiers...")
    query = "SELECT DISTINCT era_identifier FROM preprocessed_features ORDER BY era_identifier;"
    # Execute the query and fetch results directly using the connector's engine
    try:
        with connector.engine.connect() as connection:
            result = connection.execute(sqlalchemy.text(query))
            eras = [row[0] for row in result]
    except Exception as e:
        logging.error(f"Failed to fetch unique eras: {e}")
        raise

    if not eras:
        raise RuntimeError("No era identifiers found in preprocessed_features table.")

    logging.info(f"Found {len(eras)} unique eras: {eras}")
    return eras


def fetch_preprocessed_data_for_era(connector: SQLAlchemyPostgresConnector, era_identifier: str) -> pd.DataFrame:
    """Load rows from *preprocessed_features* and unpack the JSONB column.*"""

    query = "SELECT time, era_identifier, features FROM preprocessed_features WHERE era_identifier = :era_identifier"
    params = {"era_identifier": era_identifier}

    logging.info(f"Executing query for era: {era_identifier}")
    # Ensure fetch_data_to_pandas can handle raw SQL execution with parameters
    # Assuming fetch_data_to_pandas executes the query and returns a DataFrame
    try:
         # Pass the text object and params correctly
        df = connector.fetch_data_to_pandas(sql=sqlalchemy.text(query), params=params)
    except Exception as e:
        logging.error(f"Database query failed for era {era_identifier}: {e}")
        raise
        
    if df.empty:
        logging.warning(f"Fetched DataFrame for era '{era_identifier}' is empty.")
        # Return an empty DataFrame with expected base columns to avoid downstream errors if needed,
        # or handle this case more explicitly in the main loop.
        # For now, let's return it and handle potential issues later.
        return pd.DataFrame(columns=["time", "era_identifier", "features"]) # Return empty df with expected columns

    logging.info(f"Fetched {len(df)} rows for era '{era_identifier}'")

    # Parse JSONB column into real columns
    logging.info("Unpacking JSONB features → columns …")
    feature_cols = pd.json_normalize(df["features"])
    feature_cols.index = df.index  # align indices

    wide_df = pd.concat([df[["time", "era_identifier"]], feature_cols], axis=1)
    logging.info("Wide DataFrame shape after unpack: %s", wide_df.shape)

    return wide_df


def melt_for_tsfresh(wide_df: pd.DataFrame) -> pd.DataFrame:
    """Convert wide sensor matrix (numeric columns only) to long format required by *tsfresh*."""

    logging.info("Converting to long (tidy) format for tsfresh …")

    id_columns = ["era_identifier", "time"]
    # Select only numeric columns for melting, excluding the ID columns
    numeric_cols = wide_df.select_dtypes(include=np.number).columns.tolist()
    value_columns = [col for col in numeric_cols if col not in id_columns]

    if not value_columns:
        raise ValueError("No numeric columns found in wide_df to melt for tsfresh.")

    logging.info(f"Melting based on {len(value_columns)} numeric columns.")

    long_df = (
        wide_df.melt(
            id_vars=id_columns,
            value_vars=value_columns,  # Use only numeric columns
            var_name="variable",
            value_name="value",
        )
        .dropna(subset=["value"])
    )

    long_df.rename(
        columns={"era_identifier": "id"},  # tsfresh default column name is short: `id`
        inplace=True,
    )

    logging.info("Long DataFrame shape: %s", long_df.shape)

    return long_df


# -----------------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------------


def run_tsfresh(long_df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Running tsfresh feature extraction (this can take a while) …")

    features = extract_features(
        long_df,
        column_id="id",
        column_sort="time",
        column_kind="variable",
        column_value="value",
        disable_progressbar=True,
        n_jobs=0,  # use all CPUs available in container
    )

    logging.info("tsfresh generated %s features", features.shape[1])
    return features


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    # Create connector once
    connector = _get_connector()
    eras_to_process = get_unique_eras(connector)

    all_era_features: list[pd.DataFrame] = []
    total_rows_processed = 0
    all_sensor_cols = set()

    for i, era in enumerate(eras_to_process):
        logging.info(f"--- Processing Era {i+1}/{len(eras_to_process)}: {era} ---")
        try:
            wide_df_era = fetch_preprocessed_data_for_era(connector, era)
            if wide_df_era.empty or 'features' not in wide_df_era.columns or wide_df_era['features'].isnull().all():
                 logging.warning(f"Skipping era '{era}' due to empty or invalid data after fetch/unpack.")
                 continue # Skip to the next era

            total_rows_processed += len(wide_df_era)
            current_sensor_cols = {c for c in wide_df_era.columns if c not in {"time", "era_identifier"}}
            all_sensor_cols.update(current_sensor_cols)
            
            long_df_era = melt_for_tsfresh(wide_df_era)
            if long_df_era.empty:
                logging.warning(f"Skipping era '{era}' because long DataFrame became empty after melting (e.g., all numeric values were NaN).")
                continue

            tsfresh_features_era = run_tsfresh(long_df_era)
            all_era_features.append(tsfresh_features_era)
            logging.info(f"--- Completed Era: {era} ---")
        except Exception as e:
            logging.error(f"Failed processing era '{era}': {e}. Skipping this era.", exc_info=True)
            # Decide if you want to stop completely or continue with other eras
            # For now, we log and continue

    if not all_era_features:
        logging.error("No features were generated for any era. Exiting.")
        return # Exit if no features were generated at all

    logging.info("Concatenating features from all processed eras...")
    final_tsfresh_features = pd.concat(all_era_features)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_tsfresh_features.to_parquet(OUTPUT_PATH)
    logging.info("Feature set saved to %s", OUTPUT_PATH)

    # ---------------- summary report ------------------ #
    report_path = OUTPUT_PATH.parent / "tsfresh_feature_extraction_report.txt"
    logging.info("Writing summary report → %s", report_path)

    # Number of input rows and sensor columns
    # n_rows = total_rows_processed # Use the counter from the loop
    n_sensors = len(all_sensor_cols) # Use the set accumulated during the loop
    n_features = final_tsfresh_features.shape[1]
    n_eras_processed = len(all_era_features)

    with open(report_path, "w", encoding="utf-8") as rep:
        rep.write("CPU (tsfresh) Feature Extraction Report\n")
        rep.write("=======================================\n\n")
        # rep.write(f"Rows processed        : {n_rows:,}\n")
        rep.write(f"Eras processed        : {n_eras_processed} / {len(eras_to_process)}\n")
        rep.write(f"Total rows processed  : {total_rows_processed:,}\n")
        rep.write(f"Sensors considered    : {n_sensors:,}\n")
        # rep.write(f"Unique variables      : {n_sensors}\n")
        rep.write(f"Total feature columns : {n_features:,}\n\n")

        rep.write("First 25 feature names →\n")
        for fname in list(final_tsfresh_features.columns)[:25]:
            rep.write(f"  • {fname}\n")

        rep.write("\nExtraction successful.\n")

    logging.info("Report written.")


if __name__ == "__main__":
    main()
