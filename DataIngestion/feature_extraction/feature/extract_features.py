"""Feature extraction script.

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

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np  # Added for np.number
import pandas as original_pandas  # Use this for true pandas operations
import sqlalchemy
import sys  # For checking loaded modules

# Local helper for DB access
from db_utils import SQLAlchemyPostgresConnector
from tsfresh import extract_features


# Conditionally import and alias pd
if os.getenv("USE_GPU", "false").lower() == "true":
    try:
        import cudf.pandas as pd  # pd becomes cudf.pandas
        import cudf               # For explicit cudf.DataFrame, etc.
        logging.info("Running in GPU mode with cudf.pandas and cudf.")
    except ImportError as e:
        logging.error(f"Failed to import GPU libraries (cudf): {e}. Falling back to CPU mode.")
        os.environ["USE_GPU"] = "false" # Force fallback for current script execution
        pd = original_pandas      # Ensure pd is original_pandas
        # No need to re-import cudf if it failed; it won't be used.
else:
    pd = original_pandas          # pd is original_pandas
    logging.info("Running in CPU mode with pandas.")


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

# Table name for database storage of selected features
FEATURES_TABLE = os.getenv("FEATURES_TABLE", "tsfresh_selected_features")

# Global constant for reporting the feature set used
FC_PARAMS_NAME = "Custom (preprocess_config.json-Guided)"

# Placeholder for tsfresh configuration per sensor
kind_to_fc_parameters_global: dict[str, Any] = {}

# -----------------------------------------------------------------


def select_relevant_features(
    features_df: pd.DataFrame,
    correlation_threshold: float = 0.95,
    variance_threshold: float = 0.0,
) -> pd.DataFrame:
    """Perform a simple unsupervised feature selection."""

    if features_df.empty:
        return features_df

    use_gpu = os.getenv("USE_GPU", "false").lower() == "true" and "cudf" in sys.modules

    if use_gpu and isinstance(features_df, cudf.DataFrame):
        work_df = features_df.to_pandas()
    else:
        work_df = features_df.copy()

    variances = work_df.var()
    cols_to_keep = variances[variances > variance_threshold].index
    work_df = work_df[cols_to_keep]

    MAX_FEATURES_FOR_CORR = 2000   # was 200
    if work_df.shape[1] <= MAX_FEATURES_FOR_CORR:
        corr_matrix = work_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > correlation_threshold)]
        work_df = work_df.drop(columns=to_drop)

    if use_gpu:
        index_name = work_df.index.name or "index"
        pdf_reset = work_df.reset_index()
        cdf = cudf.DataFrame.from_pandas(pdf_reset)
        cdf.set_index(index_name, inplace=True)
        return cdf
    return work_df


# -----------------------------------------------------------------
# Promote a plain SQL table to Timescale hypertable if needed
# -----------------------------------------------------------------
def _make_hypertable_if_needed(conn, table_name, time_column):
    sql = """
    DO $$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb')
        THEN
            CREATE EXTENSION IF NOT EXISTS timescaledb;
        END IF;

        IF NOT EXISTS (
            SELECT 1
            FROM   timescaledb_information.hypertables
            WHERE  hypertable_schema = current_schema
              AND  hypertable_name   = $1)
        THEN
            PERFORM create_hypertable($1, $2, if_not_exists => TRUE);
        END IF;
    END;
    $$;
    """
    conn.execute(sqlalchemy.text(sql), (table_name, time_column))


def main() -> None:
    """Entry point for feature extraction."""

    logging.info("Starting feature extraction …")

    # -----------------------------------------------------------------
    # 1.  Pull data straight from TimescaleDB
    # -----------------------------------------------------------------
    SQL_WIDE = """
    SELECT f.*, e.era_id
    FROM   preprocessed_features f
    JOIN   era_labels_all      e USING (time)
    """

    connector = SQLAlchemyPostgresConnector(
        user=DB_USER, password=DB_PASSWORD,
        host=DB_HOST, port=DB_PORT, db_name=DB_NAME
    )

    if os.getenv("USE_GPU", "false").lower() == "true":
        wide_df = cudf.read_sql(SQL_WIDE, connector.engine)
    else:
        wide_df = pd.read_sql(SQL_WIDE, connector.engine)

    logging.info("Wide dataframe shape: %s", wide_df.shape)

    # -----------------------------------------------------------------
    # 2.  Melt to long format on GPU if possible
    # -----------------------------------------------------------------
    numeric_cols = [c for c in wide_df.columns
                    if c not in ("era_id", "time") and
                       wide_df[c].dtype.kind in ("i", "f")]

    melt_id_vars = ["era_id", "time"]

    long_df = wide_df.melt(
        id_vars=melt_id_vars,
        value_vars=numeric_cols,
        var_name="kind",
        value_name="value"
    )

    logging.info("Long dataframe shape (before tsfresh): %s", long_df.shape)

    if os.getenv("USE_GPU", "false").lower() == "true":
        long_df = long_df.to_pandas()

    # -----------------------------------------------------------------
    # 3.  tsfresh extraction (single shot)
    # -----------------------------------------------------------------
    features = extract_features(
        long_df,
        column_id="era_id",
        column_sort="time",
        column_kind="kind",
        column_value="value",
        kind_to_fc_parameters=kind_to_fc_parameters_global,
        default_fc_parameters=None,
        n_jobs=os.cpu_count() - 2,
    )

    if features.index.name == "era_id":
        features.reset_index(inplace=True)

    logging.info("Raw feature matrix shape: %s", features.shape)

    selected = select_relevant_features(features)

    df_for_db = selected.copy()
    df_for_db.index.name = "id"
    df_for_db = df_for_db.reset_index()
    if hasattr(df_for_db, "to_pandas"):
        df_for_db = df_for_db.to_pandas()

    connector.write_dataframe(
        df_for_db,
        FEATURES_TABLE,
        if_exists="replace",
        index=False,
    )

    with connector.engine.begin() as c:
        _make_hypertable_if_needed(c, FEATURES_TABLE, "era_id")

if __name__ == "__main__":
    main()
