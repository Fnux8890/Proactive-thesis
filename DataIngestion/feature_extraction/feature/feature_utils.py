"""Utility functions for feature extraction."""

from __future__ import annotations

import logging
import numpy as np
import sqlalchemy

from . import config

try:
    import cudf
    import cudf.pandas as cudf_pd
except Exception:  # pragma: no cover - cudf optional
    cudf = None
    cudf_pd = None

import pandas as pd


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def select_relevant_features(
    features_df: pd.DataFrame,
    correlation_threshold: float = 0.95,
    variance_threshold: float = 0.0,
) -> pd.DataFrame:
    """Perform a simple unsupervised feature selection."""

    if features_df.empty:
        return features_df

    use_gpu = config.USE_GPU_FLAG and cudf is not None and isinstance(features_df, cudf.DataFrame)

    work_df = features_df.to_pandas() if use_gpu else features_df.copy()

    variances = work_df.var()
    cols_to_keep = variances[variances > variance_threshold].index
    work_df = work_df[cols_to_keep]

    MAX_FEATURES_FOR_CORR = 2000
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


# ---------------------------------------------------------------------------
# Promote a plain SQL table to Timescale hypertable if needed
# ---------------------------------------------------------------------------

def make_hypertable_if_needed(conn, table_name: str, time_column: str) -> None:
    """Ensure a table is a TimescaleDB hypertable."""
    sql = """
    DO $$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
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
