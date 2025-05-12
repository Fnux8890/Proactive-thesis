"""
GPU feature-extraction pipeline using RAPIDS (cuDF) and tsflex.

Run inside the GPU image: uv run python extract_features_gpu.py
"""

import os
import json
import logging
from pathlib import Path
from typing import Any

import cudf                          # GPU DataFrame
import cupy as cp
import numpy as np
from tsflex.features import FeatureCollection, MultipleFeatureDescriptors
from tsflex.processing import DataFrameFunctionExtraction
from tsflex.helpers import get_window_centers
import pandas as pd
from db_utils import SQLAlchemyPostgresConnector

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")

# -------- Environment ---------------------------------------------------------
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_HOST = os.getenv("DB_HOST", "db")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
FILTER_ERA = os.getenv("ERA_IDENTIFIER")
OUT_PATH = Path(os.getenv("OUTPUT_PATH",
                          "/app/data/output/gpu_features.parquet"))
# ------------------------------------------------------------------------------


def fetch_to_cudf() -> cudf.DataFrame:
    connector = SQLAlchemyPostgresConnector(
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        db_name=DB_NAME,
    )
    base_q = "SELECT time, era_identifier, features FROM preprocessed_features"
    params: dict[str, Any] | None = None
    if FILTER_ERA:
        base_q += " WHERE era_identifier = :era"
        params = {"era": FILTER_ERA}

    pdf = connector.fetch_data_to_pandas(base_q if params is None
                                         else connector.engine.execute(base_q,
                                                                       params))
    if pdf.empty:
        raise RuntimeError("Empty fetch")

    # explode JSONB → columns
    feat_pdf = pd.json_normalize(pdf["features"].apply(json.loads))
    full_pdf = pd.concat([pdf[["time", "era_identifier"]], feat_pdf], axis=1)
    logging.info("Loaded %s rows, %s columns", *full_pdf.shape)

    cdf = cudf.from_pandas(full_pdf)
    # ensure correct dtypes
    cdf["time"] = cudf.to_datetime(cdf["time"])
    return cdf


def infer_base_freq(cdf: cudf.DataFrame) -> int:
    """Infer the sampling period in minutes based on first two timestamps."""

    t0, t1 = cdf.index[:2].to_pandas()
    delta_min = int((t1 - t0).total_seconds() / 60)
    return delta_min


def compute_rolling_features(cdf: cudf.DataFrame, sensor_cols: list[str]) -> cudf.DataFrame:
    """Compute rolling statistics fully on GPU using cuDF operations."""

    base_freq_min = infer_base_freq(cdf)

    # Mapping window label → window size in rows
    window_map = {
        "30m": max(1, 30 // base_freq_min),
        "2h": max(1, 120 // base_freq_min),
    }

    stride_rows = max(1, 15 // base_freq_min)  # 15-minute stride

    feat_frames = []

    for label, win in window_map.items():
        logging.info("Computing rolling stats for window %s (%s rows)…", label, win)

        roll_df = cudf.DataFrame()
        roll_df["time"] = cdf.index

        for col in sensor_cols:
            s = cdf[col]
            roll_mean = s.rolling(window=win, min_periods=1).mean()
            roll_std = s.rolling(window=win, min_periods=1).std()
            roll_min = s.rolling(window=win, min_periods=1).min()
            roll_max = s.rolling(window=win, min_periods=1).max()

            roll_df[f"{col}_mean_{label}"] = roll_mean
            roll_df[f"{col}_std_{label}"] = roll_std
            roll_df[f"{col}_min_{label}"] = roll_min
            roll_df[f"{col}_max_{label}"] = roll_max

        # stride sampling
        roll_df = roll_df.iloc[::stride_rows]
        feat_frames.append(roll_df.reset_index(drop=True))

    # Concatenate on rows then drop duplicate time columns by merge on time
    merged = feat_frames[0]
    for frame in feat_frames[1:]:
        merged = merged.merge(frame, on="time", how="outer")

    merged = merged.sort_values("time").reset_index(drop=True)
    return merged


def main() -> None:
    cdf_raw = fetch_to_cudf()

    # Set datetime index for rolling operations
    cdf_raw = cdf_raw.set_index("time").sort_index()

    sensor_cols = [c for c in cdf_raw.columns if c not in {"era_identifier"} and cdf_raw[c].dtype.kind in "fiu"]

    features_cdf = compute_rolling_features(cdf_raw, sensor_cols)

    n_feat_cols = len(features_cdf.columns) - 1  # excluding time

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features_cdf.to_parquet(OUT_PATH)
    logging.info("Saved GPU-optimised feature set → %s", OUT_PATH)

    # --------- summary report ---------
    report_path = OUT_PATH.parent / "gpu_feature_extraction_report.txt"
    with open(report_path, "w", encoding="utf-8") as rep:
        rep.write("GPU-Optimised Feature Extraction Report\n")
        rep.write("========================================\n\n")
        rep.write(f"Rows processed          : {len(cdf_raw):,}\n")
        rep.write(f"Sensors considered      : {len(sensor_cols):,}\n")
        rep.write("Windows (rows)          : 30m & 2h (derived)\n")
        rep.write("Stride (rows)           : every 15 min\n")
        rep.write(f"Feature columns created : {n_feat_cols:,}\n\n")

        rep.write("First 25 feature columns →\n")
        for col in list(features_cdf.columns)[1:26]:
            rep.write(f"  • {col}\n")

        rep.write("\nExtraction successful.\n")

    logging.info("Report written → %s", report_path)


if __name__ == "__main__":
    main()
