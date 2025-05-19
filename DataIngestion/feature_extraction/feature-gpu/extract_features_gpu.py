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
# import cupy as cp # This also appears unused
import numpy as np
# from tsflex.features import FeatureCollection, MultipleFeatureDescriptors # Removing unused import
# from tsflex.features import FeatureExtractor  # Removing unused import
# from tsflex.helpers import get_window_centers # Removing unused import
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

    # explode JSONB → columns (pdf["features"] is already a Series of dicts)
    feat_pdf = pd.json_normalize(pdf["features"])
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

    window_map = {
        "30m": max(1, 30 // base_freq_min),
        "2h": max(1, 120 // base_freq_min),
    }

    stride_rows = max(1, 15 // base_freq_min)  # 15-minute stride

    # Initialize the final DataFrame with the original DatetimeIndex, strided
    # This ensures the index is consistently datetime for all merges.
    all_times_strided = cdf.index[::stride_rows]
    merged_features_df = cudf.DataFrame(index=all_times_strided)
    # Create the 'time' column directly from the DatetimeIndex values
    # The new Series will naturally align with merged_features_df's index
    merged_features_df["time"] = merged_features_df.index.to_arrow().to_pylist() # Or simply merged_features_df.index.values for a CuPy array if direct assignment works

    for label, win in window_map.items():
        logging.info("Computing rolling stats for window %s (%s rows)…", label, win)

        # Calculate features on the original cdf which has a DatetimeIndex
        for col in sensor_cols:
            s = cdf[col] # cdf has DatetimeIndex
            roll_mean = s.rolling(window=win, min_periods=1).mean()
            roll_std = s.rolling(window=win, min_periods=1).std()
            roll_min = s.rolling(window=win, min_periods=1).min()
            roll_max = s.rolling(window=win, min_periods=1).max()

            # Create a temporary DataFrame for these new features, strided
            # Its index will match the original cdf's index initially
            temp_feature_df = cudf.DataFrame(index=cdf.index)
            temp_feature_df[f"{col}_mean_{label}"] = roll_mean
            temp_feature_df[f"{col}_std_{label}"] = roll_std
            temp_feature_df[f"{col}_min_{label}"] = roll_min
            temp_feature_df[f"{col}_max_{label}"] = roll_max

            # Apply stride and ensure it aligns with merged_features_df's index
            temp_feature_df_strided = temp_feature_df.iloc[::stride_rows]
            
            # Merge into the main DataFrame using their common DatetimeIndex
            # Use suffixes to avoid column name collisions if features are re-calculated with same name (though unlikely here)
            merged_features_df = merged_features_df.merge(
                temp_feature_df_strided, 
                left_index=True, 
                right_index=True, 
                how="outer",
                suffixes=("", f"_dup_{col}_{label}") # Add suffix for safety, though ideally not needed
            )

    # The "time" column might be duplicated if any merge added it back despite left_index=True, right_index=True
    # Let's ensure only one 'time' column from the index remains if needed as a column.
    # If "time" column is not part of merged_features_df.columns from the merge, this won't error.
    # Best to reset index to make 'time' a column, then drop duplicates, then sort.
    
    merged_features_df = merged_features_df.reset_index() # 'time' (original index name) becomes a column
    
    # If merges created duplicate 'time' columns (e.g. time_x, time_y), handle them.
    # However, with index-based merge, this should be less of an issue.
    # The main 'time' column is from reset_index().
    # Drop other columns that might be named 'time_x' or 'time_y' if they were created.
    cols_to_drop = [c for c in merged_features_df.columns if c.startswith('time_') and c != 'time']
    if cols_to_drop:
        merged_features_df = merged_features_df.drop(columns=cols_to_drop)

    # Ensure it's sorted by time and has a clean default RangeIndex for final output
    merged_features_df = merged_features_df.sort_values("time").reset_index(drop=True)
    return merged_features_df


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
