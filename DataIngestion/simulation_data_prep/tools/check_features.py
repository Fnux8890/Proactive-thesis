#!/usr/bin/env python
"""
check_features.py
-----------------
Quick sanity‑check utility for a single feature Parquet file.

Usage (from repo root):
    uv pip install polars   # first run once if polars not yet in host venv
    python DataIngestion/simulation_data_prep/tools/check_features.py \
           DataIngestion/simulation_data_prep/output/features_2014-04-15.parquet

The script prints:
  • file path & DataFrame shape
  • unique values of Lamp_kWh_daily (should be <= 1 per day)
  • list of columns recently added (RoC, rolling averages, energy, etc.)
"""
from __future__ import annotations

import sys
from pathlib import Path
import textwrap

import polars as pl

RECENT_KEYS: list[str] = [
    "RoC_per_s",
    "rolling_avg",
    "Lamp_kWh_daily",
    "rolling_std_",
]

def main(parquet_path: str) -> None:  # pragma: no cover
    p = Path(parquet_path)
    if not p.exists():
        sys.exit(f"File not found: {p}")

    df = pl.read_parquet(p)

    print("\n" + "=" * 60)
    print(f"File: {p.relative_to(Path.cwd()) if p.is_relative_to(Path.cwd()) else p}")
    print(f"Shape: {df.shape}\n")

    if "Lamp_kWh_daily" in df.columns:
        daily_vals = df.select("Lamp_kWh_daily").unique()
        print("Unique Lamp_kWh_daily values:")
        print(daily_vals)
    else:
        print("Column Lamp_KWh_daily not found – check pipeline.")

    new_cols = [c for c in df.columns if any(key in c for key in RECENT_KEYS)]
    print("\nDetected newly‑added columns (subset):")
    wrapper = textwrap.TextWrapper(width=100, subsequent_indent="  ")
    print(wrapper.fill(", ".join(sorted(new_cols)) or "<none found>"))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: check_features.py <parquet_path>")
    main(sys.argv[1]) 