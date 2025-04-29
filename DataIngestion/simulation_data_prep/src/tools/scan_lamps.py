#!/usr/bin/env python
"""Quick scan of *_status columns to see how often they are TRUE.

Usage:
    python scan_lamps.py /path/to/data.parquet [more_files...]

Outputs a table of columns and their true‑ratio (fraction of rows where the
value is truthy).  Helps decide which lamp status columns actually flip and
should be listed in `plant_config.json`.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import polars as pl


def scan_file(path: Path) -> pl.DataFrame:
    """Return a DataFrame of column true ratios for one file."""
    print(f"\nScanning {path} …")
    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pl.read_parquet(path, low_memory=True)
    elif path.suffix.lower() in {".csv"}:
        df = pl.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")

    lamp_cols: list[str] = [c for c in df.columns if c.endswith("_status")]
    if not lamp_cols:
        print("  No *_status columns found. Skipping.")
        return pl.DataFrame()

    # Cast bool/str/number to int8 -> 1 for truthy, 0 for falsy/NULL.
    true_ratio = (
        df.select(lamp_cols)
        .with_columns(pl.all().cast(pl.Int8))
        .mean()  # fraction of ones
        .transpose(include_header=True, header_name="column")
        .rename({"column": "lamp_status", "column_0": "true_ratio"})
        .sort("true_ratio", descending=True)
    )
    print(true_ratio)
    return true_ratio


def aggregate_ratios(ratios: Sequence[pl.DataFrame]) -> None:
    if not ratios:
        return
    combined = pl.concat(ratios)
    grouped = combined.group_by("lamp_status").mean().sort("true_ratio", descending=True)
    print("\n=== Aggregated true‑ratio across all files ===")
    print(grouped)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan lamp *_status columns for TRUE ratio")
    parser.add_argument("files", nargs="+", help="Parquet/CSV files to scan")
    args = parser.parse_args()

    ratios = []
    for file in args.files:
        path = Path(file)
        if not path.exists():
            print(f"File not found: {path}")
            continue
        try:
            r = scan_file(path)
            if r.height > 0:
                ratios.append(r)
        except Exception as e:
            print(f"Error processing {path}: {e}")
    aggregate_ratios(ratios)


if __name__ == "__main__":
    main() 