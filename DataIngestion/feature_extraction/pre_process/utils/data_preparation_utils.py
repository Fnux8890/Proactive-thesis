import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_config(path: Path) -> dict[str, Any]:
    print(f"Loading configuration from {path}...")
    try:
        with open(path) as f:
            config = json.load(f)
        print("Configuration loaded successfully.")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {path}. Exiting.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {path}. Exiting.")
        sys.exit(1)


def sort_and_prepare_df(
    df: pd.DataFrame, config: dict[str, Any], era_identifier: str
) -> pd.DataFrame:
    if df.empty:
        return df

    time_col_name = config.get("common_settings", {}).get("time_col", "time")
    if time_col_name not in df.columns:
        print(
            f"Error (Era: {era_identifier}): Time column '{time_col_name}' not found in fetched data."
        )
        return pd.DataFrame()

    df[time_col_name] = pd.to_datetime(df[time_col_name], errors="coerce", utc=True)
    df.dropna(subset=[time_col_name], inplace=True)
    df = df.sort_values(by=time_col_name).reset_index(drop=True)

    id_col_name = config.get("common_settings", {}).get("id_col", "entity_id")
    default_id_value = (
        config.get("era_definitions", {}).get(era_identifier, {}).get("description", era_identifier)
    )
    if id_col_name not in df.columns:
        print(
            f"(Era: {era_identifier}) '{id_col_name}' not found. Adding default ID column with value: '{default_id_value}'."
        )
        df[id_col_name] = default_id_value

    # --- Sentinel Value Processing ---
    preprocessing_rules = config.get("preprocessing_rules", {})
    sentinel_rules_config = preprocessing_rules.get("sentinel_value_rules", {})
    if sentinel_rules_config:
        print(f"(Era: {era_identifier}) Applying sentinel value rules...")
        for rule_name, rule_details in sentinel_rules_config.items():
            columns_to_process = rule_details.get("columns_with_sentinel", [])
            sentinel_val = rule_details.get("sentinel_value")

            if sentinel_val is None:
                print(f"  Warning: Sentinel value not defined for rule '{rule_name}'. Skipping this rule.")
                continue

            if not columns_to_process:
                print(f"  Info: No columns specified for sentinel rule '{rule_name}'.")
                continue

            print(f"  Processing rule '{rule_name}': Replacing '{sentinel_val}' with NaN for columns: {columns_to_process}")
            for col in columns_to_process:
                if col in df.columns:
                    if df[col].dtype == object and isinstance(sentinel_val, (int, float)):
                        # Attempt to convert object column to numeric if sentinel is numeric, to avoid mixed-type issues or TypeErrors with .replace()
                        # This is a common case if numbers are read as strings.
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        print(f"    Column '{col}' was object type, converted to numeric before sentinel replacement.")
                    
                    # Check if sentinel_val is NaN - replace works differently for NaN sentinel
                    if pd.isna(sentinel_val):
                        # This case is unlikely for typical sentinels like -1, but good to be aware
                        # .replace(np.nan, new_value) works, but replacing a specific value *with* np.nan is standard
                        pass # Standard .replace below handles replacing values with np.nan

                    original_non_nan_count = df[col].notna().sum()
                    df[col] = df[col].replace(sentinel_val, np.nan)
                    current_nan_count = df[col].isna().sum()
                    nan_introduced_count = current_nan_count - (df.shape[0] - original_non_nan_count)

                    if nan_introduced_count > 0:
                        print(f"    Column '{col}': Replaced {nan_introduced_count} occurrences of '{sentinel_val}' with NaN.")
                    else:
                        print(f"    Column '{col}': No occurrences of '{sentinel_val}' found or value was already NaN.")
                else:
                    print(f"    Warning: Column '{col}' for sentinel rule '{rule_name}' not found in DataFrame.")
    else:
        print(f"(Era: {era_identifier}) No sentinel value rules found in configuration.")
    # --- End Sentinel Value Processing ---

    era_specific_config = config.get("era_definitions", {}).get(era_identifier, {})
    boolean_cols_to_int = era_specific_config.get(
        "boolean_columns_to_int",
        config.get("common_settings", {}).get("boolean_columns_to_int", []),
    )
    if boolean_cols_to_int:
        print(
            f"(Era: {era_identifier}) Converting boolean columns to int (0/1): {boolean_cols_to_int}"
        )
        for col in boolean_cols_to_int:
            if col in df.columns:
                if df[col].dtype == "object":
                    df[col] = (
                        df[col]
                        .astype(str)
                        .str.lower()
                        .map(
                            {
                                "true": 1,
                                "false": 0,
                                "1": 1,
                                "0": 0,
                                "1.0": 1,
                                "0.0": 0,
                                "on": 1,
                                "off": 0,
                                "yes": 1,
                                "no": 0,
                            }
                        )
                        .astype(pd.Int64Dtype())
                    )
                    print(f"  Converted object column '{col}' to int (0/1) using mapping.")
                elif df[col].dtype == "bool":
                    df[col] = df[col].astype(pd.Int64Dtype())
                    print(f"  Converted boolean column '{col}' to int.")
                elif pd.api.types.is_numeric_dtype(df[col]):
                    is_01 = df[col].dropna().isin([0, 1]).all()
                    if is_01:
                        df[col] = df[col].astype(pd.Int64Dtype())
                        print(f"  Ensured numeric (0/1) column '{col}' is integer type.")
                    else:
                        print(
                            f"  Warning: Numeric column '{col}' is not strictly 0/1. Skipping direct 0/1 conversion. Dtype: {df[col].dtype}"
                        )
                else:
                    print(
                        f"  Warning: Column '{col}' intended for bool->int conversion has unhandled dtype: {df[col].dtype}. Skipping conversion."
                    )
            else:
                print(f"  Warning: Column '{col}' for bool->int conversion not found.")

    if "co2_status" in df.columns:
        print(f"(Era: {era_identifier}) Applying specific mapping for 'co2_status' column.")
        conditions = [
            df["co2_status"] == 0.0,
            (df["co2_status"] > 0) & (df["co2_status"] <= 1.0),
            (df["co2_status"] < 0),
        ]
        choices = [0, 1, np.nan]
        df["co2_status"] = (
            pd.Series(np.select(conditions, choices, default=np.nan), index=df.index)
            .astype(float)
            .astype(pd.Int64Dtype())
        )
        print(
            f"  'co2_status' mapped. Non-NaN value counts after mapping:\n{df['co2_status'].value_counts(dropna=False)}"
        )
    return df


def resample_data_for_era(
    df: pd.DataFrame, era_identifier: str, era_config: dict, common_config: dict
) -> pd.DataFrame:
    if df.empty:
        print(f"(Era: {era_identifier}) DataFrame is empty, skipping resampling.")
        return df

    time_col = common_config.get("time_col", "time")
    target_freq = era_config.get("target_frequency")

    if not target_freq:
        print(f"(Era: {era_identifier}) No target_frequency defined. Skipping resampling.")
        return df

    print(f"\n--- Resampling Data for Era: {era_identifier} to frequency: {target_freq} ---")

    current_df = df.copy()

    if isinstance(current_df.index, pd.DatetimeIndex) and current_df.index.name == time_col:
        print(f"  Data already has '{time_col}' as DatetimeIndex.")
    elif time_col in current_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(current_df[time_col]):
            print(
                f"  (Era: {era_identifier}) Time column '{time_col}' is not datetime. Attempting conversion."
            )
            current_df[time_col] = pd.to_datetime(current_df[time_col], errors="coerce", utc=True)
            current_df.dropna(subset=[time_col], inplace=True)
            if current_df.empty:
                print(
                    f"  (Era: {era_identifier}) DataFrame became empty after failed time conversion. Cannot resample."
                )
                return current_df

        print(f"  Setting '{time_col}' as index for resampling...")
        current_df = current_df.set_index(time_col)
        if not isinstance(current_df.index, pd.DatetimeIndex):
            print(
                f"  Warning: Index is '{time_col}' but not DatetimeIndex after set_index. Attempting final conversion."
            )
            current_df.index = pd.to_datetime(current_df.index, utc=True)
    else:
        print(
            f"  Error (Era: {era_identifier}): Time column '{time_col}' not found. Cannot resample."
        )
        return df

    if not isinstance(current_df.index, pd.DatetimeIndex):
        print(
            f"  Error (Era: {era_identifier}): Index is not DatetimeIndex before resample. Resampling may fail."
        )
        return df

    try:
        df_resampled = current_df.resample(target_freq).asfreq()
        print(
            f"Resampling complete. Shape before: {current_df.shape}, Shape after: {df_resampled.shape}"
        )
        if not df_resampled.empty:
            print(
                f"Data range after resampling: {df_resampled.index.min()} to {df_resampled.index.max()}"
            )
        else:
            print("DataFrame is empty after resampling.")
        return df_resampled
    except Exception as e:
        print(f"Error during resampling for Era '{era_identifier}': {e}")
        if time_col in df.columns:
            return df
        return current_df.reset_index()


def save_data(df: pd.DataFrame, path: Path):
    print(f"Saving processed data to {path}...")
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".parquet":
            df.to_parquet(path, index=False)
        elif path.suffix == ".csv":
            df.to_csv(path, index=False)
        else:
            raise ValueError("Unsupported file format for saving. Please use .parquet or .csv")
        print(f"Data saved successfully to {path}")
    except Exception as e:
        print(f"Error saving data to {path}: {e}")


def generate_summary_report(report_items: list, output_dir: Path, filename: str):
    report_path = output_dir / filename
    print(f"\nGenerating summary report at: {report_path}")
    try:
        with open(report_path, "w") as f:
            for item_type, content in report_items:
                f.write(f"--- {item_type} ---\n")
                if isinstance(content, pd.DataFrame | pd.Series):
                    f.write(content.to_string() + "\n\n")
                elif isinstance(content, dict):
                    for key, val in content.items():
                        f.write(f"{key}: {val}\n")
                    f.write("\n")
                else:
                    f.write(str(content) + "\n\n")
        print(f"Summary report saved successfully to {report_path}")
    except Exception as e:
        print(f"Error saving summary report: {e}")
