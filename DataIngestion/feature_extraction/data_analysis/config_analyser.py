#!/usr/bin/env -S uv run --isolated
# /// script
# dependencies = ["pandas", "sqlalchemy", "psycopg2-binary"] 
# ///

import json
import os
import pandas as pd
from sqlalchemy import create_engine, text
from typing import Dict, List, Any, Tuple

# --- Configuration ---
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_HOST = os.getenv("DB_HOST", "db")  # Use 'localhost' if running script outside Docker hitting a local DB
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_TABLE = os.getenv("DB_TABLE", "sensor_data_merged")

CONFIG_FILE_PATH = "/app/input_config/data_processing_config.json" # Absolute path in container
OUTPUT_DIR_CONTAINER = "/app/output_data" # Absolute path in container for output

ERA1_START_TIME = "2013-12-01T00:00:00Z"
ERA1_END_TIME = "2016-09-08T00:00:00Z" # Or whatever your true full end date is

# Columns to analyze from the config's outlier_rules
# If empty, it will try to get them from the config file
COLUMNS_TO_ANALYZE = [
    "air_temp_c",
    "relative_humidity_percent",
    "radiation_w_m2",
    "light_intensity_umol",
    "co2_measured_ppm",
]

# Parameters for outlier suggestion methods
IQR_MULTIPLIER = 1.5
PERCENTILE_LOWER = 0.01  # 1st percentile
PERCENTILE_UPPER = 0.99  # 99th percentile

# Hard physical limits (override statistical suggestions if they go beyond these)
PHYSICAL_LIMITS = {
    "relative_humidity_percent": {"min": 0, "max": 100},
    "radiation_w_m2": {"min": 0, "max": 2000}, # Example, adjust as needed
    "light_intensity_umol": {"min": 0, "max": 5000}, # Generous but more physical than 1e39
    "co2_measured_ppm": {"min": 0} # CO2 realistically won't be negative
    # air_temp_c can have a wide range, so statistical might be better
}

def get_db_engine():
    db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    try:
        engine = create_engine(db_url)
        with engine.connect() as connection: # Test connection
            print(f"Successfully connected to database: {DB_NAME}")
        return engine
    except Exception as e:
        print(f"Error connecting to database: {e}")
        raise

def load_json_config(file_path: str) -> Dict:
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return {}

def get_column_stats_from_db(engine, column_name: str, start_time: str, end_time: str) -> Dict[str, Any]:
    """
    Fetches statistics for a given column from the database for the specified time range.
    Handles potential issues with light_intensity_umol by applying a pre-filter for stats calc.
    """
    # Pre-filter for extremely anomalous columns like light_intensity_umol for stats calculation
    # The actual_min/max will still be from the full range.
    stats_filter_condition = ""
    if column_name == "light_intensity_umol":
        # Apply a reasonable upper bound just for calculating Q1, Q3, AVG, STDDEV
        # This avoids these stats being skewed by totally non-physical values.
        stats_filter_condition = f"AND {column_name} >= {PHYSICAL_LIMITS.get(column_name, {}).get('min', -1e10)} AND {column_name} <= {PHYSICAL_LIMITS.get(column_name, {}).get('max', 1e10)}"
    
    query = text(f"""
    WITH filtered_data AS (
        SELECT {column_name}
        FROM {DB_TABLE}
        WHERE time >= :start_time AND time <= :end_time AND {column_name} IS NOT NULL
    ),
    stats_data AS (
        SELECT {column_name}
        FROM {DB_TABLE}
        WHERE time >= :start_time AND time <= :end_time AND {column_name} IS NOT NULL {stats_filter_condition}
    )
    SELECT
        (SELECT MIN({column_name}) FROM filtered_data) AS actual_min,
        (SELECT MAX({column_name}) FROM filtered_data) AS actual_max,
        (SELECT AVG({column_name}) FROM stats_data) AS mean_value,
        (SELECT STDDEV({column_name}) FROM stats_data) AS stddev_value,
        (SELECT PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY {column_name}) FROM stats_data) AS p01,
        (SELECT PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {column_name}) FROM stats_data) AS q1,
        (SELECT PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {column_name}) FROM stats_data) AS q3,
        (SELECT PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY {column_name}) FROM stats_data) AS p99
    """)
    
    try:
        with engine.connect() as connection:
            result = connection.execute(query, {"start_time": start_time, "end_time": end_time}).fetchone()
        if result:
            # Convert RowProxy to dict
            return dict(zip(result.keys(), result)) if hasattr(result, 'keys') else dict(result._mapping)

        return {}
    except Exception as e:
        print(f"Error fetching stats for {column_name}: {e}")
        return {}

def suggest_outlier_bounds(stats: Dict[str, Any], column_name: str) -> Dict[str, float]:
    """Suggests outlier bounds based on IQR and percentiles, respecting physical limits."""
    suggestions = {}
    q1 = stats.get('q1')
    q3 = stats.get('q3')
    p01 = stats.get('p01')
    p99 = stats.get('p99')

    phys_min = PHYSICAL_LIMITS.get(column_name, {}).get('min', -float('inf'))
    phys_max = PHYSICAL_LIMITS.get(column_name, {}).get('max', float('inf'))

    # IQR method
    if q1 is not None and q3 is not None:
        iqr = q3 - q1
        iqr_lower = q1 - IQR_MULTIPLIER * iqr
        iqr_upper = q3 + IQR_MULTIPLIER * iqr
        suggestions['iqr_min'] = max(iqr_lower, phys_min)
        suggestions['iqr_max'] = min(iqr_upper, phys_max)

    # Percentile method
    if p01 is not None:
        suggestions['p01_min'] = max(p01, phys_min)
    if p99 is not None:
        suggestions['p99_max'] = min(p99, phys_max)
        
    # Actual observed min/max (can also be suggestions if conservative)
    if stats.get('actual_min') is not None:
        suggestions['actual_observed_min'] = max(stats['actual_min'], phys_min)
    if stats.get('actual_max') is not None:
        suggestions['actual_observed_max'] = min(stats['actual_max'], phys_max)


    return suggestions

def analyze_and_suggest_config_updates():
    print("Starting configuration analysis...")
    engine = get_db_engine()
    if not engine:
        return

    current_config = load_json_config(CONFIG_FILE_PATH)
    if not current_config or "preprocessing" not in current_config or "outlier_rules" not in current_config["preprocessing"]:
        print("Invalid or empty configuration loaded. Cannot proceed.")
        return

    columns_in_config = {rule["column"] for rule in current_config["preprocessing"]["outlier_rules"]}
    
    analysis_target_columns = COLUMNS_TO_ANALYZE if COLUMNS_TO_ANALYZE else list(columns_in_config)
    if not analysis_target_columns:
        print("No columns specified or found in config to analyze.")
        return

    print(f"\nAnalyzing columns for Era 1 ({ERA1_START_TIME} to {ERA1_END_TIME}): {', '.join(analysis_target_columns)}\n")

    report = []
    updated_rules = [] # For potentially creating a new config

    for rule in current_config["preprocessing"]["outlier_rules"]:
        col = rule["column"]
        if col not in analysis_target_columns:
            updated_rules.append(rule) # Keep rule as is if not analyzed
            continue

        print(f"--- Analyzing: {col} ---")
        stats = get_column_stats_from_db(engine, col, ERA1_START_TIME, ERA1_END_TIME)
        
        col_report = {"column": col, "current_min": rule.get("min_value"), "current_max": rule.get("max_value")}
        
        if not stats:
            print(f"Could not retrieve statistics for {col}. Keeping existing rule.\n")
            col_report["notes"] = "Failed to retrieve DB stats."
            report.append(col_report)
            updated_rules.append(rule)
            continue

        col_report["db_actual_min"] = stats.get('actual_min')
        col_report["db_actual_max"] = stats.get('actual_max')
        col_report["db_q1"] = stats.get('q1')
        col_report["db_q3"] = stats.get('q3')
        col_report["db_p01"] = stats.get('p01')
        col_report["db_p99"] = stats.get('p99')

        suggested = suggest_outlier_bounds(stats, col)
        col_report["suggested_iqr_min"] = suggested.get('iqr_min')
        col_report["suggested_iqr_max"] = suggested.get('iqr_max')
        col_report["suggested_p01_min"] = suggested.get('p01_min')
        col_report["suggested_p99_max"] = suggested.get('p99_max')
        
        # --- Logic for choosing which suggested bounds to use for an update ---
        # Example: Prefer IQR, fallback to P01/P99 if IQR is None, then actual observed.
        # Always respect physical limits (already handled in suggest_outlier_bounds).
        
        chosen_min = rule.get("min_value") # Default to current
        if suggested.get('iqr_min') is not None:
            chosen_min = suggested['iqr_min']
        elif suggested.get('p01_min') is not None:
            chosen_min = suggested['p01_min']
        elif suggested.get('actual_observed_min') is not None: # Fallback to actual observed if no stats
             chosen_min = suggested['actual_observed_min']


        chosen_max = rule.get("max_value") # Default to current
        if suggested.get('iqr_max') is not None:
            chosen_max = suggested['iqr_max']
        elif suggested.get('p99_max') is not None:
            chosen_max = suggested['p99_max']
        elif suggested.get('actual_observed_max') is not None: # Fallback to actual observed
            chosen_max = suggested['actual_observed_max']

        # Ensure chosen_min is not greater than chosen_max if both are numbers
        if isinstance(chosen_min, (int, float)) and isinstance(chosen_max, (int, float)) and chosen_min > chosen_max:
            print(f"Warning for {col}: Suggested min ({chosen_min}) is greater than suggested max ({chosen_max}). Reviewing logic or data. Keeping current for now.")
            # Fallback to something safer or keep current if logic results in invalid range
            chosen_min = rule.get("min_value")
            chosen_max = rule.get("max_value")


        print(f"  Current Config: min={rule.get('min_value')}, max={rule.get('max_value')}")
        print(f"  DB Actual:    min={stats.get('actual_min_filtered', stats.get('actual_min')):.2f}, max={stats.get('actual_max_filtered', stats.get('actual_max')):.2f} (Note: light_intensity_umol max might be extreme)")
        print(f"  DB Q1:        {stats.get('q1'):.2f}, Q3: {stats.get('q3'):.2f}")
        print(f"  Suggested IQR: min={suggested.get('iqr_min'):.2f}, max={suggested.get('iqr_max'):.2f}")
        print(f"  Suggested P01/P99: min={suggested.get('p01_min'):.2f}, max={suggested.get('p99_max'):.2f}")
        print(f"  Chosen for Update: min={chosen_min}, max={chosen_max}\n")
        
        col_report["chosen_new_min"] = chosen_min
        col_report["chosen_new_max"] = chosen_max
        report.append(col_report)

        new_rule = rule.copy()
        if chosen_min is not None: new_rule["min_value"] = round(chosen_min, 4) if isinstance(chosen_min, float) else chosen_min
        if chosen_max is not None: new_rule["max_value"] = round(chosen_max, 4) if isinstance(chosen_max, float) else chosen_max
        updated_rules.append(new_rule)

    print("\n--- Summary Report ---")
    report_df = pd.DataFrame(report)
    print(report_df.to_string())

    # Create the new config structure
    new_config_data = current_config.copy()
    new_config_data["preprocessing"]["outlier_rules"] = updated_rules
    
    # Save the new config to a different file for review
    base_config_filename = os.path.basename(CONFIG_FILE_PATH)
    output_filename = base_config_filename.replace(".json", "_suggested.json")
    
    # Ensure output directory exists in container (though volume mount should handle host side)
    os.makedirs(OUTPUT_DIR_CONTAINER, exist_ok=True) 
    output_config_path = os.path.join(OUTPUT_DIR_CONTAINER, output_filename)

    try:
        with open(output_config_path, 'w') as f:
            json.dump(new_config_data, f, indent=4)
        print(f"\nSuggested configuration saved to: {output_config_path}")
        print("Please review this file and, if acceptable, rename it or copy its contents to your main config.")
    except Exception as e:
        print(f"Error saving suggested config: {e}")


if __name__ == "__main__":
    # Create dummy config if it doesn't exist for testing
    # if not os.path.exists(CONFIG_FILE_PATH):
    #     print(f"Creating dummy config at {CONFIG_FILE_PATH}")
    #     dummy_config = {
    #         "preprocessing": {
    #             "outlier_rules": [
    #                 {"column": "air_temp_c", "min_value": -5, "max_value": 40, "clip": True},
    #                 {"column": "co2_measured_ppm", "min_value": 300, "max_value": 1200, "clip": True}
    #             ]
    #         }
    #     }
    #     os.makedirs(os.path.dirname(CONFIG_FILE_PATH), exist_ok=True)
    #     with open(CONFIG_FILE_PATH, 'w') as f:
    #         json.dump(dummy_config, f, indent=4)
            
    analyze_and_suggest_config_updates()
