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
import pandas as original_pandas # Use this for true pandas operations
import sqlalchemy
import sys # For checking loaded modules

# Local helper for DB access
from db_utils import SQLAlchemyPostgresConnector
from tsfresh import extract_features
from tsfresh.feature_extraction.settings import EfficientFCParameters

# Conditionally import and alias pd
if os.getenv("USE_GPU", "false").lower() == "true":
    try:
        import cudf.pandas as pd  # pd becomes cudf.pandas
        import cudf               # For explicit cudf.DataFrame, etc.
        import dask_cudf          # For Dask-cuDF operations
        import inspect            # For monkey-patching
        logging.info("Running in GPU mode with cudf.pandas, cudf, and dask_cudf.")
    except ImportError as e:
        logging.error(f"Failed to import GPU libraries (cudf, dask_cudf): {e}. Falling back to CPU mode.")
        os.environ["USE_GPU"] = "false" # Force fallback for current script execution
        pd = original_pandas      # Ensure pd is original_pandas
        # No need to re-import cudf, dask_cudf, inspect if they failed; they won't be used.
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

# Location where the parquet file will be written inside the container
DEFAULT_OUTPUT_PATH = Path("/app/data/output/tsfresh_features.parquet")
OUTPUT_PATH = Path(_env("OUTPUT_PATH", str(DEFAULT_OUTPUT_PATH)))

# Optional: only process a single era, useful during debugging
# FILTER_ERA: str | None = os.getenv("ERA_IDENTIFIER")

# Global constant for reporting the feature set used
FC_PARAMS_NAME = "EfficientFCParameters"


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
        # Bind parameters to the text query object
        prepared_query = sqlalchemy.text(query).bindparams(**params)
        # Pass the prepared query object to the fetch method
        df_host = connector.fetch_data_to_pandas(query=prepared_query) # Fetches as Pandas DF

        if os.getenv("USE_GPU", "false").lower() == "true" and 'cudf' in sys.modules:
            if not df_host.empty:
                logging.info("Converting fetched pandas DataFrame to cuDF DataFrame...")
                df = cudf.DataFrame.from_pandas(df_host)   # lives in VRAM
                logging.info("Conversion to cuDF DataFrame complete.")
            else:
                df = cudf.DataFrame(columns=df_host.columns) # Empty cuDF DataFrame with same columns
        else:
            df = df_host # Use the host pandas DataFrame directly

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

    if os.getenv("USE_GPU", "false").lower() == "true" and 'cudf' in sys.modules and isinstance(df, cudf.DataFrame):
        if not df["features"].empty:
            logging.info("GPU Mode: Using original_pandas for JSON normalization.")
            # 1. Convert cuDF Series to original_pandas Series
            original_pandas_features_series = df["features"].to_pandas()
            # 2. Normalize using original_pandas
            normalized_original_pandas_df = original_pandas.json_normalize(original_pandas_features_series)
            # 3. Convert resulting original_pandas DataFrame back to cuDF DataFrame
            feature_cols = cudf.DataFrame.from_pandas(normalized_original_pandas_df)
        else:
            logging.warning("GPU Mode: 'features' column is empty. Creating empty cuDF feature_cols.")
            feature_cols = cudf.DataFrame(index=df.index) # df.index is a cuDF index here
    else: # CPU mode or df is an original_pandas DataFrame
        if not df["features"].empty:
            # In CPU mode, pd is original_pandas. df["features"] is an original_pandas Series.
            # So, pd.json_normalize is effectively original_pandas.json_normalize
            feature_cols = pd.json_normalize(df["features"])
        else:
            logging.warning("CPU Mode: 'features' column is empty. Creating empty original_pandas feature_cols.")
            # In CPU mode, pd is original_pandas. df.index is an original_pandas index.
            feature_cols = pd.DataFrame(index=df.index)

    # Align indices. df.index should be appropriate for pd (cudf or pandas type based on mode)
    # This handles cases where feature_cols might be empty but df was not.
    if not df.empty:
        feature_cols.index = df.index
    elif feature_cols.empty and df.empty: # Both empty, ensure index consistency if any schema known
        pass # Default empty indices are usually fine

    if os.getenv("USE_GPU", "false").lower() == "true" and 'cudf' in sys.modules and isinstance(df, cudf.DataFrame):
        # Ensure both parts are cudf DataFrames
        df_time_era = df[["time", "era_identifier"]]
        # feature_cols should already be a cudf.DataFrame if in GPU mode from previous step
        wide_df = cudf.concat([df_time_era, feature_cols], axis=1)
    else:
        # Ensure both parts are original_pandas DataFrames for concatenation
        df_time_era = df[["time", "era_identifier"]]
        current_feature_cols = feature_cols
        if 'cudf' in sys.modules and isinstance(df_time_era, cudf.DataFrame):
            df_time_era = df_time_era.to_pandas()
        if 'cudf' in sys.modules and isinstance(current_feature_cols, cudf.DataFrame):
            current_feature_cols = current_feature_cols.to_pandas()
        wide_df = original_pandas.concat([df_time_era, current_feature_cols], axis=1)

    logging.info("Wide DataFrame shape after unpack: %s", wide_df.shape)

    return wide_df


def melt_for_tsfresh(wide_df: pd.DataFrame) -> pd.DataFrame:
    """Convert wide sensor matrix (numeric columns only) to long format required by *tsfresh*."""

    logging.info("Converting to long (tidy) format for tsfresh …")

    id_columns = ["era_identifier", "time"]
    # Select only numeric columns for melting, excluding the ID columns
    # In cuDF, select_dtypes(include=np.number) might not work as robustly as iterating and checking.
    # Let's refine the numeric column selection for both pandas and cudf.

    numeric_cols = []
    if 'cudf' in sys.modules and isinstance(wide_df, cudf.DataFrame):
        for col in wide_df.columns:
            if col not in id_columns and cudf.api.types.is_numeric_dtype(wide_df[col].dtype):
                numeric_cols.append(col)
    else: # Assuming original_pandas DataFrame
        numeric_cols = wide_df.select_dtypes(include=original_pandas.api.types.is_number).columns.tolist()

    value_columns = [col for col in numeric_cols if col not in id_columns]

    if not value_columns:
        raise ValueError("No numeric columns found in wide_df to melt for tsfresh.")

    logging.info(f"Melting based on {len(value_columns)} numeric columns: {value_columns}")

    # Ensure common dtype for value_vars in GPU mode before melting
    if os.getenv("USE_GPU", "false").lower() == "true" and 'cudf' in sys.modules and isinstance(wide_df, cudf.DataFrame):
        logging.info("GPU Mode: Ensuring common dtype (float32) for value_vars before melt.")
        temp_wide_df = wide_df.copy() # Work on a copy to avoid modifying the original df used elsewhere
        for col_name in value_columns:
            if col_name in temp_wide_df.columns:
                try:
                    # Ensure the column is numeric-like before casting to avoid errors on non-numeric types
                    if cudf.api.types.is_numeric_dtype(temp_wide_df[col_name].dtype):
                        temp_wide_df[col_name] = temp_wide_df[col_name].astype(np.float32)
                    else:
                        logging.warning(f"GPU Mode: Column '{col_name}' is not numeric, attempting to convert. Original dtype: {temp_wide_df[col_name].dtype}")
                        # Attempt conversion, which might fail if data is not convertible
                        temp_wide_df[col_name] = cudf.to_numeric(temp_wide_df[col_name], errors='coerce').astype(np.float32)
                except Exception as e_cast:
                    logging.error(f"GPU Mode: Failed to cast column '{col_name}' to float32. Error: {e_cast}. Skipping this column for melt.")
                    # Remove problematic column from value_columns to avoid melt error
                    value_columns.remove(col_name)
            else:
                logging.warning(f"GPU Mode: Column '{col_name}' not found in temp_wide_df for dtype conversion. This shouldn't happen.")
        wide_df_to_melt = temp_wide_df # Use the (potentially modified) copy for melting
        if not value_columns: # Check if all columns were removed due to errors
             raise ValueError("GPU Mode: No valid numeric columns left after dtype conversion attempts for melting.")
    else:
        wide_df_to_melt = wide_df # Use the original DataFrame for CPU path

    long_df = (
        wide_df_to_melt.melt(
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

# tsfresh monkey-patching for GPU - REMOVED as it was ineffective and problematic
# def _patch_tsfresh_gpu():
#     ...

# The run_tsfresh function will be removed as its logic is integrated into the main loop with per-variable processing.
# def run_tsfresh(long_df: pd.DataFrame) -> pd.DataFrame:
#     logging.info("Running tsfresh feature extraction (this can take a while) …")
# 
#     features = extract_features(
#         long_df,
#         column_id="id",
#         column_sort="time",
#         column_kind="variable",
#         column_value="value",
#         disable_progressbar=True,
#         n_jobs=0,  # use all CPUs available in container
#     )
# 
#     logging.info("tsfresh generated %s features", features.shape[1])
#     return features


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

    USE_GPU_FLAG = os.getenv("USE_GPU", "false").lower() == "true"

    # REMOVED: _patch_tsfresh_gpu() call, as the patching was not working and causing issues.
    # if USE_GPU_FLAG:
    #     _patch_tsfresh_gpu()

    for i, era in enumerate(eras_to_process):
        logging.info(f"--- Processing Era {i+1}/{len(eras_to_process)}: {era} ---")
        try:
            wide_df_era = fetch_preprocessed_data_for_era(connector, era)
            if wide_df_era.empty:
                 logging.warning(f"Skipping era '{era}' due to empty data after fetch/unpack.")
                 continue # Skip to the next era

            total_rows_processed += len(wide_df_era)
            # Using .name attribute for series if pd is cudf.pandas
            current_sensor_cols = {c.name if hasattr(c, 'name') else c for c in wide_df_era.columns if (c.name if hasattr(c, 'name') else c) not in {"time", "era_identifier"}}
            all_sensor_cols.update(current_sensor_cols)
            
            long_df_era = melt_for_tsfresh(wide_df_era)
            if long_df_era.empty:
                logging.warning(f"Skipping era '{era}' because long DataFrame became empty after melting (e.g., all numeric values were NaN).")
                continue

            if USE_GPU_FLAG and 'cudf' in sys.modules:
                logging.info(f"GPU Mode: Preparing cuDF DataFrame for era '{era}' for direct tsfresh processing...")
                # long_df_era is already a cuDF DataFrame from melt_for_tsfresh
                
                logging.info(f"GPU Mode: Converting cuDF DataFrame to pandas DataFrame before calling tsfresh for era '{era}' (shape: {long_df_era.shape})...")
                long_df_era_pandas = long_df_era.to_pandas() # Convert to pandas

                logging.info(f"GPU Mode: Running tsfresh on pandas DataFrame (converted from cuDF) for era '{era}' (shape: {long_df_era_pandas.shape})...")
                try:
                    # Call tsfresh.extract_features directly on the pandas DataFrame.
                    # n_jobs=1 is still a good practice here as underlying data might have GPU origins,
                    # and we want to control parallelism.
                    current_era_all_var_features = extract_features(
                        long_df_era_pandas, # Pass the pandas DataFrame
                        column_id="id",
                        column_sort="time",
                        column_kind="variable",
                        column_value="value",
                        default_fc_parameters=EfficientFCParameters(), # Use EfficientFCParameters
                        disable_progressbar=True,
                        n_jobs=1 # Use 1 job
                        # pivot=True is default and usually desired here
                    )
                    # The result current_era_all_var_features will be a pandas DataFrame.
                    # The final concatenation logic will handle converting it back to cuDF if needed.
                    logging.info(f"GPU Mode: tsfresh completed for era '{era}'. Features shape (pandas): {current_era_all_var_features.shape}")
                except Exception as e_tsfresh_gpu:
                    logging.error(f"GPU Mode: tsfresh extraction failed for era '{era}': {e_tsfresh_gpu}", exc_info=True)
                    continue # Skip this era on failure

                if current_era_all_var_features.empty:
                    logging.warning(f"GPU Mode: tsfresh returned no features for era '{era}'.")
                    continue
                
                all_era_features.append(current_era_all_var_features)

            else: # CPU Path - existing logic
                # New logic: Process tsfresh per variable
                unique_variables = long_df_era['variable'].unique()
                era_specific_features_list: list[pd.DataFrame] = []
                logging.info(f"CPU Mode: Found {len(unique_variables)} variables to process for era '{era}'.")

                for var_idx, var_name in enumerate(unique_variables):
                    logging.info(f"--- CPU Mode: Processing Era '{era}', Variable {var_idx+1}/{len(unique_variables)}: {var_name} ---")
                    # Use .copy() to avoid potential SettingWithCopyWarning later
                    df_single_variable = long_df_era[long_df_era['variable'] == var_name].copy()

                    # tsfresh expects columns: id, time, value for basic operation per kind
                    df_to_extract = df_single_variable[['id', 'time', 'value']]

                    if df_to_extract.empty or df_to_extract['value'].isnull().all():
                        logging.warning(f"CPU Mode: Data for variable '{var_name}' in era '{era}' is empty or all NaNs. Skipping this variable.")
                        continue

                    logging.info(f"CPU Mode: Running tsfresh for variable '{var_name}' in era '{era}' (data shape: {df_to_extract.shape})...")
                    try:
                        var_features = extract_features(
                            df_to_extract,
                            column_id="id",
                            column_sort="time",
                            column_value="value",
                            default_fc_parameters=EfficientFCParameters(), # Use EfficientFCParameters
                            disable_progressbar=True,
                            n_jobs=1,  # Reduce parallelism to 1 for CPU
                            # show_warnings=False, # Consider adding if matrix_profile warning is noisy and stumpy won't be installed
                        )
                    except Exception as e_tsfresh:
                        logging.error(f"CPU Mode: tsfresh extraction failed for variable '{var_name}' in era '{era}': {e_tsfresh}", exc_info=True)
                        continue # Skip this variable

                    if var_features.empty:
                        logging.warning(f"CPU Mode: tsfresh returned no features for variable '{var_name}' in era '{era}'.")
                        continue
                    
                    # Prepend variable name to feature names to ensure uniqueness
                    var_features.columns = [f"{var_name}__{col}" for col in var_features.columns]
                    era_specific_features_list.append(var_features)
                    logging.info(f"CPU Mode: Generated {var_features.shape[1]} features for variable '{var_name}' in era '{era}'.")

                if not era_specific_features_list:
                    logging.warning(f"CPU Mode: No features generated for any variable in era '{era}'. Skipping this era.")
                    continue

                # Concatenate features for all variables for the current era
                current_era_all_var_features = pd.concat(era_specific_features_list, axis=1)
                all_era_features.append(current_era_all_var_features)
            
            logging.info(f"--- Completed processing for Era: {era}. Total features for era: {current_era_all_var_features.shape[1]} ---")

        except Exception as e:
            logging.error(f"Failed processing era '{era}': {e}. Skipping this era.", exc_info=True)
            # Decide if you want to stop completely or continue with other eras
            # For now, we log and continue

    if not all_era_features:
        logging.error("No features were generated for any era. Exiting.")
        return # Exit if no features were generated at all

    logging.info("Concatenating features from all processed eras...")

    if os.getenv("USE_GPU", "false").lower() == "true" and 'cudf' in sys.modules:
        # Expect all_era_features to contain cudf.DataFrames
        # Add a small check and conversion for robustness, though ideally not needed.
        processed_list_for_concat = []
        for item in all_era_features:
            if isinstance(item, original_pandas.DataFrame):
                logging.warning("Found original_pandas.DataFrame in all_era_features during GPU mode; converting to cuDF.")
                processed_list_for_concat.append(cudf.DataFrame.from_pandas(item))
            else:
                processed_list_for_concat.append(item) # Assume it's a cuDF DataFrame or compatible
        
        if not processed_list_for_concat: # Handle empty list after potential filtering/errors
            logging.warning("No features available in processed_list_for_concat for GPU mode. Creating empty cuDF DataFrame.")
            final_tsfresh_features = cudf.DataFrame()
        else:
            final_tsfresh_features = cudf.concat(processed_list_for_concat)
    else:
        # Expect all_era_features to contain original_pandas.DataFrames
        # Add a small check and conversion for robustness.
        processed_list_for_concat = []
        for item in all_era_features:
            if 'cudf' in sys.modules and isinstance(item, cudf.DataFrame):
                logging.warning("Found cudf.DataFrame in all_era_features during CPU mode; converting to pandas.")
                processed_list_for_concat.append(item.to_pandas())
            else:
                processed_list_for_concat.append(item) # Assume it's an original_pandas.DataFrame or compatible

        if not processed_list_for_concat: # Handle empty list
            logging.warning("No features available in processed_list_for_concat for CPU mode. Creating empty pandas DataFrame.")
            final_tsfresh_features = original_pandas.DataFrame()
        else:
            final_tsfresh_features = original_pandas.concat(processed_list_for_concat)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_tsfresh_features.to_parquet(OUTPUT_PATH)
    logging.info("Feature set saved to %s", OUTPUT_PATH)

    # --- Gather info for the report ---
    parquet_file_path_str = str(OUTPUT_PATH.resolve()) # Get absolute path
    parquet_file_size_bytes = 0
    if OUTPUT_PATH.exists(): # Check if file exists before getting size
        parquet_file_size_bytes = OUTPUT_PATH.stat().st_size
    parquet_file_size_mb = parquet_file_size_bytes / (1024 * 1024)
    
    df_rows, df_cols = 0, 0 # Initialize
    if not final_tsfresh_features.empty:
        df_rows, df_cols = final_tsfresh_features.shape
    else:
        # df_cols is n_features, which is already used below and would be 0 if empty.
        # df_rows would be 0.
        pass 

    # ---------------- summary report ------------------ #
    report_path = OUTPUT_PATH.parent / "tsfresh_feature_extraction_report.txt"
    logging.info("Writing summary report → %s", report_path)

    # Number of input rows and sensor columns
    # n_rows = total_rows_processed # Use the counter from the loop
    n_sensors = len(all_sensor_cols) # Use the set accumulated during the loop
    n_features = final_tsfresh_features.shape[1]
    n_eras_processed = len(all_era_features) # Number of eras that successfully produced features

    with open(report_path, "w", encoding="utf-8") as rep:
        # Dynamically set report title based on GPU usage
        report_title_prefix = "GPU (tsfresh)" if USE_GPU_FLAG and 'cudf' in sys.modules else "CPU (tsfresh)"
        rep.write(f"{report_title_prefix} Feature Extraction Report\n")
        rep.write("=======================================\n\n")
        rep.write(f"Feature settings used : {FC_PARAMS_NAME}\n")
        rep.write(f"Eras processed        : {n_eras_processed} / {len(eras_to_process)}\n")
        rep.write(f"Total rows processed  : {total_rows_processed:,}\n") # From original wide_df rows
        rep.write(f"Sensors considered    : {n_sensors:,}\n")
        rep.write(f"Total feature columns : {df_cols:,}\n\n") # Use df_cols from shape

        rep.write(f"Output Parquet file   : {parquet_file_path_str}\n")
        rep.write(f"Parquet file size (MB): {parquet_file_size_mb:.2f}\n")
        rep.write(f"DataFrame shape (rows, features): ({df_rows}, {df_cols})\n\n")

        rep.write("First 25 feature names →\n")
        for fname in list(final_tsfresh_features.columns)[:25]:
            rep.write(f"  • {fname}\n")

        rep.write("\nExtraction successful.\n")

    logging.info("Report written.")


if __name__ == "__main__":
    main()
