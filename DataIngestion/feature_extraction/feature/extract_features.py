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
from tsfresh.feature_extraction.settings import EfficientFCParameters, MinimalFCParameters

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

# Where to persist the selected feature set
DEFAULT_SELECTED_OUTPUT_PATH = Path("/app/data/output/tsfresh_features_selected.parquet")
SELECTED_OUTPUT_PATH = Path(_env("SELECTED_OUTPUT_PATH", str(DEFAULT_SELECTED_OUTPUT_PATH)))

# Table name for database storage of selected features
FEATURES_TABLE = _env("FEATURES_TABLE", "tsfresh_selected_features")

# Optional: only process a single era, useful during debugging
# FILTER_ERA: str | None = os.getenv("ERA_IDENTIFIER") # This might be re-purposed for filtering loaded era definitions

# Paths for input files (to be set via env vars or args)
CONSOLIDATED_DATA_PATH_ENV = "CONSOLIDATED_DATA_PATH"
ERA_DEFINITIONS_PATH_ENV = "ERA_DEFINITIONS_PATH"
ERA_ID_COLUMN_KEY_ENV = "ERA_ID_COLUMN_KEY" # Env var to specify which 'era_level_X' to use

DEFAULT_CONSOLIDATED_DATA_PATH = "/app/data/processed/consolidated_data.jsonl"  # Example path
DEFAULT_ERA_DEFINITIONS_PATH = "/app/data/era_definitions/"    # Example: directory containing era JSONL files
DEFAULT_ERA_ID_COLUMN_KEY = "era_level_B" # Default key for era identifier in JSONL

# Global constant for reporting the feature set used
FC_PARAMS_NAME = "MinimalFCParameters"  # Updated to MinimalFCParameters


# -----------------------------------------------------------------------------
# Data loading & preparation helpers
# -----------------------------------------------------------------------------

def load_era_definitions(era_definitions_dir_path: str, era_id_key: str, USE_GPU_FLAG: bool) -> pd.DataFrame | None:
    """Load era definitions from all JSONL and Parquet files in a directory, calculate end times, and return a DataFrame."""
    jsonl_files = list(Path(era_definitions_dir_path).glob('*.jsonl'))
    parquet_files = list(Path(era_definitions_dir_path).glob('*.parquet'))
    
    if not jsonl_files and not parquet_files:
        logging.error(f"No JSONL or Parquet files found in era definitions directory: {era_definitions_dir_path}")
        return None

    all_era_dfs = []

    # Process JSONL files
    for file_path in jsonl_files:
        logging.info(f"Loading era definitions from JSONL: {file_path}")
        current_raw_eras = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_number, line in enumerate(f, 1):
                    try:
                        record = json.loads(line)
                        if 'time' not in record or era_id_key not in record:
                            logging.warning(f"Skipping record in {file_path} line {line_number}: missing 'time' or '{era_id_key}'. Record: {record}")
                            continue
                        current_raw_eras.append(record)
                    except json.JSONDecodeError:
                        logging.warning(f"Skipping invalid JSON in {file_path} line {line_number}: {line.strip()}")
            if current_raw_eras:
                df = original_pandas.DataFrame(current_raw_eras)
                 # For JSONL, assume 'time' might be ms epoch. Try parsing, then fallback.
                try:
                    df['start_time'] = original_pandas.to_datetime(df['time'], unit='ms', errors='raise')
                except (ValueError, TypeError):
                    logging.warning(f"Could not parse 'time' as ms epoch in {file_path}, trying default parsing.")
                    df['start_time'] = original_pandas.to_datetime(df['time'], errors='coerce')
                all_era_dfs.append(df)
        except Exception as e:
            logging.error(f"Error reading or parsing JSONL file {file_path}: {e}")
            continue

    # Process Parquet files
    for file_path in parquet_files:
        logging.info(f"Loading era definitions from Parquet: {file_path}")
        try:
            if USE_GPU_FLAG and 'cudf' in sys.modules:
                df = cudf.read_parquet(file_path) 
                # Convert to pandas for consistent processing initially, then convert back if needed
                # This simplifies handling diverse Parquet contents before final cuDF conversion
                df = df.to_pandas() 
            else:
                df = original_pandas.read_parquet(file_path)
            
            if 'time' not in df.columns or era_id_key not in df.columns:
                logging.warning(f"Skipping Parquet file {file_path}: missing 'time' or '{era_id_key}' column. Found columns: {df.columns.tolist()}")
                continue
            # For Parquet, assume 'time' is likely a standard datetime string or object. Fallback to ms epoch if direct fails.
            try:
                 df['start_time'] = original_pandas.to_datetime(df['time'], errors='raise')
            except (ValueError, TypeError):
                logging.warning(f"Could not parse 'time' directly in Parquet {file_path}, trying as ms epoch.")
                df['start_time'] = original_pandas.to_datetime(df['time'], unit='ms', errors='coerce')
            all_era_dfs.append(df)
        except Exception as e:
            logging.error(f"Error reading or parsing Parquet file {file_path}: {e}")
            continue

    if not all_era_dfs:
        logging.error("No valid era definitions loaded from any JSONL or Parquet file.")
        return None

    # Concatenate all loaded DataFrames
    eras_df = original_pandas.concat(all_era_dfs, ignore_index=True)
    logging.info(f"Loaded a total of {len(eras_df)} raw era records from {len(jsonl_files) + len(parquet_files)} files.")

    # Ensure 'start_time' (derived from 'time') and the specified era_id_key column exist
    if 'start_time' not in eras_df.columns or era_id_key not in eras_df.columns:
        logging.error(f"'start_time' (from 'time') or '{era_id_key}' column not found after loading all era files. Available columns: {eras_df.columns.tolist()}")
        return None

    eras_df.rename(columns={era_id_key: 'era_id'}, inplace=True)

    # Drop rows where start_time could not be parsed or era_id is missing/NaN
    eras_df.dropna(subset=['start_time', 'era_id'], inplace=True)
    if eras_df.empty:
        logging.error("No valid eras remaining after datetime conversion and NA drop.")
        return None

    # Sort by start_time to correctly determine end_times
    eras_df.sort_values('start_time', inplace=True)
    eras_df.reset_index(drop=True, inplace=True)

    # Calculate end_time: the start_time of the next era
    eras_df['end_time'] = eras_df['start_time'].shift(-1)

    # Select and reorder columns
    final_eras_df = eras_df[['era_id', 'start_time', 'end_time']].copy()
    
    # If GPU mode is active, convert the resulting pandas DataFrame to cuDF
    if USE_GPU_FLAG and 'cudf' in sys.modules:
        logging.info("Converting final loaded era definitions DataFrame to cuDF.")
        try:
            final_eras_df = cudf.DataFrame.from_pandas(final_eras_df)
        except Exception as e_cudf_conv:
            logging.error(f"Failed to convert final era definitions DataFrame to cuDF: {e_cudf_conv}. Returning pandas DataFrame.")
            # Fallback to returning pandas DataFrame if conversion fails

    logging.info(f"Processed {len(final_eras_df)} era definitions. Columns: {final_eras_df.columns.tolist()}")
    return final_eras_df


def melt_for_tsfresh(wide_df: pd.DataFrame) -> pd.DataFrame:
    """Convert wide sensor matrix (numeric columns only) to long format required by *tsfresh*."""

    logging.info("Converting to long (tidy) format for tsfresh …")

    id_columns = ["id", "time"] # Expect 'id' column directly from main
    # Select only numeric columns for melting, excluding the ID columns
    # In cuDF, select_dtypes(include=np.number) might not work as robustly as iterating and checking.
    # Let's refine the numeric column selection for both pandas and cudf.

    numeric_cols = []
    if 'cudf' in sys.modules and isinstance(wide_df, cudf.DataFrame):
        for col in wide_df.columns:
            if col not in id_columns and cudf.api.types.is_numeric_dtype(wide_df[col].dtype):
                numeric_cols.append(col)
    else: # Assuming original_pandas DataFrame
        numeric_cols = wide_df.select_dtypes(include=[np.number]).columns.tolist()

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
    long_df.rename(columns={"variable": "kind"}, inplace=True)

        # No longer renaming 'era_identifier' to 'id' here, as 'id' is expected directly.
    # if "era_identifier" in long_df.columns:
    #     long_df.rename(
    #         columns={"era_identifier": "id"}, 
    #         inplace=True,
    #     )

    logging.info("Long DataFrame shape: %s", long_df.shape)

    return long_df


def select_relevant_features(
    features_df: pd.DataFrame,
    correlation_threshold: float = 0.95,
    variance_threshold: float = 0.0,
) -> pd.DataFrame:
    """Perform a simple unsupervised feature selection.

    Features with zero (or near-zero) variance are removed and highly
    correlated features are dropped based on *correlation_threshold*.
    The returned DataFrame preserves the original index.
    """

    if features_df.empty:
        return features_df

    use_gpu = os.getenv("USE_GPU", "false").lower() == "true" and "cudf" in sys.modules

    if use_gpu and isinstance(features_df, cudf.DataFrame):
        work_df = features_df.to_pandas()
    else:
        work_df = features_df.copy()

    # Remove constant columns
    variances = work_df.var()
    cols_to_keep = variances[variances > variance_threshold].index
    work_df = work_df[cols_to_keep]

    # Drop highly correlated columns
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


# -----------------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------------

def main() -> None:
    """Main execution function."""
    logging.info("Starting feature extraction with file-based inputs...")
    USE_GPU_FLAG = os.getenv("USE_GPU", "false").lower() == "true"

    # Get file paths from environment variables or use defaults
    consolidated_data_file_path = _env(CONSOLIDATED_DATA_PATH_ENV, DEFAULT_CONSOLIDATED_DATA_PATH)
    era_definitions_dir_path = _env(ERA_DEFINITIONS_PATH_ENV, DEFAULT_ERA_DEFINITIONS_PATH)
    era_id_key_from_json = _env(ERA_ID_COLUMN_KEY_ENV, DEFAULT_ERA_ID_COLUMN_KEY)

    logging.info(f"Loading consolidated data from: {consolidated_data_file_path}")
    try:
        # Determine if using GPU for pandas operations
        # pd is already aliased to cudf.pandas or original_pandas at the top of the script
        # Determine if using GPU for pandas operations
        if USE_GPU_FLAG and 'cudf' in sys.modules:
            # Load with original_pandas then convert to cuDF for robust JSONL parsing
            temp_pandas_df = original_pandas.read_json(consolidated_data_file_path, lines=True, orient='records')
            if temp_pandas_df.empty:
                logging.error(f"Consolidated data file is empty: {consolidated_data_file_path}. Exiting.")
                return
            consolidated_df = cudf.DataFrame.from_pandas(temp_pandas_df)
            del temp_pandas_df # Free memory
            logging.info("Loaded consolidated data into cuDF DataFrame.")
        else:
            consolidated_df = original_pandas.read_json(consolidated_data_file_path, lines=True, orient='records')
            if consolidated_df.empty:
                logging.error(f"Consolidated data file is empty: {consolidated_data_file_path}. Exiting.")
                return
            logging.info("Loaded consolidated data into pandas DataFrame.")
            
        logging.info(f"Consolidated data shape: {consolidated_df.shape}")

        # Convert 'time' column to datetime objects using the appropriate pandas module (pd could be cudf.pandas or original_pandas)
        if 'time' in consolidated_df.columns:
            consolidated_df['time'] = pd.to_datetime(consolidated_df['time'], unit='ms', errors='coerce') # Assuming time is ms epoch
            consolidated_df.dropna(subset=['time'], inplace=True) # Drop rows where time conversion failed
            if consolidated_df.empty:
                logging.error("Consolidated data became empty after 'time' conversion/NA drop. Exiting.")
                return
            # Sort by time, essential for time-series operations and correct slicing later
            consolidated_df.sort_values('time', inplace=True)
            consolidated_df.reset_index(drop=True, inplace=True)
        else:
            logging.error("'time' column not found in consolidated data. Exiting.")
            return

        # --- Handle -1 as NaN using list from memory --- 
        # MEMORY a3c1c22a-cbf9-437d-bed1-382773fd58e3
        columns_to_clean_neg_one = ['dli_sum', 'air_temp_c', 'relative_humidity_percent', 'co2_measured_ppm', 'radiation_w_m2', 'light_intensity_umol']
        for col in columns_to_clean_neg_one:
            if col in consolidated_df.columns:
                # Check if column is numeric before attempting replace -1
                if pd.api.types.is_numeric_dtype(consolidated_df[col]):
                    logging.info(f"Replacing -1 with NaN in numeric column: {col}")
                    if USE_GPU_FLAG and 'cudf' in sys.modules and isinstance(consolidated_df, cudf.DataFrame):
                        # cuDF replace method
                        consolidated_df[col] = consolidated_df[col].replace(-1, cudf.NA) # cudf.NA for nullable int/float types
                    else:
                        consolidated_df[col] = consolidated_df[col].replace(-1, np.nan)
                else:
                    logging.warning(f"Column {col} is not numeric, skipping -1 replacement.")        
            else:
                logging.warning(f"Column {col} (for -1 to NaN) not found in consolidated data.")

    except FileNotFoundError:
        logging.error(f"Consolidated data file not found: {consolidated_data_file_path}. Exiting.")
        return
    except Exception as e:
        logging.error(f"Error loading consolidated data from {consolidated_data_file_path}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return

    # Load era definitions using the new function that returns a DataFrame
    eras_df = load_era_definitions(era_definitions_dir_path, era_id_key_from_json, USE_GPU_FLAG)
    if eras_df is None or eras_df.empty:
        logging.error("No era definitions loaded or DataFrame is empty. Exiting.")
        return

    # Optional filtering of eras (can adapt FILTER_ERA logic here if needed)
    # filter_era_id = os.getenv("ERA_IDENTIFIER")
    # if filter_era_id:
    #     eras_definitions = [e for e in eras_definitions if e.get('era_level_B') == filter_era_id] # Adjust key as needed
    #     logging.info(f"Filtered to process {len(eras_definitions)} eras based on ERA_IDENTIFIER: {filter_era_id}")
    #     if not eras_definitions:
    #         logging.error(f"Specified ERA_IDENTIFIER '{filter_era_id}' not found in loaded era definitions. Exiting.")
    #         return

    all_era_features = []
    total_rows_processed = 0
    all_sensor_cols = set() # To store all unique sensor columns encountered

    # Define the ID column from era definitions (e.g., 'era_level_B')
    # This will be used as the 'id' for tsfresh grouping.
    # ERA_ID_KEY_IN_DEFINITION is now handled by load_era_definitions returning 'era_id' column

    # Iterate through the eras DataFrame from load_era_definitions
    for index, era_row in eras_df.iterrows():
        era_identifier = era_row['era_id']
        start_time = era_row['start_time']
        end_time = era_row['end_time'] # This can be pd.NaT for the last era

        logging.info(f"Processing era: {era_identifier}, Start: {start_time}, End: {end_time}")

        # Filter consolidated_df for the current era's time window
        if pd.isna(end_time): # Handle the last era (end_time is NaT)
            era_data_slice_mask = (consolidated_df['time'] >= start_time)
        else:
            era_data_slice_mask = (consolidated_df['time'] >= start_time) & (consolidated_df['time'] < end_time)
        
        # Apply mask and copy to avoid SettingWithCopyWarning, and to ensure 'id' column assignment doesn't affect original df
        # Use .loc for boolean indexing for both pandas and cudf compatibility
        wide_df_era_slice = consolidated_df.loc[era_data_slice_mask].copy()

        if wide_df_era_slice.empty:
            logging.warning(f"Skipping era {era_identifier} as no data falls within its time window [{start_time} - {end_time}).")
            continue
        
        # Add the 'id' column for tsfresh (using the era_identifier for this segment)
        # Use assign for better chaining and cudf compatibility if pd is cudf.pandas
        if hasattr(wide_df_era_slice, 'assign'): # Check if 'assign' method exists (for cudf or pandas)
            wide_df = wide_df_era_slice.assign(id=str(era_identifier)) # Ensure era_identifier is string for id
        else: # Fallback if 'assign' is not available (should not happen with modern pandas/cudf)
            wide_df = wide_df_era_slice.copy() # Ensure it's a copy
            wide_df['id'] = str(era_identifier)

        if wide_df.empty:
            logging.warning("Skipping era %s due to missing or empty data after slicing.", era_identifier)
            continue

        total_rows_processed += len(wide_df) 
        
        # Identify sensor columns to melt. Exclude 'time', 'id', and other metadata columns from era_def or main_df.
        # Example of potential metadata columns in wide_df if not careful: 'source_system', 'format_type', etc.
        # We need a definitive list of SENSOR columns to use or a way to exclude metadata.
        # For now, assume all numeric columns except 'time', 'id' are sensors.
        sensor_columns_for_melt = [col for col in wide_df.columns if pd.api.types.is_numeric_dtype(wide_df[col]) and col not in ['time', 'id']]
        # Filter to desired horticultural columns based on MEMORY A3C1C22A
        desired_sensors = ['air_temp_c', 'relative_humidity_percent', 'co2_measured_ppm', 'radiation_w_m2', 'light_intensity_umol', 'dli_sum']
        sensor_columns_for_melt = [col for col in sensor_columns_for_melt if col in desired_sensors]
        all_sensor_cols.update(sensor_columns_for_melt)

        if not sensor_columns_for_melt:
            logging.warning(f"No sensor columns identified for melting for era {era_identifier}. Skipping.")
            continue
            
        # Create a DataFrame with only 'id', 'time', and the selected sensor columns for melting.
        df_to_melt = wide_df[['id', 'time'] + sensor_columns_for_melt]

        long_df = melt_for_tsfresh(df_to_melt) # Pass the filtered df_to_melt
        if long_df is None or long_df.empty:
            logging.warning("Skipping era %s due to empty data after melting.", era_identifier)
            continue

        logging.info("Extracting features for era %s with %s (%d kinds from %d sensors)...", 
                     era_identifier, FC_PARAMS_NAME, len(long_df['kind'].unique()), len(sensor_columns_for_melt))

        try:
            tsfresh_features = extract_features(
                long_df, 
                column_id="id", 
                column_sort="time", 
                column_kind="kind", 
                column_value="value", 
                default_fc_parameters=MinimalFCParameters(), # Updated to MinimalFCParameters
                n_jobs=0 if USE_GPU_FLAG and 'cudf' in sys.modules else os.cpu_count()
            )
            all_era_features.append(tsfresh_features)
            logging.info("Successfully extracted %d features for era %s.", len(tsfresh_features.columns), era_identifier)
        except Exception as e:
            logging.error("Failed to extract features for era %s: %s", era_identifier, e)
            import traceback
            logging.error(traceback.format_exc())
            continue
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

    # Perform simple feature selection
    logging.info("Selecting relevant features …")
    final_tsfresh_features = select_relevant_features(final_tsfresh_features)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_tsfresh_features.to_parquet(OUTPUT_PATH)
    logging.info("Feature set saved to %s", OUTPUT_PATH)

    # Persist the selected features as a separate parquet file for convenience
    SELECTED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_tsfresh_features.to_parquet(SELECTED_OUTPUT_PATH)
    logging.info("Selected feature set saved to %s", SELECTED_OUTPUT_PATH)

    # Optionally store the selected features in the database
    try:
        connector = SQLAlchemyPostgresConnector(
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            db_name=DB_NAME,
        )
        df_for_db = final_tsfresh_features.copy()
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
        logging.info("Selected features written to table '%s'", FEATURES_TABLE)
    except Exception as exc:
        logging.error("Failed to write features to DB: %s", exc)

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
        rep.write(f"Eras processed        : {n_eras_processed} / {len(eras_df)}\n")
        rep.write(f"Total rows processed  : {total_rows_processed:,}\n") # From original wide_df rows
        rep.write(f"Total feature columns : {n_features:,}\n\n")

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
