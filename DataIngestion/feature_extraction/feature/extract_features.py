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
import textwrap # For pretty-printing logs

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
DEFAULT_ERA_ID_COLUMN_KEY = "era_level_C" # Default key for era identifier in JSONL, chosen for stability

# Global constant for reporting the feature set used
FC_PARAMS_NAME = "Custom (preprocess_config.json-Guided)"  # Explicitly name the config file


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
    
    # load_era_definitions now always returns a pandas DataFrame.
    # The internal logic may use cuDF for reading Parquet if USE_GPU_FLAG is true,
    # but the final concatenation and return value are pandas.
    logging.info(f"Processed {len(final_eras_df)} era definitions (as pandas.DataFrame). Columns: {final_eras_df.columns.tolist()}")
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

        # --- Configurable Sentinel Value Replacement ---
        config_file_path = Path(__file__).resolve().parent.parent / "pre_process" / "preprocess_config.json"
        sentinel_map_for_replacement_parsed = {}
        
        nan_equivalent_value = np.nan # Default to numpy's NaN
        if USE_GPU_FLAG and 'cudf' in sys.modules and isinstance(consolidated_df, cudf.DataFrame):
            nan_equivalent_value = cudf.NA # Use cudf.NA for GPU dataframes if applicable

        if config_file_path.exists():
            logging.info(f"Loading sentinel value configuration from: {config_file_path}")
            try:
                with open(config_file_path, 'r') as f:
                    config_data = json.load(f)
                
                sentinel_replacements_from_config = config_data.get("sentinel_replacements")
                if sentinel_replacements_from_config and isinstance(sentinel_replacements_from_config, dict):
                    for str_key_from_config, replacement_target_from_config in sentinel_replacements_from_config.items():
                        actual_value_to_search_for = None
                        try:
                            actual_value_to_search_for = int(str_key_from_config)
                        except ValueError:
                            try:
                                actual_value_to_search_for = float(str_key_from_config)
                            except ValueError:
                                logging.warning(f"Could not parse sentinel value '{str_key_from_config}' from config to numeric. Skipping this rule.")
                                continue
                        
                        final_replacement_value = nan_equivalent_value if replacement_target_from_config is None else replacement_target_from_config
                        sentinel_map_for_replacement_parsed[actual_value_to_search_for] = final_replacement_value
                    
                    if sentinel_map_for_replacement_parsed:
                        logging.info(f"Applying sentinel replacements based on map: {sentinel_map_for_replacement_parsed}")
                        detailed_replacement_counts = {}
                        is_cudf_df = USE_GPU_FLAG and 'cudf' in sys.modules and isinstance(consolidated_df, cudf.DataFrame)

                        for col_name in consolidated_df.columns:
                            if pd.api.types.is_numeric_dtype(consolidated_df[col_name]):
                                col_specific_counts = {}
                                for val_to_find, val_to_substitute in sentinel_map_for_replacement_parsed.items():
                                    # Count occurrences of the sentinel value
                                    # Note: For cuDF, .sum() on a boolean series returns a scalar in a Series/DataFrame, so .item() is needed.
                                    # For pandas, .sum() on a boolean series returns an int/float scalar directly.
                                    try:
                                        matches = (consolidated_df[col_name] == val_to_find)
                                        count_of_sentinel = matches.sum()
                                        if is_cudf_df and hasattr(count_of_sentinel, 'item'): # Check if it's a cuDF scalar object
                                            count_of_sentinel = count_of_sentinel.item()
                                        else: # Pandas or cuDF already gave a Python scalar
                                            count_of_sentinel = int(count_of_sentinel)
                                            
                                        if count_of_sentinel > 0:
                                            consolidated_df[col_name] = consolidated_df[col_name].replace(val_to_find, val_to_substitute)
                                            col_specific_counts[str(val_to_find)] = count_of_sentinel
                                    except Exception as e:
                                        logging.error(f"Error during sentinel count/replace for col '{col_name}', value '{val_to_find}': {e}")
                                
                                if col_specific_counts:
                                    detailed_replacement_counts[col_name] = col_specific_counts
                        
                        if detailed_replacement_counts:
                            logging.info(f"Sentinel replacements report: {json.dumps(detailed_replacement_counts)}")
                            logging.info("Sentinel value replacement applied to numeric columns based on configuration.")
                        else:
                            logging.info("No sentinel values were found/replaced based on current configuration and data.")
                    else:
                        logging.info("No valid sentinel replacement rules derived from config (e.g., map is empty after parsing or sentinels not found).")
                else:
                    logging.warning("'sentinel_replacements' key not found, not a dict, or empty in config. No replacements applied.")
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON from {config_file_path}: {e}. No sentinel replacements applied.")
            except Exception as e:
                logging.error(f"Error processing sentinel config {config_file_path}: {e}. No sentinel replacements applied.")
        else:
            logging.warning(f"Sentinel value configuration file not found: {config_file_path}. No replacements applied.")

        # --- TSFRESH: Profile Validation and Parameter Map Building ---
        ALLOWED_PROFILES_CLASSES = { # Map profile name to tsfresh parameter class
            "minimal": MinimalFCParameters,
            "efficient": EfficientFCParameters,
            # "comprehensive": ComprehensiveFCParameters, # Add if/when supported
        }
        DEFAULT_TSFRESH_PROFILE_KEY = "_default"
        FALLBACK_PROFILE_NAME = "minimal" # Fallback if _default in config is invalid or config missing

        kind_to_fc_parameters_global = {}
        
        # Identify all potentially relevant numeric columns from the dataframe
        # Exclude time, id, and era_level columns as they are not sensor data for tsfresh
        all_numeric_df_cols = {
            col for col in consolidated_df.columns 
            if pd.api.types.is_numeric_dtype(consolidated_df[col]) and 
               col not in ['time', 'id'] and not col.startswith('era_level_')
        }

        configured_sensor_profiles_from_json = {}
        effective_default_profile_name = FALLBACK_PROFILE_NAME # Start with the ultimate fallback

        if config_data and isinstance(config_data.get("tsfresh_sensor_profiles"), dict):
            configured_sensor_profiles_from_json = config_data["tsfresh_sensor_profiles"]
            
            # Validate and determine the effective_default_profile_name from config
            default_from_cfg_str = configured_sensor_profiles_from_json.get(DEFAULT_TSFRESH_PROFILE_KEY, FALLBACK_PROFILE_NAME).lower()
            if default_from_cfg_str in ALLOWED_PROFILES_CLASSES:
                effective_default_profile_name = default_from_cfg_str
            else:
                logging.warning(
                    f"TSFRESH: Unknown profile '{default_from_cfg_str}' set as '{DEFAULT_TSFRESH_PROFILE_KEY}' in config. "
                    f"Falling back to '{FALLBACK_PROFILE_NAME}' for unlisted sensors."
                )
                # effective_default_profile_name remains FALLBACK_PROFILE_NAME
            logging.info(f"TSFRESH: Effective default profile for unlisted sensors: '{effective_default_profile_name}'")
        else:
            logging.warning(
                f"TSFRESH: 'tsfresh_sensor_profiles' section missing or malformed in config, or config_data not loaded. "
                f"Defaulting all unlisted sensors to '{FALLBACK_PROFILE_NAME}'."
            )
            # effective_default_profile_name remains FALLBACK_PROFILE_NAME

        # 1. Process explicitly configured sensors
        for sensor_name_from_cfg, profile_name_from_cfg in configured_sensor_profiles_from_json.items():
            if sensor_name_from_cfg == DEFAULT_TSFRESH_PROFILE_KEY:
                continue # Already processed for effective_default_profile_name

            profile_name_lower = profile_name_from_cfg.lower()

            if sensor_name_from_cfg not in all_numeric_df_cols:
                logging.warning(
                    f"TSFRESH: Configured sensor '{sensor_name_from_cfg}' (profile: '{profile_name_lower}') "
                    "is missing from DataFrame or is non-numeric. It will be SKIPPED."
                )
                continue # Skip this sensor; it won't be added to kind_to_fc_parameters_global

            # Sensor exists and is numeric; now validate its requested profile
            if profile_name_lower in ALLOWED_PROFILES_CLASSES:
                kind_to_fc_parameters_global[sensor_name_from_cfg] = ALLOWED_PROFILES_CLASSES[profile_name_lower]()
            else:
                logging.warning(
                    f"TSFRESH: Sensor '{sensor_name_from_cfg}' requested unknown profile '{profile_name_lower}'. "
                    f"Defaulting to '{FALLBACK_PROFILE_NAME}' profile for this sensor."
                )
                kind_to_fc_parameters_global[sensor_name_from_cfg] = ALLOWED_PROFILES_CLASSES[FALLBACK_PROFILE_NAME]()
        
        # 2. Apply effective default profile to numeric sensors NOT explicitly configured already
        default_param_instance = ALLOWED_PROFILES_CLASSES[effective_default_profile_name]()
        for col_name in all_numeric_df_cols:
            if col_name not in kind_to_fc_parameters_global: # If not already processed via explicit config
                kind_to_fc_parameters_global[col_name] = default_param_instance
        
        # Log the final constructed map (this logging logic is good as is)
        if kind_to_fc_parameters_global:
            # Sort items by sensor name for consistent log output
            sorted_map_items = sorted(kind_to_fc_parameters_global.items())
            map_for_logging = {k: v.__class__.__name__ for k, v in sorted_map_items}
            logging.info("TSFRESH: Final 'kind_to_fc_parameters_global' map constructed:\n%s",
                         textwrap.indent(json.dumps(map_for_logging, indent=2), prefix="  ")) # Removed sort_keys as dict is now ordered for logging
        else:
            logging.warning("TSFRESH: 'kind_to_fc_parameters_global' is empty. Tsfresh might use its own defaults or fail if no default_fc_parameters is provided later.")


    except FileNotFoundError:
        logging.error(f"Consolidated data file not found: {consolidated_data_file_path}. Exiting.")
        return
    except Exception as e:
        logging.error(f"Error loading consolidated data from {consolidated_data_file_path}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return

    # Runtime assertions for consolidated_df
    assert "par_synth_umol" in consolidated_df.columns, "Synthetic PAR missing from consolidated_df"
    if 'par_synth_umol' in consolidated_df.columns and pd.api.types.is_numeric_dtype(consolidated_df['par_synth_umol']):
        # Ensure no NaN values that could cause issues with min() if not handled by cuDF/pandas appropriately
        # For cuDF, min() skips NA by default. For pandas, skipna=True is default.
        min_par_val = consolidated_df.par_synth_umol.min()
        assert min_par_val >= 0, f"Negative PAR values in consolidated_df! Min value: {min_par_val}"
    else:
        logging.warning("par_synth_umol column not found or not numeric in consolidated_df, skipping min value assertion.")

    # Load era definitions. This function now always returns a pandas DataFrame.
    # The USE_GPU_FLAG is passed for internal Parquet reading optimization if applicable.
    eras_df = load_era_definitions(era_definitions_dir_path, era_id_key_from_json, USE_GPU_FLAG)
    # Note: eras_df will be a pandas DataFrame for itertuples, as requested.
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

    # Iterate through the eras DataFrame (pandas) from load_era_definitions
    # Using itertuples() as requested for efficiency with pandas DataFrames.
    for era in eras_df.itertuples(index=False, name='Era'): # Use index=False for direct attribute access like era.era_id
        logging.info(f"Processing era: {era.era_id}, Start: {era.start_time}, End: {era.end_time}")

        # Filter consolidated_df for the current era's time window
        # consolidated_df can be pandas or cuDF. Slicing logic should be compatible.
        if original_pandas.isna(era.end_time): # Handle the last era (end_time is NaT/None)
            # For cuDF, .loc requires boolean Series for row indexing
            time_condition = consolidated_df['time'] >= era.start_time
        else:
            time_condition = (consolidated_df['time'] >= era.start_time) & (consolidated_df['time'] < era.end_time)
        
        era_data_slice = consolidated_df.loc[time_condition].copy()

        if era_data_slice.empty:
            logging.warning(f"Era {era.era_id} contained no rows after slicing, skipping.")
            all_era_features.append(None)
            continue

        # Add 'id' column for tsfresh, using the era.era_id
        # This 'id' column is crucial for tsfresh to group time series data.
        # Ensure the assignment works for both pandas and cuDF.
        era_data_slice['id'] = str(era.era_id)

        if era_data_slice.empty:
            logging.warning("Skipping era %s due to missing or empty data after slicing.", era.era_id)
            continue

        total_rows_processed += len(era_data_slice) 
        
        # Identify sensor columns to melt. Exclude 'time', 'id', and other metadata columns from era_def or main_df.
        # Example of potential metadata columns in wide_df if not careful: 'source_system', 'format_type', etc.
        # We need a definitive list of SENSOR columns to use or a way to exclude metadata.
        # For now, assume all numeric columns except 'time', 'id' are sensors.
        current_era_numeric_cols = {
            col for col in era_data_slice.columns
            if pd.api.types.is_numeric_dtype(era_data_slice[col]) and
               col not in ['time', 'id'] and not col.startswith('era_level_')
        }
        # Intersect with keys from the globally defined and validated map
        # Also, ensure we convert to list for pd.melt id_vars expecting list-like for value_vars
        sensor_columns_for_melt = list(current_era_numeric_cols & set(kind_to_fc_parameters_global.keys()))

        if not sensor_columns_for_melt:
            logging.warning(f"ERA: {era.era_id} - No numeric sensor columns eligible for tsfresh processing after filtering with kind_to_fc_parameters_global. Skipping feature extraction for this era.")
            # Append a DataFrame with just era_id to keep structure consistent for pd.concat later
            # Use original_pandas if cuDF is active and we need a pandas DataFrame here.
            df_module = pd if not USE_GPU else original_pandas
            all_era_features.append(df_module.DataFrame({'era_id': [era.era_id]}))
            continue # to the next era

        all_sensor_cols.update(sensor_columns_for_melt) # For reporting actual sensors used

        # (Lines 591-597 - df_to_melt, melt_for_tsfresh, check if long_df is empty - can be removed as melt is restructured)
        # The melt_for_tsfresh helper might need to be deprecated or refactored if not used elsewhere.
        # For now, we'll perform the melt directly here.

        # Define melt_id_vars before the melt operation
        melt_id_vars = ['era_id', 'time'] # 'id' from original data is mapped to 'era_id' for tsfresh

        # Critical check: Ensure melt_id_vars are in era_data_slice columns
        if not all(vid in era_data_slice.columns for vid in melt_id_vars):
            logging.error(f"ERA: {era.era_id} - Critical melt ID variables {melt_id_vars} not found in era_data_slice. Columns: {era_data_slice.columns}. Skipping melt and feature extraction for this era.")
            all_era_features.append(df_module.DataFrame({'era_id': [era.era_id]})) # df_module defined earlier
            continue # to the next era

        # Perform the melt operation directly
        logging.info("Melting data for era %s...", era.era_id)
        era_data_slice_long = df_module.melt(
            era_data_slice,
            id_vars=melt_id_vars,
            value_vars=sensor_columns_for_melt, # Use the filtered list
            var_name='kind',
            value_name='value'
        )

        if era_data_slice_long.empty:
            logging.warning(f"ERA: {era.era_id} - Melted data (era_data_slice_long) is empty. Skipping feature extraction.")
            all_era_features.append(df_module.DataFrame({'era_id': [era.era_id]}))
            continue

        # Log details before attempting feature extraction
        logging.info("Extracting features for era %s with %s (%d kinds from %d sensors)...",
                     era.era_id, FC_PARAMS_NAME, era_data_slice_long['kind'].nunique(), len(sensor_columns_for_melt))

        try:
            # This check is important before building era_specific_kind_to_fc
            if 'kind' not in era_data_slice_long.columns or era_data_slice_long['kind'].nunique() == 0:
                logging.warning(f"ERA: {era.era_id} - No 'kind' in melted data or no unique kinds after melt. Appending minimal DataFrame.")
                tsfresh_features_for_era = df_module.DataFrame({'era_id': [era.era_id]})
            else:
                # Build era_specific_kind_to_fc based on kinds present in this specific era's melted data
                # and ensure those kinds are in the globally defined map.
                # sensor_columns_for_melt ensures that kinds are already in kind_to_fc_parameters_global.
                era_specific_kind_to_fc = {
                    kind: kind_to_fc_parameters_global[kind]
                    for kind in sensor_columns_for_melt # These are the kinds that went into the melt
                    if kind in era_data_slice_long['kind'].unique() # And are actually present after melt
                }

                if not era_specific_kind_to_fc:
                    logging.warning(f"ERA: {era.era_id} - 'era_specific_kind_to_fc' is empty even though melt was successful. "
                                    "This might mean no relevant kinds for tsfresh or an issue with parameter mapping. "
                                    "Tsfresh will use its own defaults or fail if no default_fc_parameters is provided (currently None).")
                
                fc_params_str = ", ".join([f"'{k}': {type(v).__name__}" for k, v in era_specific_kind_to_fc.items()])
                logging.info(f"TSFRESH: Using era_specific_kind_to_fc for era {era.era_id}: {{ {fc_params_str} }}")

                tsfresh_features_for_era = extract_features(
                    era_data_slice_long,
                    column_id="era_id", # This must match one of the id_vars used in melt
                    column_sort="time",
                    column_kind="kind",
                    column_value="value",
                    kind_to_fc_parameters=era_specific_kind_to_fc if era_specific_kind_to_fc else None,
                    default_fc_parameters=None,
                    n_jobs=0
                )

                if not tsfresh_features_for_era.empty:
                    if tsfresh_features_for_era.index.name == 'era_id': # tsfresh usually returns column_id as index
                        tsfresh_features_for_era = tsfresh_features_for_era.reset_index()
                    elif 'era_id' not in tsfresh_features_for_era.columns: # Fallback if not index and not column
                        tsfresh_features_for_era['era_id'] = era.era_id
                elif 'era_id' not in tsfresh_features_for_era.columns: # If empty, ensure era_id column for schema consistency
                    tsfresh_features_for_era['era_id'] = era.era_id
            
            all_era_features.append(tsfresh_features_for_era)

            num_actual_features = 0
            if not tsfresh_features_for_era.empty:
                num_actual_features = len(tsfresh_features_for_era.columns)
                if 'era_id' in tsfresh_features_for_era.columns:
                    num_actual_features -= 1
            
            if num_actual_features > 0:
                logging.info(f"Successfully extracted {num_actual_features} features for era {era.era_id}.")
            else:
                logging.info(f"No features extracted by tsfresh for era {era.era_id} (result was empty or only contained era_id).")

        except Exception as e:
            logging.error(f"Error during feature extraction for era {era.era_id}: {e}", exc_info=True)
            df_module_exc = pd if not USE_GPU else original_pandas
            all_era_features.append(df_module_exc.DataFrame({'era_id': [era.era_id]}))

    valid_features = [f for f in all_era_features if f is not None and not f.empty]

    if not valid_features:
        logging.warning("No features extracted for any era. Output will be empty.")
        # Create an empty DataFrame. Use original_pandas if pd might be cuDF alias.
        final_features = original_pandas.DataFrame()
    else:
        logging.info(f"Concatenating features from {len(valid_features)} eras.")
        # If USE_GPU is true and all valid_features are cuDF DataFrames, use cudf.concat.
        # Otherwise, convert all to pandas and use original_pandas.concat.
        all_are_cudf = USE_GPU_FLAG and 'cudf' in sys.modules and all(isinstance(f, cudf.DataFrame) for f in valid_features)
        
        if all_are_cudf:
            logging.info("All feature sets are cuDF and USE_GPU is true. Using cuDF concatenation.")
            try:
                final_features = cudf.concat(valid_features, ignore_index=False)
            except Exception as e_concat_cudf:
                logging.error(f"cuDF concatenation failed: {e_concat_cudf}. Falling back to pandas concatenation.")
                pandas_features_list = [f.to_pandas() for f in valid_features] # All were cuDF, so to_pandas() is safe
                final_features = original_pandas.concat(pandas_features_list, ignore_index=False)
        else:
            logging.info("Using pandas for feature concatenation (either mixed types, all pandas, or GPU off).")
            pandas_features_list = []
            for f_item in valid_features:
                if 'cudf' in sys.modules and isinstance(f_item, cudf.DataFrame):
                    pandas_features_list.append(f_item.to_pandas())
                elif isinstance(f_item, original_pandas.DataFrame):
                    pandas_features_list.append(f_item)
                # Silently skip if not a recognized DataFrame type, though this shouldn't happen with earlier checks.
            
            if pandas_features_list:
                final_features = original_pandas.concat(pandas_features_list, ignore_index=False)
            else:
                logging.warning("No valid DataFrames to concatenate after potential conversions. Output will be empty.")
                final_features = original_pandas.DataFrame()

    logging.info(f"Final features extracted. Shape: {final_features.shape}")

    # Runtime assertion for final_features
    assert not final_features.empty, "Tsfresh returned 0 features (final_features DataFrame is empty after concatenation)"

    # Save to parquet
    logging.info(f"Writing final features to: {OUTPUT_PATH}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
    try:
        if USE_GPU_FLAG and 'cudf' in sys.modules and isinstance(final_features, cudf.DataFrame):
            final_features.to_parquet(OUTPUT_PATH)
            logging.info("Successfully wrote features to Parquet using cuDF.")
        elif isinstance(final_features, original_pandas.DataFrame): # Could be pandas from start or after cudf concat failure
            final_features.to_parquet(OUTPUT_PATH, engine='pyarrow')
            logging.info("Successfully wrote features to Parquet using pandas.")
        # This case handles if final_features is cuDF but USE_GPU_FLAG became false (should not happen ideally)
        # or if cudf.concat was used but then a fallback made it pandas, covered by above elif.
        elif 'cudf' in sys.modules and isinstance(final_features, cudf.DataFrame):
            logging.warning("Final_features is cuDF, but conditions for direct cuDF save not met. Converting to pandas.")
            final_features.to_pandas().to_parquet(OUTPUT_PATH, engine='pyarray')
            logging.info("Successfully wrote features to Parquet after cuDF to pandas conversion.")
        else:
            # This case should ideally not be reached if final_features is always a DataFrame type (cudf or pandas)
            logging.error(f"Final_features is of an unexpected type ({type(final_features)}) or state. Cannot save to Parquet.")

    except Exception as e:
        logging.error(f"Failed to write features to Parquet: {e}")
        import traceback
        logging.error(traceback.format_exc())

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
    if not final_features.empty:
        df_rows, df_cols = final_features.shape
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
    n_features = final_features.shape[1]
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
        for fname in list(final_features.columns)[:25]:
            rep.write(f"  • {fname}\n")

        rep.write("\nExtraction successful.\n")

    logging.info("Report written.")


if __name__ == "__main__":
    main()
