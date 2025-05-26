"""Feature extraction script.

This script connects to the TimescaleDB instance, reads the `preprocessed_wide` hypertable
 (written by the preprocessing pipeline) where each feature already has its own column,
converts it to the long format expected by *tsfresh*, and finally extracts a rich
set of statistical features which are persisted back into the database.  The output
table is automatically promoted to a TimescaleDB hypertable so downstream steps can
query it efficiently.

Usage (inside container):

    uv run python extract_features.py

Environment variables (all have sensible defaults for docker-compose):

    DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME  –  connection parameters
    FEATURES_TABLE                                 –  destination table for the
                                                    selected features
"""

from __future__ import annotations

import logging
import os
import random
from typing import Any

import numpy as np  # Added for np.number
import pandas as original_pandas  # Use this for true pandas operations
import sqlalchemy
import sys  # For checking loaded modules
import json
import time
import textwrap
from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters
from tsfresh import extract_features
from db_utils import SQLAlchemyPostgresConnector
from . import config
from .feature_utils import (
    select_relevant_features,
    make_hypertable_if_needed,
)
from .feature_utils_supervised import select_tsfresh_features


# Conditionally import and alias pd
if config.USE_GPU_FLAG:
    try:
        import cudf.pandas as pd  # pd becomes cudf.pandas
        import cudf  # For explicit cudf.DataFrame, etc.

        logging.info("Running in GPU mode with cudf.pandas and cudf.")
    except ImportError as e:
        logging.error(
            f"Failed to import GPU libraries (cudf): {e}. Falling back to CPU mode."
        )
        os.environ["USE_GPU"] = "false"  # Force fallback for current script execution
        pd = original_pandas  # Ensure pd is original_pandas
        # No need to re-import cudf if it failed; it won't be used.
else:
    pd = original_pandas  # pd is original_pandas
    logging.info("Running in CPU mode with pandas.")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------

# Use configuration constants from ``config``
DB_USER = config.DB_USER
DB_PASSWORD = config.DB_PASSWORD
DB_HOST = config.DB_HOST
DB_PORT = config.DB_PORT
DB_NAME = config.DB_NAME
FEATURES_TABLE = config.FEATURES_TABLE
FC_PARAMS_NAME = "Custom (preprocess_config.json-Guided)"
USE_GPU_FLAG = config.USE_GPU_FLAG

# Paths
consolidated_data_file_path = config.CONSOLIDATED_DATA_FILE_PATH
era_definitions_dir_path = config.ERA_DEFINITIONS_DIR_PATH
OUTPUT_PATH = config.OUTPUT_PATH
SELECTED_OUTPUT_PATH = config.SELECTED_OUTPUT_PATH

# Supervised feature selection configuration
USE_SUPERVISED_SELECTION = config.USE_SUPERVISED_SELECTION
FDR_LEVEL = config.FDR_LEVEL
N_JOBS = config.N_JOBS
CHUNKSIZE = config.CHUNKSIZE
TARGET_COLUMN = config.TARGET_COLUMN

# Placeholder for tsfresh configuration per sensor
kind_to_fc_parameters_global: dict[str, Any] = {}

# -----------------------------------------------------------------


# -----------------------------------------------------------------
# Promote a plain SQL table to Timescale hypertable if needed
# ----------------------------------------------------------------


def perform_feature_selection(features_df, consolidated_df, target_column_name):
    """Select features either using supervised or unsupervised methods based on configuration.

    Parameters
    ----------
    features_df : pd.DataFrame
        Feature matrix with observations as rows.
    consolidated_df : pd.DataFrame
        DataFrame containing the target column.
    target_column_name : str
        Name of the target column (typically config.TARGET_COLUMN).

    Returns
    -------
    pd.DataFrame
        Selected features.
    """
    # Global configurations are used here as per the original structure
    # USE_SUPERVISED_SELECTION, FDR_LEVEL, N_JOBS, CHUNKSIZE

    if USE_SUPERVISED_SELECTION and target_column_name in consolidated_df.columns:
        logging.info(
            f"Attempting supervised feature selection with target '{target_column_name}' "
            f"using global configs (FDR: {FDR_LEVEL}, N_JOBS: {N_JOBS}, CHUNKSIZE: {CHUNKSIZE})."
        )
        try:
            # Store original features_df for fallback in case of issues or if y_series is unsuitable
            original_features_df_for_fallback = features_df.copy()
            if consolidated_df[target_column_name].empty:
                logging.warning(
                    f"Target column '{target_column_name}' is empty. "
                    "Falling back to unsupervised feature selection."
                )
                return select_relevant_features(original_features_df_for_fallback)

            if len(y_series) < len(features_df):
                logging.warning(
                    f"Target series (len {len(y_series)}) is shorter than feature matrix (len {len(features_df)}). "
                    f"Aligning feature matrix to target series length for supervised selection."
                )
                features_df_aligned = features_df.iloc[: len(y_series)]
            elif len(y_series) > len(features_df):
                logging.info(
                    f"Target series (len {len(y_series)}) is longer than feature matrix (len {len(features_df)}). "
                    f"Aligning target series to feature matrix length."
                )
                y_series_aligned = y_series.iloc[: len(features_df)]

            # ADDED: Ensure both aligned dataframes have fresh, identical 0-based RangeIndex after any iloc slicing.
            # This applies whether slicing happened or if lengths were initially equal.
            features_df_aligned = features_df_aligned.reset_index(drop=True)
            y_series_aligned = y_series_aligned.reset_index(drop=True)

            # Final check for index alignment before proceeding to supervised selection
            if not features_df_aligned.index.equals(y_series_aligned.index):
                logging.error(
                    "CRITICAL: Index mismatch detected between aligned features and target series "
                    "right before supervised selection. This should not happen after explicit resets. "
                    f"Features index type: {type(features_df_aligned.index)}, "
                    f"Target index type: {type(y_series_aligned.index)}. "
                    f"Features index head: {features_df_aligned.index[:5]}, "
                    f"Target index head: {y_series_aligned.index[:5]}. "
                    "Falling back to unsupervised feature selection to prevent errors."
                )
                return select_relevant_features(original_features_df_for_fallback)

            if y_series_aligned.empty:
                logging.warning(
                    f"Aligned target series for '{target_column_name}' is empty after attempting to match "
                    f"feature matrix (original len {len(features_df)}, aligned len {len(features_df_aligned)}). "
                    "Falling back to unsupervised selection."
                )
                return select_relevant_features(original_features_df_for_fallback)

            if features_df_aligned.empty:
                logging.warning(
                    f"Aligned feature DataFrame is empty for target '{target_column_name}' after attempting to match "
                    f"target series (original len {len(y_series)}, aligned len {len(y_series_aligned)}). "
                    "Falling back to unsupervised selection."
                )
                return select_relevant_features(original_features_df_for_fallback)
                return select_relevant_features(features_df)

            logging.info(
                f"Shape of feature matrix for selection: {features_df_aligned.shape}"
            )
            logging.info(
                f"Length of target series for selection: {len(y_series_aligned)}"
            )

            if y_series_aligned.isnull().all():
                logging.warning(
                    f"Target series '{target_column_name}' contains all NaN values after alignment. "
                    "Supervised selection is not possible. Falling back to unsupervised selection."
                )
                return select_relevant_features(features_df)

            return select_tsfresh_features(
                features_df_aligned,
                y_series_aligned,
                fdr=FDR_LEVEL,
                n_jobs=N_JOBS,
                chunksize=CHUNKSIZE,
            )
        except ValueError as ve:
            logging.error(
                f"ValueError during supervised feature selection (e.g., shape mismatch, NaNs in target): {ve}. "
                "Falling back to unsupervised feature selection."
            )
            return select_relevant_features(features_df)
        except Exception as e:
            logging.error(
                f"An unexpected error occurred during supervised feature selection: {e}. "
                "Falling back to unsupervised feature selection."
            )
            return select_relevant_features(features_df)
    else:
        if not USE_SUPERVISED_SELECTION:
            logging.info(
                "Supervised feature selection is disabled (USE_SUPERVISED_SELECTION=False). Using unsupervised selection."
            )
        elif target_column_name not in consolidated_df.columns:
            logging.warning(
                f"Target column '{target_column_name}' not found in consolidated_df. "
                "Falling back to unsupervised feature selection."
            )
        else:
            logging.info(
                "Proceeding with unsupervised feature selection as conditions for supervised selection were not fully met."
            )
        return select_relevant_features(features_df)


def main() -> None:
    """Entry point for feature extraction."""
    start_time_main = time.time()

    logging.info("Starting feature extraction …")
    logging.info(f"Loading consolidated data from: {consolidated_data_file_path}")
    try:
        # Determine if using GPU for pandas operations
        # pd is already aliased to cudf.pandas or original_pandas at the top of the script
        # Determine if using GPU for pandas operations
        if USE_GPU_FLAG and "cudf" in sys.modules:
            # Load with original_pandas then convert to cuDF for robust JSONL parsing
            temp_pandas_df = original_pandas.read_json(
                consolidated_data_file_path, lines=True, orient="records"
            )
            if temp_pandas_df.empty:
                logging.error(
                    f"Consolidated data file is empty: {consolidated_data_file_path}. Exiting."
                )
                return
            consolidated_df = cudf.DataFrame.from_pandas(temp_pandas_df)
            del temp_pandas_df  # Free memory
            logging.info("Loaded consolidated data into cuDF DataFrame.")
        else:
            consolidated_df = original_pandas.read_json(
                consolidated_data_file_path, lines=True, orient="records"
            )
            if consolidated_df.empty:
                logging.error(
                    f"Consolidated data file is empty: {consolidated_data_file_path}. Exiting."
                )
                return
            logging.info("Loaded consolidated data into pandas DataFrame.")

        logging.info(f"Consolidated data shape: {consolidated_df.shape}")

        # Convert 'time' column to datetime objects using the appropriate pandas module (pd could be cudf.pandas or original_pandas)
        if "time" in consolidated_df.columns:
            consolidated_df["time"] = pd.to_datetime(
                consolidated_df["time"], unit="ms", errors="coerce"
            )  # Assuming time is ms epoch
            consolidated_df.dropna(
                subset=["time"], inplace=True
            )  # Drop rows where time conversion failed
            if consolidated_df.empty:
                logging.error(
                    "Consolidated data became empty after 'time' conversion/NA drop. Exiting."
                )
                return
            # Sort by time, essential for time-series operations and correct slicing later
            consolidated_df.sort_values("time", inplace=True)
            # consolidated_df.reset_index(drop=True, inplace=True) # Original line: explicitly drops and replaces the index.
            logging.info(
                "Original 'consolidated_df.reset_index(drop=True)' omitted to preserve index after time sort."
            )
        else:
            logging.error("'time' column not found in consolidated data. Exiting.")
            return

        # --- Configurable Sentinel Value Replacement ---
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_file_path = os.path.join(
            base_dir, "pre_process", "preprocess_config.json"
        )
        sentinel_map_for_replacement_parsed = {}

        nan_equivalent_value = np.nan  # Default to numpy's NaN
        if (
            USE_GPU_FLAG
            and "cudf" in sys.modules
            and isinstance(consolidated_df, cudf.DataFrame)
        ):
            nan_equivalent_value = (
                cudf.NA
            )  # Use cudf.NA for GPU dataframes if applicable

        if os.path.exists(config_file_path):
            logging.info(
                f"Loading sentinel value configuration from: {config_file_path}"
            )
            try:
                with open(config_file_path, "r") as f:
                    config_data = json.load(f)

                sentinel_replacements_from_config = config_data.get(
                    "sentinel_replacements"
                )
                if sentinel_replacements_from_config and isinstance(
                    sentinel_replacements_from_config, dict
                ):
                    for (
                        str_key_from_config,
                        replacement_target_from_config,
                    ) in sentinel_replacements_from_config.items():
                        actual_value_to_search_for = None
                        try:
                            actual_value_to_search_for = int(str_key_from_config)
                        except ValueError:
                            try:
                                actual_value_to_search_for = float(str_key_from_config)
                            except ValueError:
                                logging.warning(
                                    f"Could not parse sentinel value '{str_key_from_config}' from config to numeric. Skipping this rule."
                                )
                                continue

                        final_replacement_value = (
                            nan_equivalent_value
                            if replacement_target_from_config is None
                            else replacement_target_from_config
                        )
                        sentinel_map_for_replacement_parsed[
                            actual_value_to_search_for
                        ] = final_replacement_value

                    if sentinel_map_for_replacement_parsed:
                        logging.info(
                            f"Applying sentinel replacements based on map: {sentinel_map_for_replacement_parsed}"
                        )
                        detailed_replacement_counts = {}
                        is_cudf_df = (
                            USE_GPU_FLAG
                            and "cudf" in sys.modules
                            and isinstance(consolidated_df, cudf.DataFrame)
                        )

                        for col_name in consolidated_df.columns:
                            if pd.api.types.is_numeric_dtype(consolidated_df[col_name]):
                                col_specific_counts = {}
                                for (
                                    val_to_find,
                                    val_to_substitute,
                                ) in sentinel_map_for_replacement_parsed.items():
                                    # Count occurrences of the sentinel value
                                    # Note: For cuDF, .sum() on a boolean series returns a scalar in a Series/DataFrame, so .item() is needed.
                                    # For pandas, .sum() on a boolean series returns an int/float scalar directly.
                                    try:
                                        matches = (
                                            consolidated_df[col_name] == val_to_find
                                        )
                                        count_of_sentinel = matches.sum()
                                        if is_cudf_df and hasattr(
                                            count_of_sentinel, "item"
                                        ):  # Check if it's a cuDF scalar object
                                            count_of_sentinel = count_of_sentinel.item()
                                        else:  # Pandas or cuDF already gave a Python scalar
                                            count_of_sentinel = int(count_of_sentinel)

                                        if count_of_sentinel > 0:
                                            consolidated_df[col_name] = consolidated_df[
                                                col_name
                                            ].replace(val_to_find, val_to_substitute)
                                            col_specific_counts[str(val_to_find)] = (
                                                count_of_sentinel
                                            )
                                    except Exception as e:
                                        logging.error(
                                            f"Error during sentinel count/replace for col '{col_name}', value '{val_to_find}': {e}"
                                        )

                                if col_specific_counts:
                                    detailed_replacement_counts[col_name] = (
                                        col_specific_counts
                                    )

                        if detailed_replacement_counts:
                            logging.info(
                                f"Sentinel replacements report: {json.dumps(detailed_replacement_counts)}"
                            )
                            logging.info(
                                "Sentinel value replacement applied to numeric columns based on configuration."
                            )
                        else:
                            logging.info(
                                "No sentinel values were found/replaced based on current configuration and data."
                            )
                    else:
                        logging.info(
                            "No valid sentinel replacement rules derived from config (e.g., map is empty after parsing or sentinels not found)."
                        )
                else:
                    logging.warning(
                        "'sentinel_replacements' key not found, not a dict, or empty in config. No replacements applied."
                    )
            except json.JSONDecodeError as e:
                logging.error(
                    f"Error decoding JSON from {config_file_path}: {e}. No sentinel replacements applied."
                )
            except Exception as e:
                logging.error(
                    f"Error processing sentinel config {config_file_path}: {e}. No sentinel replacements applied."
                )
        else:
            logging.warning(
                f"Sentinel value configuration file not found: {config_file_path}. No replacements applied."
            )

        # --- TSFRESH: Profile Validation and Parameter Map Building ---
        ALLOWED_PROFILES_CLASSES = {  # Map profile name to tsfresh parameter class
            "minimal": MinimalFCParameters,
            "efficient": EfficientFCParameters,
            # "comprehensive": ComprehensiveFCParameters, # Add if/when supported
        }
        DEFAULT_TSFRESH_PROFILE_KEY = "_default"
        FALLBACK_PROFILE_NAME = (
            "minimal"  # Fallback if _default in config is invalid or config missing
        )

        kind_to_fc_parameters_global = {}

        # Identify all potentially relevant numeric columns from the dataframe
        # Exclude time, id, and era_level columns as they are not sensor data for tsfresh
        all_numeric_df_cols = {
            col
            for col in consolidated_df.columns
            if pd.api.types.is_numeric_dtype(consolidated_df[col])
            and col not in ["time", "id"]
            and not col.startswith("era_level_")
        }

        configured_sensor_profiles_from_json = {}
        effective_default_profile_name = (
            FALLBACK_PROFILE_NAME  # Start with the ultimate fallback
        )

        if config_data and isinstance(config_data.get("tsfresh_sensor_profiles"), dict):
            configured_sensor_profiles_from_json = config_data[
                "tsfresh_sensor_profiles"
            ]

            # Validate and determine the effective_default_profile_name from config
            default_from_cfg_str = configured_sensor_profiles_from_json.get(
                DEFAULT_TSFRESH_PROFILE_KEY, FALLBACK_PROFILE_NAME
            ).lower()
            if default_from_cfg_str in ALLOWED_PROFILES_CLASSES:
                effective_default_profile_name = default_from_cfg_str
            else:
                logging.warning(
                    f"TSFRESH: Unknown profile '{default_from_cfg_str}' set as '{DEFAULT_TSFRESH_PROFILE_KEY}' in config. "
                    f"Falling back to '{FALLBACK_PROFILE_NAME}' for unlisted sensors."
                )
                # effective_default_profile_name remains FALLBACK_PROFILE_NAME
            logging.info(
                f"TSFRESH: Effective default profile for unlisted sensors: '{effective_default_profile_name}'"
            )
        else:
            logging.warning(
                f"TSFRESH: 'tsfresh_sensor_profiles' section missing or malformed in config, or config_data not loaded. "
                f"Defaulting all unlisted sensors to '{FALLBACK_PROFILE_NAME}'."
            )
            # effective_default_profile_name remains FALLBACK_PROFILE_NAME

        # 1. Process explicitly configured sensors
        for (
            sensor_name_from_cfg,
            profile_name_from_cfg,
        ) in configured_sensor_profiles_from_json.items():
            if sensor_name_from_cfg == DEFAULT_TSFRESH_PROFILE_KEY:
                continue  # Already processed for effective_default_profile_name

            profile_name_lower = profile_name_from_cfg.lower()

            if sensor_name_from_cfg not in all_numeric_df_cols:
                logging.warning(
                    f"TSFRESH: Configured sensor '{sensor_name_from_cfg}' (profile: '{profile_name_lower}') "
                    "is missing from DataFrame or is non-numeric. It will be SKIPPED."
                )
                continue  # Skip this sensor; it won't be added to kind_to_fc_parameters_global

            # Sensor exists and is numeric; now validate its requested profile
            if profile_name_lower in ALLOWED_PROFILES_CLASSES:
                kind_to_fc_parameters_global[sensor_name_from_cfg] = (
                    ALLOWED_PROFILES_CLASSES[profile_name_lower]()
                )
            else:
                logging.warning(
                    f"TSFRESH: Sensor '{sensor_name_from_cfg}' requested unknown profile '{profile_name_lower}'. "
                    f"Defaulting to '{FALLBACK_PROFILE_NAME}' profile for this sensor."
                )
                kind_to_fc_parameters_global[sensor_name_from_cfg] = (
                    ALLOWED_PROFILES_CLASSES[FALLBACK_PROFILE_NAME]()
                )

        # 2. Apply effective default profile to numeric sensors NOT explicitly configured already
        default_param_instance = ALLOWED_PROFILES_CLASSES[
            effective_default_profile_name
        ]()
        for col_name in all_numeric_df_cols:
            if (
                col_name not in kind_to_fc_parameters_global
            ):  # If not already processed via explicit config
                kind_to_fc_parameters_global[col_name] = default_param_instance

        # Log the final constructed map (this logging logic is good as is)
        if kind_to_fc_parameters_global:
            # Sort items by sensor name for consistent log output
            sorted_map_items = sorted(kind_to_fc_parameters_global.items())
            map_for_logging = {k: v.__class__.__name__ for k, v in sorted_map_items}
            logging.info(
                "TSFRESH: Final 'kind_to_fc_parameters_global' map constructed:\n%s",
                textwrap.indent(json.dumps(map_for_logging, indent=2), prefix="  "),
            )  # Removed sort_keys as dict is now ordered for logging
        else:
            logging.warning(
                "TSFRESH: 'kind_to_fc_parameters_global' is empty. Tsfresh might use its own defaults or fail if no default_fc_parameters is provided later."
            )

    except FileNotFoundError:
        logging.error(
            f"Consolidated data file not found: {consolidated_data_file_path}. Exiting."
        )
        return
    except Exception as e:
        logging.error(
            f"Error loading consolidated data from {consolidated_data_file_path}: {e}"
        )
        import traceback

        logging.error(traceback.format_exc())
        return

    # --- Load Era Definitions ---
    era_definitions_df = load_era_definitions(
        era_definitions_dir_path, config.ERA_ID_COLUMN_KEY
    )
    if era_definitions_df.empty:
        logging.error(
            "No era definitions loaded. Cannot proceed with era-based feature extraction. Exiting."
        )
        return
    logging.info(f"Loaded {len(era_definitions_df)} era definitions.")

    # # --- Iterate through Eras and Extract Features ---
    # all_era_features = []
    # total_rows_processed = 0
    # all_sensor_cols = set() # To track all unique sensor columns actually used
    # df_module = pd if not USE_GPU_FLAG else original_pandas # For creating small DataFrames consistently
    #
    # # Ensure 'id' column (as era_id) is string type for tsfresh compatibility if it's created
    # # This is more relevant if we create an empty df with era_id, ensure it's object/string.
    #
    # for era_idx, era in enumerate(era_definitions_df.itertuples()):
    #     logging.info(f"Processing Era {era.era_id} ({era_idx + 1}/{len(era_definitions_df)}): Start {era.start_time}, End {era.end_time}")
    #
    #     # Slice consolidated_df for the current era
    #     # Ensure 'time' in consolidated_df is datetime64[ns] for comparison
    #     if pd.api.types.is_datetime64_any_dtype(consolidated_df['time']):
    #         time_condition = (consolidated_df['time'] >= era.start_time) & (consolidated_df['time'] < era.end_time)
    #     else:
    #         # Fallback or error if time is not in expected format - though earlier checks should prevent this.
    #         logging.error(f"ERA: {era.era_id} - 'time' column in consolidated_df is not datetime. Current type: {consolidated_df['time'].dtype}. Skipping era.")
    #         all_era_features.append(df_module.DataFrame({'era_id': [str(era.era_id)]})) # Keep structure
    #         continue
    #
    #     era_data_slice = consolidated_df.loc[time_condition].copy()
    #
    #     if era_data_slice.empty:
    #         logging.warning(f"Era {era.era_id} contained no rows after slicing, skipping.")
    #         all_era_features.append(df_module.DataFrame({'era_id': [str(era.era_id)]}))
    #         continue
    #
    #     # Add 'id' column for tsfresh, using the era.era_id. Must be string.
    #     era_data_slice['id'] = str(era.era_id)
    #
    #     total_rows_processed += len(era_data_slice)
    #
    #     current_era_numeric_cols = {
    #         col for col in era_data_slice.columns
    #         if pd.api.types.is_numeric_dtype(era_data_slice[col]) and \
    #            col not in ['time', 'id'] and not col.startswith('era_level_')
    #     }
    #     sensor_columns_for_melt = list(current_era_numeric_cols & set(kind_to_fc_parameters_global.keys()))
    #
    #     if not sensor_columns_for_melt:
    #         logging.warning(f"ERA: {era.era_id} - No numeric sensor columns eligible for tsfresh. Skipping.")
    #         all_era_features.append(df_module.DataFrame({'era_id': [str(era.era_id)]}))
    #         continue
    #
    #     all_sensor_cols.update(sensor_columns_for_melt)
    #
    #     melt_id_vars = ['id', 'time'] # 'id' is now era.era_id (string)
    #
    #     logging.info(f"Melting data for era {era.era_id} using sensors: {sensor_columns_for_melt}")
    #     era_data_slice_long = df_module.melt(
    #         era_data_slice,
    #         id_vars=melt_id_vars,
    #         value_vars=sensor_columns_for_melt,
    #         var_name='kind',
    #         value_name='value'
    #     )
    #
    #     if era_data_slice_long.empty:
    #         logging.warning(f"ERA: {era.era_id} - Melted data is empty. Skipping feature extraction.")
    #         all_era_features.append(df_module.DataFrame({'era_id': [str(era.era_id)]}))
    #         continue
    #
    #     logging.info("Extracting features for era %s (%d kinds from %d sensors)...",
    #                  era.era_id, era_data_slice_long['kind'].nunique(), len(sensor_columns_for_melt))
    #
    #     try:
    #         era_specific_kind_to_fc = {
    #             kind: kind_to_fc_parameters_global[kind]
    #             for kind in sensor_columns_for_melt
    #             if kind in era_data_slice_long['kind'].unique() and kind in kind_to_fc_parameters_global
    #         }
    #
    #         if not era_specific_kind_to_fc:
    #             logging.warning(f"ERA: {era.era_id} - 'era_specific_kind_to_fc' is empty. Tsfresh might use defaults or fail.")
    #
    #         tsfresh_features_for_era = extract_features(
    #             era_data_slice_long,
    #             column_id="id",
    #             column_sort="time",
    #             column_kind="kind",
    #             column_value="value",
    #             kind_to_fc_parameters=era_specific_kind_to_fc if era_specific_kind_to_fc else None,
    #             default_fc_parameters=None, # Explicitly no global fallback here, rely on kind_to_fc_parameters_global logic
    #             n_jobs=0 # Serial processing per era, parallelization is across eras if script is run multiple times or by orchestrator
    #         )
    #
    #         if not tsfresh_features_for_era.empty:
    #             if tsfresh_features_for_era.index.name == 'id':
    #                 tsfresh_features_for_era = tsfresh_features_for_era.reset_index()
    #             elif 'id' not in tsfresh_features_for_era.columns:
    #                 tsfresh_features_for_era['id'] = str(era.era_id)
    #         else: # tsfresh_features_for_era is empty
    #             tsfresh_features_for_era = df_module.DataFrame({'id': [str(era.era_id)]})
    #
    #         all_era_features.append(tsfresh_features_for_era)
    #         num_actual_features = len(tsfresh_features_for_era.columns) -1 if 'id' in tsfresh_features_for_era.columns else len(tsfresh_features_for_era.columns)
    #         logging.info(f"Successfully extracted {num_actual_features} features for era {era.era_id}.")
    #
    #     except Exception as e:
    #         logging.error(f"Error during feature extraction for era {era.era_id}: {e}", exc_info=True)
    #         all_era_features.append(df_module.DataFrame({'id': [str(era.era_id)]}))
    #
    # logging.info(f"Total rows processed across all eras: {total_rows_processed}")
    # logging.info(f"Unique sensor columns used for melting across all eras: {all_sensor_cols}")
    #
    # valid_features = [f for f in all_era_features if f is not None and not f.empty and len(f.columns) > 1]
    #
    # if not valid_features:
    #     logging.warning("No features extracted for any era, or features only contained 'id'. Output will be empty.")
    #     final_features = original_pandas.DataFrame()
    # else:
    #     logging.info(f"Concatenating features from {len(valid_features)} eras.")
    #     # Handle potential mix of cuDF and pandas DataFrames if GPU path had issues
    #     processed_valid_features = []
    #     is_any_cudf = False
    #     if USE_GPU_FLAG and 'cudf' in sys.modules:
    #         for f_item in valid_features:
    #             if isinstance(f_item, cudf.DataFrame):
    #                 is_any_cudf = True
    #                 processed_valid_features.append(f_item)
    #             elif isinstance(f_item, original_pandas.DataFrame):
    #                 logging.info("Converting a pandas DataFrame to cuDF for concatenation.")
    #                 try:
    #                     processed_valid_features.append(cudf.DataFrame.from_pandas(f_item))
    #                     is_any_cudf = True # Mark that we now have cuDF dfs
    #                 except Exception as e_conv_pd_cudf:
    #                     logging.error(f"Failed to convert pandas df to cudf: {e_conv_pd_cudf}. Using original pandas df.")
    #                     processed_valid_features.append(f_item) # Keep as pandas
    #             else:
    #                 processed_valid_features.append(f_item) # Should not happen
    #     else: # Not using GPU or cudf not available
    #         processed_valid_features = valid_features
    #
    #     if is_any_cudf and USE_GPU_FLAG and 'cudf' in sys.modules:
    #         logging.info("Attempting concatenation with cuDF.")
    #         try:
    #             final_features = cudf.concat(processed_valid_features, ignore_index=True) # ignore_index=True as 'id' is now a column
    #         except Exception as e_concat_cudf:
    #             logging.error(f"cuDF concatenation failed: {e_concat_cudf}. Falling back to pandas concatenation.")
    #             pandas_features_list = [f.to_pandas() if ('cudf' in sys.modules and isinstance(f, cudf.DataFrame)) else f for f in processed_valid_features]
    #             final_features = original_pandas.concat(pandas_features_list, ignore_index=True)
    #     else:
    #         logging.info("Using pandas for feature concatenation.")
    #         pandas_features_list = [f.to_pandas() if ('cudf' in sys.modules and isinstance(f, cudf.DataFrame)) else f for f in processed_valid_features]
    #         final_features = original_pandas.concat(pandas_features_list, ignore_index=True)
    #
    # if 'id' in final_features.columns:
    #     final_features.rename(columns={'id': 'era_id'}, inplace=True) # Rename 'id' (which was era_id) to 'era_id'
    #     # Set era_id as index if it's not already, for consistency, though not strictly necessary for parquet.
    #     if final_features.index.name != 'era_id' and 'era_id' in final_features.columns:
    #         # Ensure era_id is unique before setting as index if that's a requirement downstream
    #         # For now, we assume it might not be unique if multiple eras produce same features (unlikely with tsfresh)
    #         pass # final_features = final_features.set_index('era_id', drop=False) # drop=False keeps it as a column too
    #
    # logging.info(f"Final features extracted. Shape: {final_features.shape}")

    # Initialize final_features to an empty DataFrame if the main loop is commented out.
    # This is to prevent NameError for the code below and to make the script parsable,
    # especially when debugging by commenting out the era processing loop.
    if "final_features" not in locals():
        logging.info(
            "final_features was not defined (era processing loop likely commented out). Initializing to empty DataFrame."
        )
        if USE_GPU_FLAG and "cudf" in sys.modules:
            final_features = cudf.DataFrame()
        else:
            final_features = original_pandas.DataFrame()

    if final_features.empty:
        logging.error(
            "Tsfresh returned 0 features or all eras failed. final_features DataFrame is empty. Exiting."
        )
        return

    # Save to parquet
    logging.info(f"Writing final features to: {OUTPUT_PATH}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        df_to_save = final_features.copy()
        if (
            USE_GPU_FLAG
            and "cudf" in sys.modules
            and isinstance(df_to_save, cudf.DataFrame)
        ):
            df_to_save.to_parquet(OUTPUT_PATH)
        else:
            if hasattr(
                df_to_save, "to_pandas"
            ):  # If it's cuDF but USE_GPU_FLAG is false, or mixed type result
                df_to_save = df_to_save.to_pandas()
            df_to_save.to_parquet(OUTPUT_PATH, engine="pyarrow")
        logging.info(f"Successfully wrote full feature set to {OUTPUT_PATH}")
    except Exception as e:
        logging.error(f"Failed to write full features to Parquet: {e}", exc_info=True)

    # --- Perform Feature Selection (First Instance) ---
    logging.info(
        "Selecting relevant features from the full feature set (%s columns)...",
        final_features.shape[1],
    )
    # Pass final_features (which has era_id as a column) and consolidated_df (which has TARGET_COLUMN)
    # The helper function needs to align them based on row count if supervised.
    # TARGET_COLUMN is a global from config.
    selected_features = perform_feature_selection(
        final_features, consolidated_df, TARGET_COLUMN
    )
    logging.info(
        f"Selected features shape after first selection: {selected_features.shape}"
    )

    # --- Save Selected Features (First Instance) ---
    if not selected_features.empty:
        SELECTED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            df_to_save_selected = selected_features.copy()
            if (
                USE_GPU_FLAG
                and "cudf" in sys.modules
                and isinstance(df_to_save_selected, cudf.DataFrame)
            ):
                df_to_save_selected.to_parquet(SELECTED_OUTPUT_PATH)
            else:
                if hasattr(df_to_save_selected, "to_pandas"):
                    df_to_save_selected = df_to_save_selected.to_pandas()
                df_to_save_selected.to_parquet(SELECTED_OUTPUT_PATH, engine="pyarrow")
            logging.info(
                f"Successfully wrote selected feature set to {SELECTED_OUTPUT_PATH}"
            )
        except Exception as e:
            logging.error(
                f"Failed to write selected features to Parquet {SELECTED_OUTPUT_PATH}: {e}",
                exc_info=True,
            )
    else:
        logging.warning(
            f"Selected features DataFrame is empty after first selection. Skipping save to {SELECTED_OUTPUT_PATH}"
        )

    # --- Optionally Store Selected Features in DB (First Instance) ---
    try:
        if not selected_features.empty:
            # Ensure 'era_id' is a column if it was an index, before writing to DB
            db_ready_features = selected_features.copy()
            if db_ready_features.index.name == "era_id":
                db_ready_features = db_ready_features.reset_index()
            elif (
                "era_id" not in db_ready_features.columns
                and "id" in db_ready_features.columns
            ):
                db_ready_features.rename(
                    columns={"id": "era_id"}, inplace=True
                )  # if 'id' was used from tsfresh output directly

            # Ensure 'era_id' is present
            if "era_id" not in db_ready_features.columns:
                logging.error(
                    f"'era_id' column missing from selected_features before DB write. Columns: {db_ready_features.columns}"
                )
            else:
                connector = SQLAlchemyPostgresConnector(
                    user=DB_USER,
                    password=DB_PASSWORD,
                    host=DB_HOST,
                    port=DB_PORT,
                    db_name=DB_NAME,
                )
                # Convert to pandas if it's cuDF before writing to DB
                if hasattr(db_ready_features, "to_pandas"):
                    db_ready_features = db_ready_features.to_pandas()

                # Ensure 'era_id' is first column for consistency, if it exists
                if "era_id" in db_ready_features.columns:
                    cols = ["era_id"] + [
                        col for col in db_ready_features.columns if col != "era_id"
                    ]
                    db_ready_features = db_ready_features[cols]

                connector.write_dataframe(
                    db_ready_features, FEATURES_TABLE, if_exists="replace", index=False
                )
                logging.info(
                    f"Selected features (first pass) written to table '{FEATURES_TABLE}'"
                )
                with connector.engine.begin() as c:
                    # Attempt to make FEATURES_TABLE a hypertable.
                    # 'era_id' is a common key for feature tables.
                    make_hypertable_if_needed(c, FEATURES_TABLE, "era_id")
    except sqlalchemy.exc.SQLAlchemyError as e_db_initial_select:
        logging.error(
            f"Database error during initial selected features storage: {e_db_initial_select}"
        )
        logging.error(traceback.format_exc())
    except Exception as e_initial_select:
        logging.error(
            f"Unexpected error during initial selected features storage: {e_initial_select}"
        )
        logging.error(traceback.format_exc())
    finally:
        # Ensure connector is closed if opened, or other cleanup if necessary
        if "connector" in locals() and connector.engine:
            connector.engine.dispose()  # Dispose of the engine to close connections
            logging.info(
                "SQLAlchemy engine disposed after initial selected features DB operations."
            )
    if not valid_features:
        logging.warning("No features extracted for any era. Output will be empty.")
        # Create an empty DataFrame. Use original_pandas if pd might be cuDF alias.
        final_features = original_pandas.DataFrame()
    else:
        logging.info(f"Concatenating features from {len(valid_features)} eras.")
        # If USE_GPU is true and all valid_features are cuDF DataFrames, use cudf.concat.
        # Otherwise, convert all to pandas and use original_pandas.concat.
        all_are_cudf = (
            USE_GPU_FLAG
            and "cudf" in sys.modules
            and all(isinstance(f, cudf.DataFrame) for f in valid_features)
        )

        if all_are_cudf:
            logging.info(
                "All feature sets are cuDF and USE_GPU is true. Using cuDF concatenation."
            )
            try:
                final_features = cudf.concat(valid_features, ignore_index=False)
            except Exception as e_concat_cudf:
                logging.error(
                    f"cuDF concatenation failed: {e_concat_cudf}. Falling back to pandas concatenation."
                )
                pandas_features_list = [
                    f.to_pandas() for f in valid_features
                ]  # All were cuDF, so to_pandas() is safe
                final_features = original_pandas.concat(
                    pandas_features_list, ignore_index=False
                )
        else:
            logging.info(
                "Using pandas for feature concatenation (either mixed types, all pandas, or GPU off)."
            )
            pandas_features_list = []
            for f_item in valid_features:
                if "cudf" in sys.modules and isinstance(f_item, cudf.DataFrame):
                    pandas_features_list.append(f_item.to_pandas())
                elif isinstance(f_item, original_pandas.DataFrame):
                    pandas_features_list.append(f_item)
                # Silently skip if not a recognized DataFrame type, though this shouldn't happen with earlier checks.

            if pandas_features_list:
                final_features = original_pandas.concat(
                    pandas_features_list, ignore_index=False
                )
            else:
                logging.warning(
                    "No valid DataFrames to concatenate after potential conversions. Output will be empty."
                )
                final_features = original_pandas.DataFrame()

    logging.info(f"Final features extracted. Shape: {final_features.shape}")

    # Runtime assertion for final_features
    assert (
        not final_features.empty
    ), "Tsfresh returned 0 features (final_features DataFrame is empty after concatenation)"

    # Save to parquet
    logging.info(f"Writing final features to: {OUTPUT_PATH}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    try:
        if (
            USE_GPU_FLAG
            and "cudf" in sys.modules
            and isinstance(final_features, cudf.DataFrame)
        ):
            final_features.to_parquet(OUTPUT_PATH)
            logging.info("Successfully wrote features to Parquet using cuDF.")
        elif isinstance(
            final_features, original_pandas.DataFrame
        ):  # Could be pandas from start or after cudf concat failure
            final_features.to_parquet(OUTPUT_PATH, engine="pyarrow")
            logging.info("Successfully wrote features to Parquet using pandas.")
        # This case handles if final_features is cuDF but USE_GPU_FLAG became false (should not happen ideally)
        # or if cudf.concat was used but then a fallback made it pandas, covered by above elif.
        elif "cudf" in sys.modules and isinstance(final_features, cudf.DataFrame):
            logging.warning(
                "Final_features is cuDF, but conditions for direct cuDF save not met. Converting to pandas."
            )
            final_features.to_pandas().to_parquet(OUTPUT_PATH, engine="pyarrow")
            logging.info(
                "Successfully wrote features to Parquet after cuDF to pandas conversion."
            )
        else:
            # This case should ideally not be reached if final_features is always a DataFrame type (cudf or pandas)
            logging.error(
                f"Final_features is of an unexpected type ({type(final_features)}) or state. Cannot save to Parquet."
            )

        SELECTED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Determine how to save based on whether it's cuDF or pandas
        if (
            USE_GPU_FLAG
            and "cudf" in sys.modules
            and isinstance(selected_features, cudf.DataFrame)
        ):
            selected_features.to_parquet(SELECTED_OUTPUT_PATH)
            logging.info(
                "Selected feature set (cuDF) saved to %s", SELECTED_OUTPUT_PATH
            )
        elif isinstance(selected_features, original_pandas.DataFrame):
            selected_features.to_parquet(SELECTED_OUTPUT_PATH, engine="pyarrow")
            logging.info(
                "Selected feature set (pandas) saved to %s", SELECTED_OUTPUT_PATH
            )
        elif "cudf" in sys.modules and isinstance(
            selected_features, cudf.DataFrame
        ):  # Should be caught by first if, but as a fallback
            logging.warning(
                "Selected_features is cuDF, but conditions for direct cuDF save not met. Converting to pandas."
            )
            selected_features.to_pandas().to_parquet(
                SELECTED_OUTPUT_PATH, engine="pyarrow"
            )
            logging.info(
                "Selected feature set (cuDF to pandas) saved to %s",
                SELECTED_OUTPUT_PATH,
            )
        else:
            logging.error(
                f"Selected_features is of an unexpected type ({type(selected_features)}) or state. Cannot save to {SELECTED_OUTPUT_PATH}."
            )
    except FileNotFoundError as e_parquet_fnf:
        logging.error(
            f"Parquet save error: File not found - {e_parquet_fnf}. Check paths.",
            exc_info=True,
        )
    except PermissionError as e_parquet_perm:
        logging.error(
            f"Parquet save error: Permission denied - {e_parquet_perm}. Check permissions.",
            exc_info=True,
        )
    except Exception as e_parquet_generic:
        logging.error(
            f"Generic error during Parquet saving: {e_parquet_generic}", exc_info=True
        )
    finally:
        logging.info("Parquet saving attempt concluded.")
    if selected_features.empty:
        logging.warning(
            f"Selected features DataFrame is empty. Skipping save to {SELECTED_OUTPUT_PATH}"
        )
    else:
        logging.info(f"Selected features saved to: {SELECTED_OUTPUT_PATH}")

    # Optionally store the selected features in the database
    try:
        if not selected_features.empty:
            connector = SQLAlchemyPostgresConnector(
                user=DB_USER,
                password=DB_PASSWORD,
                host=DB_HOST,
                port=DB_PORT,
                db_name=DB_NAME,
            )
            df_for_db = selected_features.copy()
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
    except sqlalchemy.exc.SQLAlchemyError as e_db_final_select:
        logging.error(
            f"Database error during final selected features storage: {e_db_final_select}",
            exc_info=True,
        )
    except Exception as e_final_select:
        logging.error(
            f"Unexpected error during final selected features storage: {e_final_select}",
            exc_info=True,
        )
    finally:
        # Ensure connector is closed if opened
        if (
            "connector" in locals()
            and hasattr(connector, "engine")
            and connector.engine is not None
        ):
            try:
                connector.engine.dispose()
                logging.info(
                    "SQLAlchemy engine disposed after final selected features DB operations."
                )
            except Exception as e_dispose:
                logging.error(f"Error disposing SQLAlchemy engine: {e_dispose}")

    # --- Gather info for the report ---
    parquet_file_path_str = str(OUTPUT_PATH.resolve())  # Get absolute path
    parquet_file_size_bytes = 0
    if OUTPUT_PATH.exists():  # Check if file exists before getting size
        parquet_file_size_bytes = OUTPUT_PATH.stat().st_size
    parquet_file_size_mb = parquet_file_size_bytes / (1024 * 1024)

    df_rows, df_cols = 0, 0  # Initialize
    if not final_features.empty:
        df_rows, df_cols = final_features.shape


    # -----------------------------------------------------------------
    # 2.  Melt to long format on GPU if possible
    # -----------------------------------------------------------------
    numeric_cols = [
        c
        for c in wide_df.columns
        if c not in ("era_id", "time") and wide_df[c].dtype.kind in ("i", "f")
    ]

    melt_id_vars = ["era_id", "time"]

    long_df = wide_df.melt(
        id_vars=melt_id_vars,
        value_vars=numeric_cols,
        var_name="kind",
        value_name="value",
    )

    logging.info("Long dataframe shape (before tsfresh): %s", long_df.shape)

    if os.getenv("USE_GPU", "false").lower() == "true":
        long_df = long_df.to_pandas()

    # -----------------------------------------------------------------
    # 3.  tsfresh extraction (single shot)
    # -----------------------------------------------------------------
    features = extract_features(
        long_df,
        column_id="era_id",
        column_sort="time",
        column_kind="kind",
        column_value="value",
        kind_to_fc_parameters=kind_to_fc_parameters_global,
        default_fc_parameters=None,
        n_jobs=os.cpu_count() - 2,
    )

    if features.index.name == "era_id":
        features.reset_index(inplace=True)

    logging.info("Raw feature matrix shape: %s", features.shape)

    selected = perform_feature_selection(features, consolidated_df, TARGET_COLUMN)

    df_for_db = selected.copy()
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

    with connector.engine.begin() as c:
        make_hypertable_if_needed(c, FEATURES_TABLE, "era_id")


if __name__ == "__main__":
    main()
