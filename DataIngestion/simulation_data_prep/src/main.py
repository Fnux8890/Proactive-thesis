import logging
import os # Import os to read environment variables
import json # <-- Import json library
from datetime import datetime, timezone # Import timezone
from pathlib import Path # <-- Import Path
import pandas as pd # <-- Import pandas
from db_connector import get_db_connection, close_db_connection
from retriever import retrieve_data
# Import feature calculators
from feature_calculator import (
    calculate_vpd, calculate_dli, calculate_gdd, calculate_dif,
    calculate_co2_difference, # <-- Import CO2 calculator
    calculate_daily_actuator_summaries, # <-- Import Actuator Summary calculator
    calculate_delta,
    calculate_rate_of_change,
    calculate_rolling_average,
    # V-- Add new advanced feature functions --V
    calculate_rolling_std_dev,
    calculate_lag_feature,
    calculate_distance_from_range_midpoint,
    calculate_in_range_flag,
    calculate_night_stress_flag
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_data_window(
    start_time: datetime | None, # Allow None
    end_time: datetime | None,   # Allow None
    columns_to_get: list[str] | None,
    gdd_config: dict, # Pass GDD config
    dif_config: dict, # Pass DIF config
    actuator_config: dict, # Pass Actuator Summary config
    feature_params: dict, # Pass config for basic features (delta, roc, rolling)
    adv_feature_params: dict, # Pass config for advanced features
    objective_params: dict, # Pass optimal ranges etc. for domain features
    stress_params: dict # Pass stress thresholds for domain features
):
    """Process data for a given time window or the entire dataset.

    Includes basic derived features and advanced statistical/domain features.

    Args:
        start_time: Start time for data retrieval (UTC, inclusive).
        end_time: End time for data retrieval (UTC, exclusive).
        columns_to_get: Specific columns to retrieve from the database.
        gdd_config: Dictionary containing GDD calculation parameters.
        dif_config: Dictionary containing DIF calculation parameters.
        actuator_config: Dictionary containing Actuator Summary parameters.
        feature_params: Dictionary containing parameters for delta, rate of change,
                        and rolling average features.
        adv_feature_params: Dictionary containing parameters for advanced features
                            (rolling std dev, lag, domain flags, etc.).
        objective_params: Dictionary containing optimal ranges/setpoints from
                          objective_function_parameters config.
        stress_params: Dictionary containing stress thresholds from config.

    Returns:
        None. Processes data and saves output or logs errors.
    """
    conn = None
    data_df = None # Initialize data_df
    try:
        conn = get_db_connection()
        if not conn:
            logger.warning("Could not establish database connection. Exiting process.")
            return

        if start_time and end_time:
            logger.info(f"Processing data from {start_time} to {end_time}")
        else:
            logger.info("Processing all available data (no time window specified).")

        # --- Auto-add required columns based on config --- #
        if columns_to_get is not None: # Only modify if a specific list was given
            logger.debug(f"Initial requested columns: {columns_to_get}")
            required_feature_cols = set()
            required_feature_cols.add('time') # Always need time for index

            # Base cols for EXISTING basic features
            required_feature_cols.update(["air_temp_c", "relative_humidity_percent"]) # VPD
            required_feature_cols.add('light_intensity_umol') # DLI
            required_feature_cols.add('air_temp_c') # GDD/DIF
            required_feature_cols.update(['co2_measured_ppm', 'co2_required_ppm']) # CO2 Diff
            if dif_config.get('day_definition') == 'lamp_status':
                required_feature_cols.update(dif_config.get('lamp_status_columns', []))
            required_feature_cols.update(actuator_config.get('percent_columns_for_average', []))
            required_feature_cols.update(actuator_config.get('percent_columns_for_changes', []))
            required_feature_cols.update(actuator_config.get('binary_columns_for_on_time', []))
            for _, cols_pair in feature_params.get('delta_cols', {}).items():
                if isinstance(cols_pair, list) and len(cols_pair) == 2:
                    required_feature_cols.update(cols_pair)
            required_feature_cols.update(feature_params.get('rate_of_change_cols', []))
            required_feature_cols.update(feature_params.get('rolling_average_cols', {}).keys())

            # Base cols for NEW advanced features
            required_feature_cols.update(adv_feature_params.get('rolling_std_dev_cols', {}).keys())
            required_feature_cols.update(adv_feature_params.get('lag_features', {}).keys())
            # Domain features need specific columns from their config sections
            for _, col_name in adv_feature_params.get('distance_from_optimal_midpoint', {}).items():
                 if isinstance(col_name, str): required_feature_cols.add(col_name)
            for _, col_name in adv_feature_params.get('in_optimal_range_flag', {}).items():
                 if isinstance(col_name, str): required_feature_cols.add(col_name)
            for _, stress_cfg in adv_feature_params.get('night_stress_flags', {}).items():
                if isinstance(stress_cfg, dict) and isinstance(stress_cfg.get('input_temp_col'), str):
                    required_feature_cols.add(stress_cfg['input_temp_col'])
                # Also need lamp columns if using lamp status for night definition
                if dif_config.get('day_definition') == 'lamp_status':
                    required_feature_cols.update(dif_config.get('lamp_status_columns', []))

            initial_user_cols = set(columns_to_get)
            # Filter out potential None/empty strings that might sneak into config lists
            required_feature_cols = {col for col in required_feature_cols if isinstance(col, str) and col}
            final_cols_to_get = list(initial_user_cols.union(required_feature_cols))

            added_cols = list(required_feature_cols - initial_user_cols)
            if added_cols:
                logger.info(f"Added required columns for feature calculation to requested list: {sorted(added_cols)}")
            columns_to_get = sorted(final_cols_to_get) # Sort for consistency
            logger.info(f"Final columns to retrieve: {columns_to_get}")
        # -------------------------------------------------------------- #

        # Retrieve data
        logger.info(f"Attempting to retrieve columns: {columns_to_get if columns_to_get else 'ALL'}")
        data_df = retrieve_data(conn, start_time, end_time, columns=columns_to_get)

        if data_df is None:
            logger.error("Failed to retrieve data.")
            return
        if data_df.empty:
            logger.info("No records found for the specified criteria.")
            return

        logger.info(f"Retrieved data shape: {data_df.shape}")
        logger.info(f"Retrieved columns: {data_df.columns.tolist()}")
        # logger.info(f"Original data head:\n{data_df.head().to_string()}") # Reduce logging verbosity

        # Ensure DataFrame has a DatetimeIndex
        if 'time' not in data_df.columns:
            logger.error("'time' column is missing from retrieved data. Cannot proceed.")
            return
        try:
            data_df['time_dt'] = pd.to_datetime(data_df['time'], errors='coerce')
            original_len = len(data_df)
            data_df = data_df.dropna(subset=['time_dt'])
            if len(data_df) < original_len:
                logger.warning(f"Dropped {original_len - len(data_df)} rows due to unparseable 'time' values.")
            data_df = data_df.set_index('time_dt', drop=False)
            if not isinstance(data_df.index, pd.DatetimeIndex):
                 raise TypeError("Index is not DatetimeIndex after conversion.")
            data_df = data_df.sort_index()
            logger.info("DataFrame index set to 'time_dt' (datetime object).")
            time_idx = data_df.index
        except Exception as idx_e:
             logger.exception(f"Failed to set DatetimeIndex: {idx_e}")
             return

        # --- Calculate Derived Features --- #
        feature_results = {
            'vpd': False, 'dli': False, 'gdd': False, 'dif': False,
            'co2_diff': False, 'actuator_summaries': False,
            'delta': False, 'rate_of_change': False, 'rolling_average': False,
            'rolling_std_dev': False, 'lag': False, 'domain_flags': False # Add new advanced features
        }
        processed_basic_feature = False
        processed_adv_feature = False

        # --- Basic Time-Series Features (VPD, Delta, RoC, RollingAvg) --- #
        # VPD
        required_vpd_cols = ["air_temp_c", "relative_humidity_percent"]
        if all(col in data_df.columns for col in required_vpd_cols):
            try:
                data_df['vpd_kpa'] = calculate_vpd(data_df["air_temp_c"], data_df["relative_humidity_percent"])
                feature_results['vpd'] = True; processed_basic_feature = True
            except Exception as e: logger.exception(f"Error during VPD calculation: {e}")
        else: logger.warning(f"VPD calc skipped: Missing {required_vpd_cols}")
        # Delta
        delta_config = feature_params.get('delta_cols', {})
        if delta_config:
            for out_col, inp_cols in delta_config.items():
                if isinstance(inp_cols, list) and len(inp_cols) == 2 and all(c in data_df.columns for c in inp_cols):
                    try:
                        data_df[out_col] = calculate_delta(data_df[inp_cols[0]], data_df[inp_cols[1]])
                        feature_results['delta'] = True; processed_basic_feature = True
                    except Exception as e: logger.exception(f"Error calculating delta '{out_col}': {e}")
                else: logger.warning(f"Delta '{out_col}' skipped. Missing/invalid cols: {inp_cols}")
        # Rate of Change
        roc_cols = feature_params.get('rate_of_change_cols', [])
        if roc_cols:
            for col in roc_cols:
                if col in data_df.columns:
                    try:
                        out_col = f"{col}_RoC_per_s"
                        data_df[out_col] = calculate_rate_of_change(data_df[col], time_idx)
                        feature_results['rate_of_change'] = True; processed_basic_feature = True
                    except Exception as e: logger.exception(f"Error calculating RoC for '{col}': {e}")
                else: logger.warning(f"RoC for '{col}' skipped. Column not found.")
        # Rolling Average
        rolling_config = feature_params.get('rolling_average_cols', {})
        if rolling_config:
            for col, window_min in rolling_config.items():
                if col in data_df.columns and isinstance(window_min, int) and window_min > 0:
                    try:
                        out_col = f"{col}_RollingAvg_{window_min}min"
                        data_df[out_col] = calculate_rolling_average(data_df[col], window_min, time_idx)
                        feature_results['rolling_average'] = True; processed_basic_feature = True
                    except Exception as e: logger.exception(f"Error rolling avg '{col}' w={window_min}: {e}")
                else: logger.warning(f"Rolling avg '{col}' skipped. Col missing or invalid window: {window_min}")

        if processed_basic_feature: logger.info("Completed basic feature calculations (VPD, Delta, RoC, RollAvg).")

        # --- Advanced Statistical Features (Rolling StdDev, Lag) --- #
        # Rolling Std Dev
        rolling_std_config = adv_feature_params.get('rolling_std_dev_cols', {})
        if rolling_std_config:
            for col, window_min in rolling_std_config.items():
                 if col in data_df.columns and isinstance(window_min, int) and window_min > 0:
                    try:
                        out_col = f"{col}_RollingStd_{window_min}min"
                        data_df[out_col] = calculate_rolling_std_dev(data_df[col], window_min, time_idx)
                        feature_results['rolling_std_dev'] = True; processed_adv_feature = True
                    except Exception as e: logger.exception(f"Error rolling std '{col}' w={window_min}: {e}")
                 else: logger.warning(f"Rolling std '{col}' skipped. Col missing or invalid window: {window_min}")
        # Lag Features
        lag_config = adv_feature_params.get('lag_features', {})
        if lag_config:
             for col, lag_min in lag_config.items():
                 if col in data_df.columns and isinstance(lag_min, int) and lag_min > 0:
                    try:
                        out_col = f"{col}_Lag_{lag_min}min"
                        data_df[out_col] = calculate_lag_feature(data_df[col], lag_min, time_idx)
                        feature_results['lag'] = True; processed_adv_feature = True
                    except Exception as e: logger.exception(f"Error lag feature '{col}' lag={lag_min}: {e}")
                 else: logger.warning(f"Lag feature '{col}' skipped. Col missing or invalid lag: {lag_min}")

        if processed_adv_feature: logger.info("Completed advanced statistical features (RollStd, Lag).")
        processed_adv_feature = False # Reset for domain features

        # --- Advanced Domain Features (Dist from Midpoint, In Range Flag, Stress Flag) --- #
        optimal_ranges = objective_params.get('optimal_ranges', {})
        # Distance from Midpoint
        dist_config = adv_feature_params.get('distance_from_optimal_midpoint', {})
        if dist_config and optimal_ranges:
            for range_key, col_name in dist_config.items():
                if col_name in data_df.columns and range_key in optimal_ranges and isinstance(optimal_ranges[range_key], dict):
                    bounds = optimal_ranges[range_key]
                    lower = bounds.get('lower'); upper = bounds.get('upper')
                    if lower is not None and upper is not None:
                        try:
                            out_col = f"{col_name}_DistFromOptMid"
                            data_df[out_col] = calculate_distance_from_range_midpoint(data_df[col_name], lower, upper)
                            feature_results['domain_flags'] = True; processed_adv_feature = True
                        except Exception as e: logger.exception(f"Error dist midpoint '{col_name}': {e}")
                    else: logger.warning(f"Dist midpoint '{col_name}' skipped. Invalid bounds in config: {bounds}")
                else: logger.warning(f"Dist midpoint for '{range_key}' skipped. Col '{col_name}' missing or range undefined.")
        # In Range Flag
        in_range_config = adv_feature_params.get('in_optimal_range_flag', {})
        if in_range_config and optimal_ranges:
            for range_key, col_name in in_range_config.items():
                if col_name in data_df.columns and range_key in optimal_ranges and isinstance(optimal_ranges[range_key], dict):
                    bounds = optimal_ranges[range_key]
                    lower = bounds.get('lower'); upper = bounds.get('upper')
                    if lower is not None and upper is not None:
                        try:
                            out_col = f"{col_name}_InOptRangeFlag"
                            data_df[out_col] = calculate_in_range_flag(data_df[col_name], lower, upper)
                            feature_results['domain_flags'] = True; processed_adv_feature = True
                        except Exception as e: logger.exception(f"Error in-range flag '{col_name}': {e}")
                    else: logger.warning(f"In-range flag '{col_name}' skipped. Invalid bounds in config: {bounds}")
                else: logger.warning(f"In-range flag for '{range_key}' skipped. Col '{col_name}' missing or range undefined.")
        # Night Stress Flag
        stress_flag_config = adv_feature_params.get('night_stress_flags', {})
        if stress_flag_config and stress_params:
             for flag_name, flag_cfg in stress_flag_config.items():
                 if isinstance(flag_cfg, dict):
                     temp_col = flag_cfg.get('input_temp_col')
                     thresh_key = flag_cfg.get('threshold_config_key')
                     thresh_subkey = flag_cfg.get('threshold_sub_key')
                     if temp_col in data_df.columns and thresh_key in stress_params and thresh_subkey in stress_params[thresh_key]:
                         threshold = stress_params[thresh_key][thresh_subkey]
                         if isinstance(threshold, (int, float)):
                             try:
                                 out_col = f"NightStress_{flag_name}_Flag"
                                 data_df[out_col] = calculate_night_stress_flag(data_df[temp_col], threshold, dif_config, data_df)
                                 feature_results['domain_flags'] = True; processed_adv_feature = True
                             except Exception as e: logger.exception(f"Error night stress flag '{flag_name}': {e}")
                         else: logger.warning(f"Night stress flag '{flag_name}' skipped. Invalid threshold value: {threshold}")
                     else: logger.warning(f"Night stress flag '{flag_name}' skipped. Temp col or threshold config path invalid.")
                 else: logger.warning(f"Skipping invalid night stress flag config: {flag_name}")

        if processed_adv_feature: logger.info("Completed advanced domain features (DistMid, InRange, StressFlag).")

        # --- Daily Summary Features (DLI, GDD, DIF, Actuator Sum) --- #
        # These are calculated based on daily summaries, map back to original index.
        # DLI
        ppfd_col = 'light_intensity_umol'
        if ppfd_col in data_df.columns:
            try:
                dli_daily = calculate_dli(data_df[ppfd_col], time_idx, ppfd_col)
                data_df['DLI_mol_m2_d'] = data_df.index.normalize().map(dli_daily)
                feature_results['dli'] = True
            except Exception as e: logger.exception(f"Error during DLI calculation: {e}")
        else: logger.warning(f"DLI calc skipped: '{ppfd_col}' missing.")
        # GDD & DIF
        temp_col_gdd_dif = 'air_temp_c'
        if temp_col_gdd_dif in data_df.columns:
            # GDD
            t_base = gdd_config.get('t_base_celsius')
            t_cap = gdd_config.get('t_cap_celsius')
            if t_base is not None:
                 try:
                     gdd_daily_df = calculate_gdd(data_df[temp_col_gdd_dif], t_base, t_cap, temp_col_gdd_dif)
                     data_df['GDD_daily'] = data_df.index.normalize().map(gdd_daily_df['GDD'])
                     data_df['GDD_cumulative'] = data_df.index.normalize().map(gdd_daily_df['Cumulative_GDD'])
                     feature_results['gdd'] = True
                 except Exception as e: logger.exception(f"Error during GDD calculation: {e}")
            else: logger.error(f"GDD calc skipped: 't_base_celsius' missing.")
            # DIF
            try:
                dif_daily = calculate_dif(data_df[temp_col_gdd_dif], dif_config, data_df)
                if not dif_daily.empty:
                    data_df['DIF_daily'] = data_df.index.normalize().map(dif_daily)
                    feature_results['dif'] = True
            except Exception as e: logger.exception(f"Error during DIF calculation: {e}")
        else: logger.warning(f"GDD/DIF calc skipped: '{temp_col_gdd_dif}' missing.")
        # CO2 Diff
        measured_co2_col = 'co2_measured_ppm'
        required_co2_col = 'co2_required_ppm'
        if measured_co2_col in data_df.columns and required_co2_col in data_df.columns:
            try:
                data_df['CO2_diff_ppm'] = calculate_co2_difference(data_df[measured_co2_col], data_df[required_co2_col])
                feature_results['co2_diff'] = True
            except Exception as e: logger.exception(f"Error during CO2 diff calc: {e}")
        else: logger.warning(f"CO2 diff calc skipped: Missing {[c for c in [measured_co2_col, required_co2_col] if c not in data_df.columns]}")
        # Actuator Summaries
        try:
            actuator_summaries_df = calculate_daily_actuator_summaries(data_df, actuator_config)
            if not actuator_summaries_df.empty:
                # Merge results back, preserving original index type
                data_df = pd.merge(data_df.reset_index(), # Use reset_index to merge on normalized date
                                   actuator_summaries_df, how='left',
                                   left_on=data_df.index.normalize(),
                                   right_index=True).set_index('time_dt') # Set index back to original DatetimeIndex
                feature_results['actuator_summaries'] = True
                logger.info(f"Added actuator summary columns: {actuator_summaries_df.columns.tolist()}")
        except Exception as e: logger.exception(f"Error during Actuator Summary calc: {e}")

        # --- End Feature Calculation --- #

        # Log summary of calculations performed
        logger.info(f"Feature calculation summary: {feature_results}")

        # Log head/tail again after all calculations
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        logger.info(f"Data head after all feature calculations:\n{data_df.head().to_string()}")
        logger.info(f"Data tail after all feature calculations:\n{data_df.tail().to_string()}")

        # --- Save Processed Data --- #
        output_dir = Path("./output")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"processed_data_{timestamp_str}.csv"
        try:
            # Ensure original 'time' string column exists for output
            if 'time' not in data_df.columns:
                 data_df['time'] = data_df.index.strftime('%Y-%m-%d %H:%M:%S.%f') # Recreate from DatetimeIndex
            data_df.to_csv(output_dir / output_filename, index=False)
            logger.info(f"Processed data saved to {output_dir / output_filename}")
        except Exception as save_e:
             logger.exception(f"Failed to save processed data to CSV: {save_e}")

    except Exception as data_e:
        logger.exception(f"An error occurred during data retrieval or processing: {data_e}")
    finally:
        if conn:
            close_db_connection(conn)

def main():
    """Main entry point for the data preparation script.

    Loads configuration, parses environment variables for time window and columns,
    and calls process_data_window to perform data retrieval and feature calculation.
    """
    logger.info("Data preparation script started.")

    # --- Load Configuration --- #
    config_path = Path(__file__).parent / "plant_config.json"
    config = {}
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
    except FileNotFoundError:
         logger.error(f"Configuration file not found at {config_path}. Some features may not be calculated.")
         # Allow continuing with empty config sections
    except json.JSONDecodeError as json_e:
         logger.error(f"Error decoding JSON from {config_path}: {json_e}. Some features may not be calculated.")
         # Allow continuing with empty config sections
    except Exception as config_e:
        logger.exception(f"Failed to load configuration from {config_path}: {config_e}. Assuming defaults.")
        # Allow continuing with empty config sections

    # --- Extract Config Sections --- #
    gdd_params = config.get('gdd_parameters', {})
    gdd_profile_name = gdd_params.get('crop_profile', 'default')
    gdd_config = gdd_params.get('profiles', {}).get(gdd_profile_name, {})
    dif_config = config.get('dif_parameters', {})
    actuator_config = config.get('actuator_summary_parameters', {})
    feature_params = config.get('feature_parameters', {}) # Basic features
    adv_feature_params = config.get('advanced_feature_parameters', {}) # New advanced features
    objective_params = config.get('objective_function_parameters', {}) # For domain features
    stress_params = config.get('stress_thresholds', {}) # For domain features

    logger.info(f"Using GDD profile: '{gdd_profile_name}' with {gdd_config}")
    logger.info(f"Using DIF parameters: {dif_config}")
    logger.info(f"Using Actuator Summary parameters: {actuator_config}")
    logger.info(f"Using Basic Feature Parameters: {feature_params}")
    logger.info(f"Using Advanced Feature Parameters: {adv_feature_params}")
    logger.info(f"Using Objective Function Parameters: {objective_params}")
    logger.info(f"Using Stress Thresholds: {stress_params}")
    # ----------------------------- #

    # Read time window and columns from environment variables (unchanged)
    start_time_str = os.getenv("START_TIME")
    end_time_str = os.getenv("END_TIME")
    required_cols_str = os.getenv("REQUIRED_RAW_COLUMNS")
    columns_to_get = None
    if required_cols_str:
        columns_to_get = [col.strip() for col in required_cols_str.split(',') if col.strip()]
        logger.info(f"User requested columns from REQUIRED_RAW_COLUMNS: {columns_to_get}")
    else:
        logger.info("REQUIRED_RAW_COLUMNS not set. Retrieving all columns (recommended).")

    # Parse time window (unchanged)
    start_time = None
    end_time = None
    parse_error = False
    if start_time_str and end_time_str:
        try:
            start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
            end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
        except ValueError:
            try:
                 dt_format = '%Y-%m-%d %H:%M:%S'
                 start_time = datetime.strptime(start_time_str, dt_format)
                 end_time = datetime.strptime(end_time_str, dt_format)
                 logger.warning("Parsed time using format YYYY-MM-DD HH:MM:SS. Assuming UTC.")
            except ValueError:
                logger.error(f"Invalid START_TIME/END_TIME format. Use ISO 8601 or YYYY-MM-DD HH:MM:SS.")
                parse_error = True
        if not parse_error:
            if start_time.tzinfo is None: start_time = start_time.replace(tzinfo=timezone.utc)
            if end_time.tzinfo is None: end_time = end_time.replace(tzinfo=timezone.utc)
            if start_time >= end_time:
                logger.error(f"Start time ({start_time}) must be before end time ({end_time}).")
                parse_error = True
    elif start_time_str or end_time_str:
         logger.warning("START_TIME and END_TIME must both be set to filter by time. Retrieving all data.")
    else:
        logger.info("START_TIME and/or END_TIME not set. Retrieving all data.")

    # Proceed only if time parsing was successful (or not attempted)
    if not parse_error:
        try:
            # Pass all loaded config sections to the processing function
            process_data_window(
                start_time, end_time, columns_to_get,
                gdd_config, dif_config, actuator_config, feature_params,
                adv_feature_params, objective_params, stress_params # Add new params
            )
        except Exception as e:
            logger.exception(f"An unexpected error occurred during processing: {e}")

    logger.info("Data preparation script finished.")

if __name__ == "__main__":
    main() 