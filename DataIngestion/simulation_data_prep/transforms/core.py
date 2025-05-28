from __future__ import annotations

"""Core feature-engineering pipeline.

For now we simply expose ``transform_features`` (copied verbatim from the
original ``flow.py``).  In subsequent refactors we can split this further into
lighting / climate / flags helpers – but moving the code here already removes
~800 lines from Prefect flow module and lets other tools import it without
pulling in Prefect.
"""

from datetime import timedelta
from typing import TYPE_CHECKING, Optional, List, Dict, Any, Tuple

import polars as pl
from prefect import task

# Make sure config is importable from the parent directory or adjust path as needed
# If src is a package, relative import might work: from ..config import ...
# Assuming config.py is accessible:
from src.config import PlantConfig, DataProcessingConfig, OptimalRange, NightStressFlagDetail
from prefect.logging import get_run_logger
from prefect.exceptions import MissingContextError

import warnings
import logging
from datetime import timedelta, time
from typing import Dict, Any

from polars import col

# Internal imports - Use absolute paths from project root (/app)
from src.config import PlantConfig, OptimalRange, NightStressFlagDetail
from src.feature_calculator import (
    calculate_dli,
    calculate_gdd,
    calculate_dif,
    calculate_rate_of_change,
    calculate_rolling_average,
    calculate_rolling_std_dev,
    calculate_lag_feature,
    calculate_distance_from_range_midpoint,
    calculate_in_range_flag,
    calculate_night_stress_flag,
    calculate_daily_actuator_summaries
)

# Import the new time feature function
from src.feature_engineering import create_time_features

if TYPE_CHECKING:  # pragma: no cover
    import logging

# Remove the unnecessary pandas import attempt
# try:
#     import pandas as pd
# except ImportError:
#     pd = None # type: ignore

def _logger() -> "logging.Logger":
    try:
        return get_run_logger()
    except MissingContextError:  # when run in plain script / pytest
        logger = logging.getLogger("transform_features_local")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger


# ===== Updated transform_features function ==============================

def transform_features(
    df: pl.DataFrame,
    plant_cfg: PlantConfig,
    data_proc_cfg: DataProcessingConfig,
    segment_name: Optional[str] = None
) -> pl.DataFrame:
    """Generate all engineered features for a cleaned, imputed sensor Polars frame.

    Args:
        df: Input Polars DataFrame (cleaned and imputed).
        plant_cfg: Loaded PlantConfig object.
        data_proc_cfg: Loaded DataProcessingConfig object.
        segment_name: Optional name of the data segment being processed.

    Returns:
        Polars DataFrame with added feature columns.
    """

    logger = _logger()
    # Determine display name for segment before f-string
    segment_display_name = segment_name or "N/A"
    logger.info(f"Starting feature transformation for segment '{segment_display_name}' on DataFrame with shape {df.shape}")
    logger.info(f"Input columns: {df.columns}")

    # --- Load Segment-Specific or Global Feature Configuration ---
    current_segment_feature_config: Optional[SegmentFeatureConfig] = None
    if segment_name and data_proc_cfg.segment_feature_configs and segment_name in data_proc_cfg.segment_feature_configs:
        current_segment_feature_config = data_proc_cfg.segment_feature_configs[segment_name]
        logger.info(f"Using feature configuration specific to segment: '{segment_name}'.")
    elif data_proc_cfg.global_feature_config:
        current_segment_feature_config = data_proc_cfg.global_feature_config
        logger.info("Using global feature configuration.")
    else:
        logger.warning("No segment-specific or global feature configuration found. Many features might be skipped.")
        # Create a default empty config to prevent attribute errors later, though most features will be skipped.
        current_segment_feature_config = SegmentFeatureConfig(
            feature_parameters=FeatureParameters(), 
            advanced_feature_parameters=AdvancedFeatureParametersPlaceholder()
        )

    # Extract specific parameter groups for easier access, defaulting to empty if None
    feature_params = current_segment_feature_config.feature_parameters if current_segment_feature_config.feature_parameters else FeatureParameters()
    adv_feature_params = current_segment_feature_config.advanced_feature_parameters if current_segment_feature_config.advanced_feature_parameters else AdvancedFeatureParametersPlaceholder()
    active_optimal_keys = current_segment_feature_config.active_optimal_condition_keys if current_segment_feature_config.active_optimal_condition_keys else []

    if df.is_empty():
        logger.warning("Input DataFrame is empty, skipping transformations.")
        return df

    if "time" not in df.columns:
        logger.error("Input DataFrame must contain a 'time' column.")
        raise ValueError("Missing 'time' column for feature transformation.")

    if df["time"].dtype != pl.Datetime:
        try:
            df = df.with_columns(pl.col("time").str.to_datetime().alias("time"))
            logger.info("Converted 'time' column to datetime.")
        except Exception as e:
            logger.error(f"Failed to convert 'time' column to datetime: {e}")
            raise

    df = df.sort("time")

    # --- Add Date Column --- (Often useful for joining daily aggregates)
    try:
        df = df.with_columns(pl.col("time").dt.date().alias("date"))
    except Exception as e:
        logger.error(f"Failed to create 'date' column: {e}")
        raise

    # --- Add Time-Based Features ---
    try:
        # Call the function from feature_engineering
        df = create_time_features(df, time_col="time")
        # Logging is handled within create_time_features
    except Exception as e:
        logger.exception(f"Error adding time-based features: {e}")
        # Continue, but log the error

    # ================================================================\n    # === SEGMENT-AWARE FEATURE CALCULATION LOGIC STARTS HERE ===\n    # ================================================================\n    # Based on segment_name and data_proc_cfg, decide which features to calculate.\n    # Example: Only calculate certain actuator features if segment_name == \'Era2_Actuators\'\n    # Example: Get segment-specific lists for rolling windows/lags if configured.

    processed_features_list = [] # Track added features

    # --- Derived Delta Features ---
    delta_added = False
    if feature_params.delta_cols:
        delta_config = feature_params.delta_cols
        logger.info(f"Calculating delta features for segment '{segment_name or "global"}' based on config: {list(delta_config.keys())}")
        for out_col, cols in delta_config.items():
            if isinstance(cols, list) and len(cols) == 2:
                c1, c2 = cols
                if c1 in df.columns and c2 in df.columns:
                    try:
                        df = df.with_columns((pl.col(c1) - pl.col(c2)).alias(out_col))
                        processed_features_list.append(out_col)
                        delta_added = True
                    except Exception as e_delta:
                        logger.exception(f"Error calculating delta '{out_col}': {e_delta}")
                else:
                    logger.warning(f"Skipping delta '{out_col}' for segment '{segment_name}': required columns {cols} not found.")
            else:
                logger.warning(f"Delta config for '{out_col}' invalid. Skipping.")
        if delta_added: logger.info(f"Shape after delta features: {df.shape}")
    else: logger.info("No feature_parameters.delta_cols configured; skipping delta feature generation.")

    # --- Supplemental Lighting (PPF & kW) ---
    if hasattr(plant_cfg, "lamp_groups") and plant_cfg.lamp_groups:
        ppf_exprs = []
        kw_exprs = []
        logger.info("Calculating supplemental lighting features...")
        for col_name, detail in plant_cfg.lamp_groups.items():
            if col_name in df.columns:
                ppf_exprs.append(
                    pl.when(pl.col(col_name).cast(pl.Int8, strict=False).fill_null(0) == 1)
                    .then(pl.lit(detail.ppf_umol_s * detail.count))
                    .otherwise(0)
                )
                kw_exprs.append(
                    pl.when(pl.col(col_name).cast(pl.Int8, strict=False).fill_null(0) == 1)
                    .then(pl.lit(detail.power_kw * detail.count))
                    .otherwise(0)
                )
            else:
                logger.warning(f"Lamp status column '{col_name}' configured but not found.")

        if ppf_exprs:
            try:
                df = df.with_columns([
                    sum(ppf_exprs).alias("supplemental_ppf_umol_s"),
                    sum(kw_exprs).alias("lamp_power_kw")
                ])
                processed_features_list.extend(["supplemental_ppf_umol_s", "lamp_power_kw"])
                logger.info("Added supplemental_ppf_umol_s and lamp_power_kw columns.")
            except Exception as e_suppl:
                 logger.exception(f"Error calculating supplemental lighting sums: {e_suppl}")
        else:
            logger.warning("No matching lamp status columns found in data for configured lamp_groups.")
    else: logger.info("No lamp_groups configured; skipping supplemental lighting calculations.")

    # --- Combine natural and supplemental PPFD ---
    total_ppfd_col = None
    light_col = 'light_intensity_umol'
    supp_light_col = 'supplemental_ppf_umol_s'
    target_total_col = "ppfd_total"
    ppfd_expressions_to_add = []

    if light_col in df.columns:
        light_expr = pl.col(light_col).cast(pl.Float64, strict=False).fill_null(0)
        if supp_light_col in df.columns:
            logger.info(f"Combining '{light_col}' and '{supp_light_col}' into '{target_total_col}'.")
            supp_light_expr = pl.col(supp_light_col).cast(pl.Float64, strict=False).fill_null(0)
            ppfd_expressions_to_add.append((light_expr + supp_light_expr).alias(target_total_col))
            total_ppfd_col = target_total_col
        else:
            logger.info(f"Using '{light_col}' as '{target_total_col}'.")
            if target_total_col != light_col:
                ppfd_expressions_to_add.append(light_expr.alias(target_total_col))
            # If no rename needed, set total_ppfd_col to the original name
            total_ppfd_col = target_total_col if target_total_col != light_col else light_col 
    elif supp_light_col in df.columns:
         logger.info(f"Using supplemental light '{supp_light_col}' as '{target_total_col}'.")
         supp_light_expr = pl.col(supp_light_col).cast(pl.Float64, strict=False).fill_null(0)
         if target_total_col != supp_light_col:
             ppfd_expressions_to_add.append(supp_light_expr.alias(target_total_col))
         total_ppfd_col = target_total_col if target_total_col != supp_light_col else supp_light_col
    else:
        logger.warning(f"Neither '{light_col}' nor '{supp_light_col}' found, cannot calculate DLI or total PPFD features.")

    if ppfd_expressions_to_add:
        df = df.with_columns(ppfd_expressions_to_add)
        if total_ppfd_col: processed_features_list.append(total_ppfd_col)

    # --- Feature Calculations (Polars native) ---

    # VPD (Polars) - This was already Polars native
    if 'air_temp_c' in df.columns and 'relative_humidity_percent' in df.columns:
        try:
            # Assuming calculate_vpd is imported from feature_calculator and is Polars native
            from src.feature_calculator import calculate_vpd # Ensure it's imported
            vpd_series = calculate_vpd(df["air_temp_c"], df["relative_humidity_percent"])
            df = df.with_columns(vpd_series)
            processed_features_list.append('vpd_kpa')
            logger.info("Calculated VPD (Polars).")
        except Exception as e: logger.exception(f"Error VPD calculation (Polars): {e}")

    # CO2 Difference (Polars) - This was already Polars native
    if 'co2_measured_ppm' in df.columns and 'co2_required_ppm' in df.columns:
         try:
            # Assuming calculate_co2_difference is imported
            from src.feature_calculator import calculate_co2_difference # Ensure it's imported
            co2_diff_series = calculate_co2_difference(df['co2_measured_ppm'], df['co2_required_ppm'])
            df = df.with_columns(co2_diff_series)
            processed_features_list.append('CO2_diff_ppm')
            logger.info("Calculated CO2 Difference (Polars).")
         except Exception as e: logger.exception(f"Error CO2 diff calc (Polars): {e}")

    # --- DLI (Polars native) ---
    if total_ppfd_col and total_ppfd_col in df.columns:
        try:
            # calculate_dli is already imported from feature_calculator
            # It expects df, time_col, ppfd_col
            dli_daily_df = calculate_dli(df, time_col="time", ppfd_col=total_ppfd_col)
            if not dli_daily_df.is_empty() and "DLI_mol_m2_d" in dli_daily_df.columns:
                # Ensure 'date' column in dli_daily_df is of the same type as in df for joining
                # df already has a 'date' column from create_time_features or added manually
                df = df.join(dli_daily_df.select(["date", "DLI_mol_m2_d"]), on="date", how="left")
                processed_features_list.append('DLI_mol_m2_d')
                logger.info("Calculated DLI (Polars).")
            else:
                logger.warning(f"Polars DLI calculation for '{total_ppfd_col}' returned empty or missing DLI column.")
                if "DLI_mol_m2_d" not in df.columns: # Add a null column if it wasn't added
                    df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("DLI_mol_m2_d"))
        except Exception as e:
            logger.exception(f"Error DLI calculation (Polars): {e}")
            if "DLI_mol_m2_d" not in df.columns:
                df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("DLI_mol_m2_d"))


    # --- GDD (Polars native) ---
    if 'air_temp_c' in df.columns and hasattr(plant_cfg, 'gdd_parameters'):
        try:
            gdd_profile_name = plant_cfg.gdd_parameters.crop_profile
            gdd_profile = plant_cfg.gdd_parameters.profiles.get(gdd_profile_name)
            if gdd_profile:
                t_base = gdd_profile.t_base_celsius
                t_cap = gdd_profile.t_cap_celsius
                # calculate_gdd is already imported
                # It expects df, time_col, temp_col, t_base, t_cap
                gdd_daily_df_pl = calculate_gdd(df, time_col="time", temp_col="air_temp_c", t_base=t_base, t_cap=t_cap)
                if not gdd_daily_df_pl.is_empty() and "GDD_daily" in gdd_daily_df_pl.columns:
                    df = df.join(gdd_daily_df_pl.select(["date", "GDD_daily", "GDD_cumulative"]), on="date", how="left")
                    processed_features_list.extend(['GDD_daily', 'GDD_cumulative'])
                    logger.info("Calculated GDD (Polars).")
                else:
                    logger.warning(f"Polars GDD calculation for 'air_temp_c' returned empty or missing GDD columns.")
                    if "GDD_daily" not in df.columns:
                        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("GDD_daily"))
                    if "GDD_cumulative" not in df.columns:
                        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("GDD_cumulative"))
            else:
                logger.warning(f"GDD profile '{gdd_profile_name}' not found in config.")
        except Exception as e:
            logger.exception(f"Error GDD calculation (Polars): {e}")
            if "GDD_daily" not in df.columns:
                df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("GDD_daily"))
            if "GDD_cumulative" not in df.columns:
                df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("GDD_cumulative"))

    # --- DIF (Polars native) ---
    if 'air_temp_c' in df.columns and hasattr(plant_cfg, 'dif_parameters'):
        try:
            # Convert Pydantic model to dict for calculate_dif if it expects a dict
            # The calculate_dif in feature_calculator.py expects a dict.
            dif_cfg_dict = plant_cfg.dif_parameters.model_dump() if hasattr(plant_cfg.dif_parameters, 'model_dump') else dict(plant_cfg.dif_parameters)

            # calculate_dif is already imported
            # It expects df, time_col, temp_col, dif_config (as dict)
            dif_daily_df_pl = calculate_dif(df, time_col="time", temp_col="air_temp_c", dif_config=dif_cfg_dict)
            if dif_daily_df_pl is not None and not dif_daily_df_pl.is_empty() and "DIF_daily" in dif_daily_df_pl.columns:
                df = df.join(dif_daily_df_pl.select(["date", "DIF_daily"]), on="date", how="left")
                processed_features_list.append('DIF_daily')
                logger.info("Calculated DIF (Polars).")
            else:
                logger.warning(f"Polars DIF calculation for 'air_temp_c' returned empty, None, or missing DIF_daily column.")
                if "DIF_daily" not in df.columns:
                    df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("DIF_daily"))
        except Exception as e:
            logger.exception(f"Error DIF calculation (Polars): {e}")
            if "DIF_daily" not in df.columns:
                df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("DIF_daily"))


    # --- Actuator Summaries (Polars, partial implementation from feature_calculator) ---
    if hasattr(plant_cfg, 'actuator_summary_parameters') and plant_cfg.actuator_summary_parameters:
        try:
            summary_cfg_dict = plant_cfg.actuator_summary_parameters.model_dump() if hasattr(plant_cfg.actuator_summary_parameters, 'model_dump') else dict(plant_cfg.actuator_summary_parameters)
            actuator_daily_df = calculate_daily_actuator_summaries(df, time_col="time", summary_config=summary_cfg_dict)
            if not actuator_daily_df.is_empty():
                # Join all columns from actuator_daily_df except 'date'
                cols_to_join = [col for col in actuator_daily_df.columns if col != "date"]
                if cols_to_join:
                    df = df.join(actuator_daily_df.select(["date"] + cols_to_join), on="date", how="left")
                    processed_features_list.extend(cols_to_join)
                    logger.info(f"Calculated Actuator Summaries (Polars, partial): {cols_to_join}")
                else:
                    logger.info("Actuator Summaries (Polars) did not produce columns to join.")
            else:
                logger.warning("Actuator Summaries (Polars) returned empty DataFrame.")
        except Exception as e:
            logger.exception(f"Error Actuator Summaries calculation (Polars): {e}")


    # --- Rate of Change, Rolling Avg/Std, Lag, Domain Features (Polars native Series outputs) ---
    # These functions from feature_calculator.py typically return a Polars Series.
    # We will add them using with_columns.

    # Rate of Change
    if feature_params.rate_of_change_cols:
        logger.info(f"Calculating Rate of Change for segment '{segment_name or "global"}' for: {feature_params.rate_of_change_cols}")
        for col_name in feature_params.rate_of_change_cols:
            if col_name in df.columns:
                try:
                    roc_series = calculate_rate_of_change(df, value_col=col_name, time_col="time")
                    df = df.with_columns(roc_series)
                    processed_features_list.append(roc_series.name)
                except Exception as e_roc:
                    logger.exception(f"Error calculating RoC for '{col_name}': {e_roc}")
            else:
                logger.warning(f"Skipping RoC for '{col_name}': column not found.")

    # Rolling Average
    if feature_params.rolling_average_cols:
        logger.info(f"Calculating Rolling Average for segment '{segment_name or "global"}' for: {feature_params.rolling_average_cols}")
        for col_name, roll_config_dict in feature_params.rolling_average_cols.items(): # Expect dict-like object (Pydantic model)
            if col_name in df.columns:
                 # Check if the config has .get method (like dict/Pydantic model)
                 if not hasattr(roll_config_dict, 'get'):
                     logger.warning(f"Skipping rolling avg for '{col_name}' due to unexpected config type: {type(roll_config_dict)}")
                     continue
                 try:
                    # Access values using .get() for safety
                    window_minutes = roll_config_dict.get("window_minutes")
                    min_p = roll_config_dict.get("min_periods") # Can be None

                    if window_minutes is None:
                        logger.warning(f"Missing 'window_minutes' in rolling average config for '{col_name}'. Skipping.")
                        continue

                    window_str = f"{window_minutes}m"
                    
                    roll_avg_series = calculate_rolling_average(
                        df, value_col=col_name, time_col="time", window_str=window_str, min_periods=min_p
                    )
                    df = df.with_columns(roll_avg_series)
                    processed_features_list.append(roll_avg_series.name)
                 except Exception as e_roll_avg:
                    logger.exception(f"Error calculating Rolling Average for '{col_name}' with config '{roll_config_dict}': {e_roll_avg}")
            else:
                logger.warning(f"Skipping Rolling Average for '{col_name}': column not found.")

    # Rolling Standard Deviation
    if adv_feature_params.rolling_std_dev_cols:
        logger.info(f"Calculating Rolling Std Dev for segment '{segment_name or "global"}' for: {adv_feature_params.rolling_std_dev_cols}")
        for col_name, roll_config_dict in adv_feature_params.rolling_std_dev_cols.items(): 
            if col_name in df.columns:
                # Check if the config has .get method
                if not hasattr(roll_config_dict, 'get'):
                    logger.warning(f"Skipping Rolling Std Dev for '{col_name}' due to unexpected config type: {type(roll_config_dict)}")
                    continue
                try:
                    window_minutes = roll_config_dict.get("window_minutes")
                    min_p = roll_config_dict.get("min_periods")

                    if window_minutes is None:
                        logger.warning(f"Missing 'window_minutes' in rolling std dev config for '{col_name}'. Skipping.")
                        continue

                    window_str = f"{window_minutes}m"
                    
                    roll_std_series = calculate_rolling_std_dev(
                        df, value_col=col_name, time_col="time", window_str=window_str, min_periods=min_p
                    )
                    df = df.with_columns(roll_std_series)
                    processed_features_list.append(roll_std_series.name)
                except Exception as e_roll_std:
                    logger.exception(f"Error calculating Rolling Std Dev for '{col_name}' with config '{roll_config_dict}': {e_roll_std}")
            else:
                logger.warning(f"Skipping Rolling Std Dev for '{col_name}': column not found.")

    # Lag Features
    if adv_feature_params.lag_features:
        logger.info(f"Calculating Lag Features for segment '{segment_name or "global"}' for: {adv_feature_params.lag_features}")
        for col_name, lag_config_value in adv_feature_params.lag_features.items():
            if col_name in df.columns:
                try:
                    lag_series = None
                    output_alias = f"{col_name}_lag_unknown"

                    if isinstance(lag_config_value, int): # Assuming it's lag_minutes
                        lag_minutes = lag_config_value
                        if hasattr(plant_cfg, 'data_frequency_minutes') and plant_cfg.data_frequency_minutes and plant_cfg.data_frequency_minutes > 0:
                            lag_periods = int(lag_minutes / plant_cfg.data_frequency_minutes)
                            if lag_periods > 0:
                                output_alias = f"{col_name}_lag_{lag_periods}p"
                                lag_series = calculate_lag_feature(df, value_col=col_name, time_col="time", lag_periods=lag_periods)
                                lag_series = lag_series.alias(output_alias)
                            else:
                                logger.warning(f"Lag period for '{col_name}' is not positive ({lag_periods}) based on lag_minutes {lag_minutes} and data_freq {plant_cfg.data_frequency_minutes}. Skipping.")
                        else:
                            lag_duration_str = f"{lag_minutes}m"
                            output_alias = f"{col_name}_lag_{lag_duration_str}"
                            logger.warning(f"Data frequency not available or invalid for '{col_name}'. Attempting duration-based lag with {lag_duration_str}, which uses simple shift.")
                            lag_series = calculate_lag_feature(df, value_col=col_name, time_col="time", lag_duration=lag_duration_str)
                            lag_series = lag_series.alias(output_alias)
                    elif isinstance(lag_config_value, str): # Assuming it's a duration string like "1h"
                        lag_duration_str = lag_config_value
                        output_alias = f"{col_name}_lag_{lag_duration_str}"
                        logger.warning(f"Attempting duration-based lag for '{col_name}' with {lag_duration_str}, which uses simple shift.")
                        lag_series = calculate_lag_feature(df, value_col=col_name, time_col="time", lag_duration=lag_duration_str)
                        lag_series = lag_series.alias(output_alias)
                    else:
                        logger.warning(f"Invalid lag configuration for '{col_name}': {lag_config_value}. Expected int (minutes) or str (duration). Skipping.")

                    if lag_series is not None:
                        df = df.with_columns(lag_series)
                        processed_features_list.append(lag_series.name)

                except Exception as e_lag:
                    logger.exception(f"Error calculating Lag Feature for '{col_name}': {e_lag}")
            else:
                logger.warning(f"Skipping Lag Feature for '{col_name}': column not found.")

    # Domain-Specific Optimal Range Features
    optimal_conditions_map = plant_cfg.optimal_conditions
    # Filter optimal_conditions_map based on active_optimal_keys from segment_config
    if optimal_conditions_map and active_optimal_keys:
        logger.info(f"Calculating optimal range features for segment '{segment_name or "global"}' using active keys: {active_optimal_keys}")
        for feature_key in active_optimal_keys:
            # Use .get() on the optimal_conditions_map which is a Pydantic model / behaves like dict
            opt_cond_group = getattr(optimal_conditions_map, feature_key, None) 
            if opt_cond_group is None:
                logger.warning(f"Optimal condition key '{feature_key}' (active for segment) not found in plant_cfg.optimal_conditions. Skipping.")
                continue
            
            # Now iterate through the inner dict (e.g., {"growth": TempTarget(...)} )
            for condition_name, opt_range_model in opt_cond_group.items():
                # Map feature_key to DataFrame column name
                target_col_for_optimal = None
                if feature_key == "temperature_celsius" and "air_temp_c" in df.columns:
                    target_col_for_optimal = "air_temp_c"
                elif feature_key == "vpd_kpa" and "vpd_kpa" in df.columns:
                    target_col_for_optimal = "vpd_kpa"
                # Add more mappings... Example for DLI:
                # elif feature_key == "dli_mol_m2_day" and "DLI_mol_m2_d" in df.columns:
                #     target_col_for_optimal = "DLI_mol_m2_d"
    
                if target_col_for_optimal and hasattr(opt_range_model, 'min') and hasattr(opt_range_model, 'max'): # Check for .min and .max
                    lower_b = opt_range_model.min
                    upper_b = opt_range_model.max
                    if lower_b is not None and upper_b is not None:
                        logger.info(f"Calculating optimal range features for '{target_col_for_optimal}' (Condition: {condition_name}, Range: {lower_b}-{upper_b}).")
                        try:
                            dist_series = calculate_distance_from_range_midpoint(df[target_col_for_optimal], lower_bound=lower_b, upper_bound=upper_b)
                            flag_series = calculate_in_range_flag(df[target_col_for_optimal], lower_bound=lower_b, upper_bound=upper_b)
                            # Add condition_name to output column name for clarity if needed
                            dist_col_name = f"{dist_series.name}_{condition_name}"
                            flag_col_name = f"{flag_series.name}_{condition_name}"
                            df = df.with_columns([
                                dist_series.alias(dist_col_name),
                                flag_series.alias(flag_col_name)
                            ])
                            processed_features_list.extend([dist_col_name, flag_col_name])
                        except Exception as e_opt_range:
                            logger.exception(f"Error calculating optimal range features for '{target_col_for_optimal}': {e_opt_range}")
                    else:
                        logger.warning(f"Min/Max values not fully defined for '{feature_key}.{condition_name}' on column '{target_col_for_optimal}'. Skipping range features.")
                elif target_col_for_optimal:
                    logger.warning(f"Optimal range model for '{feature_key}.{condition_name}' (col '{target_col_for_optimal}') is missing or lacks min/max attributes. Skipping range features.")


    # Night Stress Flags
    if adv_feature_params.night_stress_flags:
        logger.info(f"Calculating Night Stress Flags for segment '{segment_name or "global"}' based on config.")
        dif_cfg_for_stress = plant_cfg.dif_parameters.model_dump() if plant_cfg.dif_parameters else {}

        for flag_name, flag_detail_union in adv_feature_params.night_stress_flags.items():
            if not isinstance(flag_detail_union, NightStressFlagDetail):
                if isinstance(flag_detail_union, str): # It's a comment
                    logger.info(f"Skipping night stress flag entry '{flag_name}' as it appears to be a comment: {flag_detail_union}")
                else:
                    logger.warning(f"Skipping night stress flag entry '{flag_name}' due to unexpected configuration type: {type(flag_detail_union)}")
                continue

            flag_detail: NightStressFlagDetail = flag_detail_union
            
            input_col = flag_detail.input_temp_col
            stress_type = flag_detail.stress_type
            output_col_suffix = flag_detail.output_col_suffix or f"{flag_detail.threshold_config_key}_{flag_detail.threshold_sub_key or 'direct'}_{stress_type}"
            output_col_name = f"{input_col}_night_stress_{output_col_suffix}"

            if input_col not in df.columns:
                logger.warning(f"Input column '{input_col}' for night stress flag '{flag_name}' not found in DataFrame. Skipping.")
                continue

            try:
                # Get the threshold value from plant_cfg.stress_thresholds
                threshold_obj = getattr(plant_cfg.stress_thresholds, flag_detail.threshold_config_key, None)
                actual_threshold_val: Optional[float] = None

                if threshold_obj is None:
                    logger.error(f"Threshold config key '{flag_detail.threshold_config_key}' not found in plant_cfg.stress_thresholds for flag '{flag_name}'. Skipping.")
                    continue

                if flag_detail.threshold_sub_key:
                    actual_threshold_val = getattr(threshold_obj, flag_detail.threshold_sub_key, None)
                elif isinstance(threshold_obj, (float, int)):
                    actual_threshold_val = float(threshold_obj)
                else:
                    logger.error(f"Cannot determine threshold value for '{flag_name}'. '{flag_detail.threshold_config_key}' is an object but no sub_key provided, or it's not a direct float/int. Skipping.")
                    continue
                
                if actual_threshold_val is None:
                    logger.error(f"Threshold value is None for '{flag_name}' (key: {flag_detail.threshold_config_key}, sub_key: {flag_detail.threshold_sub_key}). Skipping.")
                    continue

                logger.info(f"Calculating night stress flag '{output_col_name}' for column '{input_col}', type: '{stress_type}', threshold: {actual_threshold_val}")
                
                night_stress_series = calculate_night_stress_flag(
                    df,
                    time_col="time",
                    temp_col=input_col,
                    stress_threshold_temp=actual_threshold_val,
                    dif_config=dif_cfg_for_stress,
                    stress_type=stress_type
                )
                df = df.with_columns(night_stress_series.alias(output_col_name))
                processed_features_list.append(output_col_name)

            except AttributeError as attr_e:
                logger.exception(f"AttributeError while processing night stress flag '{flag_name}' (Details: key='{flag_detail.threshold_config_key}', sub_key='{flag_detail.threshold_sub_key}'): {attr_e}")
            except Exception as e_ns:
                logger.exception(f"Error calculating night stress flag '{flag_name}' for column '{input_col}': {e_ns}")

    # --- Availability Flags (New) ---
    if adv_feature_params.availability_flags_for_cols:
        logger.info(f"Calculating Availability Flags for segment '{segment_name or "global"}' for columns: {adv_feature_params.availability_flags_for_cols}")
        availability_exprs = []
        for col_to_check in adv_feature_params.availability_flags_for_cols:
            if col_to_check in df.columns:
                availability_exprs.append(
                    pl.col(col_to_check).is_not_null().cast(pl.Int8).alias(f"{col_to_check}_is_available")
                )
                processed_features_list.append(f"{col_to_check}_is_available")
            else:
                # If original column isn't there, flag will be all 0s (or we can skip)
                logger.warning(f"Source column '{col_to_check}' for availability flag not found. Flag will indicate all missing or be skipped.")
                # To ensure the column exists if other logic expects it:
                # availability_exprs.append(pl.lit(0, dtype=pl.Int8).alias(f"{col_to_check}_is_available"))
        if availability_exprs:
            df = df.with_columns(availability_exprs)

    # Remove the main Pandas conversion block
    # The df_pandas, time_idx_pd variables and the large if/else pandas_needed block are now removed.
    logger.info(f"Completed Polars-native feature calculations. Processed features list (subset): {list(set(processed_features_list))[:10]}")


    # --- Final Cleanup ---
    # Drop intermediate columns used only for daily joins if they still exist
    # The 'date' column is often kept as it's a useful feature itself.
    # If it was *only* for joining and needs to be removed:
    # if 'date' in df.columns and not keep_date_col_explicitly: # Add a flag if needed
    #     df = df.drop('date')
    #     logger.info("Dropped intermediate 'date' column.")
    # For now, 'date' is assumed to be a desired output feature from create_time_features

    logger.info(f"Finished feature transformation. Final output shape: {df.shape}")
    return df 