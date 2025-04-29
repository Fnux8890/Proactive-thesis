from __future__ import annotations

"""Core feature-engineering pipeline.

For now we simply expose ``transform_features`` (copied verbatim from the
original ``flow.py``).  In subsequent refactors we can split this further into
lighting / climate / flags helpers – but moving the code here already removes
~800 lines from Prefect flow module and lets other tools import it without
pulling in Prefect.
"""

from datetime import timedelta
from typing import TYPE_CHECKING

import polars as pl
from prefect import task

# Make sure config is importable from the parent directory or adjust path as needed
# If src is a package, relative import might work: from ..config import ...
# Assuming config.py is accessible:
from config import PlantConfig, OptimalRange, NightStressFlagDetail
from prefect.logging import get_run_logger
from prefect.exceptions import MissingContextError

if TYPE_CHECKING:  # pragma: no cover
    import logging


def _logger() -> "logging.Logger":
    try:
        return get_run_logger()
    except MissingContextError:  # when run in plain script / pytest
        import logging
        return logging.getLogger("transform_features")


# ===== copy of the original transform_features ==============================
# (trimmed import lines and replaced get_logger with _logger)

# Note: Removing @task decorator here. The wrapper in flow.py keeps it a task.
# If run standalone, it's just a regular function.
def transform_features(df: pl.DataFrame, config: PlantConfig) -> pl.DataFrame:  # noqa: C901 (big function, legacy)
    """Generate all engineered features for a raw sensor Polars frame."""

    logger = _logger()
    logger.info(f"Starting feature transformation on DataFrame with shape {df.shape}")

    # --- BEGIN Phase 1 Verification ---
    logger.info(f"Incoming columns: {df.columns}")
    if "lamp_group" in df.columns:
        logger.info(f"lamp_group dtype: {df['lamp_group'].dtype}")
    else:
        logger.warning("lamp_group column NOT found in incoming DataFrame!")
    # --- END Phase 1 Verification ---

    if df.is_empty():
        logger.warning("Input DataFrame is empty, skipping transformations.")
        return df

    # --- Establish Canonical 10-minute Time Grid & Join --- 
    if df.height == 0:
        logger.warning("Input DataFrame is empty after initial checks, returning empty frame.")
        return df # Return the empty frame structure

    t_min = df["time"].min()
    t_max = df["time"].max()

    # Create the 10-minute grid using Polars range (returns a Series when eager=True)
    time_grid = pl.datetime_range(t_min, t_max, interval="10m", eager=True)

    # Validate grid – len() works for Series in all Polars versions
    if len(time_grid) == 0:
        logger.error(
            "Failed to generate time grid from %s to %s. Check input time range.",
            t_min,
            t_max,
        )
        return df  # Gracefully return original frame

    # Wrap into a DataFrame for the join
    grid_df = pl.DataFrame({"time": time_grid})
    expected_rows = grid_df.height
    logger.info(f"Created 10-minute time grid with {expected_rows} intervals from {t_min} to {t_max}.")

    # Left join original data onto the grid. This ensures all 10-min slots are present.
    # This also resolves the previous row inflation issue.
    df = grid_df.join(df, on="time", how="left")
    logger.info(f"Shape after joining to 10-min grid: {df.shape} (expected {expected_rows} rows)")

    # --- Hybrid Forward Fill + Flag for Whitelisted Columns ---
    whitelist_ffill_cols = [
        "outside_temp_c", 
        "curtain_4_percent", 
        "flow_temp_1_c"
        # Add other SLOWLY changing variables here if deemed appropriate
    ]

    fill_expressions = []
    for col_name in whitelist_ffill_cols:
        if col_name in df.columns:
            flag_col_name = f"{col_name}_is_filled"
            # Create the flag *before* filling, based on current nulls
            fill_expressions.append(pl.col(col_name).is_null().cast(pl.Int8).alias(flag_col_name))
            # Apply forward fill
            fill_expressions.append(pl.col(col_name).forward_fill().alias(col_name))
            logger.debug(f"Applying forward fill and adding flag for {col_name}.")
        else:
            logger.warning(f"Column '{col_name}' not found for forward filling.")

    if fill_expressions:
        df = df.with_columns(fill_expressions)
        logger.info(f"Applied forward fill and flags for {len(fill_expressions)//2} columns.")
        logger.info(f"Shape after forward fill: {df.shape}")

    # --- Add Date Column (Needed for Daily Aggregations) --- 
    df = df.with_columns(pl.col("time").dt.date().alias("date")) # Add date column for joins

    # --- Derived Delta Features (config-driven) ---
    delta_config = config.feature_parameters.delta_cols if hasattr(config, 'feature_parameters') else None
    if delta_config:
        delta_added = False
        for out_col, cols in delta_config.items():
            if isinstance(cols, list) and len(cols) == 2:
                c1, c2 = cols
                if c1 in df.columns and c2 in df.columns:
                    df = df.with_columns((pl.col(c1) - pl.col(c2)).alias(out_col))
                    delta_added = True
                else:
                    logger.warning(f"Skipping delta '{out_col}': required columns {cols} not found.")
            else:
                logger.warning(f"Delta config for '{out_col}' invalid (needs exactly 2 columns). Skipping.")
        if delta_added:
            logger.info(f"Shape after delta features: {df.shape}")
    else:
        logger.info("No delta_cols configured; skipping delta feature generation.")

    # --- Supplemental Lighting (PPF & kW) ---
    if getattr(config, "lamp_groups", None):
        ppf_exprs = []
        kw_exprs = []
        for col_name, detail in config.lamp_groups.items():
            if col_name not in df.columns:
                continue  # skip missing columns
            ppf_exprs.append(
                pl.when(pl.col(col_name).cast(pl.Int8) == 1)
                .then(pl.lit(detail.ppf_umol_s * detail.count))
                .otherwise(0)
            )
            kw_exprs.append(
                pl.when(pl.col(col_name).cast(pl.Int8) == 1)
                .then(pl.lit(detail.power_kw * detail.count))
                .otherwise(0)
            )

        if ppf_exprs:
            df = df.with_columns([
                sum(ppf_exprs).alias("supplemental_ppf_umol_s"),
                sum(kw_exprs).alias("lamp_power_kw")
            ])
            logger.info("Added supplemental_ppf_umol_s and lamp_power_kw columns.")
        else:
            logger.warning("No matching lamp status columns found in data for configured lamp_groups.")
    else:
        logger.info("No lamp_groups configured; skipping supplemental lighting calculations.")

    # --- Feature Calculations ---

    # Decide which PPFD column to integrate: natural + supplemental if available
    total_ppfd_col = None
    if "light_intensity_umol" in df.columns:
        if "supplemental_ppf_umol_s" in df.columns:
            # NOTE: we assume greenhouse area = 1 m2 for now; adjust if area available.
            df = df.with_columns((pl.col("light_intensity_umol") + pl.col("supplemental_ppf_umol_s")).alias("ppfd_total"))
            total_ppfd_col = "ppfd_total"
        else:
            total_ppfd_col = "light_intensity_umol"

    if total_ppfd_col:
        # 1. DLI (Daily Light Integral)
        # Unit Test Target: 1hr constant 500 µmol -> 1.8 mol/m²/d
        logger.info("Calculating DLI...")
        dli_daily = (
            df.select(["time", "date", total_ppfd_col]) # time, date, PPFD total
            .sort("time")
            .with_columns(
                # Calculate time delta in seconds, fill first null with 0
                pl.col("time").diff().dt.total_seconds().fill_null(0).alias("delta_t_s")
            )
            .with_columns(
                # Calculate moles for the interval (PAR * delta_t / 1e6)
                (pl.col(total_ppfd_col) * pl.col("delta_t_s") / 1_000_000).alias("mol_chunk")
            )
            .group_by("date", maintain_order=True) # Use standard group_by on date
            .agg(
                pl.sum("mol_chunk").alias("DLI_mol_m2_d")
            )
            .select(["date", "DLI_mol_m2_d"]) # Keep only date and DLI for join
        )
        # Join daily DLI back to the main dataframe - REVERTED FROM MAP
        logger.info(f"Calculated daily DLI summary with shape: {dli_daily.shape}")
        df = df.join(dli_daily, on="date", how="left")
    else:
        logger.warning(f"Column 'light_intensity_umol' not found, skipping DLI calculation.")
    logger.info(f"Shape after DLI: {df.shape}")

    # 1.5 DLI Supplemental (DLI from only supplemental lamps)
    if "supplemental_ppf_umol_s" in df.columns:
        logger.info("Calculating Supplemental DLI...")
        dli_suppl_daily = (
            df.select(["time", "date", "supplemental_ppf_umol_s"])
            .sort("time")
            .with_columns(
                pl.col("time").diff().dt.total_seconds().fill_null(0).alias("delta_t_s")
            )
            .with_columns(
                (pl.col("supplemental_ppf_umol_s") * pl.col("delta_t_s") / 1_000_000).alias("mol_chunk_suppl")
            )
            .group_by("date", maintain_order=True)
            .agg(
                pl.sum("mol_chunk_suppl").alias("DLI_suppl_mol_m2_d")
            )
            .select(["date", "DLI_suppl_mol_m2_d"])
        )
        logger.info(f"Calculated supplemental DLI summary with shape: {dli_suppl_daily.shape}")
        df = df.join(dli_suppl_daily, on="date", how="left")
        logger.info(f"Shape after Supplemental DLI: {df.shape}")
    else:
        logger.warning("supplemental_ppf_umol_s column not found, skipping Supplemental DLI calculation.")

    # 2.5 VPD (Vapor Pressure Deficit) – needs to run before GDD to define T, RH
    logger.info("Calculating VPD (Buck 1981)...")
    T = pl.col("air_temp_c")
    RH = pl.col("relative_humidity_percent")
    svp_expr = 0.61121 * ((17.502 * T) / (T + 240.97)).exp()

    # --- Insert debug prints --- 
    # print("--- DataFrame before VPD calculation ---")
    # print(df.head(3))
    # print("---dtypes:", df.dtypes)
    # --- End debug prints --- 

    avp_expr = svp_expr * (RH / 100.0)
    df = df.with_columns((svp_expr - avp_expr).alias("vpd_kpa"))
    logger.info(f"Shape after VPD: {df.shape}")

    # 3. GDD (Growing Degree Days - Daily & Cumulative)
    # Unit Test Target: Day (mean=20C, base=10C) -> 14 GDD (based on config)
    logger.info("Calculating GDD...")
    gdd_profile = config.gdd_parameters.profiles[config.gdd_parameters.crop_profile]
    t_base = gdd_profile.t_base_celsius
    t_cap = gdd_profile.t_cap_celsius
    # Calculate the increment per row first, handling potential None caps
    gdd_increment_expr = pl.when(T > t_base)
    if t_cap is not None:
        gdd_increment_expr = gdd_increment_expr.then(pl.min_horizontal(T, pl.lit(t_cap)) - t_base)
    else:
        gdd_increment_expr = gdd_increment_expr.then(T - t_base)
    gdd_increment_expr = gdd_increment_expr.otherwise(0).alias("gdd_increment")
    df = df.with_columns(gdd_increment_expr)

    # Calculate daily GDD sum
    gdd_daily_sum = (
        df.select(["date", "gdd_increment"]) # Select only necessary columns for aggregation
        .group_by("date", maintain_order=True)
        .agg(pl.sum("gdd_increment").alias("GDD_daily"))
    )
    logger.info(f"Calculated daily GDD summary with shape: {gdd_daily_sum.shape}")

    # Join daily GDD sum back using a standard left join
    df = df.join(gdd_daily_sum.select(["date", "GDD_daily"]), on="date", how="left")

    # Calculate cumulative GDD (still within-day)
    # Check if GDD_daily exists before calculating cumulative
    if "GDD_daily" in df.columns:
        df = df.with_columns(
            pl.col("GDD_daily").cum_sum().over("date").alias("GDD_cumulative_within_day")
        )
        logger.warning("GDD_cumulative calculated within day; true cumulative needs historical context.")
    else:
         logger.warning("Skipping GDD_cumulative calculation as GDD_daily column was not created.")
    logger.info(f"Shape after GDD: {df.shape}")

    # 4. DIF (Day-minus-Night Temperature) - Explicit Calculation
    # Unit Test Target: Day=25C, Night=18C -> +7C
    logger.info("Calculating DIF (Explicit Method)...")
    if config.dif_parameters.day_definition == "lamp_status":
        lamp_cols = config.dif_parameters.lamp_status_columns
        # <-- ADD Lamp Value Debugging (Optional)-->
        try:
            logger.info(f"Unique values in lamp status columns ({lamp_cols}):")
            for col_name in lamp_cols:
                 if col_name in df.columns:
                     unique_vals = df[col_name].unique().to_list()
                     logger.info(f"  {col_name}: {unique_vals}")
                 else:
                     logger.warning(f"  Lamp column {col_name} not found in DataFrame.")
        except Exception as e:
            logger.warning(f"Could not log unique lamp values: {e}")
        # <-- END Lamp Value Debugging -->

        if not lamp_cols or not any(c in df.columns for c in lamp_cols):
            logger.warning("DIF calc by lamp_status specified, but no valid lamp_status_columns found or defined.")
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias("DIF_daily")) # Add null column
        else:
            # Create masks
            # FIX: Treat None as 0 (OFF) before comparing lamp status
            day_mask_expr = pl.any_horizontal([pl.col(c).fill_null(0).eq(1) for c in lamp_cols if c in df.columns])
            night_mask_expr = ~day_mask_expr

            # Calculate daily mean day temperature
            day_temps_daily = (
                df.filter(day_mask_expr)
                .group_by("date", maintain_order=True)
                .agg(pl.mean("air_temp_c").alias("_dayT_mean"))
            )

            # Calculate daily mean night temperature
            night_temps_daily = (
                df.filter(night_mask_expr)
                .group_by("date", maintain_order=True)
                .agg(pl.mean("air_temp_c").alias("_nightT_mean"))
            )

            # <-- ADD DIF DEBUG LOGGING -->
            logger.info(f"Day temps rows: {day_temps_daily.shape}, Night temps rows: {night_temps_daily.shape}")

            # Join day and night means together
            # Use outer_coalesce if polars >= 0.20.11, otherwise use outer and fillna
            try:
                # Try outer_coalesce first
                dif_stats_daily = day_temps_daily.join(
                    night_temps_daily, on="date", how="outer_coalesce"
                )
            except Exception: # Broad exception for potential older polars version
                logger.warning("outer_coalesce join failed, falling back to outer join for DIF stats.")
                dif_stats_daily = day_temps_daily.join(
                    night_temps_daily, on="date", how="outer"
                )

            # Calculate DIF daily, filling missing day/night means with 0 before subtraction
            dif_stats_daily = dif_stats_daily.with_columns(
                (pl.col("_dayT_mean").fill_null(0.0) - pl.col("_nightT_mean").fill_null(0.0)).alias("DIF_daily")
            ).select(["date", "DIF_daily"]) # Keep only date and final DIF

            # Join the calculated daily DIF back to the main DataFrame - REVERTED FROM MAP
            logger.info(f"Calculated daily DIF summary with shape: {dif_stats_daily.shape}")
            df = df.join(dif_stats_daily, on="date", how="left")

    else:
        logger.warning(f"DIF calculation method '{config.dif_parameters.day_definition}' not implemented.")
        df = df.with_columns(pl.lit(None).cast(pl.Float64).alias("DIF_daily")) # Add null column
    logger.info(f"Shape after DIF: {df.shape}")

    # 4.5 Basic Statistical Features: Rate of Change & Rolling Average (config-driven)
    feature_cfg = config.feature_parameters
    if feature_cfg:
        # --- Rate of Change ---
        if feature_cfg.rate_of_change_cols:
            for col_name in feature_cfg.rate_of_change_cols:
                if col_name not in df.columns:
                    logger.warning(f"Skipping rate-of-change for '{col_name}': column not found.")
                    continue
                try:
                    roc_col = f"{col_name}_RoC_per_s"
                    df = df.with_columns([
                        (pl.col(col_name).diff() / pl.col("time").diff().dt.total_seconds()).alias(roc_col)
                    ])
                except Exception as e:
                    logger.error(f"Failed RoC calculation for {col_name}: {e}")
            logger.info(f"Shape after rate-of-change features: {df.shape}")

        # --- Rolling Average ---
        if feature_cfg.rolling_average_cols:
            for col_name, window_min in feature_cfg.rolling_average_cols.items():
                if col_name not in df.columns:
                    logger.warning(f"Skipping rolling average for '{col_name}': column not found.")
                    continue
                try:
                    window_str = f"{window_min}m"
                    roll_avg_col = f"{col_name}_rolling_avg_{window_min}m"
                    roll_df = (
                        df.select(["time", col_name])
                          .rolling(index_column="time", period=window_str, closed="left")
                          .agg(pl.mean(col_name).alias(roll_avg_col))
                          .unique(subset=["time"], keep="first")
                    )
                    df = df.join(roll_df, on="time", how="left")
                except Exception as e:
                    logger.error(f"Failed rolling average for {col_name} ({window_min}m): {e}")
            logger.info(f"Shape after rolling average features: {df.shape}")

    # End basic statistical features

    # 5. Advanced Flags
    logger.info("Calculating advanced flags...")
    # Rolling SD / Lag (Using duration syntax)
    # Access advanced feature parameters from configuration (fixed attribute name)
    advanced_config = config.advanced_feature_parameters
    objective_config = config.objective_function_parameters

    # --- Rolling Standard Deviation ---
    # Safeguard: only iterate if rolling_std_dev_cols is provided
    if advanced_config.rolling_std_dev_cols:
        for col_name, window_minutes in advanced_config.rolling_std_dev_cols.items():
            if col_name.startswith("_"):
                logger.debug(f"Skipping rolling std dev for config key: {col_name}")
                continue
            if col_name not in df.columns:
                logger.warning(f"Skipping rolling std dev for '{col_name}': column not found.")
                continue
            try:
                window_duration_str = f"{window_minutes}m"
                # Aggregate separately
                rolling_agg_df = df.select(["time", col_name]) \
                                    .rolling(index_column="time", period=window_duration_str, closed="left") \
                                    .agg(pl.std(col_name).alias(f"{col_name}_rolling_std"))

                # FIX: Add unique step to prevent row explosion before joining
                rolling_agg_df = rolling_agg_df.unique(subset=["time"], keep="first")

                # Join back
                df = df.join(rolling_agg_df, on="time", how="left") \
                      .rename({f"{col_name}_rolling_std": f"{col_name}_rolling_std_{window_minutes}m"})

                # Construct the new column name first
                new_col_name = f"{col_name}_rolling_std_{window_minutes}m"
                # Re-check if column exists after potential join failures
                if new_col_name in df.columns:
                     logger.info(f"Shape after rolling std dev for {col_name}: {df.shape}")
                else:
                     # Use the pre-constructed column name variable
                     logger.warning(f"Rolling std dev column {new_col_name} was not successfully added.")

            except Exception as e:
                logger.error(f"Failed in-place rolling std dev calculation for {col_name} with period {window_duration_str}: {e}")

    # Lag features processing
    if advanced_config.lag_features:
        logger.info("Calculating time-based lag features using asof_join...")
        for col_name, lag_minutes in advanced_config.lag_features.items():
            if col_name.startswith("_"):
                logger.debug(f"Skipping lag feature for config key: {col_name}")
                continue
            if col_name not in df.columns:
                logger.warning(f"Skipping lag feature for '{col_name}': column not found.")
                continue

            try:
                # Define lag duration and new column name
                lag_duration = timedelta(minutes=lag_minutes)
                lagged_col_name = f"{col_name}_lag_{lag_minutes}m"
                logger.debug(f"Processing lag for {col_name} with duration {lag_duration} -> {lagged_col_name}")

                # Prepare a temporary DataFrame with time shifted *forward*
                df_to_lag = df.select(['time', col_name])
                df_lagged_time = df_to_lag.with_columns(
                    (pl.col('time') + lag_duration).alias('time_join_key')
                )

                # Perform the asof join
                # Find the last value in df_lagged_time whose 'time_join_key' is less than or equal to the current row's 'time'
                df = df.join_asof(
                    df_lagged_time.select(['time_join_key', col_name]).rename({col_name: lagged_col_name}),
                    left_on='time',          # Original time
                    right_on='time_join_key', # Time shifted forward by lag
                    strategy='backward'      # Match the latest record at or before the left timestamp
                )
                logger.info(f"Shape after lag feature for {col_name}: {df.shape}")

            except Exception as e:
                 logger.error(f"Failed processing lag feature for {col_name} with {lag_minutes}m lag: {e}", exc_info=True)

    # Distance-from-midpoint & in-range flag
    logger.info("Calculating distance-from-midpoint and in-range flags...")
    processed_dist_range = False
    # Get the mapping from config key to actual column name
    dist_midpoint_mapping = advanced_config.distance_from_optimal_midpoint
    if not dist_midpoint_mapping:
        logger.warning("Config key 'distance_from_optimal_midpoint' not found or empty in advanced_feature_parameters, skipping calculation.")
    else:
        for config_key, optimal_range in objective_config.optimal_ranges.items():
            if config_key.startswith("_"):
                logger.debug(f"Skipping midpoint/range for config key: {config_key}")
                continue

            # Get the actual input column name from the mapping
            input_col_name = dist_midpoint_mapping.get(config_key)

            # Check if the mapping exists for this config_key
            if not input_col_name:
                 logger.warning(f"Skipping distance-from-midpoint for config key '{config_key}': No corresponding input column defined in 'distance_from_optimal_midpoint'.")
                 continue

            # Check if the mapped input column exists in the DataFrame
            if input_col_name not in df.columns:
                logger.warning(f"Skipping distance-from-midpoint/in-range for config key '{config_key}': Mapped input column '{input_col_name}' not found.")
                continue

            if isinstance(optimal_range, OptimalRange) and optimal_range.lower is not None and optimal_range.upper is not None:
                lower_val = optimal_range.lower
                upper_val = optimal_range.upper
                midpoint = (lower_val + upper_val) / 2.0
                # Use the mapped input_col_name for calculation and alias
                df = df.with_columns(
                    (pl.col(input_col_name) - midpoint).abs().alias(f"{input_col_name}_dist_opt_mid")
                )
                processed_dist_range = True
            else:
                logger.warning(f"Skipping distance-from-midpoint for config key '{config_key}' (column '{input_col_name}'): Optimal range not found or invalid in config.")

    if config.advanced_feature_parameters.in_optimal_range_flag:
        # Get the mapping for in_optimal_range flags as well
        in_range_mapping = config.advanced_feature_parameters.in_optimal_range_flag
        if not in_range_mapping:
             logger.warning("Config key 'in_optimal_range_flag' not found or empty, skipping these flags.")
        else:
             for target_key, input_col in in_range_mapping.items(): # Original logic used target_key and input_col from here
                if not isinstance(input_col, str) or not input_col:
                     logger.warning(f"Skipping in-optimal-range flag for '{target_key}': invalid input column name '{input_col}'.")
                     continue
                if input_col not in df.columns:
                     logger.warning(f"Skipping in-optimal-range flag for '{target_key}': input column '{input_col}' not found.")
                     continue

                # Find corresponding optimal range using the target_key
                optimal_range = config.objective_function_parameters.optimal_ranges.get(target_key)
                if isinstance(optimal_range, OptimalRange) and optimal_range.lower is not None and optimal_range.upper is not None: # Ensure range is valid
                    lower = optimal_range.lower
                    upper = optimal_range.upper
                    df = df.with_columns(
                        pl.when((pl.col(input_col) >= lower) & (pl.col(input_col) <= upper))
                        .then(1)
                        .otherwise(0)
                        .cast(pl.Int8) # Use Int8 for flags
                        .alias(f"{input_col}_in_opt_range") # Alias uses the actual input column
                    )
                    processed_dist_range = True
                else:
                    logger.warning(f"Skipping in-optimal-range flag for '{target_key}' (column '{input_col}'): Optimal range not found or invalid in objective_function_parameters.")
    if processed_dist_range:
        logger.info(f"Shape after midpoint/range flags: {df.shape}")

    # Night-stress flag
    processed_night_stress = False
    if config.dif_parameters.day_definition == "lamp_status" and config.advanced_feature_parameters.night_stress_flags:
        # Ensure night_mask_expr is defined for this scope
        lamp_cols = config.dif_parameters.lamp_status_columns
        night_mask_expr = None # Default to None
        if lamp_cols:
            # Recreate night_mask_expr for this scope
            day_mask_expr = pl.any_horizontal([pl.col(c) == 1 for c in lamp_cols if c in df.columns])
            night_mask_expr = ~day_mask_expr # Define it here for use below

        if night_mask_expr is not None:
            logger.info("Calculating night stress flags...")
            for flag_name, details in config.advanced_feature_parameters.night_stress_flags.items():
                 if flag_name.startswith("_"):
                     logger.debug(f"Skipping night stress flag for config key: {flag_name}")
                     continue

                 # FIX: Check instance using imported class directly
                 if isinstance(details, NightStressFlagDetail):
                    logger.info(f"Processing night stress flag: name='{flag_name}', type={type(details)}, details={details}")
                    input_col = details.input_temp_col
                    threshold_key = details.threshold_config_key
                    threshold_sub_key = details.threshold_sub_key

                    if input_col not in df.columns:
                        logger.warning(f"Skipping night stress flag '{flag_name}': Input column '{input_col}' not found.")
                        continue

                    try:
                        threshold_val = getattr(getattr(config.stress_thresholds, threshold_key), threshold_sub_key)
                        df = df.with_columns(
                            # Ensure night_mask_expr is valid before using
                            pl.when(night_mask_expr & (pl.col(input_col) > threshold_val))
                            .then(1)
                            .otherwise(0)
                            .cast(pl.Int8) # Use Int8 for flags
                            .alias(flag_name)
                        )
                        processed_night_stress = True
                    except AttributeError:
                         logger.warning(f"Could not find threshold '{threshold_key}.{threshold_sub_key}' in config for night stress flag '{flag_name}'.")
                    except Exception as e:
                         logger.warning(f"Error processing night stress flag '{flag_name}': {e}")
                 else:
                     logger.warning(f"Skipping night stress flag '{flag_name}': Invalid config format.")
            if processed_night_stress:
                logger.info(f"Shape after night stress flags: {df.shape}")
        else:
            logger.warning("Cannot calculate night stress flags: Lamp status columns not defined or missing.")

    # 2.6 Lamp Energy (kWh) daily aggregation
    if "lamp_power_kw" in df.columns:
        try:
            logger.info("Aggregating lamp energy (kWh) per day ...")
            energy_daily = (
                df.select(["time", "date", "lamp_power_kw"])
                  .sort("time")
                  .with_columns(
                      pl.col("time").diff().dt.total_seconds().fill_null(0).alias("delta_t_s_energy")
                  )
                  .with_columns(
                      (pl.col("lamp_power_kw") * pl.col("delta_t_s_energy") / 3600.0).alias("kwh_chunk")
                  )
                  .group_by("date", maintain_order=True)
                  .agg(pl.sum("kwh_chunk").alias("Lamp_kWh_daily"))
            )
            df = df.join(energy_daily, on="date", how="left")
            logger.info(f"Shape after Lamp_kWh_daily: {df.shape}")
        except Exception as e:
            logger.error(f"Failed lamp energy aggregation: {e}")
    else:
        logger.info("lamp_power_kw column not found; skipping lamp energy aggregation.")

    # --- ADDED: Lamp Energy (kWh) per Group (Phase 3) ---
    if "lamp_power_kw" in df.columns and "lamp_group" in df.columns:
        try:
            logger.info("Aggregating lamp energy (kWh) per group per day...")
            # Ensure delta_t_s_energy and kwh_chunk are calculated. Re-calculate if needed.
            # If kwh_chunk isn't already on df, recalculate it here:
            if "kwh_chunk" not in df.columns:
                 df = df.with_columns(
                    pl.col("time").diff().dt.total_seconds().fill_null(0).alias("_delta_t_s_temp")
                 ).with_columns(
                    (pl.col("lamp_power_kw") * pl.col("_delta_t_s_temp") / 3600.0).alias("kwh_chunk")
                 ).drop("_delta_t_s_temp")

            energy_per_group_daily = (
                df.filter(pl.col("lamp_group").is_not_null() & pl.col("kwh_chunk").is_not_null())
                  .group_by(["date", "lamp_group"], maintain_order=True)
                  .agg(pl.sum("kwh_chunk").alias("Lamp_kWh_daily_per_group"))
            )
            logger.info(f"Calculated daily kWh per lamp group summary:\n{energy_per_group_daily}")
            # Note: This result (energy_per_group_daily) is currently only logged.
            # It could be joined back, pivoted, or saved separately as needed.
        except Exception as e:
            logger.error(f"Failed per-group lamp energy aggregation: {e}")
    else:
        logger.info("lamp_power_kw or lamp_group column not found; skipping per-group lamp energy aggregation.")
    # --- END: Lamp Energy (kWh) per Group ---

    # --- Final Cleanup --- (Optional)
    if "date" in df.columns:
        df = df.drop(["date"]) # Drop intermediate date col
        logger.info(f"Shape after dropping date column: {df.shape}")

    logger.info(f"Finished feature transformation. Final shape: {df.shape}")
    return df 