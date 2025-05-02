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
from src.config import PlantConfig, OptimalRange, NightStressFlagDetail
from prefect.logging import get_run_logger
from prefect.exceptions import MissingContextError

import warnings
import logging
from datetime import timedelta, time
from typing import Dict, Any

import pandas as pd
import numpy as np
from polars import col

# Internal imports - Use absolute paths from project root (/app)
from src.config import PlantConfig, OptimalRange, NightStressFlagDetail
from src.feature_calculator import (
    calculate_dli,
    calculate_gdd,
    calculate_vpd,
    calculate_dif,
    # TODO: Add imports for other helpers if/when called (e.g., calculate_rolling_std_dev)
)

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

    # --- END GDD CALCULATION --- 

    # DIF
    dif_temp_col = "air_temp_c"
    if dif_temp_col in df.columns:
        logger.info("Calling calculate_dif helper...")
        try:
            # Convert Polars inputs to Pandas before calling the Pandas-based function
            # Ensure the source Polars df HAS a 'time' column for index setting
            if 'time' not in df.columns:
                logger.error("Cannot call calculate_dif: Polars DataFrame missing 'time' column for index.")
                raise ValueError("Missing 'time' column in Polars DataFrame for DIF calculation")

            # Convert column to pandas Series and set DatetimeIndex
            temp_series_pd = df[dif_temp_col].to_pandas()
            time_index_pd = pd.to_datetime(df['time'].to_pandas(), errors='coerce') # Convert time column to DatetimeIndex
            temp_series_pd.index = time_index_pd

            # Convert full DataFrame and set DatetimeIndex
            data_df_pd = df.to_pandas() # Pass the whole df if needed for lamp status
            data_df_pd.index = time_index_pd

            # Call the function with Pandas objects that now have a DatetimeIndex
            dif_daily_result_pd = calculate_dif(temp_series_pd, config.dif_parameters, data_df_pd)

            # Convert the result back to Polars if needed downstream (currently mapped directly)
            # If dif_daily_result_pd is a Pandas Series as expected:
            if dif_daily_result_pd is not None and not dif_daily_result_pd.empty:
                # Ensure index name is 'date' BEFORE mapping
                dif_daily_result_pd.index.name = 'date'
                logger.info(f"Calculated daily DIF summary (Pandas result shape): {dif_daily_result_pd.shape}")
                if 'date' in df.columns: # Check in the original Polars df
                    # Map the Pandas Series result to the Polars DataFrame
                    # Ensure the Series name is unique or handle potential conflicts
                    dif_col_name = "DIF_C" # Example name, adjust if needed
                    if hasattr(dif_daily_result_pd, 'name') and dif_daily_result_pd.name:
                        dif_col_name = dif_daily_result_pd.name
                    # Convert Pandas result Series to Polars Series for mapping
                    # Convert the Pandas Series result back to a Polars DataFrame
                    dif_daily_result_pl = pl.from_pandas(dif_daily_result_pd.reset_index())
                    # Ensure the 'date' column in the result is cast to pl.Date to match the main df
                    dif_daily_result_pl = dif_daily_result_pl.with_columns(
                        pl.col('date').cast(pl.Date)
                    )

                    # Perform the map/join in Polars
                    # Since map is tricky between pandas/polars, let's use a join
                    df = df.join(dif_daily_result_pl.select(['date', dif_daily_result_pd.name]), on='date', how='left')

                    # Original mapping logic (might fail across types):
                    # df[dif_col_name] = df['date'].map(dif_daily_result_pd)
                    logger.info(f"Shape after DIF join: {df.shape}")
                else:
                     logger.warning("Cannot map/join DIF results, 'date' column missing from main Polars df.")
                 else:
                 logger.warning("calculate_dif returned empty or None result.")
        except AttributeError as ae:
             logger.exception(f"AttributeError calling calculate_dif (likely Pandas/Polars mismatch): {ae}")
        except Exception as e:
            logger.exception(f"Error calling calculate_dif: {e}")
    else:
         logger.warning(f"Skipping DIF calculation: Temp column '{dif_temp_col}' not found.")

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

    # 2.6 Lamp Energy (kWh) daily aggregation (Total)
    # This calculates the TOTAL lamp energy per day
    if "lamp_power_kw" in df.columns:
        try:
            logger.info("Aggregating TOTAL lamp energy (kWh) per day ...")
            # Calculate time delta in seconds if not already present
            if "delta_t_s_energy" not in df.columns:
                 df = df.with_columns(
                      pl.col("time").diff().dt.total_seconds().fill_null(0).alias("delta_t_s_energy")
                  )
            # Calculate kWh chunk for the total power
            if "kwh_chunk_total" not in df.columns: # Use a distinct name
                 df = df.with_columns(
                      (pl.col("lamp_power_kw") * pl.col("delta_t_s_energy") / 3600.0).alias("kwh_chunk_total")
                 )

            energy_daily_total = (
                df.select(["date", "kwh_chunk_total"])
                  .group_by("date", maintain_order=True)
                  .agg(pl.sum("kwh_chunk_total").alias("Lamp_kWh_daily")) # Keep original output column name
            )
            df = df.join(energy_daily_total, on="date", how="left")
            logger.info(f"Shape after Lamp_kWh_daily (Total): {df.shape}")
        except Exception as e:
            logger.error(f"Failed TOTAL lamp energy aggregation: {e}")
    else:
        logger.info("lamp_power_kw column not found; skipping TOTAL lamp energy aggregation.")


    # --- REVISED: Lamp Energy (kWh) per Group ---
    # This calculates energy PER GROUP based on individual status columns
    logger.info("Attempting per-group lamp energy calculation...")
    all_groups_energy = []
    # Ensure delta_t_s_energy is calculated (needed for per-group chunks)
    if "delta_t_s_energy" not in df.columns and "time" in df.columns:
        df = df.with_columns(
            pl.col("time").diff().dt.total_seconds().fill_null(0).alias("delta_t_s_energy")
        )

    if getattr(config, "lamp_groups", None) and "delta_t_s_energy" in df.columns:
        logger.debug(f"Found lamp_groups in config: {list(config.lamp_groups.keys())}")
        for col_name, detail in config.lamp_groups.items():
            if col_name in df.columns:
                try:
                    # Extract group identifier (e.g., '1' from 'lamp_grp1_no3_status')
                    # This assumes a consistent naming convention. Adjust regex if needed.
                    group_match = pl.lit(col_name).str.extract(r"lamp_grp(\d+)_.*", 1)
                    # Handle cases where regex might not match - default to a placeholder?
                    group_id_expr = pl.coalesce(group_match.cast(pl.Int8), pl.lit(-1).cast(pl.Int8)).alias("lamp_group") # Use Int8 for space

                    # Calculate power for this specific group
                    group_power_kw = detail.power_kw * detail.count

                    # Calculate kWh chunk specifically for this group when it's ON
                    # Ensure status column is treated as binary (0 or 1)
                    status_col_binary = pl.col(col_name).cast(pl.Int8).fill_null(0) # Cast to Int8, fill nulls as 0 (OFF)
                    kwh_chunk_group_expr = (
                        pl.lit(group_power_kw) * status_col_binary * pl.col("delta_t_s_energy") / 3600.0
                    ).alias("kwh_chunk_group")

                    # Aggregate daily kWh for this group
                    energy_this_group_daily = (
                        df.select(["date", col_name, "delta_t_s_energy"]) # Select necessary columns
                          .with_columns([
                              group_id_expr,
                              kwh_chunk_group_expr
                          ])
                          .filter(pl.col("kwh_chunk_group") > 0) # Only sum when power was consumed
                  .group_by(["date", "lamp_group"], maintain_order=True)
                          .agg(pl.sum("kwh_chunk_group")) # Sum the group-specific chunk
                          # Rename the aggregated column here
                          .rename({"kwh_chunk_group": "Lamp_kWh_daily_per_group"})
                    )
                    if not energy_this_group_daily.is_empty():
                        all_groups_energy.append(energy_this_group_daily)
                        logger.debug(f"Calculated daily kWh for group derived from {col_name}.")
                    else:
                        logger.debug(f"No energy calculated for group derived from {col_name} (likely always off or missing data).")

        except Exception as e:
                     logger.error(f"Failed processing energy for lamp group column '{col_name}': {e}", exc_info=True)
            else:
                 logger.warning(f"Configured lamp group column '{col_name}' not found in DataFrame. Skipping.")

        if all_groups_energy:
            # Combine results from all groups
            energy_per_group_combined = pl.concat(all_groups_energy)
            logger.info(f"Calculated daily kWh per lamp group summary:\n{energy_per_group_combined}")
            # --- OPTIONAL: Join back to main df ---
            # Joining this back might make the main df very long if many groups.
            # Consider if this joined data is needed or if the summary table is sufficient.
            # Example join (uncomment and adapt if needed):
            # df = df.join(energy_per_group_combined, on="date", how="left", suffix="_group_agg")
            # logger.info(f"Shape after joining per-group kWh summary: {df.shape}")
            # --------------------------------------
        else:
             logger.info("No per-group lamp energy data was generated.")

    else:
         logger.info("Skipping per-group lamp energy calculation: lamp_groups not configured, delta_t_s missing, or time missing.")
    # --- END: REVISED Lamp Energy (kWh) per Group ---


    # --- Final Cleanup --- (Optional)
    if "date" in df.columns:
        df = df.drop(["date"]) # Drop intermediate date col
        logger.info(f"Shape after dropping date column: {df.shape}")

    logger.info(f"Finished feature transformation. Final shape: {df.shape}")
    return df 