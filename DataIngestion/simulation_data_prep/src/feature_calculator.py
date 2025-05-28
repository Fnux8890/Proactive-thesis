import polars as pl
import numpy as np
import logging
from typing import List, Dict, Optional, Any, Tuple
from datetime import timedelta, time

logger = logging.getLogger(__name__)


def calculate_saturation_vapor_pressure(temp_c: pl.Series) -> pl.Series:
    """Calculate saturation vapor pressure (SVP) from air temperature using Polars.

    Uses the Tetens formula. Result is SVP in kilopascals (kPa).

    Args:
        temp_c: Polars Series of air temperature in Celsius (°C).

    Returns:
        pl.Series: Saturation vapor pressure in kPa.
    """
    A = 0.61078 # kPa
    B = 17.27
    C = 237.3  # °C
    # Ensure input is float, nulls remain null
    temp_c_numeric = temp_c.cast(pl.Float64, strict=False)
    svp = A * ((B * temp_c_numeric) / (C + temp_c_numeric)).exp()
    logger.debug("Calculated SVP (Polars).")
    return svp.alias("svp_kpa")


def calculate_vpd(
    temp_c: pl.Series,
    relative_humidity_percent: pl.Series
) -> pl.Series:
    """Calculate Vapor Pressure Deficit (VPD) from temperature and humidity using Polars.

    Args:
        temp_c: Polars Series of air temperature in Celsius (°C).
        relative_humidity_percent: Polars Series of relative humidity (% RH).

    Returns:
        pl.Series: Vapor Pressure Deficit in kPa.
    """
    temp_c_num = temp_c.cast(pl.Float64, strict=False)
    rh_percent_num = relative_humidity_percent.cast(pl.Float64, strict=False)

    svp_kpa = calculate_saturation_vapor_pressure(temp_c_num)

    # Calculate VPD directly: SVP * (1 - RH/100)
    vpd_kpa = svp_kpa * (1 - (rh_percent_num / 100.0))

    # VPD cannot be negative
    vpd_kpa = vpd_kpa.clip(lower_bound=0)

    logger.info("VPD calculation performed (Polars).")
    return vpd_kpa.alias("vpd_kpa")


# --- NEW DLI CALCULATION ---
def calculate_dli(
    df: pl.DataFrame,
    time_col: str = "time",
    ppfd_col: str = "ppfd_total"
) -> pl.DataFrame:
    """Calculate Daily Light Integral (DLI) from PPFD readings using Polars.

    Integrates PPFD over 24-hour periods.

    Args:
        df: Polars DataFrame containing timestamp and PPFD columns.
        time_col: Name of the datetime column.
        ppfd_col: Name of the PPFD column (μmol·m⁻²·s⁻¹).

    Returns:
        pl.DataFrame: DataFrame with columns ['date', 'DLI_mol_m2_d'].
    """
    if ppfd_col not in df.columns or time_col not in df.columns:
        logger.warning(f"Required columns '{time_col}' or '{ppfd_col}' not found. Cannot calculate DLI.")
        return pl.DataFrame({"date": [], "DLI_mol_m2_d": []}, schema={"date": pl.Date, "DLI_mol_m2_d": pl.Float64})

    if df.is_empty():
        logger.warning("Input DataFrame for DLI calculation is empty.")
        return pl.DataFrame({"date": [], "DLI_mol_m2_d": []}, schema={"date": pl.Date, "DLI_mol_m2_d": pl.Float64})

    # Ensure sorted by time for diff()
    df_sorted = df.sort(time_col)

    df_with_delta = df_sorted.with_columns([
        pl.col(time_col).diff().dt.total_seconds().fill_null(0).alias("delta_t_s"),
        pl.col(ppfd_col).cast(pl.Float64, strict=False).clip(lower_bound=0).fill_null(0).alias("_ppfd_numeric")
    ])

    df_with_moles = df_with_delta.with_columns(
        (pl.col("_ppfd_numeric") * pl.col("delta_t_s") / 1_000_000).alias("mol_chunk")
    )

    dli_daily = (
        df_with_moles.group_by(pl.col(time_col).dt.date().alias("date"))
        .agg(
            pl.sum("mol_chunk").alias("DLI_mol_m2_d")
        )
        .sort("date")
    )

    logger.info(f"DLI calculation performed using column '{ppfd_col}' (Polars).")
    return dli_daily
# --- END DLI CALCULATION --- 

# --- GDD CALCULATION ---
def calculate_gdd(
    df: pl.DataFrame,
    time_col: str = "time",
    temp_col: str = "air_temp_c",
    t_base: float = 10.0, # Example default
    t_cap: Optional[float] = None
) -> pl.DataFrame:
    """Calculate daily and cumulative Growing Degree Days (GDD) using Polars.

    Args:
        df: Polars DataFrame with time and temperature columns.
        time_col: Name of the datetime column.
        temp_col: Name of the temperature column (°C).
        t_base: Base temperature threshold (°C).
        t_cap: Optional upper temperature threshold (°C).

    Returns:
        pl.DataFrame: DataFrame with ['date', 'GDD_daily', 'GDD_cumulative'].
    """
    if temp_col not in df.columns or time_col not in df.columns:
        logger.warning(f"Required columns '{time_col}' or '{temp_col}' not found. Cannot calculate GDD.")
        return pl.DataFrame({"date": [], "GDD_daily": [], "GDD_cumulative": []}, schema={"date": pl.Date, "GDD_daily": pl.Float64, "GDD_cumulative": pl.Float64})

    if df.is_empty():
        logger.warning("Input DataFrame for GDD is empty.")
        return pl.DataFrame({"date": [], "GDD_daily": [], "GDD_cumulative": []}, schema={"date": pl.Date, "GDD_daily": pl.Float64, "GDD_cumulative": pl.Float64})

    temp_series_numeric = df.with_columns(pl.col(temp_col).cast(pl.Float64, strict=False))

    daily_min_max = (
        temp_series_numeric.group_by(pl.col(time_col).dt.date().alias("date"))
        .agg([
            pl.min(temp_col).alias("Tmin"),
            pl.max(temp_col).alias("Tmax")
        ])
        .drop_nulls() # Drop days where min or max couldn't be calculated
        .sort("date")
    )

    if daily_min_max.is_empty():
        logger.warning(f"No valid daily min/max temps found in '{temp_col}'. Cannot calculate GDD.")
        return pl.DataFrame({"date": [], "GDD_daily": [], "GDD_cumulative": []}, schema={"date": pl.Date, "GDD_daily": pl.Float64, "GDD_cumulative": pl.Float64})

    adj_tmin_expr = pl.max_horizontal(pl.col("Tmin"), pl.lit(t_base))
    
    adj_tmax_expr = pl.col("Tmax")
    if t_cap is not None:
        adj_tmax_expr = pl.min_horizontal(adj_tmax_expr, pl.lit(t_cap))
    
    adj_tmax_expr = pl.max_horizontal(adj_tmax_expr, pl.lit(t_base))
    
    # Ensure adj_tmin <= adj_tmax after t_cap application by re-evaluating adj_tmin against the (potentially capped) adj_tmax
    # This requires creating adj_tmax as an intermediate column to use in adj_tmin definition if t_cap is applied
    # Simpler: calculate avg_temp with potentially unconstrained adj_tmin, then ensure GDD >= 0

    # Simplified logic: apply adj_tmin, then adj_tmax, then ensure adj_tmin <= adj_tmax for average calculation
    # This can be done by applying adjustments sequentially and then using another pl.min_horizontal
    # or by calculating average and then clipping GDD to 0.
    
    # Calculate GDD for the day directly with expressions
    gdd_calculated = daily_min_max.with_columns([
        adj_tmin_expr.alias("_adj_tmin"),
        adj_tmax_expr.alias("_adj_tmax"),
    ]).with_columns([
        pl.min_horizontal(pl.col("_adj_tmin"), pl.col("_adj_tmax")).alias("final_adj_tmin"), # Ensure Tmin <= Tmax
        pl.col("_adj_tmax").alias("final_adj_tmax")
    ]).with_columns([
        (((pl.col("final_adj_tmax") + pl.col("final_adj_tmin")) / 2.0) - t_base).clip(lower_bound=0).alias("GDD_daily")
    ]).with_columns(
        pl.col("GDD_daily").cum_sum().alias("GDD_cumulative")
    ).drop(["_adj_tmin", "_adj_tmax", "final_adj_tmin", "final_adj_tmax"]) # Drop intermediate columns

    logger.info(f"GDD calculation performed using '{temp_col}' (Tbase={t_base}, Tcap={t_cap}) (Polars).")
    return gdd_calculated.select(["date", "GDD_daily", "GDD_cumulative"])
# --- END GDD CALCULATION --- 

# --- DIF CALCULATION ---
def calculate_dif(
    df: pl.DataFrame,
    time_col: str = "time",
    temp_col: str = "air_temp_c",
    dif_config: Optional[Dict[str, Any]] = None # Pass config dict directly
) -> pl.DataFrame:
    """Calculate the daily Temperature Differential (DIF) using Polars.

    Args:
        df: Polars DataFrame with time, temp, and potentially lamp status cols.
        time_col: Name of the datetime column.
        temp_col: Name of the temperature column (°C).
        dif_config: Dictionary with DIF parameters (day_definition, etc.).

    Returns:
        pl.DataFrame: DataFrame with ['date', 'DIF_daily'].
    """
    if temp_col not in df.columns or time_col not in df.columns:
        logger.warning(f"Required columns '{time_col}' or '{temp_col}' not found. Cannot calculate DIF.")
        return pl.DataFrame({"date": [], "DIF_daily": []}, schema={"date": pl.Date, "DIF_daily": pl.Float64})

    if df.is_empty():
        logger.warning("Input DataFrame for DIF is empty.")
        return pl.DataFrame({"date": [], "DIF_daily": []}, schema={"date": pl.Date, "DIF_daily": pl.Float64})

    if dif_config is None: 
        logger.warning("DIF config not provided. Skipping DIF calculation.")
        return pl.DataFrame({"date": [], "DIF_daily": []}, schema={"date": pl.Date, "DIF_daily": pl.Float64})

    day_definition = dif_config.get('day_definition', 'fixed_time').lower()
    
    # Select necessary columns including potential lamp_status columns
    select_cols = [pl.col(time_col), pl.col(temp_col)]
    active_lamp_cols_names = [] # Keep track of actual lamp column names used

    if day_definition == 'fixed_time':
        day_start = dif_config.get('fixed_time_day_start_hour', 6)
        day_end = dif_config.get('fixed_time_day_end_hour', 18)
        period_expr = pl.when(
            (pl.col(time_col).dt.hour() >= day_start) & (pl.col(time_col).dt.hour() < day_end)
        ).then(pl.lit("Day")).otherwise(pl.lit("Night"))
    elif day_definition == 'lamp_status':
        lamp_cols_config = dif_config.get('lamp_status_columns', [])
        for col_name in lamp_cols_config:
            if col_name in df.columns:
                select_cols.append(pl.col(col_name))
                active_lamp_cols_names.append(col_name)
        
        if not active_lamp_cols_names: # If none of the configured lamp_cols exist in df
            logger.error(f"DIF calc skipped: Lamp status columns {lamp_cols_config} not found in DataFrame.")
            return pl.DataFrame({"date": [], "DIF_daily": []}, schema={"date": pl.Date, "DIF_daily": pl.Float64})

        lamp_on_expr = pl.lit(False)
        for col_name_iter in active_lamp_cols_names: # Iterate over actual column names
            lamp_on_expr = lamp_on_expr | (pl.col(col_name_iter).cast(pl.Int8, strict=False).fill_null(0) == 1)
        
        period_expr = pl.when(lamp_on_expr).then(pl.lit("Day")).otherwise(pl.lit("Night"))
        # Fallback logic would typically be handled before this function or by re-calling
        # with a different config if this path fails to produce 'Day' and 'Night' periods.
    else:
        logger.error(f"Invalid 'day_definition' for DIF: '{day_definition}'.")
        return pl.DataFrame({"date": [], "DIF_daily": []}, schema={"date": pl.Date, "DIF_daily": pl.Float64})

    if period_expr is not None:
        # Apply the period expression to the DataFrame that has the necessary columns
        df_with_period = df.select(select_cols).with_columns(period_expr.alias("period"))
    else:
        # This case should ideally not be reached if day_definition is validated earlier
        return pl.DataFrame({"date": [], "DIF_daily": []}, schema={"date": pl.Date, "DIF_daily": pl.Float64})

    # Calculate daily average temp per period
    daily_avg_temps = (
        df_with_period.group_by([
            pl.col(time_col).dt.date().alias("date"), 
            pl.col("period") # This should now exist
        ])
        .agg(pl.mean(temp_col).alias("avg_temp"))
        .pivot(index="date", on="period", values="avg_temp") # Changed 'columns' to 'on'
        .sort("date")
    )

    # Calculate DIF
    if "Day" in daily_avg_temps.columns and "Night" in daily_avg_temps.columns:
        dif_daily = daily_avg_temps.with_columns(
            (pl.col("Day") - pl.col("Night")).alias("DIF_daily")
        ).select(["date", "DIF_daily"])
        logger.info(f"DIF calculation performed using '{day_definition}' definition (Polars).")
        return dif_daily
    else:
        logger.warning("Could not calculate DIF: Missing 'Day' or 'Night' pivoted columns.")
        return pl.DataFrame({"date": [], "DIF_daily": []}, schema={"date": pl.Date, "DIF_daily": pl.Float64})

# --- CO2 DIFFERENCE CALCULATION ---
def calculate_co2_difference(
    measured_co2: pl.Series,
    required_co2: pl.Series
) -> pl.Series:
    """Calculate the difference between measured and required CO2 levels using Polars.

    Args:
        measured_co2: Polars Series of measured CO2 values (ppm).
        required_co2: Polars Series of required/target CO2 values (ppm).

    Returns:
        pl.Series: The CO2 difference (ppm).
    """
    measured_numeric = measured_co2.cast(pl.Float64, strict=False)
    required_numeric = required_co2.cast(pl.Float64, strict=False)
    co2_diff = measured_numeric - required_numeric
    logger.info("CO2 difference calculation performed (Polars).")
    return co2_diff.alias("CO2_diff_ppm")

# --- END CO2 DIFFERENCE CALCULATION ---

# --- ACTUATOR SUMMARY CALCULATIONS ---

def calculate_daily_actuator_summaries(
    df: pl.DataFrame,
    time_col: str = "time",
    summary_config: Optional[Dict[str, List[str]]] = None
) -> pl.DataFrame:
    """Calculate daily summaries for specified actuators using Polars.

    Args:
        df: Polars DataFrame with time and actuator columns.
        time_col: Name of the datetime column.
        summary_config: Dict with keys like 'percent_columns_for_average', etc.

    Returns:
        pl.DataFrame: DataFrame indexed by date with summary columns.
    """
    if summary_config is None: summary_config = {}
    if time_col not in df.columns or df.is_empty():
        logger.warning(f"Time column '{time_col}' missing or DataFrame empty for actuator summaries.")
        return pl.DataFrame({"date": []}, schema={"date": pl.Date})

    df_with_date = df.with_columns(pl.col(time_col).dt.date().alias("date"))
    
    agg_expressions = []
    processed_any_avg = False

    # 1. Daily Average
    avg_cols = summary_config.get('percent_columns_for_average', [])
    for col_name in avg_cols:
        if col_name in df_with_date.columns:
            # Ensure column is numeric before aggregation
            df_with_date = df_with_date.with_columns(pl.col(col_name).cast(pl.Float64, strict=False).alias(col_name))
            agg_expressions.append(pl.mean(col_name).alias(f"Avg_{col_name}_Daily"))
            processed_any_avg = True
        else:
            logger.warning(f"Column '{col_name}' for daily average not found.")
            # Add a null column to maintain schema if other aggregations exist
            agg_expressions.append(pl.lit(None, dtype=pl.Float64).alias(f"Avg_{col_name}_Daily")) 

    # Placeholder for Daily Change Count & ON Hours - requires more complex Polars logic
    # For Daily Change Count: df.group_by("date").agg(pl.col(col_name).diff().is_not_null().sum())
    # For ON Hours: Requires calculating time deltas and summing when status is ON.

    if not processed_any_avg: # If no average columns, and others not implemented
        logger.warning("No valid columns/configs for actuator summary calculation (Polars version is partial)." )
        return pl.DataFrame({"date": df_with_date["date"].unique().sort()}, schema={"date": pl.Date})

    results_df = (
        df_with_date
        .group_by("date")
        .agg(agg_expressions)
        .sort("date")
    )
    
    logger.info("Partial actuator summary (averages) calculations performed (Polars). Change count & ON hours need full Polars implementation.")
    return results_df

# --- END ACTUATOR SUMMARY CALCULATIONS ---


# --- BASIC DIFFERENCE CALCULATION ---
def calculate_delta(
    series1: pl.Series,
    series2: pl.Series
) -> pl.Series:
    """Calculate the element-wise difference between two Polars series.

    Args:
        series1: The first polars Series (e.g., inside temperature).
        series2: The second polars Series (e.g., outside temperature).

    Returns:
        pl.Series: The difference.
    """
    num1 = series1.cast(pl.Float64, strict=False)
    num2 = series2.cast(pl.Float64, strict=False)
    delta = num1 - num2
    # Infer name or use generic
    name1 = series1.name or 'series1'
    name2 = series2.name or 'series2'
    logger.debug(f"Delta calculation performed between {name1} and {name2} (Polars).")
    return delta.alias(f"delta_{name1}_{name2}")

# --- END BASIC DIFFERENCE CALCULATION ---


# --- RATE OF CHANGE CALCULATION ---
def calculate_rate_of_change(
    df: pl.DataFrame,
    value_col: str,
    time_col: str = "time"
) -> pl.Series:
    """Calculate the rate of change of a series per second using Polars.

    Args:
        df: Polars DataFrame with time and value columns.
        value_col: Name of the value column.
        time_col: Name of the time column.

    Returns:
        pl.Series: The rate of change per second.
    """
    alias_name = f"{value_col}_RoC_per_s"
    if value_col not in df.columns or time_col not in df.columns:
        logger.warning(f"Missing '{value_col}' or '{time_col}' for RoC calculation.")
        # Return series of nulls with same length as df
        return pl.Series(name=alias_name, values=[None] * len(df), dtype=pl.Float64)

    df_sorted = df.sort(time_col) # Ensure sorted
    rate_of_change = (
        df_sorted.select([
            pl.col(value_col).diff().alias("val_diff"),
            pl.col(time_col).diff().dt.total_seconds().alias("time_diff_s")
        ])
        .with_columns(
            (pl.col("val_diff") / pl.col("time_diff_s").replace(0, None)) # Avoid division by zero
             .alias(alias_name)
        )
    )[alias_name]
    logger.debug(f"Rate of change calculation performed for {value_col} (Polars).")
    return rate_of_change

# --- END RATE OF CHANGE CALCULATION ---


# --- ROLLING AVERAGE CALCULATION ---
def calculate_rolling_average(
    df: pl.DataFrame,
    value_col: str,
    time_col: str = "time",
    window_str: str = "1h", # Polars time window string e.g., "1h", "15m", "1d"
    min_periods: Optional[int] = None # Added min_periods
) -> pl.Series:
    """Calculate the rolling average over a specified time window using Polars.

    Args:
        df: Polars DataFrame with time and value columns.
        value_col: Name of the value column.
        time_col: Name of the time column.
        window_str: Polars time window string e.g., "1h", "15m", "1d".
        min_periods: Optional minimum number of observations in window required to have a value.

    Returns:
        pl.Series: The rolling average.
    """
    alias_name = f"{value_col}_rolling_avg_{window_str}"
    if value_col not in df.columns or time_col not in df.columns:
        logger.warning(f"Missing '{value_col}' or '{time_col}' for rolling average.")
        return pl.Series(name=alias_name, values=[None] * len(df), dtype=pl.Float64)
       
    rolling_avg = (
        df.sort(time_col) # Crucial for rolling by time
        # DataFrame.rolling with a time period string does not accept min_periods directly.
        # The number of nulls will depend on the window size and 'closed' parameter.
        .rolling(index_column=time_col, period=window_str, closed="left", min_periods=min_periods)
        .agg(
            pl.mean(value_col).alias(alias_name)
        )
    )[alias_name]
    logger.debug(f"Rolling average ({window_str}) calculated for {value_col} (Polars).")
    return rolling_avg

# --- END ROLLING AVERAGE CALCULATION ---


# --- TIME INTEGRAL (TOTAL SUM) CALCULATION ---
def calculate_total_integral(
    series: pl.Series,
    time_index: pl.Series,
    unit_factor: float = 1.0
) -> float:
    """Calculate the total integral of a series over time.

    Approximates the integral by summing (value * time_interval) for each step.
    Useful for calculating total energy (Power * time) or total PAR (PPFD * time).

    Args:
        series: Polars Series of values (e.g., power in kW, PPFD in umol/m2/s).
        time_index: Polars Series with dtype pl.Datetime for the series.
        unit_factor: Optional factor to apply (e.g., 1/3600 to convert kW*seconds to kWh).

    Returns:
        float: The total integrated value over the series duration.
    """
    series_name = series.name if series is not None and series.name else 'series'
    if series is None:
        logger.warning(f"Input series '{series_name}' is None. Cannot calculate total integral.")
        return np.nan
    if not isinstance(time_index, pl.Series) or time_index.dtype != pl.Datetime:
        logger.error("Provided time_index is not a Polars Series of Datetime. Cannot calculate total integral.")
        return np.nan
    if len(series) != len(time_index):
        logger.warning(f"Series and time_index have different lengths for '{series_name}'. This might lead to issues.")

    series_numeric = series.cast(pl.Float64, strict=False)
    valid_mask = series_numeric.is_not_null()

    # Calculate time interval in seconds
    seconds_per_interval = time_index.diff().dt.total_seconds().fill_null(strategy="backward")

    # Calculate value * time_interval for valid points
    value_x_interval_values = series_numeric * seconds_per_interval

    # Apply valid_mask. Polars operations often handle nulls, but let's be explicit if needed
    total_sum = value_x_interval_values.sum()

    result = total_sum * unit_factor
    logger.debug(f"Total integral calculated for {series_name}. Raw sum: {total_sum}, Result: {result}")
    return result

# --- END TIME INTEGRAL CALCULATION ---


# --- MEAN ABSOLUTE DEVIATION CALCULATION ---
def calculate_mean_absolute_deviation(
    measured_series: pl.Series,
    setpoint_series: pl.Series | float
) -> float:
    """Calculate the Mean Absolute Deviation (MAD) from a setpoint.

    MAD = mean(abs(measured_value - setpoint_value))

    Args:
        measured_series: Polars Series of measured values.
        setpoint_series: Polars Series of setpoint values (must align with measured_series)
                         or a single float setpoint value.

    Returns:
        float: The calculated MAD.
    """
    measured_name = measured_series.name if measured_series is not None and measured_series.name else 'measured'
    if measured_series is None:
        logger.warning(f"Input series '{measured_name}' is None. Cannot calculate MAD.")
        return np.nan

    measured_numeric = pl.to_series(measured_series, dtype=pl.Float64)

    if isinstance(setpoint_series, pl.Series):
        setpoint_name = setpoint_series.name if setpoint_series.name else 'setpoint'
        if not measured_numeric.index.equals(setpoint_series.index):
            logger.warning(f"Measured index and setpoint series index do not match for {measured_name}/{setpoint_name}. Reindexing setpoint.")
            setpoint_series = setpoint_series.reindex(measured_numeric.index)
        setpoint_numeric = pl.to_series(setpoint_series, dtype=pl.Float64)
    elif isinstance(setpoint_series, (int, float)):
        setpoint_numeric = float(setpoint_series)
        setpoint_name = str(setpoint_numeric)
    else:
        logger.error("Setpoint must be a polars Series or a single number. Cannot calculate MAD.")
        return np.nan

    absolute_deviation = (measured_numeric - setpoint_numeric).abs()
    mean_abs_deviation = absolute_deviation.mean(skipna=True)

    logger.debug(f"MAD calculated for {measured_name} against setpoint {setpoint_name}.")
    return mean_abs_deviation

# --- END MEAN ABSOLUTE DEVIATION CALCULATION ---


# --- TIME IN RANGE CALCULATION ---
def calculate_time_in_range_percent(
    series: pl.Series,
    lower_bound: float,
    upper_bound: float
) -> float:
    """Calculate the percentage of time a series is within a specified range.

    Range is inclusive: [lower_bound, upper_bound].

    Args:
        series: Polars Series of measured values.
        lower_bound: The lower bound of the optimal range.
        upper_bound: The upper bound of the optimal range.

    Returns:
        float: Percentage of time (0.0 to 100.0) the series is within the range.
    """
    series_name = series.name if series is not None and series.name else 'series'
    if series is None:
        logger.warning(f"Input series '{series_name}' is None. Cannot calculate time in range.")
        return np.nan

    series_numeric = pl.to_series(series, dtype=pl.Float64)

    # Create boolean series: True if within bounds
    in_range_bool = (series_numeric >= lower_bound) & (series_numeric <= upper_bound)

    # Calculate mean directly on boolean series (True=1, False=0), ignoring NaNs
    fraction_in_range = in_range_bool.mean(skipna=True) 

    # Convert fraction to percentage
    percent_in_range = fraction_in_range * 100.0

    logger.debug(f"Time in range [{lower_bound}, {upper_bound}] calculated for {series_name}.")
    return percent_in_range

# --- END TIME IN RANGE CALCULATION ---


# --- STATISTICAL: ROLLING STANDARD DEVIATION ---
def calculate_rolling_std_dev(
    df: pl.DataFrame,
    value_col: str,
    time_col: str = "time",
    window_str: str = "1h",
    min_periods: Optional[int] = None # Added min_periods
) -> pl.Series:
    """Calculate the rolling standard deviation over a time window using Polars.

    Args:
        df: Polars DataFrame with time and value columns.
        value_col: Name of the value column.
        time_col: Name of the time column.
        window_str: Polars time window string e.g., "1h", "15m", "1d".
        min_periods: Optional minimum number of observations in window required to have a value.

    Returns:
        pl.Series: The rolling standard deviation.
    """
    alias_name = f"{value_col}_rolling_std_{window_str}"
    if value_col not in df.columns or time_col not in df.columns:
        logger.warning(f"Missing '{value_col}' or '{time_col}' for rolling std dev.")
        return pl.Series(name=alias_name, values=[None] * len(df), dtype=pl.Float64)

    # min_periods_for_std_test = 3 # This was for a test expectation that might differ from Polars default

    rolling_std = (
        df.sort(time_col)
        # DataFrame.rolling with a time period string does not accept min_periods directly.
        .rolling(index_column=time_col, period=window_str, closed="left", min_periods=min_periods)
        .agg(
            pl.std(value_col).alias(alias_name)
        )
    )[alias_name]
    logger.debug(f"Rolling std dev ({window_str}) calculated for {value_col} (Polars).")
    return rolling_std

# --- END ROLLING STANDARD DEVIATION ---


# --- STATISTICAL: LAG FEATURE ---
def calculate_lag_feature(
    df: pl.DataFrame,
    value_col: str,
    time_col: str = "time",
    lag_periods: Optional[int] = None,
    lag_duration: Optional[str] = None
) -> pl.Series:
    """Create a lagged version of a series using Polars (by period or time duration).

    Args:
        df: Polars DataFrame with time and value columns.
        value_col: Name of the value column.
        time_col: Name of the time column.
        lag_periods: Optional number of periods to lag.
        lag_duration: Optional time duration string e.g., "1h".

    Returns:
        pl.Series: The lagged series.
    """
    alias_name = f"{value_col}_lag_{lag_periods}p" if lag_periods else f"{value_col}_lag_invalid"
    if value_col not in df.columns:
        logger.warning(f"Missing '{value_col}' for lag feature.")
        return pl.Series(name=alias_name, values=[None] * len(df), dtype=pl.Float64)
       
    if lag_periods is not None and lag_periods > 0:
        # Ensure using Series.shift which takes 'n'
        lagged_series = df.get_column(value_col).shift(n=lag_periods)
        logger.debug(f"Lag feature (periods={lag_periods}) calculated for {value_col} (Polars).")
        return lagged_series.alias(alias_name)
    elif lag_duration is not None:
        # Time-based lag is trickier in Polars directly on series. Often done via joins.
        # Alternative: shift by an estimated number of periods based on avg frequency.
        # For simplicity, current Polars shift is primarily by row count.
        # Returning null series for time duration lag for now.
        logger.warning(f"Time duration lag ('{lag_duration}') not directly implemented via simple shift in this Polars version. Use period lag or join-based approach.")
        return pl.Series(name=alias_name, values=[None] * len(df), dtype=pl.Float64)
    else:
        logger.error("Invalid lag parameters: specify either lag_periods (int) or lag_duration (str).")
        return pl.Series(name=alias_name, values=[None] * len(df), dtype=pl.Float64)

# --- END LAG FEATURE ---


# --- DOMAIN: DISTANCE FROM OPTIMAL RANGE MIDPOINT ---
def calculate_distance_from_range_midpoint(
    series: pl.Series,
    lower_bound: float,
    upper_bound: float
) -> pl.Series:
    """Calculate signed distance from the midpoint of a range using Polars."""
    alias_name = f"{series.name}_dist_opt_mid"
    try:
        midpoint = (float(lower_bound) + float(upper_bound)) / 2.0
    except (TypeError, ValueError):
        logger.error(f"Invalid bounds for midpoint calculation: lower={lower_bound}, upper={upper_bound}")
        return pl.Series(name=alias_name, values=[None] * series.len(), dtype=pl.Float64)

    series_numeric = series.cast(pl.Float64, strict=False)
    distance = series_numeric - midpoint
    logger.debug(f"Distance from range midpoint ({midpoint}) calculated for {series.name} (Polars).")
    return distance.alias(alias_name)

# --- END DISTANCE FROM OPTIMAL RANGE MIDPOINT ---


# --- DOMAIN: IN OPTIMAL RANGE FLAG ---
def calculate_in_range_flag(
    series: pl.Series,
    lower_bound: float,
    upper_bound: float
) -> pl.Series:
    """Create a binary flag (0 or 1) if value is within a range using Polars."""
    alias_name = f"{series.name}_in_opt_range"
    try:
        lower = float(lower_bound)
        upper = float(upper_bound)
    except (TypeError, ValueError):
        logger.error(f"Invalid bounds for in-range flag: lower={lower_bound}, upper={upper_bound}")
        # Return a series of Nones (or 0s) with the correct length and type
        return pl.Series(name=alias_name, values=[0] * series.len(), dtype=pl.Int8) # Ensure it is a series

    series_numeric = series.cast(pl.Float64, strict=False)
    
    # Create a boolean Series by applying the condition directly to the input Series
    condition = (series_numeric >= lower) & (series_numeric <= upper)
    
    # Cast the boolean Series to Int8 (True becomes 1, False becomes 0)
    # and fill any nulls that might have been in series_numeric (and thus in condition) with 0.
    in_range_flag = condition.cast(pl.Int8).fill_null(0)
    
    logger.debug(f"In-range flag [{lower}, {upper}] calculated for {series.name} (Polars).")
    return in_range_flag.alias(alias_name)

# --- END IN OPTIMAL RANGE FLAG ---


# --- DOMAIN: NIGHT STRESS THRESHOLD FLAG ---
def calculate_night_stress_flag(
    df: pl.DataFrame,
    time_col: str = "time",
    temp_col: str = "air_temp_c",
    stress_threshold_temp: float = 10.0,
    dif_config: Optional[Dict[str, Any]] = None,
    stress_type: str = "high"  # Added stress_type parameter
) -> pl.Series:
    """Create a binary flag if night temperature is outside a stress threshold using Polars.

    Args:
        df: Polars DataFrame with time and value columns.
        time_col: Name of the datetime column.
        temp_col: Name of the temperature column.
        stress_threshold_temp: The temperature threshold.
        dif_config: Dictionary with DIF parameters for night definition.
        stress_type: Type of stress, 'high' (temp > threshold) or 'low' (temp < threshold).

    Returns:
        pl.Series: Binary flag (0 or 1) for stress condition.
    """
    alias_name = f"night_{stress_type}_stress_{temp_col}"
    if temp_col not in df.columns or time_col not in df.columns:
        logger.warning(f"Missing '{temp_col}' or '{time_col}' for night stress flag.")
        return pl.Series(name=alias_name, values=[0] * len(df), dtype=pl.Int8)
    if dif_config is None: 
        logger.warning("DIF config needed for night definition in night stress flag. Using default fixed_time (6-18)." )
        dif_config = {"day_definition": "fixed_time", "fixed_time_day_start_hour": 6, "fixed_time_day_end_hour": 18}

    day_definition = dif_config.get('day_definition', 'fixed_time').lower()
    period_expr = None

    if day_definition == 'fixed_time':
        day_start = dif_config.get('fixed_time_day_start_hour', 6)
        day_end = dif_config.get('fixed_time_day_end_hour', 18)
        period_expr = pl.when(
            (pl.col(time_col).dt.hour() >= day_start) & (pl.col(time_col).dt.hour() < day_end)
        ).then(pl.lit("Day")).otherwise(pl.lit("Night"))
    elif day_definition == 'lamp_status':
        lamp_cols = dif_config.get('lamp_status_columns', [])
        active_lamp_cols = [col for col in lamp_cols if col in df.columns]
        if active_lamp_cols:
            lamp_on_expr = pl.lit(False)
            for col_name_loop in active_lamp_cols: 
                lamp_on_expr = lamp_on_expr | (pl.col(col_name_loop).cast(pl.Int8, strict=False).fill_null(0) == 1)
            period_expr = pl.when(lamp_on_expr).then(pl.lit("Day")).otherwise(pl.lit("Night"))
        else:
            logger.warning("Lamp status columns not found for night stress flag's DIF definition. Falling back to fixed time.")
            day_start = dif_config.get('fixed_time_day_start_hour', 6)
            day_end = dif_config.get('fixed_time_day_end_hour', 18)
            period_expr = pl.when(
                (pl.col(time_col).dt.hour() >= day_start) & (pl.col(time_col).dt.hour() < day_end)
            ).then(pl.lit("Day")).otherwise(pl.lit("Night")) 
    else:
         logger.error(f"Invalid 'day_definition' for night stress flag: '{day_definition}'.")
         return pl.Series(name=alias_name, values=[0] * len(df), dtype=pl.Int8)

    temp_numeric = df[temp_col].cast(pl.Float64, strict=False)
    # Corrected chaining for Polars when/then/otherwise
    
    condition = None
    if stress_type == "high":
        condition = (period_expr == "Night") & (temp_numeric > stress_threshold_temp)
    elif stress_type == "low":
        condition = (period_expr == "Night") & (temp_numeric < stress_threshold_temp)
    else:
        logger.error(f"Invalid stress_type '{stress_type}' for night stress flag. Must be 'high' or 'low'.")
        return pl.Series(name=alias_name, values=[0] * len(df), dtype=pl.Int8)
        
    stress_flag_expr = (
        pl.when(condition)
        .then(pl.lit(1, dtype=pl.Int8))
        .otherwise(pl.lit(0, dtype=pl.Int8))
    ).fill_null(0) # Treat null temps as not stressed

    logger.debug(f"Night stress flag (Threshold={stress_threshold_temp}°C) calculated (Polars).")
    # Evaluate the expression on the df and return the resulting Series
    result_series = df.select(stress_flag_expr.alias(alias_name)).get_column(alias_name)
    return result_series

# --- END NIGHT STRESS THRESHOLD FLAG ---