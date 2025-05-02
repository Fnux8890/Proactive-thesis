import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def calculate_saturation_vapor_pressure(temp_c: pd.Series) -> pd.Series:
    """Calculate saturation vapor pressure (SVP) from air temperature.

    Uses the Tetens formula, a variation of the Magnus formula, commonly used
    in meteorology and horticulture for temperatures above freezing.
    Result is SVP in kilopascals (kPa).

    See feature report Section 3.2 and reference [6].

    Args:
        temp_c: Pandas Series of air temperature in Celsius (°C).

    Returns:
        pd.Series: Saturation vapor pressure in kPa. Returns NaN where input
                   temperature is NaN or cannot be converted to numeric.
    """
    # Magnus formula coefficients (Tetens variation from report [6])
    A = 0.61078 # kPa
    B = 17.27
    C = 237.3  # °C
    # Ensure input is numeric, coercing errors
    temp_c_numeric = pd.to_numeric(temp_c, errors='coerce')
    svp = A * np.exp((B * temp_c_numeric) / (C + temp_c_numeric))
    logger.debug("Calculated SVP.")
    return svp


def calculate_vpd(
    temp_c: pd.Series,
    relative_humidity_percent: pd.Series
) -> pd.Series:
    """Calculate Vapor Pressure Deficit (VPD) from temperature and humidity.

    VPD is the difference between saturation vapor pressure (SVP) at a given
    temperature and the actual vapor pressure (AVP) derived from relative
    humidity. Result is VPD in kilopascals (kPa).

    VPD = SVP(temp_c) * (1 - (relative_humidity_percent / 100))

    See feature report Section 3.2.

    Args:
        temp_c: Pandas Series of air temperature in Celsius (°C).
        relative_humidity_percent: Pandas Series of relative humidity (% RH).

    Returns:
        pd.Series: Vapor Pressure Deficit in kPa. Returns NaN where inputs
                   are NaN or non-numeric. Negative results are clipped to 0.
    """
    if temp_c is None or relative_humidity_percent is None:
        logger.warning("Input Series for VPD calculation are None.")
        # Return an empty Series with the correct dtype to avoid downstream errors
        return pd.Series(dtype=np.float64)

    # Ensure inputs are numeric, coercing errors to NaN
    temp_c_num = pd.to_numeric(temp_c, errors='coerce')
    rh_percent_num = pd.to_numeric(relative_humidity_percent, errors='coerce')

    # Calculate saturation vapor pressure (SVP)
    svp_kpa = calculate_saturation_vapor_pressure(temp_c_num)

    # Calculate actual vapor pressure (AVP) - intermediate step not strictly needed
    # avp_kpa = svp_kpa * (rh_percent_num / 100.0)
    # Calculate VPD directly
    # vpd_kpa = svp_kpa - avp_kpa
    vpd_kpa = svp_kpa * (1 - (rh_percent_num / 100.0))

    # VPD cannot be negative (handle potential floating point inaccuracies or weird data)
    vpd_kpa = vpd_kpa.clip(lower=0) # Use Series.clip method

    logger.info("VPD calculation performed.")
    return vpd_kpa


# --- NEW DLI CALCULATION ---
def calculate_dli(
    ppfd_series: pd.Series,
    time_index: pd.DatetimeIndex,
    ppfd_col_name: str = 'light_intensity_umol'
) -> pd.Series:
    """Calculate Daily Light Integral (DLI) from PPFD readings.

    Integrates Photosynthetic Photon Flux Density (PPFD) over a 24-hour period
    to get the total moles of PAR photons per square meter per day.
    Assumes input ppfd_series contains PPFD values in μmol·m⁻²·s⁻¹.
    Requires a DatetimeIndex to determine time intervals between readings.

    See feature report Section 3.1.

    Args:
        ppfd_series: Pandas Series containing instantaneous PPFD data (μmol·m⁻²·s⁻¹).
                     Must share the same index as `time_index`.
        time_index: Pandas DatetimeIndex from the original DataFrame, sorted ascending.
                    Used to calculate the duration of each measurement interval.
        ppfd_col_name: Name of the PPFD column being used (for logging).

    Returns:
        pd.Series: Daily DLI values in mol·m⁻²·d⁻¹, indexed by day (start of day).
                   Returns an empty Series if inputs are invalid or no valid DLI
                   can be calculated.
    """
    if ppfd_series is None or ppfd_series.empty:
        logger.warning(f"Input PPFD series '{ppfd_col_name}' is None or empty. Cannot calculate DLI.")
        return pd.Series(dtype=np.float64)

    if not isinstance(time_index, pd.DatetimeIndex):
         logger.error("Provided time_index is not a DatetimeIndex. Cannot calculate DLI.")
         return pd.Series(dtype=np.float64)

    if not time_index.is_monotonic_increasing:
        logger.warning("Provided time_index is not sorted. Sorting index for interval calculation.")
        time_index = time_index.sort_values()
        # Reindex ppfd_series in case sorting changed order relative to original df
        ppfd_series = ppfd_series.reindex(time_index)

    # Ensure PPFD is numeric and non-negative
    ppfd_numeric = pd.to_numeric(ppfd_series, errors='coerce').clip(lower=0)
    # Keep track of original NaNs or non-numeric to avoid calculating interval for them
    valid_mask = ppfd_numeric.notna()

    # Calculate seconds per measurement interval based on the main time_index
    # Use forward fill for the first interval, assuming it's the same as the second
    seconds_per_interval = time_index.to_series().diff().dt.total_seconds().bfill()

    # Calculate total micromoles per interval ONLY for valid readings
    photons_umol_m2_interval = pd.Series(index=time_index, dtype=np.float64)
    photons_umol_m2_interval[valid_mask] = ppfd_numeric[valid_mask] * seconds_per_interval[valid_mask]

    # Resample daily, sum the micromoles, and convert to moles for DLI
    # Summing NaNs results in NaN for that day, which is appropriate
    dli_daily = photons_umol_m2_interval.resample('D').sum(min_count=1) / 1_000_000 # min_count=1 ensures days with only NaN input result in NaN output

    logger.info(f"DLI calculation performed using column '{ppfd_col_name}'.")
    return dli_daily
# --- END DLI CALCULATION --- 

# --- GDD CALCULATION ---
def calculate_gdd(
    temp_series: pd.Series,
    t_base: float,
    t_cap: float | None = None,
    temp_col_name: str = 'air_temp_c'
) -> pd.DataFrame:
    """Calculate daily and cumulative Growing Degree Days (GDD).

    GDD quantifies the accumulation of heat relevant for plant development above
    a base temperature (Tbase), optionally capped at an upper threshold (Tcap).
    Uses the Modified Average method: adjusts daily min/max temps to thresholds
    before averaging and calculating GDD.

    See feature report Section 3.3.

    Args:
        temp_series: Pandas Series of temperature data (e.g., air_temp_c in °C).
                     Index must be a DatetimeIndex.
        t_base: Base temperature threshold (°C) below which development is assumed
                to stop. Specific to the crop.
        t_cap: Optional upper temperature threshold (°C) above which development rate
               does not increase further. If None, no upper cap is applied.
        temp_col_name: Name of the temperature column being used (for logging).

    Returns:
        pd.DataFrame: DataFrame indexed by day (start of day), containing columns
                      'GDD' (daily calculated GDD in °C-day) and 'Cumulative_GDD'.
                      Returns an empty DataFrame if inputs are invalid or no valid
                      daily temperatures are found.
    """
    if temp_series is None or temp_series.empty:
        logger.warning(f"Input temperature series '{temp_col_name}' is None or empty. Cannot calculate GDD.")
        return pd.DataFrame(columns=['GDD', 'Cumulative_GDD'])

    if not isinstance(temp_series.index, pd.DatetimeIndex):
        logger.error("Input temperature series must have a DatetimeIndex. Cannot calculate GDD.")
        return pd.DataFrame(columns=['GDD', 'Cumulative_GDD'])

    # Ensure temperature is numeric
    temp_numeric = pd.to_numeric(temp_series, errors='coerce')

    # Resample to get daily min and max temperatures
    # Drop days where min or max couldn't be calculated (e.g., all NaNs for that day)
    daily_temps = temp_numeric.resample('D').agg(['min', 'max']).dropna()
    if daily_temps.empty:
        logger.warning(f"No valid daily min/max temperature data found in '{temp_col_name}' after resampling. Cannot calculate GDD.")
        return pd.DataFrame(columns=['GDD', 'Cumulative_GDD'])

    daily_temps.rename(columns={'min': 'Tmin', 'max': 'Tmax'}, inplace=True)

    # --- Internal function to calculate daily GDD for a row --- 
    def _calc_daily_gdd(row, base, cap):
        tmin = row['Tmin']
        tmax = row['Tmax']

        # Apply base threshold adjustments
        adj_tmin = max(tmin, base)
        adj_tmax = max(tmax, base) # Ensures Tmax >= base if Tmin was adjusted

        # Apply upper cap threshold if specified
        if cap is not None:
            adj_tmax = min(adj_tmax, cap)

        # Ensure Tmin does not exceed Tmax after adjustments
        adj_tmin = min(adj_tmin, adj_tmax)

        # Calculate average of adjusted temperatures
        avg_temp = (adj_tmax + adj_tmin) / 2.0

        # Calculate GDD (cannot be negative)
        gdd = max(0, avg_temp - base)
        return gdd
    # --- End internal function ---

    # Apply the function row-wise to the daily min/max dataframe
    daily_temps['GDD'] = daily_temps.apply(lambda row: _calc_daily_gdd(row, t_base, t_cap), axis=1)

    # Calculate cumulative GDD
    # If you need to reset cumulative sum yearly, uncomment the groupby line
    # daily_temps['Cumulative_GDD'] = daily_temps.groupby(daily_temps.index.year)['GDD'].cumsum()
    daily_temps['Cumulative_GDD'] = daily_temps['GDD'].cumsum()

    logger.info(f"GDD calculation performed using column '{temp_col_name}' (Tbase={t_base}, Tcap={t_cap}).")

    return daily_temps[['GDD', 'Cumulative_GDD']]
# --- END GDD CALCULATION --- 

# --- DIF CALCULATION ---
def calculate_dif(
    temp_series: pd.Series,
    dif_config: dict,
    data_df: pd.DataFrame # Pass the whole DataFrame for lamp status access
) -> pd.Series:
    """Calculate the daily Temperature Differential (DIF).

    DIF = Average Day Temperature - Average Night Temperature.
    Day/Night periods are defined based on the 'day_definition' parameter
    in `dif_config` ('fixed_time' or 'lamp_status').

    See feature report Section 3.4.

    Args:
        temp_series: Pandas Series of temperature data (e.g., air_temp_c in °C).
                     Index must be a DatetimeIndex.
        dif_config: Dictionary containing DIF parameters from config file:
            {'day_definition': str, 'lamp_status_columns': list | None,
             'fixed_time_day_start_hour': int, 'fixed_time_day_end_hour': int}
        data_df: The full DataFrame (must contain columns specified in
                 `dif_config['lamp_status_columns']` if using 'lamp_status').

    Returns:
        pd.Series: Daily DIF values (°C), indexed by day (start of day).
                   Returns an empty Series if calculation cannot be performed due
                   to missing config, missing columns, or errors during processing.
    """
    temp_col_name = temp_series.name if temp_series.name else 'temperature'
    if temp_series is None or temp_series.empty:
        logger.warning(f"Input temperature series '{temp_col_name}' is None or empty. Cannot calculate DIF.")
        return pd.Series(dtype=np.float64)

    if not isinstance(temp_series.index, pd.DatetimeIndex):
        logger.error(f"Input temperature series '{temp_col_name}' must have a DatetimeIndex. Cannot calculate DIF.")
        return pd.Series(dtype=np.float64)

    # Ensure temperature is numeric
    temp_numeric = pd.to_numeric(temp_series, errors='coerce')

    # Create a temporary Series for period assignment, aligning with temp_numeric index
    period_assignment = pd.Series(index=temp_numeric.index, dtype=str)

    # Access Pydantic model attributes directly
    day_definition = getattr(dif_config, 'day_definition', 'fixed_time').lower()

    if day_definition == 'fixed_time':
        logger.info("Calculating DIF using fixed time definition.")
        # Access attributes directly, provide defaults if they might be missing
        day_start = getattr(dif_config, 'fixed_time_day_start_hour', 6)
        day_end = getattr(dif_config, 'fixed_time_day_end_hour', 18)
        hours = temp_numeric.index.hour
        period_assignment = np.where((hours >= day_start) & (hours < day_end), 'Day', 'Night')

    elif day_definition == 'lamp_status':
        logger.info("Calculating DIF using lamp status definition.")
        # Access attribute directly
        lamp_cols = getattr(dif_config, 'lamp_status_columns', None)
        if not lamp_cols:
            logger.error("DIF calculation skipped: 'lamp_status_columns' missing in config for 'lamp_status' definition.")
            return pd.Series(dtype=np.float64)

        # Ensure data_df index matches temp_numeric index for alignment
        if not data_df.index.equals(temp_numeric.index):
             logger.warning("DataFrame index does not match temperature series index for lamp status DIF. Aligning...")
             data_df = data_df.reindex(temp_numeric.index)

        active_lamp_cols = [col for col in lamp_cols if col in data_df.columns]
        if not active_lamp_cols:
             logger.error(f"DIF calculation skipped: None of the required lamp status columns found in DataFrame: {lamp_cols}")
             return pd.Series(dtype=np.float64)
        elif len(active_lamp_cols) < len(lamp_cols):
             logger.warning(f"Missing some lamp status columns for DIF: {list(set(lamp_cols) - set(active_lamp_cols))}. Proceeding with available: {active_lamp_cols}")

        # Assume Day = ANY specified lamp is ON (status == 1)
        lamp_on = pd.Series(index=temp_numeric.index, data=False)
        all_lamp_data_null = True # Flag to track if all input lamp columns are null
        for col in active_lamp_cols:
            # Ensure column is numeric, default non-numeric/NaN to 0 (OFF)
            lamp_status_numeric = pd.to_numeric(data_df[col], errors='coerce').fillna(0)
            lamp_on = lamp_on | (lamp_status_numeric == 1)
            if not data_df[col].isnull().all(): # Check if *any* value was not null before fillna(0)
                 all_lamp_data_null = False

        # --- START FALLBACK LOGIC ---
        if not lamp_on.any() or all_lamp_data_null:
            if all_lamp_data_null:
                 logger.warning("Lamp status columns contain only nulls. Falling back to fixed time for DIF day/night definition.")
            else:
                 logger.warning("No lamp ON status detected for the period. Falling back to fixed time for DIF day/night definition.")
            # Fallback to fixed time definition
            day_start = getattr(dif_config, 'fixed_time_day_start_hour', 6)
            day_end = getattr(dif_config, 'fixed_time_day_end_hour', 18)
            hours = temp_numeric.index.hour
            period_assignment = np.where((hours >= day_start) & (hours < day_end), 'Day', 'Night')
            day_definition = 'fixed_time (fallback)' # Update definition label for logging
        else:
            # Use the lamp status definition if lamps were detected
            period_assignment = np.where(lamp_on, 'Day', 'Night')
        # --- END FALLBACK LOGIC ---

    else:
        logger.error(f"Invalid 'day_definition' in DIF config: '{day_definition}'. Cannot calculate DIF.")
        return pd.Series(dtype=np.float64)

    # Calculate average temperature per period per date
    try:
        # Group temperature series by date and the assigned period
        period_temps = temp_numeric.groupby([temp_numeric.index.date, period_assignment]).mean().unstack()
    except Exception as group_e:
        logger.error(f"Error grouping data for DIF calculation: {group_e}")
        return pd.Series(dtype=np.float64)

    # Calculate DIF
    if 'Day' in period_temps.columns and 'Night' in period_temps.columns:
        dif_daily = period_temps['Day'] - period_temps['Night']
        # Ensure the index is a DatetimeIndex for consistency
        dif_daily.index = pd.to_datetime(dif_daily.index)
        logger.info(f"DIF calculation performed using '{day_definition}' definition.")
        return dif_daily
    else:
        logger.warning("Could not calculate DIF: Missing 'Day' or 'Night' averages for some dates.")
        # Fill missing Day/Night with NaN before returning, ensuring index matches expected days
        expected_days = temp_numeric.resample('D').size().index
        dif_daily = period_temps.reindex(expected_days).apply(lambda row: row.get('Day', np.nan) - row.get('Night', np.nan), axis=1)
        return dif_daily

# --- CO2 DIFFERENCE CALCULATION ---
def calculate_co2_difference(
    measured_co2: pd.Series,
    required_co2: pd.Series
) -> pd.Series:
    """Calculate the difference between measured and required CO2 levels.

    CO2_Difference = co2_measured_ppm - co2_required_ppm.
    Result is the instantaneous difference in ppm.

    See feature report Section 3.6.

    Args:
        measured_co2: Pandas Series of measured CO2 values (ppm).
        required_co2: Pandas Series of required/target CO2 values (ppm).

    Returns:
        pd.Series: The CO2 difference (ppm). Returns NaN where inputs are NaN
                   or non-numeric. Returns empty Series if inputs are None.
    """
    if measured_co2 is None or required_co2 is None:
        logger.warning("Input Series for CO2 difference calculation are None.")
        return pd.Series(dtype=np.float64)

    # Ensure inputs are numeric
    measured_numeric = pd.to_numeric(measured_co2, errors='coerce')
    required_numeric = pd.to_numeric(required_co2, errors='coerce')

    # Calculate difference, result will be NaN if either input was NaN
    co2_diff = measured_numeric - required_numeric

    logger.info("CO2 difference calculation performed.")
    return co2_diff

# --- END CO2 DIFFERENCE CALCULATION ---

# --- ACTUATOR SUMMARY CALCULATIONS ---

def calculate_daily_actuator_summaries(
    data_df: pd.DataFrame,
    summary_config: dict
) -> pd.DataFrame:
    """Calculate daily summaries for specified greenhouse actuator columns.

    Computes:
    1. Daily Average: For columns specified in 'percent_columns_for_average'.
    2. Daily Change Count: For columns specified in 'percent_columns_for_changes'.
       Counts the number of times the value differs from the previous reading.
    3. Daily ON Hours: For columns specified in 'binary_columns_for_on_time'.
       Assumes 1/True is ON, 0/False/NaN is OFF. Calculates total hours ON per day.

    See feature report Section 3.5.

    Args:
        data_df: DataFrame with a DatetimeIndex (sorted ascending) and containing
                 the actuator columns specified in the config.
        summary_config: Dictionary containing actuator summary parameters:
            {'percent_columns_for_average': list[str],
             'percent_columns_for_changes': list[str],
             'binary_columns_for_on_time': list[str]}

    Returns:
        pd.DataFrame: DataFrame indexed by day (start of day), containing the
                      calculated summary columns. Column names are prefixed
                      (e.g., 'Avg_Vent1_%', 'Changes_Vent1_%', 'OnHours_Lamp1').
                      Returns an empty DataFrame if config is missing, index is not
                      a DatetimeIndex, or no valid summaries can be calculated.
    """
    if not summary_config:
        logger.warning("Actuator summary config is empty. Skipping calculations.")
        return pd.DataFrame()

    if not isinstance(data_df.index, pd.DatetimeIndex):
        logger.error("Input DataFrame must have a DatetimeIndex for actuator summaries.")
        return pd.DataFrame()

    if not data_df.index.is_monotonic_increasing:
        logger.warning("Input DataFrame index is not sorted. Sorting for actuator summary calculations.")
        data_df = data_df.sort_index()

    # Create results DataFrame with daily index
    daily_index = data_df.index.normalize().unique()
    results_df = pd.DataFrame(index=daily_index)
    processed_any = False

    # 1. Calculate Daily Average for Percentage Columns
    avg_cols = summary_config.get('percent_columns_for_average', [])
    for col in avg_cols:
        if col in data_df.columns:
            try:
                col_numeric = pd.to_numeric(data_df[col], errors='coerce')
                # Use min_count=1 to return NaN if all inputs for a day are NaN
                daily_avg = col_numeric.resample('D').mean(numeric_only=False) # Keep numeric_only=False if NaNs are expected
                results_df[f'Avg_{col}_Daily'] = daily_avg
                logger.debug(f"Calculated daily average for {col}.")
                processed_any = True
            except Exception as e:
                logger.exception(f"Error calculating daily average for '{col}': {e}") # Log stack trace
        else:
            logger.warning(f"Column '{col}' for daily average not found in DataFrame.")

    # 2. Calculate Daily Change Count for Percentage Columns
    change_cols = summary_config.get('percent_columns_for_changes', [])
    for col in change_cols:
        if col in data_df.columns:
            try:
                col_numeric = pd.to_numeric(data_df[col], errors='coerce')
                # Detect changes from the previous time step (non-zero difference)
                # Fill NaN in diff() to avoid counting NaN transitions as changes initially
                # Then check for non-equal to handle NaN propagation correctly
                changes = col_numeric.diff().ne(0)
                # Sum the number of True values (changes) per day
                daily_changes = changes.resample('D').sum()
                results_df[f'Changes_{col}_Daily'] = daily_changes
                logger.debug(f"Calculated daily change count for {col}.")
                processed_any = True
            except Exception as e:
                logger.exception(f"Error calculating daily changes for '{col}': {e}") # Log stack trace
        else:
            logger.warning(f"Column '{col}' for daily changes not found in DataFrame.")

    # 3. Calculate Daily ON Hours for Binary Columns
    binary_cols = summary_config.get('binary_columns_for_on_time', [])
    for col in binary_cols:
        if col in data_df.columns:
            try:
                # Treat True/1 as ON, False/0/NaN/Other as OFF
                col_binary = pd.to_numeric(data_df[col], errors='coerce').fillna(0)
                status_on = (col_binary == 1)
                # Calculate fraction of day ON using mean (robust to irregular sampling)
                on_fraction_daily = status_on.resample('D').mean(numeric_only=False)
                daily_on_hours = on_fraction_daily * 24
                results_df[f'OnHours_{col}_Daily'] = daily_on_hours
                logger.debug(f"Calculated daily ON hours for {col}.")
                processed_any = True
            except Exception as e:
                logger.exception(f"Error calculating daily ON hours for '{col}': {e}") # Log stack trace
        else:
            logger.warning(f"Column '{col}' for daily ON hours not found in DataFrame.")

    if processed_any:
        logger.info("Actuator summary calculations performed.")
    else:
        logger.warning("No actuator summaries calculated (check config and column availability).")

    # Ensure results_df index matches the expected daily frequency if empty
    if results_df.empty and not daily_index.empty:
         results_df = pd.DataFrame(index=daily_index)

    return results_df

# --- END ACTUATOR SUMMARY CALCULATIONS ---


# --- BASIC DIFFERENCE CALCULATION ---
def calculate_delta(
    series1: pd.Series,
    series2: pd.Series
) -> pd.Series:
    """Calculate the element-wise difference between two series.

    Result = series1 - series2.

    Args:
        series1: The first pandas Series (e.g., inside temperature).
        series2: The second pandas Series (e.g., outside temperature).

    Returns:
        pd.Series: The difference. Returns NaN where inputs are NaN or non-numeric.
                   Returns empty Series if inputs are None.
    """
    name1 = series1.name if series1 is not None and series1.name else 'series1'
    name2 = series2.name if series2 is not None and series2.name else 'series2'
    if series1 is None or series2 is None:
        logger.warning(f"Input Series for delta calculation ({name1}, {name2}) are None.")
        return pd.Series(dtype=np.float64)

    # Ensure inputs are numeric
    num1 = pd.to_numeric(series1, errors='coerce')
    num2 = pd.to_numeric(series2, errors='coerce')

    delta = num1 - num2
    logger.debug(f"Delta calculation performed between {name1} and {name2}.")
    return delta

# --- END BASIC DIFFERENCE CALCULATION ---


# --- RATE OF CHANGE CALCULATION ---
def calculate_rate_of_change(
    series: pd.Series,
    time_index: pd.DatetimeIndex
) -> pd.Series:
    """Calculate the rate of change of a time series per second.

    Rate = (Current Value - Previous Value) / Time Interval (seconds).

    Args:
        series: Pandas Series of time-varying data.
        time_index: Pandas DatetimeIndex corresponding to the series.

    Returns:
        pd.Series: The rate of change per second. The first value will be NaN.
                   Returns NaN for non-numeric inputs or calculation errors.
                   Returns empty Series if inputs are None or invalid.
    """
    series_name = series.name if series is not None and series.name else 'series'
    if series is None:
        logger.warning(f"Input series '{series_name}' is None. Cannot calculate rate of change.")
        return pd.Series(dtype=np.float64)
    if not isinstance(time_index, pd.DatetimeIndex):
        logger.error("Provided time_index is not a DatetimeIndex. Cannot calculate rate of change.")
        return pd.Series(dtype=np.float64)
    if not series.index.equals(time_index):
        logger.warning(f"Series index and time_index do not match for '{series_name}'. Reindexing series.")
        series = series.reindex(time_index)

    # Ensure series is numeric
    series_numeric = pd.to_numeric(series, errors='coerce')

    # Calculate difference in value and time (in seconds)
    value_diff = series_numeric.diff()
    time_diff_seconds = time_index.to_series().diff().dt.total_seconds()

    # Avoid division by zero or NaN time difference
    # Replace 0 or NaN time diffs with NaN to prevent errors and propagate appropriately
    time_diff_seconds_safe = time_diff_seconds.replace(0, np.nan)

    rate_of_change = value_diff / time_diff_seconds_safe

    logger.debug(f"Rate of change calculation performed for {series_name}.")
    return rate_of_change

# --- END RATE OF CHANGE CALCULATION ---


# --- ROLLING AVERAGE CALCULATION ---
def calculate_rolling_average(
    series: pd.Series,
    window_minutes: int,
    time_index: pd.DatetimeIndex
) -> pd.Series:
    """Calculate the rolling average over a specified time window.

    Uses a time-based window (e.g., last X minutes).

    Args:
        series: Pandas Series of time-varying data.
        window_minutes: The size of the rolling window in minutes.
        time_index: Pandas DatetimeIndex for the series, used for time windowing.

    Returns:
        pd.Series: The rolling average. Initial values will be NaN until the
                   window is filled. Returns empty Series if inputs are None/invalid.
    """
    series_name = series.name if series is not None and series.name else 'series'
    if series is None:
        logger.warning(f"Input series '{series_name}' is None. Cannot calculate rolling average.")
        return pd.Series(dtype=np.float64)
    if not isinstance(time_index, pd.DatetimeIndex):
        logger.error("Provided time_index is not a DatetimeIndex. Cannot calculate rolling average.")
        return pd.Series(dtype=np.float64)
    if not series.index.equals(time_index):
        logger.warning(f"Series index and time_index do not match for '{series_name}'. Reindexing series.")
        series = series.reindex(time_index)

    # Ensure series is numeric
    series_numeric = pd.to_numeric(series, errors='coerce')

    # Ensure the series has the DatetimeIndex needed for time-based rolling
    series_numeric.index = time_index

    window_str = f'{window_minutes}min'
    try:
        # Use closed='right' (default) - window includes the current point
        rolling_avg = series_numeric.rolling(window=window_str, closed='right').mean()
        logger.debug(f"Rolling average ({window_str}) calculated for {series_name}.")
        return rolling_avg
    except Exception as e:
        logger.exception(f"Error calculating rolling average for '{series_name}' with window '{window_str}': {e}")
        return pd.Series(dtype=np.float64)

# --- END ROLLING AVERAGE CALCULATION ---


# --- TIME INTEGRAL (TOTAL SUM) CALCULATION ---
def calculate_total_integral(
    series: pd.Series,
    time_index: pd.DatetimeIndex,
    unit_factor: float = 1.0
) -> float:
    """Calculate the total integral of a series over time.

    Approximates the integral by summing (value * time_interval) for each step.
    Useful for calculating total energy (Power * time) or total PAR (PPFD * time).

    Args:
        series: Pandas Series of values (e.g., power in kW, PPFD in umol/m2/s).
        time_index: Pandas DatetimeIndex for the series.
        unit_factor: Optional factor to apply (e.g., 1/3600 to convert kW*seconds to kWh).

    Returns:
        float: The total integrated value over the series duration.
               Returns np.nan if inputs are invalid or calculation fails.
    """
    series_name = series.name if series is not None and series.name else 'series'
    if series is None:
        logger.warning(f"Input series '{series_name}' is None. Cannot calculate total integral.")
        return np.nan
    if not isinstance(time_index, pd.DatetimeIndex):
        logger.error("Provided time_index is not a DatetimeIndex. Cannot calculate total integral.")
        return np.nan
    if not series.index.equals(time_index):
        logger.warning(f"Series index and time_index do not match for '{series_name}'. Reindexing series.")
        series = series.reindex(time_index)

    series_numeric = pd.to_numeric(series, errors='coerce')
    valid_mask = series_numeric.notna()

    # Calculate time interval in seconds
    seconds_per_interval = time_index.to_series().diff().dt.total_seconds().bfill()

    # Calculate value * time_interval for valid points
    value_x_interval = pd.Series(index=time_index, dtype=np.float64)
    value_x_interval[valid_mask] = series_numeric[valid_mask] * seconds_per_interval[valid_mask]

    # Sum over the entire period, ignoring NaNs
    total_sum = value_x_interval.sum(skipna=True)

    result = total_sum * unit_factor
    logger.debug(f"Total integral calculated for {series_name}. Raw sum: {total_sum}, Result: {result}")
    return result

# --- END TIME INTEGRAL CALCULATION ---


# --- MEAN ABSOLUTE DEVIATION CALCULATION ---
def calculate_mean_absolute_deviation(
    measured_series: pd.Series,
    setpoint_series: pd.Series | float
) -> float:
    """Calculate the Mean Absolute Deviation (MAD) from a setpoint.

    MAD = mean(abs(measured_value - setpoint_value))

    Args:
        measured_series: Pandas Series of measured values.
        setpoint_series: Pandas Series of setpoint values (must align with measured_series)
                         or a single float setpoint value.

    Returns:
        float: The calculated MAD. Returns np.nan if inputs are invalid.
    """
    measured_name = measured_series.name if measured_series is not None and measured_series.name else 'measured'
    if measured_series is None:
        logger.warning(f"Input series '{measured_name}' is None. Cannot calculate MAD.")
        return np.nan

    measured_numeric = pd.to_numeric(measured_series, errors='coerce')

    if isinstance(setpoint_series, pd.Series):
        setpoint_name = setpoint_series.name if setpoint_series.name else 'setpoint'
        if not measured_numeric.index.equals(setpoint_series.index):
            logger.warning(f"Measured index and setpoint series index do not match for {measured_name}/{setpoint_name}. Reindexing setpoint.")
            setpoint_series = setpoint_series.reindex(measured_numeric.index)
        setpoint_numeric = pd.to_numeric(setpoint_series, errors='coerce')
    elif isinstance(setpoint_series, (int, float)):
        setpoint_numeric = float(setpoint_series)
        setpoint_name = str(setpoint_numeric)
    else:
        logger.error("Setpoint must be a pandas Series or a single number. Cannot calculate MAD.")
        return np.nan

    absolute_deviation = (measured_numeric - setpoint_numeric).abs()
    mean_abs_deviation = absolute_deviation.mean(skipna=True)

    logger.debug(f"MAD calculated for {measured_name} against setpoint {setpoint_name}.")
    return mean_abs_deviation

# --- END MEAN ABSOLUTE DEVIATION CALCULATION ---


# --- TIME IN RANGE CALCULATION ---
def calculate_time_in_range_percent(
    series: pd.Series,
    lower_bound: float,
    upper_bound: float
) -> float:
    """Calculate the percentage of time a series is within a specified range.

    Range is inclusive: [lower_bound, upper_bound].

    Args:
        series: Pandas Series of measured values.
        lower_bound: The lower bound of the optimal range.
        upper_bound: The upper bound of the optimal range.

    Returns:
        float: Percentage of time (0.0 to 100.0) the series is within the range.
               Returns np.nan if inputs are invalid.
    """
    series_name = series.name if series is not None and series.name else 'series'
    if series is None:
        logger.warning(f"Input series '{series_name}' is None. Cannot calculate time in range.")
        return np.nan

    series_numeric = pd.to_numeric(series, errors='coerce')

    # Create boolean series: True if within bounds
    in_range = (series_numeric >= lower_bound) & (series_numeric <= upper_bound)

    # Calculate mean (True=1, False=0), handling NaNs
    # The mean of the boolean series gives the fraction of time in range (ignoring NaNs)
    fraction_in_range = in_range.mean(skipna=True)

    # Convert fraction to percentage
    percent_in_range = fraction_in_range * 100.0

    logger.debug(f"Time in range [{lower_bound}, {upper_bound}] calculated for {series_name}.")
    return percent_in_range

# --- END TIME IN RANGE CALCULATION ---


# --- STATISTICAL: ROLLING STANDARD DEVIATION ---
def calculate_rolling_std_dev(
    series: pd.Series,
    window_minutes: int,
    time_index: pd.DatetimeIndex
) -> pd.Series:
    """Calculate the rolling standard deviation over a specified time window.

    Uses a time-based window (e.g., last X minutes).

    Args:
        series: Pandas Series of time-varying data.
        window_minutes: The size of the rolling window in minutes.
        time_index: Pandas DatetimeIndex for the series, used for time windowing.

    Returns:
        pd.Series: The rolling standard deviation. Initial values will be NaN until the
                   window is filled. Returns empty Series if inputs are None/invalid.
    """
    series_name = series.name if series is not None and series.name else 'series'
    if series is None:
        logger.warning(f"Input series '{series_name}' is None. Cannot calculate rolling std dev.")
        return pd.Series(dtype=np.float64)
    if not isinstance(time_index, pd.DatetimeIndex):
        logger.error("Provided time_index is not a DatetimeIndex. Cannot calculate rolling std dev.")
        return pd.Series(dtype=np.float64)
    if not series.index.equals(time_index):
        logger.warning(f"Series index and time_index do not match for '{series_name}'. Reindexing series for rolling std dev.")
        series = series.reindex(time_index)

    # Ensure series is numeric
    series_numeric = pd.to_numeric(series, errors='coerce')
    # Ensure the series has the DatetimeIndex needed for time-based rolling
    series_numeric.index = time_index

    window_str = f'{window_minutes}min'
    try:
        rolling_std = series_numeric.rolling(window=window_str, closed='right').std()
        logger.debug(f"Rolling std dev ({window_str}) calculated for {series_name}.")
        return rolling_std
    except Exception as e:
        logger.exception(f"Error calculating rolling std dev for '{series_name}' with window '{window_str}': {e}")
        return pd.Series(dtype=np.float64)

# --- END ROLLING STANDARD DEVIATION ---


# --- STATISTICAL: LAG FEATURE ---
def calculate_lag_feature(
    series: pd.Series,
    lag_minutes: int,
    time_index: pd.DatetimeIndex
) -> pd.Series:
    """Create a lagged version of a time series.

    Finds the value from approximately 'lag_minutes' ago.
    Requires a reasonably regular time index for meaningful results.

    Args:
        series: Pandas Series of time-varying data.
        lag_minutes: The desired lag period in minutes.
        time_index: Pandas DatetimeIndex for the series.

    Returns:
        pd.Series: The lagged series. Initial values will be NaN.
                   Returns empty Series if inputs are None/invalid.
    """
    series_name = series.name if series is not None and series.name else 'series'
    if series is None:
        logger.warning(f"Input series '{series_name}' is None. Cannot calculate lag feature.")
        return pd.Series(dtype=np.float64)
    if not isinstance(time_index, pd.DatetimeIndex):
        logger.error("Provided time_index is not a DatetimeIndex. Cannot calculate lag feature.")
        return pd.Series(dtype=np.float64)
    if not series.index.equals(time_index):
        logger.warning(f"Series index and time_index do not match for '{series_name}'. Reindexing series for lag feature.")
        series = series.reindex(time_index)

    # Ensure series is numeric (lagging non-numeric usually not intended)
    series_numeric = pd.to_numeric(series, errors='coerce')
    # Ensure the series has the DatetimeIndex
    series_numeric.index = time_index

    # Calculate average time delta to estimate number of periods for shift
    avg_delta_seconds = time_index.to_series().diff().mean().total_seconds()
    if pd.isna(avg_delta_seconds) or avg_delta_seconds <= 0:
        logger.warning(f"Could not determine average time delta for '{series_name}'. Cannot calculate reliable lag shifts.")
        # Fallback: Try shifting by 1 period if lag is requested?
        # For now, return empty to indicate failure.
        return pd.Series(dtype=np.float64)

    periods_to_shift = round((lag_minutes * 60) / avg_delta_seconds)
    if periods_to_shift <= 0:
        logger.warning(f"Calculated lag period ({periods_to_shift}) is zero or negative for {lag_minutes} min lag. Cannot shift.")
        return pd.Series(dtype=np.float64)

    try:
        lagged_series = series_numeric.shift(periods=periods_to_shift)
        logger.debug(f"Lag feature ({lag_minutes} min, approx {periods_to_shift} periods) calculated for {series_name}.")
        return lagged_series
    except Exception as e:
        logger.exception(f"Error calculating lag feature for '{series_name}' with lag {lag_minutes} min: {e}")
        return pd.Series(dtype=np.float64)

# --- END LAG FEATURE ---


# --- DOMAIN: DISTANCE FROM OPTIMAL RANGE MIDPOINT ---
def calculate_distance_from_range_midpoint(
    series: pd.Series,
    lower_bound: float,
    upper_bound: float
) -> pd.Series:
    """Calculate the signed distance from the midpoint of an optimal range.

    Distance = Value - (Lower + Upper) / 2
    Positive means above midpoint, negative means below.

    Args:
        series: Pandas Series of measured values.
        lower_bound: The lower bound of the optimal range.
        upper_bound: The upper bound of the optimal range.

    Returns:
        pd.Series: The signed distance from the range midpoint.
                   Returns NaN where input is NaN or non-numeric.
                   Returns empty Series if inputs are None/invalid.
    """
    series_name = series.name if series is not None and series.name else 'series'
    if series is None:
        logger.warning(f"Input series '{series_name}' is None. Cannot calculate distance from midpoint.")
        return pd.Series(dtype=np.float64)

    try:
        midpoint = (float(lower_bound) + float(upper_bound)) / 2.0
    except (TypeError, ValueError):
        logger.error(f"Invalid bounds for midpoint calculation: lower={lower_bound}, upper={upper_bound}")
        return pd.Series(dtype=np.float64)

    series_numeric = pd.to_numeric(series, errors='coerce')
    distance = series_numeric - midpoint

    logger.debug(f"Distance from range midpoint ({midpoint}) calculated for {series_name}.")
    return distance

# --- END DISTANCE FROM OPTIMAL RANGE MIDPOINT ---


# --- DOMAIN: IN OPTIMAL RANGE FLAG ---
def calculate_in_range_flag(
    series: pd.Series,
    lower_bound: float,
    upper_bound: float
) -> pd.Series:
    """Create a binary flag (0 or 1) indicating if a value is within a range.

    Range is inclusive: [lower_bound, upper_bound].
    Flag is 1 if within range, 0 otherwise. NaN inputs result in NaN output.

    Args:
        series: Pandas Series of measured values.
        lower_bound: The lower bound of the optimal range.
        upper_bound: The upper bound of the optimal range.

    Returns:
        pd.Series: Binary flag (0 or 1, dtype float64 to accommodate NaN).
                   Returns empty Series if inputs are None/invalid.
    """
    series_name = series.name if series is not None and series.name else 'series'
    if series is None:
        logger.warning(f"Input series '{series_name}' is None. Cannot calculate in-range flag.")
        return pd.Series(dtype=np.float64)

    try:
        lower = float(lower_bound)
        upper = float(upper_bound)
    except (TypeError, ValueError):
        logger.error(f"Invalid bounds for in-range flag calculation: lower={lower_bound}, upper={upper_bound}")
        return pd.Series(dtype=np.float64)

    series_numeric = pd.to_numeric(series, errors='coerce')

    # Create boolean series: True if within bounds
    in_range_bool = (series_numeric >= lower) & (series_numeric <= upper)

    # Convert boolean to float (1.0 for True, 0.0 for False)
    # NaN inputs in series_numeric will propagate as NaN here.
    in_range_flag = in_range_bool.astype(float)

    logger.debug(f"In-range flag [{lower}, {upper}] calculated for {series_name}.")
    return in_range_flag

# --- END IN OPTIMAL RANGE FLAG ---


# --- DOMAIN: NIGHT STRESS THRESHOLD FLAG ---
def calculate_night_stress_flag(
    temp_series: pd.Series,
    stress_threshold_temp: float,
    dif_config: dict,
    data_df: pd.DataFrame # Pass the whole DataFrame for accessing other columns if needed
) -> pd.Series:
    """Create a binary flag (0 or 1) if night temperature exceeds a stress threshold.

    Uses the same logic as calculate_dif to determine 'Night' periods.
    Flag is 1 if temp > threshold during Night, 0 otherwise (including Day periods).

    Args:
        temp_series: Pandas Series of temperature data (e.g., air_temp_c in °C).
        stress_threshold_temp: The temperature threshold (°C) for stress detection.
        dif_config: Dictionary containing DIF parameters (used for day/night definition).
        data_df: The full DataFrame (needed for 'lamp_status' day definition).

    Returns:
        pd.Series: Binary flag (0 or 1, dtype float64). NaN for invalid inputs.
                   Returns empty Series if inputs are None/invalid or night periods cannot be determined.
    """
    temp_col_name = temp_series.name if temp_series.name else 'temperature'
    if temp_series is None or temp_series.empty:
        logger.warning(f"Input temperature series '{temp_col_name}' is None or empty. Cannot calculate night stress flag.")
        return pd.Series(dtype=np.float64)

    if not isinstance(temp_series.index, pd.DatetimeIndex):
        logger.error(f"Input temperature series '{temp_col_name}' must have a DatetimeIndex. Cannot calculate night stress flag.")
        return pd.Series(dtype=np.float64)

    # Ensure temperature is numeric
    temp_numeric = pd.to_numeric(temp_series, errors='coerce')

    # Determine Day/Night periods (copied & adapted from calculate_dif)
    period_assignment = pd.Series(index=temp_numeric.index, dtype=str)
    day_definition = getattr(dif_config, 'day_definition', 'fixed_time').lower()

    if day_definition == 'fixed_time':
        day_start = getattr(dif_config, 'fixed_time_day_start_hour', 6)
        day_end = getattr(dif_config, 'fixed_time_day_end_hour', 18)
        hours = temp_numeric.index.hour
        period_assignment = np.where((hours >= day_start) & (hours < day_end), 'Day', 'Night')
    elif day_definition == 'lamp_status':
        lamp_cols = getattr(dif_config, 'lamp_status_columns', None)
        if not lamp_cols:
            logger.error("Night stress flag skipped: 'lamp_status_columns' missing in config for 'lamp_status' definition.")
            return pd.Series(dtype=np.float64)
        if not data_df.index.equals(temp_numeric.index):
            logger.warning("Aligning DataFrame index for lamp status night stress check.")
            data_df = data_df.reindex(temp_numeric.index)
        missing_lamp_cols = [col for col in lamp_cols if col not in data_df.columns]
        if missing_lamp_cols:
            logger.error(f"Night stress flag skipped: Required lamp status columns missing: {missing_lamp_cols}")
            return pd.Series(dtype=np.float64)
        lamp_on = pd.Series(index=temp_numeric.index, data=False)
        for col in lamp_cols:
            lamp_status_numeric = pd.to_numeric(data_df[col], errors='coerce').fillna(0)
            lamp_on = lamp_on | (lamp_status_numeric == 1)
        period_assignment = np.where(lamp_on, 'Day', 'Night')
    else:
        logger.error(f"Invalid 'day_definition' in DIF config: '{day_definition}'. Cannot calculate night stress flag.")
        return pd.Series(dtype=np.float64)

    # Calculate flag: 1 if Night and Temp > Threshold, 0 otherwise
    # NaN temperature results in NaN flag
    stress_flag_bool = (period_assignment == 'Night') & (temp_numeric > stress_threshold_temp)
    stress_flag = stress_flag_bool.astype(float) # Convert True/False to 1.0/0.0, keeping NaNs

    logger.debug(f"Night stress flag (Threshold={stress_threshold_temp}°C) calculated using '{day_definition}' definition.")
    return stress_flag

# --- END NIGHT STRESS THRESHOLD FLAG ---