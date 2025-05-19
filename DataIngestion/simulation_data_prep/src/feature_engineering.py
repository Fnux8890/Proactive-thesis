# feature_engineering.py
# import pandas as pd # Removed unused import
import numpy as np
import polars as pl
import logging
from datetime import datetime
# from . import config # To access feature engineering parameters like lag periods, window sizes

logger = logging.getLogger(__name__)

def create_time_features(df: pl.DataFrame, time_col: str = "time") -> pl.DataFrame:
    """Creates time-based and cyclical features from a datetime column using Polars.

    Args:
        df: Input Polars DataFrame.
        time_col: Name of the datetime column (must be pl.Datetime dtype).

    Returns:
        Polars DataFrame with added time-based features.
    """
    if time_col not in df.columns:
        logger.error(f"Time column '{time_col}' not found in DataFrame. Cannot create time features.")
        return df
    
    if df[time_col].dtype != pl.Datetime:
        logger.warning(f"Time column '{time_col}' is not of Polars Datetime type (is {df[time_col].dtype}). Attempting conversion.")
        try:
            df = df.with_columns(pl.col(time_col).str.to_datetime().alias(time_col))
        except Exception as e:
            logger.error(f"Failed to convert time column '{time_col}' to Datetime: {e}. Cannot create time features.")
            return df
        if df[time_col].dtype != pl.Datetime:
            logger.error(f"Conversion of time column '{time_col}' to Datetime failed. Final type: {df[time_col].dtype}. Cannot create time features.")
            return df

    logger.info(f"Creating time-based features from column: {time_col}")

    df_with_features = df.with_columns([
        pl.col(time_col).dt.hour().alias("hour_of_day"),
        pl.col(time_col).dt.weekday().alias("day_of_week"),        # 1-7 for Monday-Sunday
        pl.col(time_col).dt.day().alias("day_of_month"),
        pl.col(time_col).dt.month().alias("month_of_year"),
        pl.col(time_col).dt.year().alias("year"),
        pl.col(time_col).dt.ordinal_day().alias("day_of_year"),    # Day of year (1-366)
        pl.col(time_col).dt.week().alias("week_of_year"),      # Week of year (1-53)
        # Note: Polars might not have a direct quarter. If needed, can derive from month:
        # ((pl.col(time_col).dt.month() - 1) / 3).floor().cast(pl.Int8) + 1).alias("quarter_of_year")
    ])

    # Cyclical features
    # For day_of_week, Polars weekday() is 1-7. For sin/cos, usually 0-6 or 0-N-1 is easier.
    # Adjusting day_of_week to be 0-6 for cyclical calculation (0=Monday, 6=Sunday)
    df_with_features = df_with_features.with_columns([
        (np.sin(2 * np.pi * pl.col("hour_of_day") / 24.0)).alias("hour_sin"),
        (np.cos(2 * np.pi * pl.col("hour_of_day") / 24.0)).alias("hour_cos"),
        (np.sin(2 * np.pi * (pl.col("day_of_week") - 1) / 7.0)).alias("dayofweek_sin"), 
        (np.cos(2 * np.pi * (pl.col("day_of_week") - 1) / 7.0)).alias("dayofweek_cos"),
        (np.sin(2 * np.pi * pl.col("month_of_year") / 12.0)).alias("month_sin"),
        (np.cos(2 * np.pi * pl.col("month_of_year") / 12.0)).alias("month_cos"),
    ])
    
    logger.info(f"Added time features. New columns: {df_with_features.columns[-10:]}") # Show last few added cols
    return df_with_features

# TODO: Function create_lag_features(df, cols_to_lag, lag_periods) -> pd.DataFrame.
# Example:
# def create_lag_features(df: pd.DataFrame, columns_to_lag: list, lag_periods: list) -> pd.DataFrame:
#     """Creates lag features for specified columns and lag periods."""
#     df_features = df.copy()
#     for col in columns_to_lag:
#         if col in df_features.columns:
#             for lag in lag_periods:
#                 df_features[f'{col}_lag_{lag}'] = df_features[col].shift(lag)
#     return df_features

# TODO: Function create_rolling_window_features(df, cols_to_roll, windows, aggs) -> pd.DataFrame.
# Example:
# def create_rolling_window_features(df: pd.DataFrame, columns_to_roll: list, window_sizes: list, aggregations: list) -> pd.DataFrame:
#     """Creates rolling window features for specified columns, window sizes, and aggregations."""
#     df_features = df.copy()
#     for col in columns_to_roll:
#         if col in df_features.columns:
#             for window in window_sizes:
#                 for agg_func_name in aggregations:
#                     # Pandas rolling().agg() can take string names of common functions
#                     # For numpy functions, you might need to pass the function object itself e.g. np.mean, np.std
#                     try:
#                         # Attempt to use the aggregation function string directly
#                         rolled_series = df_features[col].rolling(window=window, min_periods=1).agg(agg_func_name)
#                         df_features[f'{col}_rolling_{agg_func_name}_{window}'] = rolled_series
#                     except Exception as e:
#                         print(f"Could not apply '{agg_func_name}' directly for column '{col}' with window '{window}'. Error: {e}")
#                         # Example for numpy functions if direct string doesn't work or for custom functions
#                         # if agg_func_name == 'mean':
#                         #     df_features[f'{col}_rolling_mean_{window}'] = df_features[col].rolling(window=window, min_periods=1).mean()
#                         # elif agg_func_name == 'std':
#                         #     df_features[f'{col}_rolling_std_{window}'] = df_features[col].rolling(window=window, min_periods=1).std()
#     return df_features 

if __name__ == '__main__':
    # Example Usage:
    logging.basicConfig(level=logging.INFO)
    start_time = datetime(2023, 1, 1, 0, 0, 0)
    periods = 48
    interval = "1h"

    sample_df = pl.DataFrame({
        "time": pl.datetime_range(start_time, None, interval=interval, eager=True).head(periods),
        "some_value": np.random.rand(periods) * 100
    })

    logger.info(f"Original DataFrame:\n{sample_df.head()}")
    
    df_with_time_features = create_time_features(sample_df, time_col="time")
    logger.info(f"DataFrame with Time Features:\n{df_with_time_features.head()}")
    print(df_with_time_features.select(pl.all().exclude("some_value")).head()) 