import cudf
import cupy
from typing import List, Dict, Any
import logging

# Configure logging for this module
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')

# Helper UDFs: These receive a window (cudf.Series) that has ALREADY been filled if necessary by the caller
def _udf_quantile(window_s: cudf.Series, q_val: float):
    if window_s.empty: # NaNs should have been filled before this UDF is called by .apply()
        return cupy.nan
    return window_s.quantile(q_val, interpolation='linear')

def _udf_skew(window_s: cudf.Series):
    # NaNs should have been filled before this UDF is called by .apply()
    if window_s.dropna().shape[0] < 3: 
        return cupy.nan
    return window_s.skew() # Series.skew() should handle float dtype

def _udf_kurtosis(window_s: cudf.Series):
    # NaNs should have been filled before this UDF is called by .apply()
    if window_s.dropna().shape[0] < 4: 
        return cupy.nan
    return window_s.kurtosis() # Series.kurtosis() should handle float dtype

class RollingStatisticalFeatures:
    def __init__(self, statistics: List[str] = None, default_min_periods: int = 1):
        """
        Initializes the calculator for rolling statistical features.

        Args:
            statistics: List of statistics to compute.
                        Supported: ['mean', 'std', 'min', 'max', 'median', 'var', 
                                    'skew', 'kurtosis', 'sum', 'quantile_25', 'quantile_50', 'quantile_75'].
                        If None, defaults to a standard set.
            default_min_periods: Default minimum number of observations in window required to have a value.
        """
        self.default_min_periods = default_min_periods
        default_stat_list = ['mean', 'std', 'min', 'max']
        self.allowed_stats = default_stat_list + ['median', 'var', 'skew', 'kurtosis', 'sum', 
                                             'quantile_25', 'quantile_50', 'quantile_75']
        
        self.requested_statistics = statistics if statistics is not None else default_stat_list
        self.statistics_to_compute = [s for s in self.requested_statistics if s in self.allowed_stats]
        
        if len(self.statistics_to_compute) != len(self.requested_statistics):
            unrecognized = set(self.requested_statistics) - set(self.statistics_to_compute)
            logger.warning(f"Some provided statistics were not recognized: {unrecognized}. Computing: {self.statistics_to_compute}")
        
        logger.info(f"RollingStatisticalFeatures initialized to compute: {self.statistics_to_compute}")

    def compute(self, input_series: cudf.Series, window_size_rows: int, feature_label_suffix: str, min_periods: int = None) -> cudf.DataFrame:
        """
        Computes specified rolling statistical features on a GPU Series.

        Args:
            input_series: The cuDF Series to compute features on (e.g., cdf[col]).
                          The series should have a DatetimeIndex if time-based context is important, 
                          though rolling here is by row count.
            window_size_rows: The window size in number of rows.
            feature_label_suffix: Suffix for feature names (e.g., "30m", "2h") to identify the window.
            min_periods: Minimum number of observations in window required to have a value; 
                         defaults to self.default_min_periods.

        Returns:
            A cuDF DataFrame with new feature columns. Each column name will be
            f"{input_series.name}_{statistic_name}_{feature_label_suffix}".
            The DataFrame will have the same index as input_series.
        """
        if input_series.empty:
            logger.warning(f"Input series '{input_series.name if input_series.name else ''}' is empty. Returning empty DataFrame.")
            return cudf.DataFrame(index=input_series.index)

        features_df = cudf.DataFrame(index=input_series.index)
        base_col_name = input_series.name if input_series.name else "value"
        
        current_min_periods = min_periods if min_periods is not None else self.default_min_periods
        actual_min_periods = min(max(1, current_min_periods), window_size_rows)

        # Ensure base series is float64 for consistency, especially for stats like skew/kurtosis
        try:
            series_for_rolling_native_nan = input_series.astype('float64')
        except Exception as e_astype:
            logger.error(f"Error converting series {base_col_name} to float64: {e_astype}. Returning empty features.")
            return features_df

        # For UDFs that error with NaNs in the window for .apply(), create a 0-filled version
        # This is a workaround for the "Handling UDF with null values is not yet supported" error.
        # This filled series will ONLY be used for those specific UDF-based stats.
        series_for_udf_apply = series_for_rolling_native_nan.fillna(0) 

        try:
            # Rolling window object based on series with original NaNs (for direct methods)
            rolling_window_native_nan = series_for_rolling_native_nan.rolling(window=window_size_rows, min_periods=actual_min_periods)
            # Rolling window object based on 0-filled series (for UDF apply methods)
            rolling_window_filled = series_for_udf_apply.rolling(window=window_size_rows, min_periods=actual_min_periods)
        except Exception as e:
            logger.error(f"Error creating rolling window for '{base_col_name}': {e}")
            return features_df 

        for stat_name in self.statistics_to_compute:
            feature_col_name = f"{base_col_name}_{stat_name}_{feature_label_suffix}"
            logger.debug(f"Attempting to compute {stat_name} for {base_col_name}...")
            result_series = cudf.Series([cupy.nan] * len(input_series), dtype='float64', index=input_series.index)
            try:
                if stat_name == 'mean': result_series = rolling_window_native_nan.mean()
                elif stat_name == 'std': result_series = rolling_window_native_nan.std()
                elif stat_name == 'min': result_series = rolling_window_native_nan.min()
                elif stat_name == 'max': result_series = rolling_window_native_nan.max()
                elif stat_name == 'var': result_series = rolling_window_native_nan.var()
                elif stat_name == 'sum': result_series = rolling_window_native_nan.sum()
                # For these, try direct first. If AttributeError, they are not on Rolling obj.
                elif stat_name == 'skew': result_series = rolling_window_native_nan.skew() 
                elif stat_name == 'kurtosis': result_series = rolling_window_native_nan.kurtosis()
                # For these, use .apply() on the 0-filled rolling window
                elif stat_name == 'median': result_series = rolling_window_filled.apply(_udf_quantile, q_val=0.5)
                elif stat_name == 'quantile_25': result_series = rolling_window_filled.apply(_udf_quantile, q_val=0.25)
                elif stat_name == 'quantile_50': result_series = rolling_window_filled.apply(_udf_quantile, q_val=0.50)
                elif stat_name == 'quantile_75': result_series = rolling_window_filled.apply(_udf_quantile, q_val=0.75)
                else:
                    logger.warning(f"Statistic '{stat_name}' is recognized but not handled in dispatch.")
                    continue
                features_df[feature_col_name] = result_series
            except AttributeError as ae:
                 logger.error(f"AttributeError for '{stat_name}' on '{base_col_name}' (window '{feature_label_suffix}'): {ae}. Trying with .apply if applicable or UDF needs fix.")
                 # Fallback for skew/kurtosis if direct call failed (should have used apply from start if known)
                 if stat_name == 'skew':
                     try: result_series = rolling_window_filled.apply(_udf_skew)
                     except Exception as e_apply: logger.error(f"Apply fallback for skew also failed: {e_apply}")
                 elif stat_name == 'kurtosis':
                     try: result_series = rolling_window_filled.apply(_udf_kurtosis)
                     except Exception as e_apply: logger.error(f"Apply fallback for kurtosis also failed: {e_apply}")
                 features_df[feature_col_name] = result_series
            except Exception as e:
                logger.error(f"Generic error computing rolling '{stat_name}' for '{base_col_name}' (window '{feature_label_suffix}'): {e}")
                features_df[feature_col_name] = result_series # Contains NaNs by default
        
        return features_df
