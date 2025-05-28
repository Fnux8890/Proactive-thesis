#!/usr/bin/env -S uv run --isolated
# /// script
# dependencies = ["pandas", "numpy", "matplotlib", "seaborn", "psycopg2-binary"]
# ///

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# --- Add path to import db_utils --- 
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Path for db_utils.py (one level up, then into db_connection)
path_to_db_connection_folder = os.path.abspath(os.path.join(current_script_dir, '..', 'db_connection'))

if path_to_db_connection_folder not in sys.path:
    sys.path.insert(0, path_to_db_connection_folder)

try:
    # This import should now find the db_utils.py in the ../db_connection/ directory
    from db_utils import get_db_connection, load_data_from_db 
    print("Successfully imported db_utils from local classes folder.")
except ImportError as e:
    print(f"Error importing db_utils: {e}")
    print(f"sys.path check: {sys.path}")
    print(f"Attempted to load db_utils from: {path_to_db_connection_folder}")
    print("Please ensure db_utils.py exists at this location and has correct hardcoded DB credentials.")
    sys.exit(1)

class OutlierDetector:
    """A class to encapsulate outlier detection methods."""

    def __init__(self):
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = (15, 7)

    def identify_iqr_outliers(self, series: pd.Series, factor: float = 1.5) -> tuple[pd.Series, float, float]:
        if not isinstance(series, pd.Series):
            raise TypeError("Input must be a pandas Series.")
        if series.empty or series.isnull().all():
             return pd.Series([False] * len(series), index=series.index, dtype=bool), np.nan, np.nan

        series_cleaned = series.dropna()
        if series_cleaned.empty:
            return pd.Series([False] * len(series), index=series.index, dtype=bool), np.nan, np.nan

        q1 = series_cleaned.quantile(0.25)
        q3 = series_cleaned.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr

        outliers_cleaned = (series_cleaned < lower_bound) | (series_cleaned > upper_bound)
        
        outlier_mask_aligned = pd.Series(False, index=series.index, dtype=bool)
        if not series_cleaned.empty:
            outlier_mask_aligned.loc[series_cleaned.index] = outliers_cleaned
        
        return outlier_mask_aligned, lower_bound, upper_bound

    def identify_rolling_zscore_outliers(self, 
                                         series: pd.Series, 
                                         window_size: int = 24, 
                                         threshold: float = 3.0) -> tuple[pd.Series, pd.Series, pd.Series]:
        if not isinstance(series, pd.Series):
            raise TypeError("Input must be a pandas Series.")
        if series.empty or series.isnull().all():
            return pd.Series([False] * len(series), index=series.index), pd.Series(np.nan, index=series.index), pd.Series(np.nan, index=series.index)

        if len(series.dropna()) < 1: 
            return pd.Series([False] * len(series), index=series.index), pd.Series(np.nan, index=series.index), pd.Series(np.nan, index=series.index)

        rolling_mean = series.rolling(window=window_size, min_periods=1).mean()
        rolling_std = series.rolling(window=window_size, min_periods=1).std()

        rolling_std_safe = rolling_std.replace(0, np.nan).ffill().bfill()
        
        if rolling_std_safe.isnull().all():
            z_scores = pd.Series(np.nan, index=series.index)
        else:
            z_scores = (series - rolling_mean) / rolling_std_safe
        
        outliers = np.abs(z_scores) > threshold
        outliers = outliers.fillna(False)

        return outliers, rolling_mean, rolling_std

if __name__ == '__main__':
    print("Running Outlier Detection on actual data from database (using Jupyter/classes/db_utils.py)...")
    detector = OutlierDetector()

    output_image_dir = os.path.join(os.path.dirname(__file__), "images")
    os.makedirs(output_image_dir, exist_ok=True)
    print(f"Saving plots to: {output_image_dir}")

    # --- Hardcoded Default Parameters for this standalone script ---
    DEFAULT_IQR_FACTOR = 1.5
    DEFAULT_ZSCORE_WINDOW = 24 * 7 # Default to weekly window for hourly-like data
    DEFAULT_ZSCORE_THRESHOLD = 3.0
    
    # Updated list based on actual sensor_data_merged schema
    DEFAULT_POTENTIAL_NUMERIC_COLS = [
        'air_temp_c', 'air_temp_middle_c', 'outside_temp_c', 
        'relative_humidity_percent', 'humidity_deficit_g_m3', 
        'radiation_w_m2', 'light_intensity_lux', 'light_intensity_umol', 'outside_light_w_m2',
        'co2_measured_ppm', 'co2_required_ppm', 'co2_dosing_status', 'co2_status',
        # 'rain_status', # boolean, handle separately if needed for numeric analysis
        'vent_pos_1_percent', 'vent_pos_2_percent', 
        'vent_lee_afd3_percent', 'vent_wind_afd3_percent', 'vent_lee_afd4_percent', 'vent_wind_afd4_percent',
        'curtain_1_percent', 'curtain_2_percent', 'curtain_3_percent', 'curtain_4_percent',
        'window_1_percent', 'window_2_percent',
        # 'lamp_grp1_no3_status', etc. (booleans)
        'heating_setpoint_c', 'pipe_temp_1_c', 'pipe_temp_2_c', 'flow_temp_1_c', 'flow_temp_2_c',
        'temperature_forecast_c', 'sun_radiation_forecast_w_m2', 
        'temperature_actual_c', 'sun_radiation_actual_w_m2',
        'vpd_hpa',
        'humidity_deficit_afd3_g_m3', 'relative_humidity_afd3_percent', 
        'humidity_deficit_afd4_g_m3', 'relative_humidity_afd4_percent',
        'behov', # integer
        'timer_on', 'timer_off', # integer
        'dli_sum', 
        'lampe_timer_on', 'lampe_timer_off', # bigint
        'value' 
    ]
    # Columns to perform outlier analysis on FOR THIS SCRIPT RUN
    DEFAULT_ACTUAL_COLUMNS_TO_ANALYZE = ['air_temp_c', 'co2_measured_ppm', 'relative_humidity_percent', 'radiation_w_m2'] 

    print(f"Using parameters: IQR Factor={DEFAULT_IQR_FACTOR}, Z-Window={DEFAULT_ZSCORE_WINDOW}, Z-Threshold={DEFAULT_ZSCORE_THRESHOLD}")
    print(f"Columns to analyze for outliers: {DEFAULT_ACTUAL_COLUMNS_TO_ANALYZE}")

    df_real_data = None
    connection = None # Renamed from db_engine for clarity as it's a raw psycopg2 conn here
    try:
        connection = get_db_connection() # Uses hardcoded values from db_utils.py
        query = "SELECT * FROM public.sensor_data_merged ORDER BY time ASC;" 
        print(f"Executing query: {query}")
        df_real_data = load_data_from_db(query, connection)
        print(f"Loaded {len(df_real_data)} rows from sensor_data_merged.")
    except Exception as e:
        print(f"Error loading data from database: {e}")
        print("Ensure database connection details are set correctly in Jupyter/classes/db_connection/db_utils.py.")
        sys.exit(1)
    finally:
        if connection is not None and not connection.closed:
            # Assuming db_utils.get_db_connection returns a psycopg2 connection which has a close() method
            try:
                connection.close()
                print("Database connection closed.")
            except Exception as close_e:
                print(f"Error closing database connection: {close_e}")

    if df_real_data is None or df_real_data.empty:
        print("No data loaded from database. Exiting.")
        sys.exit(1)

    df_real_data['time'] = pd.to_datetime(df_real_data['time'])
    df_real_data = df_real_data.set_index('time')

    columns_to_convert = DEFAULT_POTENTIAL_NUMERIC_COLS
    print(f"Attempting to convert columns to numeric: {columns_to_convert}")
    for col in columns_to_convert:
        if col in df_real_data.columns:
            df_real_data[col] = pd.to_numeric(df_real_data[col], errors='coerce')
        else:
            print(f"Warning: Column '{col}' for numeric conversion (from DEFAULT_POTENTIAL_NUMERIC_COLS) not found in DataFrame.")

    ACTUAL_COLUMNS_TO_ANALYZE = DEFAULT_ACTUAL_COLUMNS_TO_ANALYZE

    for col_name in ACTUAL_COLUMNS_TO_ANALYZE:
        if col_name not in df_real_data.columns:
            print(f"Column '{col_name}' for analysis not found in loaded data. Skipping.")
            continue
        
        if not pd.api.types.is_numeric_dtype(df_real_data[col_name]):
            print(f"Column '{col_name}' is not numeric after conversion attempts. Skipping outlier analysis.")
            continue
        
        series_to_analyze = df_real_data[col_name].dropna()
        if series_to_analyze.empty:
            print(f"Column '{col_name}' has no non-NaN data after dropping NaNs. Skipping.")
            continue

        print(f"\n--- Analyzing column: {col_name} ---")

        print(f"Running IQR Outlier Detection for {col_name} (Factor={DEFAULT_IQR_FACTOR})...")
        iqr_outliers, low_b, high_b = detector.identify_iqr_outliers(df_real_data[col_name], factor=DEFAULT_IQR_FACTOR)
        
        plt.figure()
        plt.plot(df_real_data.index, df_real_data[col_name], label='Original Data', alpha=0.7, zorder=1)
        if not df_real_data[col_name][iqr_outliers].empty:
            plt.scatter(df_real_data.index[iqr_outliers], df_real_data[col_name][iqr_outliers], color='red', s=50, label='IQR Outlier', zorder=2)
        if pd.notnull(low_b) and pd.notnull(high_b):
            plt.axhline(y=high_b, color='orange', linestyle='--', label=f'Upper Bound ({high_b:.2f})')
            plt.axhline(y=low_b, color='orange', linestyle='--', label=f'Lower Bound ({low_b:.2f})')
        plt.title(f'IQR Outlier Detection: {col_name}')
        plt.xlabel('Time')
        plt.ylabel(col_name)
        plt.legend()
        plt.tight_layout()
        plot_filename = os.path.join(output_image_dir, f"outliers_iqr_{col_name.replace('/', '_')}.png")
        plt.savefig(plot_filename)
        print(f"IQR plot for {col_name} saved to {plot_filename}")
        plt.close()

        z_window = DEFAULT_ZSCORE_WINDOW
        z_thresh = DEFAULT_ZSCORE_THRESHOLD
        print(f"Running Rolling Z-score Outlier Detection for {col_name} (Window={z_window}, Threshold={z_thresh})...")
        z_outliers, roll_m, roll_s = detector.identify_rolling_zscore_outliers(df_real_data[col_name], window_size=z_window, threshold=z_thresh)
        
        plt.figure()
        plt.plot(df_real_data.index, df_real_data[col_name], label='Original Data', alpha=0.6, zorder=1)
        if roll_m is not None and not roll_m.isnull().all():
            plt.plot(roll_m.index, roll_m, label=f'Rolling Mean (w={z_window})', color='cyan', linestyle=':', zorder=2)
        if roll_m is not None and roll_s is not None and not roll_m.isnull().all() and not roll_s.isnull().all():
            upper_z_bound = roll_m + z_thresh * roll_s
            lower_z_bound = roll_m - z_thresh * roll_s
            plt.plot(upper_z_bound.index, upper_z_bound, label=f'Upper Z-Bound', color='orange', linestyle='--', zorder=3)
            plt.plot(lower_z_bound.index, lower_z_bound, label=f'Lower Z-Bound', color='orange', linestyle='--', zorder=3)
        if not df_real_data[col_name][z_outliers].empty:
            plt.scatter(df_real_data.index[z_outliers], df_real_data[col_name][z_outliers], color='red', s=50, label=f'Z-score Outlier (t={z_thresh})', zorder=4)
        plt.title(f'Rolling Z-score Detection: {col_name} (Win={z_window}, Thr={z_thresh})')
        plt.xlabel('Time')
        plt.ylabel(col_name)
        plt.legend()
        plt.tight_layout()
        plot_filename = os.path.join(output_image_dir, f"outliers_zscore_{col_name.replace('/', '_')}.png")
        plt.savefig(plot_filename)
        print(f"Z-score plot for {col_name} saved to {plot_filename}")
        plt.close()

    print("\nOutlier detection script finished.") 