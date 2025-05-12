#!/usr/bin/env -S uv run --isolated
# /// script
# dependencies = ["pandas", "numpy", "matplotlib", "seaborn", "psycopg2-binary"]
# ///

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import sys
from typing import List, Dict

# --- Add path to import db_utils --- 
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Path for db_utils.py (one level up, then into db_connection)
path_to_db_connection_folder = os.path.abspath(os.path.join(current_script_dir, '..', 'db_connection'))

if path_to_db_connection_folder not in sys.path:
    sys.path.insert(0, path_to_db_connection_folder)

try:
    from db_utils import get_db_connection, load_data_from_db 
    print("Successfully imported db_utils from local classes folder.")
except ImportError as e:
    print(f"Error importing db_utils: {e}")
    print(f"sys.path check: {sys.path}")
    print(f"Attempted to load db_utils from: {path_to_db_connection_folder}")
    print("Please ensure db_utils.py exists at this location and has correct hardcoded DB credentials.")
    sys.exit(1)


def plot_data_availability(df: pd.DataFrame, column_name: str, output_dir: str):
    """Plots the data availability for a single column as a line plot where non-NaN is 1, NaN is 0."""
    if column_name not in df.columns:
        print(f"Column '{column_name}' not found in DataFrame. Skipping plot.")
        return

    plt.figure(figsize=(20, 3.5)) # Adjusted for better aspect ratio
    availability_series = df[column_name].notna().astype(int)
    
    plt.plot(df.index, availability_series, label='Data Present (1=Yes, 0=No)', drawstyle='steps-post')
    
    plt.title(f'Data Availability for: {column_name}', fontsize=10)
    plt.ylabel('Available (1) / Missing (0)', fontsize=8)
    plt.xlabel('Time', fontsize=8)
    plt.yticks([0, 1])
    plt.ylim(-0.1, 1.1) 
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m')) # Show Year-Month
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10])) # Ticks every 3 months
    plt.gcf().autofmt_xdate(rotation=45, ha='right') 
    plt.tick_params(axis='both', which='major', labelsize=8)

    plt.legend(fontsize=8)
    plt.tight_layout()
    
    plot_filename = os.path.join(output_dir, f"availability_{column_name.replace('/', '_').replace('%', 'pct')}.png")
    try:
        plt.savefig(plot_filename, dpi=150)
        print(f"Availability plot for {column_name} saved to {plot_filename}")
    except Exception as e_save:
        print(f"Error saving plot for {column_name}: {e_save}")
    plt.close()

def plot_missing_data_heatmap_grouped(df: pd.DataFrame, column_group: List[str], group_name: str, output_dir: str, resample_freq='D'):
    """Plots a heatmap of missing data for a group of columns, resampled by frequency."""
    if not column_group:
        print(f"No columns provided for missing data heatmap group: {group_name}.")
        return
    
    # Ensure only existing columns are used
    existing_columns = [col for col in column_group if col in df.columns]
    if not existing_columns:
        print(f"None of the columns in group '{group_name}' found in DataFrame. Skipping heatmap.")
        return
    
    df_subset = df[existing_columns].copy()
    if df_subset.empty:
        print(f"DataFrame subset for heatmap group '{group_name}' is empty.")
        return

    # Resample: 1 if all data points in period are NaN, 0 otherwise
    df_resampled_isnull = df_subset.resample(resample_freq).apply(lambda x: 1 if x.isnull().all() else 0).astype(int)

    if df_resampled_isnull.empty:
        print(f"Resampled data for heatmap group '{group_name}' is empty (freq '{resample_freq}').")
        return

    plt.figure(figsize=(20, max(4, len(existing_columns) * 0.35))) 
    sns.heatmap(df_resampled_isnull.transpose(), cbar=False, cmap="binary_r", yticklabels=True, vmin=0, vmax=1)
    plt.title(f"Missing Data Heatmap: {group_name} (1 = Fully Missing in Period '{resample_freq}')", fontsize=12)
    plt.xlabel("Time Period", fontsize=10)
    plt.ylabel("Sensor Column", fontsize=10)
    plt.xticks(fontsize=8, rotation=90)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    
    plot_filename = os.path.join(output_dir, f"heatmap_missing_{group_name.replace(' ', '_').lower()}_{resample_freq}.png")
    try:
        plt.savefig(plot_filename, dpi=150)
        print(f"Missing data heatmap for group '{group_name}' saved to {plot_filename}")
    except Exception as e_save:
        print(f"Error saving heatmap for group '{group_name}': {e_save}")
    plt.close()


if __name__ == '__main__':
    print("Running Data Availability Analysis Script...")
    sns.set_theme(style="whitegrid")

    output_image_dir = os.path.join(os.path.dirname(__file__), "images")
    os.makedirs(output_image_dir, exist_ok=True)
    print(f"Saving plots to: {output_image_dir}")

    # --- Database Connection and Data Loading ---
    df_full_data = None
    connection = None
    all_columns_from_db = [] # To store actual column names from DB
    try:
        connection = get_db_connection()
        query = "SELECT * FROM public.sensor_data_merged ORDER BY time ASC;"
        print(f"Executing query: {query}")
        df_full_data = load_data_from_db(query, connection)
        print(f"Loaded {len(df_full_data)} rows from sensor_data_merged.")
        if not df_full_data.empty:
            all_columns_from_db = df_full_data.columns.tolist()
    except Exception as e:
        print(f"Error loading data from database: {e}")
        print("Ensure database connection details are set correctly in Jupyter/classes/db_connection/db_utils.py.")
        sys.exit(1)
    finally:
        if connection is not None and not connection.closed:
            try: connection.close(); print("Database connection closed.")
            except Exception as close_e: print(f"Error closing database connection: {close_e}")

    if df_full_data is None or df_full_data.empty:
        print("No data loaded from database. Exiting.")
        sys.exit(1)

    try:
        df_full_data['time'] = pd.to_datetime(df_full_data['time'])
        df_full_data = df_full_data.set_index('time')
    except Exception as e_time:
        print(f"Error processing time column: {e_time}. Ensure 'time' column exists and is datetime convertible.")
        sys.exit(1)
    
    # --- Define Column Groups for Heatmaps --- 
    # Exclude pure metadata columns unless specifically needed for an availability check
    metadata_cols = ['source_system', 'source_file', 'format_type', 'uuid', 'time'] # 'time' is index now
    
    # All columns except metadata, for heatmap grouping
    data_cols_for_heatmap = [col for col in all_columns_from_db if col not in metadata_cols]

    # Split data_cols_for_heatmap into manageable chunks for heatmaps (e.g., 15-20 per heatmap)
    chunk_size = 20 
    column_chunks = [data_cols_for_heatmap[i:i + chunk_size] for i in range(0, len(data_cols_for_heatmap), chunk_size)]

    print(f"\n--- Generating Missing Data Heatmaps (Daily Resampling) ---")
    for i, chunk in enumerate(column_chunks):
        plot_missing_data_heatmap_grouped(df_full_data, chunk, f"Group_{i+1}", output_image_dir, resample_freq='D')

    # --- Generate Individual Availability Line Plots for Specific Key Columns ---
    # These are columns you want a detailed line plot for, ensure they exist.
    KEY_COLUMNS_FOR_LINE_PLOTS = [
        'air_temp_c', 'co2_measured_ppm', 'relative_humidity_percent', 'radiation_w_m2',
        'light_intensity_umol', 'heating_setpoint_c', 'pipe_temp_1_c',
        'vent_lee_afd3_percent', 'vent_wind_afd3_percent',
        'rain_status', 'lamp_grp1_no3_status' # Example boolean/status columns
    ]
    print(f"\n--- Generating Individual Data Availability Line Plots ---")
    existing_key_cols = [col for col in KEY_COLUMNS_FOR_LINE_PLOTS if col in df_full_data.columns]
    if not existing_key_cols:
        print("None of the specified KEY_COLUMNS_FOR_LINE_PLOTS found in the DataFrame.")
    else:
        for col_name in existing_key_cols:
            plot_data_availability(df_full_data, col_name, output_image_dir)

    print("\nData availability analysis script finished.") 