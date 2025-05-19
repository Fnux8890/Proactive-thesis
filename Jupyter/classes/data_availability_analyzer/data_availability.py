#!/usr/bin/env -S uv run --isolated
# /// script
# dependencies = ["pandas", "numpy", "matplotlib", "seaborn", "psycopg2-binary", "sqlalchemy"]
# ///

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import sys
from typing import List, Dict
import pathlib

# --- Add path to import db_utils --- 
current_script_dir = pathlib.Path(__file__).parent
# Path for db_utils.py (one level up, then into db_connection)
path_to_db_connection_folder = (current_script_dir / '..' / 'db_connection').resolve()

if str(path_to_db_connection_folder) not in sys.path:
    sys.path.insert(0, str(path_to_db_connection_folder))

try:
    from db_utils import get_db_engine, load_data_from_db 
    # print("Successfully imported db_utils from local classes folder.") # Reduced print
except ImportError as e:
    print(f"Critical Error: Could not import db_utils: {e}", file=sys.stderr)
    print(f"sys.path check: {sys.path}", file=sys.stderr)
    print(f"Attempted to load db_utils from: {path_to_db_connection_folder}", file=sys.stderr)
    print("Please ensure db_utils.py exists at this location and has correct hardcoded DB credentials.", file=sys.stderr)
    sys.exit(1)


def plot_data_availability(df: pd.DataFrame, column_name: str, output_dir: str):
    """Plots the data availability for a single column as a line plot where non-NaN is 1, NaN is 0."""
    if column_name not in df.columns:
        print(f"Warning: Column '{column_name}' not found in DataFrame. Skipping plot.", file=sys.stderr)
        return

    plt.figure(figsize=(20, 3.5)) 
    availability_series = df[column_name].notna().astype(int)
    
    plt.plot(df.index, availability_series, label='Data Present (1=Yes, 0=No)', drawstyle='steps-post')
    
    plt.title(f'Data Availability for: {column_name}', fontsize=10)
    plt.ylabel('Available (1) / Missing (0)', fontsize=8)
    plt.xlabel('Time', fontsize=8)
    plt.yticks([0, 1])
    plt.ylim(-0.1, 1.1) 
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    plt.gcf().autofmt_xdate(rotation=45, ha='right') 
    plt.tick_params(axis='both', which='major', labelsize=8)

    plt.legend(fontsize=8)
    plt.tight_layout()
    
    plot_filename = os.path.join(output_dir, f"availability_{column_name.replace('/', '_').replace('%', 'pct')}.png")
    try:
        plt.savefig(plot_filename, dpi=150)
        # print(f"Availability plot for {column_name} saved to {plot_filename}") # Reduced print
    except Exception as e_save:
        print(f"Error saving plot for {column_name}: {e_save}", file=sys.stderr)
    plt.close()

def plot_missing_data_heatmap_grouped(df: pd.DataFrame, column_group: List[str], group_name: str, output_dir: str, resample_freq='D'):
    """Plots a heatmap of missing data for a group of columns, resampled by frequency."""
    if not column_group:
        print(f"Warning: No columns provided for missing data heatmap group: {group_name}.", file=sys.stderr)
        return
    
    existing_columns = [col for col in column_group if col in df.columns]
    if not existing_columns:
        print(f"Warning: None of the columns in group '{group_name}' found in DataFrame. Skipping heatmap.", file=sys.stderr)
        return
    
    df_subset = df[existing_columns].copy()
    if df_subset.empty:
        print(f"Warning: DataFrame subset for heatmap group '{group_name}' is empty.", file=sys.stderr)
        return

    df_resampled_isnull = df_subset.resample(resample_freq).apply(lambda x: 1 if x.isnull().all() else 0).astype(int)

    if df_resampled_isnull.empty:
        print(f"Warning: Resampled data for heatmap group '{group_name}' is empty (freq '{resample_freq}').", file=sys.stderr)
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
        # print(f"Missing data heatmap for group '{group_name}' saved to {plot_filename}") # Reduced print
    except Exception as e_save:
        print(f"Error saving heatmap for group '{group_name}': {e_save}", file=sys.stderr)
    plt.close()

def plot_overall_data_availability(df: pd.DataFrame, data_columns: List[str], output_dir: str):
    """Plots the overall data availability across a list of specified data columns."""
    if not data_columns:
        print("Warning: No data columns provided for overall availability plot. Skipping.", file=sys.stderr)
        return

    existing_data_columns = [col for col in data_columns if col in df.columns]
    if not existing_data_columns:
        print("Warning: None of the specified data columns exist in the DataFrame for overall availability plot. Skipping.", file=sys.stderr)
        return
    
    plt.figure(figsize=(20, 3.5))
    # 1 if all specified columns are present, 0 otherwise
    overall_availability_series = df[existing_data_columns].notna().all(axis=1).astype(int)
    
    plt.plot(df.index, overall_availability_series, label='All Data Columns Present (1=Yes, 0=No)', drawstyle='steps-post', color='green')
    
    plt.title('Overall Data Availability (All Specified Data Columns)', fontsize=10)
    plt.ylabel('All Present (1) / Any Missing (0)', fontsize=8)
    plt.xlabel('Time', fontsize=8)
    plt.yticks([0, 1])
    plt.ylim(-0.1, 1.1)
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    plt.gcf().autofmt_xdate(rotation=45, ha='right')
    plt.tick_params(axis='both', which='major', labelsize=8)

    plt.legend(fontsize=8)
    plt.tight_layout()
    
    plot_filename = os.path.join(output_dir, "availability_overall_dataset.png")
    try:
        plt.savefig(plot_filename, dpi=150)
        # print(f"Overall availability plot saved to {plot_filename}") # Reduced print
    except Exception as e_save:
        print(f"Error saving overall availability plot: {e_save}", file=sys.stderr)
    plt.close()

def plot_stacked_area_availability(df: pd.DataFrame, data_columns: List[str], output_dir: str, resample_freq: str | None = 'D'):
    """Plots a stacked area chart of data availability for the given columns, with optional resampling."""
    if not data_columns:
        print("Warning: No data columns provided for stacked area availability plot. Skipping.", file=sys.stderr)
        return

    existing_data_columns = [col for col in data_columns if col in df.columns]
    if not existing_data_columns:
        print("Warning: None of the specified data columns exist in the DataFrame for stacked area plot. Skipping.", file=sys.stderr)
        return
    
    plot_df = df[existing_data_columns].copy()

    # Resample data if a frequency is provided
    resampled_index = plot_df.index
    if resample_freq:
        try:
            # For each column, 1 if any data present in period, 0 if all missing
            availability_df_resampled = plot_df.notna().resample(resample_freq).max().astype(int)
            if availability_df_resampled.empty:
                print(f"Warning: Resampled data for stacked area plot is empty (freq '{resample_freq}'). Skipping plot.", file=sys.stderr)
                return
            availability_series_list = [availability_df_resampled[col].values for col in existing_data_columns]
            resampled_index = availability_df_resampled.index
        except Exception as e_resample:
            print(f"Error during resampling for stacked area plot (freq '{resample_freq}'): {e_resample}. Attempting to plot without resampling.", file=sys.stderr)
            availability_series_list = [plot_df[col].notna().astype(int).values for col in existing_data_columns]
            resample_freq = None # Clear resample_freq if resampling failed
    else:
        availability_series_list = [plot_df[col].notna().astype(int).values for col in existing_data_columns]
        
    if not any(s.any() for s in availability_series_list):
        print("Warning: All specified columns have no data (possibly after resampling) for the stacked area plot. Skipping plot.", file=sys.stderr)
        return

    plt.figure(figsize=(20, 7))
    
    plot_title = f'Stacked Data Availability ({len(existing_data_columns)} Columns)'
    if resample_freq:
        plot_title += f' - {resample_freq} Resampled'

    try:
        plt.stackplot(resampled_index, availability_series_list, labels=existing_data_columns, alpha=0.7)
    except Exception as e_stackplot:
        print(f"Error during stackplot generation: {e_stackplot}", file=sys.stderr)
        print("Falling back to plotting total count of available columns due to stackplot error.", file=sys.stderr)
        # Prepare sum for fallback based on whether data was resampled or not
        if resample_freq and 'availability_df_resampled' in locals():
             # Summing boolean values (True=1, False=0) across columns
            sum_available = availability_df_resampled.sum(axis=1)
        else:
            sum_available = plot_df.notna().sum(axis=1)
        plt.plot(resampled_index, sum_available, label=f'Count of Available Columns (out of {len(existing_data_columns)})')
        plt.ylabel(f'Number of Available Columns (Max {len(existing_data_columns)})', fontsize=8)
        if len(existing_data_columns) > 15:
             plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=6)
        else:
            plt.legend(fontsize=8)

    plt.title(plot_title, fontsize=10)
    # Check if fallback occurred by looking for the fallback-specific label text or if an error was printed by stackplot
    is_fallback = any(label.get_text().startswith('Count of Available Columns') for label in plt.gca().get_lines())
    if not is_fallback:
        plt.ylabel(f'Number of Available Columns (Max {len(existing_data_columns)})', fontsize=8)

    plt.xlabel('Time', fontsize=8)
    plt.ylim(0, len(existing_data_columns) * 1.05)
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    plt.gcf().autofmt_xdate(rotation=45, ha='right')
    plt.tick_params(axis='both', which='major', labelsize=8)

    if not is_fallback:
        if len(existing_data_columns) <= 15:
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=6)
        else:
            plt.text(0.99, 0.98, 'Too many columns for individual legend items', 
                     horizontalalignment='right', verticalalignment='top',
                     transform=plt.gca().transAxes, fontsize=7, color='gray')

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    plot_filename = os.path.join(output_dir, "availability_stacked_area.png")
    try:
        plt.savefig(plot_filename, dpi=150)
    except Exception as e_save:
        print(f"Error saving stacked area availability plot: {e_save}", file=sys.stderr)
    plt.close()

def plot_value_distribution_barchart(df: pd.DataFrame, column_name: str, output_dir: str, top_n_threshold: int = 30, top_n_display: int = 20):
    """Plots a bar chart of the value distribution for a single column."""
    if column_name not in df.columns:
        print(f"Warning: Column '{column_name}' not found in DataFrame. Skipping bar chart.", file=sys.stderr)
        return
    if column_name in ['time', 'uuid']: # 'time' is usually index, 'uuid' is identifier
        print(f"Skipping bar chart for unsuitable column: {column_name}", file=sys.stderr)
        return

    plt.figure(figsize=(12, 7))
    counts = df[column_name].value_counts(dropna=False) # include NaNs
    
    num_unique_values = len(counts)
    title = f'Value Distribution for: {column_name}'

    if num_unique_values == 0:
        print(f"Warning: Column '{column_name}' has no values to plot. Skipping bar chart.", file=sys.stderr)
        plt.close()
        return

    # Prepare data for plotting
    if num_unique_values > top_n_threshold:
        counts_to_plot = counts.nlargest(top_n_display) # Already sorted by frequency
        title += f' (Top {top_n_display} of {num_unique_values} Unique Values)'
        plot_index = [str(idx) if pd.notna(idx) else '<NaN>' for idx in counts_to_plot.index]
        plot_values = counts_to_plot.values
    else:
        counts_sorted = counts.sort_index() # Sort by value/category name
        plot_index = [str(idx) if pd.notna(idx) else '<NaN>' for idx in counts_sorted.index]
        plot_values = counts_sorted.values
    
    try:
        sns.barplot(x=plot_index, y=plot_values, palette="viridis", order=plot_index)
    except Exception as e_barplot:
        print(f"Error during barplot for {column_name}: {e_barplot}. Skipping.", file=sys.stderr)
        plt.close()
        return

    plt.title(title, fontsize=10)
    plt.ylabel('Frequency Count', fontsize=8)
    plt.xlabel(f'Value in {column_name}', fontsize=8)
    
    current_xticks, _ = plt.xticks()
    if len(current_xticks) > 15 or any(len(str(lab.get_text())) > 10 for lab in plt.gca().get_xticklabels()):
        plt.xticks(rotation=45, ha="right", fontsize=8)
    else:
        plt.xticks(rotation=0, ha="center", fontsize=8)
    
    plt.yticks(fontsize=8)
    plt.tight_layout()
    
    plot_filename = os.path.join(output_dir, f"barchart_dist_{column_name.replace('/', '_').replace('%', 'pct').replace(' ', '_')}.png")
    try:
        plt.savefig(plot_filename, dpi=150)
    except Exception as e_save:
        print(f"Error saving bar chart for {column_name}: {e_save}", file=sys.stderr)
    plt.close()


if __name__ == '__main__':
    # print("Running Data Availability Analysis Script...") # Reduced print
    sns.set_theme(style="whitegrid")

    output_image_dir = os.path.join(os.path.dirname(__file__), "images")
    os.makedirs(output_image_dir, exist_ok=True)
    # print(f"Saving plots to: {output_image_dir}") # Reduced print

    df_full_data = None
    engine = None
    all_columns_from_db = []
    try:
        engine = get_db_engine()
        query = "SELECT * FROM public.sensor_data_merged ORDER BY time ASC;"
        # print(f"Executing query: {query}") # Reduced print
        df_full_data = load_data_from_db(query, engine)
        # print(f"Loaded {len(df_full_data)} rows from sensor_data_merged.") # Reduced print
        if not df_full_data.empty:
            all_columns_from_db = df_full_data.columns.tolist()
    except Exception as e:
        print(f"Critical Error: Error loading data from database: {e}", file=sys.stderr)
        print("Ensure database connection details are set correctly in Jupyter/classes/db_connection/db_utils.py.", file=sys.stderr)
        sys.exit(1)
    finally:
        if engine is not None: # Check if engine was successfully created
            try: 
                engine.dispose() # Dispose SQLAlchemy engine
                # print("SQLAlchemy engine disposed.") # Reduced print
            except Exception as dispose_e: 
                print(f"Error disposing SQLAlchemy engine: {dispose_e}", file=sys.stderr)

    if df_full_data is None or df_full_data.empty:
        print("Critical Error: No data loaded from database. Exiting.", file=sys.stderr)
        sys.exit(1)

    try:
        df_full_data['time'] = pd.to_datetime(df_full_data['time'])
        df_full_data = df_full_data.set_index('time')
    except Exception as e_time:
        print(f"Critical Error: Error processing time column: {e_time}. Ensure 'time' column exists and is datetime convertible.", file=sys.stderr)
        sys.exit(1)
    
    metadata_cols = ['source_system', 'source_file', 'format_type', 'uuid', 'time'] 
    data_cols_for_heatmap_and_overall = [col for col in all_columns_from_db if col not in metadata_cols]

    # --- Generate Overall Data Availability Plot ---
    # print(f"\n--- Generating Overall Data Availability Plot ---") # Reduced print
    plot_overall_data_availability(df_full_data, data_cols_for_heatmap_and_overall, output_image_dir)
    
    # --- Generate Stacked Area Availability Plot (with Daily Resampling by default) ---
    # print(f"\n--- Generating Stacked Area Data Availability Plot ---") # Reduced print
    plot_stacked_area_availability(df_full_data, data_cols_for_heatmap_and_overall, output_image_dir, resample_freq='D')

    chunk_size = 20 
    column_chunks = [data_cols_for_heatmap_and_overall[i:i + chunk_size] for i in range(0, len(data_cols_for_heatmap_and_overall), chunk_size)]

    # print(f"\n--- Generating Missing Data Heatmaps (Daily Resampling) ---") # Reduced print
    for i, chunk in enumerate(column_chunks):
        plot_missing_data_heatmap_grouped(df_full_data, chunk, f"Group_{i+1}", output_image_dir, resample_freq='D')

    KEY_COLUMNS_FOR_LINE_PLOTS = [
        'air_temp_c', 'co2_measured_ppm', 'relative_humidity_percent', 'radiation_w_m2',
        'light_intensity_umol', 'heating_setpoint_c', 'pipe_temp_1_c',
        'vent_lee_afd3_percent', 'vent_wind_afd3_percent',
        'rain_status', 'lamp_grp1_no3_status'
    ]
    # print(f"\n--- Generating Individual Data Availability Line Plots ---") # Reduced print
    existing_key_cols = [col for col in KEY_COLUMNS_FOR_LINE_PLOTS if col in df_full_data.columns]
    if not existing_key_cols:
        print("Warning: None of the specified KEY_COLUMNS_FOR_LINE_PLOTS found in the DataFrame.", file=sys.stderr)
    else:
        for col_name in existing_key_cols:
            plot_data_availability(df_full_data, col_name, output_image_dir)

    # --- Generate Value Distribution Bar Charts ---
    print(f"\n--- Generating Value Distribution Bar Charts ---")
    
    all_potential_barchart_cols = [
        "source_system", "source_file", "format_type", "lamp_group", "air_temp_c", 
        "air_temp_middle_c", "outside_temp_c", "relative_humidity_percent", 
        "humidity_deficit_g_m3", "radiation_w_m2", "light_intensity_lux", 
        "light_intensity_umol", "outside_light_w_m2", "co2_measured_ppm", 
        "co2_required_ppm", "co2_dosing_status", "co2_status", "rain_status", 
        "vent_pos_1_percent", "vent_pos_2_percent", "vent_lee_afd3_percent", 
        "vent_wind_afd3_percent", "vent_lee_afd4_percent", "vent_wind_afd4_percent", 
        "curtain_1_percent", "curtain_2_percent", "curtain_3_percent", "curtain_4_percent", 
        "window_1_percent", "window_2_percent", "lamp_grp1_no3_status", 
        "lamp_grp2_no3_status", "lamp_grp3_no3_status", "lamp_grp4_no3_status", 
        "lamp_grp1_no4_status", "lamp_grp2_no4_status", "measured_status_bool", 
        "heating_setpoint_c", "pipe_temp_1_c", "pipe_temp_2_c", "flow_temp_1_c", 
        "flow_temp_2_c", "temperature_forecast_c", "sun_radiation_forecast_w_m2", 
        "temperature_actual_c", "sun_radiation_actual_w_m2", "vpd_hpa", 
        "humidity_deficit_afd3_g_m3", "relative_humidity_afd3_percent", 
        "humidity_deficit_afd4_g_m3", "relative_humidity_afd4_percent", "dli_sum"
    ]

    actual_cols_for_barchart = [col for col in all_potential_barchart_cols if col in df_full_data.columns]
    
    if not actual_cols_for_barchart:
        print("Warning: No suitable columns found or specified for bar chart generation from the provided list.", file=sys.stderr)
    else:
        for col_name in actual_cols_for_barchart:
            plot_value_distribution_barchart(df_full_data, col_name, output_image_dir)

    # print("\nData availability analysis script finished.") # Reduced print 