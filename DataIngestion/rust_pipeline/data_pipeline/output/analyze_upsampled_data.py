#!/usr/bin/env -S uv run --isolated
# /// script
# dependencies = ["pandas", "numpy", "matplotlib", "seaborn"]
# ///

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Define the expected TARGET_COLUMNS to help with type conversion and selection
# This should ideally match the order and names from your Rust pipeline's TARGET_COLUMNS
TARGET_COLUMNS = [
    "time", "source_system", "source_file", "format_type", "uuid", "lamp_group",
    "air_temp_c", "air_temp_middle_c", "outside_temp_c",
    "relative_humidity_percent", "humidity_deficit_g_m3",
    "radiation_w_m2", "light_intensity_lux", "light_intensity_umol",
    "outside_light_w_m2", "co2_measured_ppm", "co2_required_ppm",
    "co2_dosing_status", "co2_status", "rain_status",
    "vent_pos_1_percent", "vent_pos_2_percent", "vent_lee_afd3_percent",
    "vent_wind_afd3_percent", "vent_lee_afd4_percent", "vent_wind_afd4_percent",
    "curtain_1_percent", "curtain_2_percent", "curtain_3_percent",
    "curtain_4_percent", "window_1_percent", "window_2_percent",
    "lamp_grp1_no3_status", "lamp_grp2_no3_status", "lamp_grp3_no3_status",
    "lamp_grp4_no3_status", "lamp_grp1_no4_status", "lamp_grp2_no4_status",
    "measured_status_bool",
    "heating_setpoint_c", "pipe_temp_1_c", "pipe_temp_2_c",
    "flow_temp_1_c", "flow_temp_2_c",
    "temperature_forecast_c", "sun_radiation_forecast_w_m2",
    "temperature_actual_c", "sun_radiation_actual_w_m2",
    "vpd_hpa", "humidity_deficit_afd3_g_m3", "relative_humidity_afd3_percent",
    "humidity_deficit_afd4_g_m3", "relative_humidity_afd4_percent",
    "behov", "status_str", "timer_on", "timer_off", "dli_sum",
    "oenske_ekstra_lys", "lampe_timer_on", "lampe_timer_off",
    "value"
]

# Define column types based on expectations from Rust and SQL scripts
# This helps in converting them correctly after loading from CSV
NUMERIC_FLOAT_COLS = [
    "air_temp_c", "air_temp_middle_c", "outside_temp_c",
    "relative_humidity_percent", "humidity_deficit_g_m3",
    "radiation_w_m2", "light_intensity_lux", "light_intensity_umol",
    "outside_light_w_m2", "co2_measured_ppm", "co2_required_ppm",
    "co2_dosing_status", "co2_status", # These were double precision
    "vent_pos_1_percent", "vent_pos_2_percent", "vent_lee_afd3_percent",
    "vent_wind_afd3_percent", "vent_lee_afd4_percent", "vent_wind_afd4_percent",
    "curtain_1_percent", "curtain_2_percent", "curtain_3_percent",
    "curtain_4_percent", "window_1_percent", "window_2_percent",
    "heating_setpoint_c", "pipe_temp_1_c", "pipe_temp_2_c",
    "flow_temp_1_c", "flow_temp_2_c",
    "temperature_forecast_c", "sun_radiation_forecast_w_m2",
    "temperature_actual_c", "sun_radiation_actual_w_m2",
    "vpd_hpa", "humidity_deficit_afd3_g_m3", "relative_humidity_afd3_percent",
    "humidity_deficit_afd4_g_m3", "relative_humidity_afd4_percent",
    "dli_sum", "value"
]

NUMERIC_INT_COLS = [ # Pandas will likely make these float if NaNs are present
    "behov", "timer_on", "timer_off", "lampe_timer_on", "lampe_timer_off"
]

BOOLEAN_COLS = [
    "rain_status", "lamp_grp1_no3_status", "lamp_grp2_no3_status",
    "lamp_grp3_no3_status", "lamp_grp4_no3_status", "lamp_grp1_no4_status",
    "lamp_grp2_no4_status", "measured_status_bool"
]


def analyze_data(csv_path: Path, output_dir: Path):
    """
    Performs analysis on the upsampled sensor data.
    """
    print(f"--- Starting Analysis for {csv_path} ---")

    if not csv_path.exists():
        print(f"ERROR: CSV file not found at {csv_path}")
        return

    # Create output directory for plots
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    try:
        # Read empty strings as NaN, and ensure correct boolean parsing
        na_values = [
            '', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan',
            '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'nan',
            'None', 'none'
        ]
        
        df = pd.read_csv(csv_path, na_values=na_values, low_memory=False)
        print(f"Successfully loaded {len(df)} rows from {csv_path}")
    except Exception as e:
        print(f"ERROR: Could not load CSV: {e}")
        return

    # Convert 'time' column
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df = df.set_index('time').sort_index()
    else:
        print("ERROR: 'time' column not found.")
        return

    # Convert column types
    for col in NUMERIC_FLOAT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    for col in NUMERIC_INT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce') 

    for col in BOOLEAN_COLS:
        if col in df.columns:
            df[col] = df[col].replace({'true': True, 'false': False, 'True': True, 'False': False, 
                                       '1': True, '0': False, '1.0': True, '0.0': False,
                                       1: True, 0: False, 1.0: True, 0.0: False})
            try:
                df[col] = df[col].astype('boolean') 
            except Exception as e:
                print(f"Warning: Could not convert column {col} to boolean directly: {e}. Leaving as object.")


    print("\n--- 1. Data Availability (Percentage Missing) ---")
    missing_percentage = (df.isnull().sum() * 100 / len(df)).sort_values(ascending=False)
    print(missing_percentage)

    print("\n--- 2. Descriptive Statistics (for numerical columns) ---")
    numeric_cols_for_desc = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols_for_desc:
        print(df[numeric_cols_for_desc].describe().transpose())
    else:
        print("No numeric columns found for descriptive statistics.")

    print("\n--- 3. Basic Outlier Detection (IQR method for selected columns) ---")
    outlier_cols = ['air_temp_c', 'light_intensity_umol', 'relative_humidity_percent', 'co2_measured_ppm']
    for col in outlier_cols:
        if col in df.columns and df[col].dtype in ['float64', 'int64', 'float32', 'int32']: # Check if numeric
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            print(f"Outliers for '{col}': {len(outliers)} rows (Lower: {lower_bound:.2f}, Upper: {upper_bound:.2f})")
            if not outliers.empty:
                 print(f"Sample outliers for {col}:\n{outliers[[col]].head()}")
        else:
            print(f"Skipping outlier detection for '{col}' (not numeric or not found).")
    
    print("\n--- 4. Value Counts for Status/Categorical Columns ---")
    status_cols_to_check = ['rain_status', 'co2_dosing_status', 'lamp_grp1_no3_status']
    for col in status_cols_to_check:
        if col in df.columns:
            print(f"\nValue counts for '{col}':")
            print(df[col].value_counts(dropna=False))
        else:
            print(f"Column '{col}' not found for value counts.")

    print("\n--- 5. Time Series Plots ---")
    plot_cols = {
        'air_temp_c': 'Air Temperature (°C)',
        'light_intensity_umol': 'Light Intensity (µmol/m²/s)',
        'dli_sum': 'Daily Light Integral (mol/m²/day)',
        'relative_humidity_percent': 'Relative Humidity (%)'
    }
    
    sample_df = df
    if len(df) > 50000: 
        print(f"Dataset is large ({len(df)} rows), sampling first 50000 rows for plots.")
        sample_df = df.iloc[:50000]

    for col, title in plot_cols.items():
        if col in sample_df.columns and pd.api.types.is_numeric_dtype(sample_df[col].dropna()): #dropna for numeric check
            if not sample_df[col].dropna().empty: # Check if column has any non-NaN numeric data
                plt.figure(figsize=(15, 5))
                sample_df[col].plot()
                plt.title(f'Time Series of {title}')
                plt.xlabel('Time')
                plt.ylabel(title)
                plot_path = output_dir / f"{col}_timeseries.png"
                plt.savefig(plot_path)
                print(f"Saved plot: {plot_path}")
                plt.close()
            else:
                print(f"Skipping plot for '{col}' (all NaN values after potential conversion).")
        else:
            print(f"Skipping plot for '{col}' (not numeric, not found, or all NaN).")
            
    print("\n--- 6. Correlation Heatmap (for numerical columns) ---")
    if numeric_cols_for_desc:
        heatmap_cols = [col for col in numeric_cols_for_desc if df[col].nunique() > 1] # Exclude constant columns

        if len(heatmap_cols) > 20:
            print(f"Many numeric columns ({len(heatmap_cols)}), selecting a subset for heatmap.")
            prioritized_cols = [c for c in [
                'air_temp_c', 'relative_humidity_percent', 'light_intensity_umol',
                'co2_measured_ppm', 'radiation_w_m2', 'vpd_hpa', 'dli_sum'
            ] if c in heatmap_cols]
            remaining_cols = [c for c in heatmap_cols if c not in prioritized_cols]
            final_heatmap_cols = prioritized_cols
            if len(final_heatmap_cols) < 15 and remaining_cols:
                 final_heatmap_cols.extend(remaining_cols[:15-len(final_heatmap_cols)])
            heatmap_cols = final_heatmap_cols
        
        if len(heatmap_cols) > 1: # Need at least 2 columns for correlation
            corr_matrix = df[heatmap_cols].corr()
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".1f", linewidths=.5)
            plt.title(f'Correlation Matrix (Subset: {len(heatmap_cols)} features)')
            plot_path = output_dir / "correlation_heatmap.png"
            plt.savefig(plot_path)
            print(f"Saved plot: {plot_path}")
            plt.close()
        else:
            print("Not enough unique numeric columns for correlation heatmap after selection.")
    else:
        print("No numeric columns for correlation heatmap.")

    print(f"\n--- Analysis Complete. Plots saved to {output_dir} ---")

if __name__ == "__main__":
    # Assuming the script is in DataIngestion/rust_pipeline/data_pipeline/
    # and the CSV is in ./output/
    # For script placed in ./output/ alongside the CSV:
    script_dir = Path(__file__).parent
    csv_file_path = script_dir / "sensor_data_upsampled.csv"
    plot_output_dir = script_dir / "analysis_plots"
    
    analyze_data(csv_file_path, plot_output_dir) 