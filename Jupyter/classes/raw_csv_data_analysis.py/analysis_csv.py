#!/usr/bin/env -S uv run --isolated
# /// script
# dependencies = ["pandas", "numpy", "matplotlib", "seaborn", "psycopg2-binary"]
# ///

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob # For finding files, though Path.glob is used
import csv # Import Python's built-in CSV module
from typing import List, Optional, Union, Dict
from collections import Counter # For making headers unique

# Helper function to make list of headers unique
def make_headers_unique(headers: List[str]) -> List[str]:
    counts = Counter(headers)
    new_headers = []
    seen_counts = Counter()
    for header in headers:
        if counts[header] > 1:
            seen_counts[header] += 1
            new_headers.append(f"{header}_{seen_counts[header]}")
        else:
            new_headers.append(header)
    return new_headers

# Function 1: Load Data
def load_knudjepsen_csvs(data_source: Union[Path, str], time_col: Optional[str] = 'timestamp') -> tuple[pd.DataFrame, Dict[str, set]]:
    """
    Loads and concatenates CSV files from a directory or a single CSV file.
    Attempts to infer delimiter and identify the time column if not standard.
    Parses the time column and sets it as the DataFrame index.
    Attempts to convert columns with 'status' in their name to integer type (0 or 1).

    Args:
        data_source (Union[Path, str]): Path to a directory containing CSV files,
                                         or path to a single CSV file.
        time_col (Optional[str]): Preferred name of the column to be parsed as datetime.
                                  If None or not found, the function will try to infer it.
                        Defaults to 'timestamp'.

    Returns:
        tuple[pd.DataFrame, Dict[str, set]]: 
            - A single DataFrame with all data, time-indexed and sorted.
            - A dictionary mapping filename to a set of its column names.
    """
    data_path = Path(data_source)
    all_dfs: List[pd.DataFrame] = []
    file_column_map: Dict[str, set] = {} # To store columns per file
    
    common_time_column_names = ['timestamp', 'time', 'date', 'datetime', 
                                'Timestamp', 'Time', 'Date', 'Datetime',
                                'dato', 'tid'] # Add Danish common names

    if data_path.is_dir():
        csv_files = list(data_path.glob("*.csv"))
        if not csv_files:
            print(f"No CSV files found in directory: {data_path}")
            return pd.DataFrame(), file_column_map
    elif data_path.is_file() and data_path.suffix.lower() == '.csv':
        csv_files = [data_path]
    else:
        print(f"Invalid data_source: {data_source}. Not a CSV file or directory.")
        return pd.DataFrame(), file_column_map

    print(f"Found {len(csv_files)} CSV files to process: {[f.name for f in csv_files]}")

    for file_path in csv_files:
        print(f"\nProcessing file: {file_path.name}")
        actual_time_col = None
        try:
            # 1. Sniff the CSV dialect (delimiter, etc.)
            sniffer = csv.Sniffer()
            detected_delimiter = ',' # Default delimiter
            header_line = ""
            try:
                with open(file_path, 'r', encoding='utf-8-sig') as f: # utf-8-sig handles BOM
                    # Read a sample for sniffing, ensuring we don't read too much of a large file
                    sample = ""
                    for i, line in enumerate(f):
                        if i == 0:
                            header_line = line.strip()
                        sample += line
                        if i >= 10: # Sniff based on first 10 lines or so
                            break
                    if not sample:
                        print(f"Warning: File {file_path.name} is empty or could not be read for sniffing. Skipping.")
                        continue
                    
                    dialect = sniffer.sniff(sample)
                    detected_delimiter = dialect.delimiter
                    print(f"  Detected delimiter for {file_path.name}: '{detected_delimiter}'")
            except Exception as sniff_err:
                print(f"  Warning: Could not reliably sniff delimiter for {file_path.name}. Defaulting to ','. Error: {sniff_err}")
                # Try to get header with default comma if sniffing failed but header_line was read
                if not header_line:
                    try:
                        with open(file_path, 'r', encoding='utf-8-sig') as f:
                            header_line = f.readline().strip()
                    except Exception:
                        print(f"  Could not read header line for {file_path.name}. Skipping file.")
                        continue
            
            # 2. Identify header and time column
            if not header_line:
                print(f"  Could not read header from {file_path.name}. Skipping.")
                continue
            
            # Split headers first
            raw_headers = [h.strip().strip('"') for h in header_line.split(detected_delimiter)]
            # If the first raw header is empty, pandas might name it 'Unnamed: 0'.
            if raw_headers and not raw_headers[0]:
                raw_headers[0] = 'Unnamed: 0' # Tentatively name it
            
            # Make headers unique *before* passing to pandas
            headers = make_headers_unique(raw_headers)
            print(f"  Processed (unique) headers: {headers}")

            if time_col and time_col in headers:
                actual_time_col = time_col
            else:
                if time_col: # only print warning if a specific one was requested but not found
                    print(f"  Warning: Specified time column '{time_col}' not found in {file_path.name}.")
                
                # Check if 'Unnamed: 0' (potential first col) can be our time col
                if 'Unnamed: 0' in headers:
                    print(f"  Checking if the first column ('Unnamed: 0') can be used as time column.")
                    # We'll attempt to parse it later, for now, mark it as a candidate.
                    # actual_time_col = 'Unnamed: 0' # Let pd.read_csv handle index_col=0 first

                if not actual_time_col: # If 'Unnamed: 0' wasn't explicitly set or if we want to search more
                    print(f"  Attempting to infer time column from common names: {common_time_column_names}")
                    for common_name in common_time_column_names:
                        if common_name in headers:
                            actual_time_col = common_name
                            print(f"  Inferred time column: '{actual_time_col}'")
                            break
                
                if not actual_time_col and 'Unnamed: 0' in headers:
                    print(f"  Defaulting to use first column 'Unnamed: 0' as potential time column candidate for parsing.")
                    actual_time_col = 'Unnamed: 0' # Explicitly try to use it for parsing later
                
                if not actual_time_col:
                    print(f"  Warning: Could not infer a time column for {file_path.name}. Will try to load without specific time parsing for now, or skip if crucial.")

            # 3. Read CSV with pandas
            read_csv_params = {
                'delimiter': detected_delimiter,
                'low_memory': False,
                'header': 0, # We read the first line for sniffing, so data starts next line effectively
                'names': headers, # Pass our processed, unique headers
                'skiprows': [0] # Skip the original header row since we provide 'names'
            }

            df = None
            try:
                print(f"  Attempting to read {file_path.name} with processed headers.")
                df = pd.read_csv(file_path, **read_csv_params)
                print(f"    Successfully read {file_path.name}.")
            except pd.errors.ParserError as pe:
                print(f"    Pandas ParserError for {file_path.name}: {pe}. Trying with quoting.")
                try:
                    read_csv_params['quoting'] = csv.QUOTE_MINIMAL
                    df = pd.read_csv(file_path, **read_csv_params)
                    print(f"    Successfully read {file_path.name} with quoting={csv.QUOTE_MINIMAL}.")
                except Exception as pe_quoted:
                    print(f"    Failed to read {file_path.name} even with QUOTE_MINIMAL. Error: {pe_quoted}. Skipping file.")
                    continue
            except Exception as e_pandas:
                print(f"  Error reading {file_path.name} with pandas: {e_pandas}. Skipping file.")
                continue
            
            if df is None: # Should be caught by exceptions, but as a safeguard
                print(f"  DataFrame for {file_path.name} is None after read attempts. Skipping.")
                continue

            # Store column names for this file *before* adding source_group or changing index
            # This reflects the columns as they were in the file initially (after header processing)
            file_column_map[file_path.name] = set(df.columns)

            # Add a source_group column based on the file name stem
            df['source_group'] = file_path.stem
            print(f"  Added 'source_group': {file_path.stem} to DataFrame from {file_path.name}")

            # 4. Process time column if found/inferred
            target_for_time_parsing = None
            # is_index_col = False # This will always be false now before explicit set_index for time

            # Since we are not using index_col in read_csv for the time column initially,
            # the index will be a default RangeIndex. We always look for actual_time_col in df.columns.
            if isinstance(df.index, pd.DatetimeIndex):
                # This case would only occur if a non-time column was somehow set as index before this point,
                # or if a future modification sets index earlier. Unlikely with current flow for time col.
                print(f"  Index of {file_path.name} is already DatetimeIndex. Current time column logic might re-evaluate.")
            
            if actual_time_col and actual_time_col in df.columns: 
                target_for_time_parsing = df[actual_time_col]
                print(f"  Identified column '{actual_time_col}' for time parsing.")
            # Removed other elif branches for target_for_time_parsing as they are not reachable
            # if we load 'Unnamed: 0' as a column initially.
            
            if target_for_time_parsing is not None:
                try:
                    parsed_time_series = pd.to_datetime(target_for_time_parsing, errors='coerce')
                    
                    if parsed_time_series.isnull().all():
                        print(f"  Warning: Time column '{actual_time_col}' in {file_path.name} resulted in all NaT values after parsing. Check format.")
                    else:
                        # Assign the parsed series back to the column first
                        df[actual_time_col] = parsed_time_series
                        
                        # Drop rows where the crucial time column is NaT
                        df.dropna(subset=[actual_time_col], inplace=True)

                        if df.empty:
                            print(f"  Data in {file_path.name} became empty after dropping rows with invalid time values in '{actual_time_col}'.")
                            continue
                        
                        # Now, set this validated column as index
                        df = df.set_index(actual_time_col)
                        print(f"  Successfully parsed and set '{actual_time_col}' as DatetimeIndex for {file_path.name}.")

                except Exception as e_time:
                    print(f"  Warning: Could not parse or set time column '{actual_time_col}' in {file_path.name}. Error: {e_time}. Data loaded without time index.")
            elif actual_time_col:
                 print(f"  Warning: Identified time column candidate '{actual_time_col}' was not found in DataFrame columns after loading {file_path.name}. Columns are: {df.columns.tolist()}")
            else: # No actual_time_col was determined
                print(f"  No time column identified for {file_path.name}; will not set DatetimeIndex.")
            
            # 5. Attempt to convert other object columns to numeric
            print(f"  Attempting to convert object columns to numeric for {file_path.name}...")
            for col in df.columns:
                if col == 'source_group': # Explicitly skip this column
                    print(f"    Skipping numeric conversion for column: {col} (intended as string group identifier).")
                    continue 
                if df[col].dtype == 'object':
                    print(f"    Checking column: {col}")
                    try:
                        # Attempt direct conversion first
                        df[col] = pd.to_numeric(df[col], errors='raise')
                        print(f"      Successfully converted column '{col}' to numeric directly.")
                    except (ValueError, TypeError):
                        # If direct conversion fails, try replacing comma with period for decimal
                        print(f"      Direct numeric conversion failed for '{col}'. Trying with comma replacement.")
                        try:
                            # Ensure it's string type before replace
                            if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == 'object':
                                # Apply to series, then convert
                                str_series = df[col].astype(str).str.replace(',', '.', regex=False)
                                df[col] = pd.to_numeric(str_series, errors='coerce') # Coerce errors to NaN
                                if not df[col].isnull().all(): # Check if conversion was somewhat successful
                                     print(f"      Successfully converted column '{col}' to numeric after comma replacement.")
                                else:
                                     print(f"      Column '{col}' became all NaN after comma replacement and numeric conversion. Might not be purely numeric or different issue.")
                            else: # Should not happen if dtype was object, but as a safe guard
                                print(f"      Column '{col}' is not string/object type for comma replacement.")
                        except Exception as e_conv_comma:
                            print(f"      Error during comma replacement/conversion for column '{col}': {e_conv_comma}")
            
            df.columns = [str(c).replace('.1', '_1').replace('.2', '_2').replace('.3', '_3').replace('.4', '_4').replace('.5', '_5') for c in df.columns]


            all_dfs.append(df)
        except Exception as e:
            print(f"  Overall error processing {file_path.name}: {e}. Skipping file.")

    if not all_dfs:
        print("\nNo data successfully loaded from any files.")
        return pd.DataFrame(), file_column_map

    print("\nConcatenating all loaded DataFrames...")
    # Before concat, ensure all DFs have compatible column types or handle mismatches.
    # This is a complex problem. For now, we rely on pandas' concat abilities.
    # A more robust solution might involve schema alignment.
    try:
        combined_df = pd.concat(all_dfs) # Removed ignore_index=True if some DFs already have DatetimeIndex
    except Exception as e_concat:
        print(f"Error during concatenation of DataFrames: {e_concat}")
        print("Attempting concatenation by aligning columns and inferring types (can be slow/lossy).")
        # Fallback: try to make columns consistent if simple concat fails due to type or column mismatches
        # This is a basic attempt; more sophisticated schema merging might be needed
        processed_dfs_for_concat = []
        if not all_dfs: return pd.DataFrame(), file_column_map

        # Get all unique column names from all dataframes
        all_cols = set()
        for d in all_dfs:
            all_cols.update(d.columns)
        
        for i, df_item in enumerate(all_dfs):
            # Reindex to ensure all dataframes have all columns, filling missing with NaN
            # This helps if some files have slightly different columns
            temp_df = df_item.reindex(columns=list(all_cols))
            processed_dfs_for_concat.append(temp_df)
        
        try:
            combined_df = pd.concat(processed_dfs_for_concat) # ignore_index still off
            print("Successfully concatenated with reindexed columns.")
        except Exception as e_concat_aligned:
            print(f"Further error during aligned concatenation: {e_concat_aligned}. Returning empty DataFrame.")
            return pd.DataFrame(), file_column_map


    # If index is not already a DatetimeIndex (e.g. if some files had no time_col)
    # and a common time column name was inferred and exists after concat.
    # This part is tricky because individual DFs might or might not have had their index set.
    # For simplicity, we'll assume if a time_col was set for *any* file, it should be the index.
    # If different files had different time columns set as index, concat might behave unexpectedly.
    # The ideal scenario is that all files can use the *same* time column name.
    
    # If the index is not a DatetimeIndex after concat, and a primary time_col was aimed for:
    final_time_col_to_check = time_col # The user's original preference
    if not final_time_col_to_check and all_dfs: # if no preference, take from first loaded df's index name if DT
        if isinstance(all_dfs[0].index, pd.DatetimeIndex):
            final_time_col_to_check = all_dfs[0].index.name

    if not isinstance(combined_df.index, pd.DatetimeIndex) and final_time_col_to_check and final_time_col_to_check in combined_df.columns:
        print(f"Combined DataFrame index is not DatetimeIndex. Attempting to set '{final_time_col_to_check}' as index.")
        try:
            combined_df[final_time_col_to_check] = pd.to_datetime(combined_df[final_time_col_to_check], errors='coerce')
            combined_df.dropna(subset=[final_time_col_to_check], inplace=True)
            if not combined_df.empty:
                combined_df = combined_df.set_index(final_time_col_to_check)
                print(f"Successfully set '{final_time_col_to_check}' as index on combined DataFrame.")
            else:
                print(f"Combined DataFrame became empty after trying to set and clean '{final_time_col_to_check}'.")
        except Exception as e_set_index_combined:
            print(f"Could not set '{final_time_col_to_check}' as index on combined DataFrame: {e_set_index_combined}")
    
    if isinstance(combined_df.index, pd.DatetimeIndex):
        print("Sorting combined DataFrame by time index.")
    combined_df = combined_df.sort_index()
        print("Removing rows with NaT in DatetimeIndex.")
        combined_df = combined_df[combined_df.index.notna()]
    else:
        print("Warning: Combined DataFrame does not have a DatetimeIndex. Sorting by index might not be time-based.")
        # combined_df = combined_df.sort_index() # Sort by whatever index it has
    
    # Convert potential light status columns to int (0 or 1)
    # This logic remains similar but operates on the combined_df
    print("\nConverting status columns...")
    for col in combined_df.columns:
        if 'status' in col.lower(): # Heuristic for status columns
            if pd.api.types.is_bool_dtype(combined_df[col]):
                combined_df[col] = combined_df[col].astype(int)
            elif combined_df[col].dtype == 'object' or pd.api.types.is_string_dtype(combined_df[col]):
                try:
                         # More robust mapping for mixed types, case-insensitive for strings
                        def map_status(val):
                        if pd.isna(val): return pd.NA
                            if isinstance(val, str):
                                val_lower = val.lower()
                            if val_lower in ['on', 'true', '1', 'active', 'yes']: return 1
                            if val_lower in ['off', 'false', '0', 'inactive', 'no']: return 0
                            elif isinstance(val, bool):
                                return 1 if val else 0
                            elif pd.api.types.is_numeric_dtype(val): # handles int/float
                            return int(val) if val in [0,1] else pd.NA 
                        return pd.NA 

                        combined_df[col] = combined_df[col].apply(map_status)
                    # Attempt conversion to nullable integer type
                        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce').astype('Int64')
                    print(f"  Processed status column '{col}'. Unique values now: {combined_df[col].dropna().unique()[:5]}")

                except Exception as e_status:
                    print(f"  Warning: Could not reliably convert status column '{col}' to 0/1. Error: {e_status}")
            elif pd.api.types.is_numeric_dtype(combined_df[col]):
                # If numeric, ensure it's Int64 if it contains only 0, 1, and NaN
                # Avoid converting if it has other numbers that might be valid for that status column
                unique_numeric_vals = combined_df[col].dropna().unique()
                if all(val in [0, 1] for val in unique_numeric_vals):
                    combined_df[col] = combined_df[col].astype('Int64') # Use nullable integer
                    print(f"  Ensured numeric status column '{col}' is Int64 (contains only 0,1,NaN).")

    print("\nData loading and pre-processing complete.")
    return combined_df, file_column_map

# Function 2: Plot hourly light profile (status or intensity) by group
def plot_hourly_light_profile_by_group(
    df: pd.DataFrame, 
    group_col: str,
    light_status_cols: Optional[List[str]] = None,
    light_intensity_cols: Optional[List[str]] = None,
    aggregation_method: str = 'mean'
):
    """
    Plots the average daily profile (by hour) of light status (proportion ON) 
    or light intensity for specified light columns, grouped by `group_col`.

    Args:
        df (pd.DataFrame): DataFrame with a DatetimeIndex.
        group_col (str): Column name for grouping (e.g., 'group_id').
        light_status_cols (Optional[List[str]]): List of column names for light status (0 or 1).
        light_intensity_cols (Optional[List[str]]): List of column names for light intensity.
        aggregation_method (str): 'mean' for proportion ON (status) or avg intensity. 'sum' for total.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in DataFrame.")

    plot_cols: List[str] = []
    y_label_base: str = ""
    main_plot_title: str = ""

    if light_status_cols:
        plot_cols.extend(light_status_cols)
        y_label_base = f"Proportion of Time Lights ON ({aggregation_method})"
        main_plot_title = "Hourly Light Status Profile by Group"
    elif light_intensity_cols: # Changed to elif to prioritize status if both provided
        plot_cols.extend(light_intensity_cols)
        y_label_base = f"Light Intensity ({aggregation_method})"
        main_plot_title = "Hourly Light Intensity Profile by Group"
    
    if not plot_cols: # No specific columns given, try to find them
        if light_status_cols is None and light_intensity_cols is None: # Auto-detect if nothing specified
            print("Auto-detecting light columns...")
            light_status_cols = [col for col in df.columns if 'light' in col.lower() and 'status' in col.lower()]
            light_intensity_cols = [col for col in df.columns if 'light' in col.lower() and 'intensity' in col.lower()]
            if light_status_cols:
                plot_cols.extend(light_status_cols)
                y_label_base = f"Proportion of Time Lights ON ({aggregation_method})"
                main_plot_title = "Hourly Light Status Profile by Group"
            elif light_intensity_cols:
                plot_cols.extend(light_intensity_cols)
                y_label_base = f"Light Intensity ({aggregation_method})"
                main_plot_title = "Hourly Light Intensity Profile by Group"


    if not plot_cols:
        raise ValueError("No light columns specified or found to plot.")

    for col in plot_cols:
        if col not in df.columns:
            raise ValueError(f"Light column '{col}' not found in DataFrame.")

    df_copy = df.copy()
    df_copy['hour'] = df_copy.index.hour
    
    # num_plot_cols = len(plot_cols) # No longer creating one large figure
    script_dir = Path(__file__).parent
    output_plot_dir = script_dir / "plots"
    output_plot_dir.mkdir(parents=True, exist_ok=True)

    # fig, axes = plt.subplots(num_plot_cols, 1, figsize=(15, 5 * num_plot_cols), squeeze=False)
    
    for i, light_col_to_plot in enumerate(plot_cols):
        fig, ax = plt.subplots(figsize=(15, 5)) # Create a new figure and axis for each plot
        
        # Use a more specific title for each individual plot
        current_plot_title = f"{main_plot_title} - {light_col_to_plot}"

        summary_df = df_copy.groupby([group_col, 'hour'])[light_col_to_plot].agg(aggregation_method).unstack(level=0)
        
        if summary_df.empty or summary_df.isnull().all().all():
            ax.text(0.5, 0.5, f"No data to plot for {light_col_to_plot}", ha='center', va='center')
            ax.set_title(f"{current_plot_title} (No Data)")
        else:
        summary_df.plot(kind='line', ax=ax)
            ax.set_title(current_plot_title)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel(y_label_base)
        ax.set_xticks(range(24))
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(title=group_col)

    plt.tight_layout()
        
        # Generate filename from the specific plot title
        safe_specific_title = current_plot_title.replace(" ", "_").replace("/", "-").replace(":", "-").replace("-", "_").lower()
        filename = f"{safe_specific_title}.png"
        plot_save_path = output_plot_dir / filename
        
        try:
            plt.savefig(plot_save_path, bbox_inches='tight')
            print(f"Plot saved to {plot_save_path}")
        except Exception as e_save:
            print(f"Error saving plot {plot_save_path}: {e_save}")
            
        plt.show() # Show and close the current figure before creating the next one
        plt.close(fig) # Explicitly close the figure to free memory

# Function 3: Compare sensor values by group and light state
def plot_sensor_values_by_group_and_light_state(
    df: pd.DataFrame, 
    sensor_col: str, 
    primary_light_status_col: str, 
    group_col: str
):
    """
    Visualizes distributions of a sensor_col, grouped by group_col and 
    the state of primary_light_status_col (ON/OFF, assumed to be 0 or 1).

    Args:
        df (pd.DataFrame): Input DataFrame.
        sensor_col (str): Column name of the sensor to analyze (e.g., 'temperature').
        primary_light_status_col (str): Column name for light status (must be 0 or 1).
        group_col (str): Column name for grouping (e.g., 'group_id').
    """
    for col in [sensor_col, primary_light_status_col, group_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
    
    if not df[primary_light_status_col].dropna().isin([0, 1]).all():
        raise ValueError(f"Column '{primary_light_status_col}' must contain only 0s and 1s.")

    # Reset index to avoid issues with duplicate labels in index if present
    df_plot = df.reset_index() # Work with a copy for plotting

    # Aggressively drop NaNs from key columns for this specific plot
    key_cols_for_boxplot = [group_col, sensor_col, primary_light_status_col]
    df_plot.dropna(subset=key_cols_for_boxplot, inplace=True)

    # Ensure the hue column is treated as categorical (string) by seaborn
    # This can sometimes help with internal seaborn/matplotlib errors
    if primary_light_status_col in df_plot:
        df_plot[primary_light_status_col] = df_plot[primary_light_status_col].astype(str)

    # Define plot_title before the try block where it might be used in an exception message
    plot_title = f"{sensor_col} Distribution by {group_col} and Light Status ({primary_light_status_col})"

    fig, ax = plt.subplots(figsize=(12, 7)) # Explicitly create fig and ax
    try:
        if df_plot.empty:
            print(f"No data available for '{plot_title}' after NaN filtering. Skipping plot.")
            ax.text(0.5, 0.5, f"No data for: {plot_title}\after NaN filtering", ha='center', va='center', wrap=True)
        else:
            sns.boxplot(data=df_plot, x=group_col, y=sensor_col, hue=primary_light_status_col, ax=ax) # Pass ax
            ax.set_title(plot_title)
            ax.set_xlabel(group_col)
            ax.set_ylabel(sensor_col)
            ax.tick_params(axis='x', rotation=45)
            # For legend, if using specific ax, it might need to be handled differently or ensure it's placed well.
            # plt.legend should still work if referring to the current axes implicitly handled by it.
            handles, labels = ax.get_legend_handles_labels()
            if handles: # Only add legend if there are handles (i.e., hue was effective)
                ax.legend(handles, labels, title=f"{primary_light_status_col} (0=OFF, 1=ON)", loc='best')
            else:
                # If no legend items (e.g. hue column had only one value), remove any auto-generated empty legend
                if ax.get_legend() is not None:
                    ax.get_legend().remove()

            ax.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout() # Apply tight_layout to the figure
    except Exception as e_boxplot:
        print(f"Error during sns.boxplot call for '{plot_title}': {e_boxplot}")
        # Optionally, clear the axes if plotting failed to avoid showing an empty/broken plot
        ax.clear()
        ax.text(0.5, 0.5, f"Plotting failed for: {plot_title}\nError: {e_boxplot}", ha='center', va='center', wrap=True)
    plt.tight_layout()

    script_dir = Path(__file__).parent
    output_plot_dir = script_dir / "plots"
    output_plot_dir.mkdir(parents=True, exist_ok=True)
    safe_title = plot_title.replace(" ", "_").replace("/", "-").lower()
    filename = f"{safe_title}.png"
    save_path = output_plot_dir / filename
    try:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    except Exception as e_save:
        print(f"Error saving plot {save_path}: {e_save}")
        
    plt.show()

# Function 4: Plot total daily light ON duration per group
def plot_daily_total_light_on_duration_per_group(
    df: pd.DataFrame, 
    light_status_cols: List[str], 
    group_col: str,
    sample_interval_minutes: Optional[float] = None # Interval in minutes between samples
):
    """
    Calculates and plots the total daily ON duration (in hours) for specified 
    light_status_cols, summed across these lights, for each group. 
    Assumes light_status_cols are 0 (OFF) or 1 (ON).

    Args:
        df (pd.DataFrame): DataFrame with a DatetimeIndex.
        light_status_cols (List[str]): List of column names for light status.
        group_col (str): Column name for grouping.
        sample_interval_minutes (Optional[float]): The sampling interval of the data in minutes.
                                                   If None, tries to infer from timestamps,
                                                   otherwise assumes sum of statuses is count of ON periods.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")
    for col in light_status_cols + [group_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    df_copy = df.copy()
    
    # Sum status across specified light columns for each row
    # This represents the number of active light sources at that timestamp.
    df_copy['total_light_activity_signal'] = df_copy[light_status_cols].sum(axis=1)
    
    # Determine time_unit_multiplier
    time_unit_multiplier: float = 1.0
    y_axis_unit_label: str = "Sum of Active Light Signals"

    if sample_interval_minutes is None:
        # Try to infer from median difference of sorted unique timestamps
        if len(df_copy.index.unique()) > 1:
            median_diff_seconds = pd.Series(df_copy.index.unique()).sort_values().diff().median().total_seconds()
            if median_diff_seconds > 0:
                sample_interval_minutes = median_diff_seconds / 60
                print(f"Inferred sample interval: {sample_interval_minutes:.2f} minutes.")
            else:
                print("Warning: Could not infer sample interval. Duration will be sum of ON signals.")
        else:
            print("Warning: Not enough data points to infer sample interval. Duration will be sum of ON signals.")
            
    if sample_interval_minutes is not None and sample_interval_minutes > 0:
        time_unit_multiplier = sample_interval_minutes / 60 # Convert sum of ON signals to hours
        y_axis_unit_label = "Total Light-Hours"


    # Resample to daily sums of this 'total_light_activity_signal' for each group
    daily_sum_signal = df_copy.groupby(group_col)['total_light_activity_signal'].resample('D').sum()
    daily_on_duration = daily_sum_signal * time_unit_multiplier
    
    # Prepare for plotting
    daily_on_duration_df = daily_on_duration.reset_index()
    time_col_name = df.index.name if df.index.name else 'timestamp' # Default if index has no name
    if time_col_name not in daily_on_duration_df.columns: # Find the datetime column if name changed
        dt_col_candidates = [col for col in daily_on_duration_df.columns if pd.api.types.is_datetime64_any_dtype(daily_on_duration_df[col])]
        if not dt_col_candidates:
            raise ValueError("Could not find datetime column after resampling.")
        time_col_name = dt_col_candidates[0]


    if daily_on_duration_df.empty:
        print("No data to plot for daily light duration.")
        return

    plot_title = f"Daily Total Light Activity per Group"
    plt.figure(figsize=(15, 7))
    sns.lineplot(data=daily_on_duration_df, x=time_col_name, y='total_light_activity_signal', hue=group_col)
    
    plt.title(plot_title)
    plt.xlabel("Date")
    plt.ylabel(y_axis_unit_label)
    plt.legend(title=group_col)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    script_dir = Path(__file__).parent
    output_plot_dir = script_dir / "plots"
    output_plot_dir.mkdir(parents=True, exist_ok=True)
    safe_title = plot_title.replace(" ", "_").replace("/", "-").lower()
    filename = f"{safe_title}.png"
    save_path = output_plot_dir / filename
    try:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    except Exception as e_save:
        print(f"Error saving plot {save_path}: {e_save}")
        
    plt.show()

# Function 5: Plot correlation heatmap for lights and other sensors within groups
def plot_light_sensor_correlation_by_group(
    df: pd.DataFrame, 
    light_cols: List[str], 
    sensor_cols: List[str], 
    group_col: str
):
    """
    Plots correlation heatmaps for specified light columns and sensor columns,
    calculated separately for each group.

    Args:
        df (pd.DataFrame): Input DataFrame.
        light_cols (List[str]): List of column names for light data (status or intensity).
        sensor_cols (List[str]): List of column names for other sensor data.
        group_col (str): Column name for grouping.
    """
    for col in light_cols + sensor_cols + [group_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    unique_groups = df[group_col].unique()
    num_groups = len(unique_groups)

    if num_groups == 0:
        print("No groups found to analyze for correlation.")
        return
    
    # These light_cols and sensor_cols are now treated as *candidates* or *preferred* columns.
    # The function will select from these based on what's available per group.
    # if not light_cols and not sensor_cols:
    #     print("No candidate light or sensor columns specified for correlation heatmap.")
    #     return # Or, try to auto-detect all numeric as a fallback?

    ncols_plot = int(np.ceil(np.sqrt(num_groups))) if num_groups > 0 else 1
    nrows_plot = int(np.ceil(num_groups / ncols_plot)) if num_groups > 0 else 1

    fig, axes = plt.subplots(nrows_plot, ncols_plot, figsize=(6 * ncols_plot, 5 * nrows_plot), squeeze=False)
    axes_flat = axes.flatten()
    
    mappable = None # For shared colorbar
    plot_main_title = "Correlation_Heatmaps_by_Group"

    for i, group_val in enumerate(unique_groups):
        ax = axes_flat[i]
        group_df_full = df[df[group_col] == group_val] # Full data for the current group

        # Dynamically select available columns for this specific group from candidates or all numeric
        current_group_cols_for_corr = []
        if light_cols or sensor_cols: # If specific candidates were provided
            available_light_cols = [col for col in light_cols if col in group_df_full.columns and not group_df_full[col].isnull().all()]
            available_sensor_cols = [col for col in sensor_cols if col in group_df_full.columns and not group_df_full[col].isnull().all()]
            current_group_cols_for_corr = list(set(available_light_cols + available_sensor_cols)) # Use set to ensure uniqueness if overlap
        else: # Fallback: if no candidates, try all numeric columns in this group
            print(f"No specific light/sensor candidates for group {group_val}, trying all numeric columns.")
            current_group_cols_for_corr = [col for col in group_df_full.columns if pd.api.types.is_numeric_dtype(group_df_full[col]) and not group_df_full[col].isnull().all()]
        
        # Remove the group_col itself if it was accidentally included (e.g. if it was numeric and no candidates given)
        if group_col in current_group_cols_for_corr:
            current_group_cols_for_corr.remove(group_col)

        if not current_group_cols_for_corr or len(current_group_cols_for_corr) < 2:
            msg = f"Not enough distinct, non-empty columns\n(min 2 required) for correlation in Group {group_val}.\nFound: {current_group_cols_for_corr}"
            ax.text(0.5, 0.5, msg, ha='center', va='center', wrap=True)
            ax.set_title(f"Group {group_val} - Insufficient Data")
            ax.axis('off')
            continue

        group_df_subset = group_df_full[current_group_cols_for_corr].copy()
        # Drop rows that are all NaN for the selected subset of columns for this group
        group_df_subset.dropna(how='all', inplace=True)
        # Also drop columns that, after row-wise dropna, have become all NaN (if any)
        group_df_subset.dropna(axis=1, how='all', inplace=True)
        
        if group_df_subset.shape[0] < 2 or group_df_subset.shape[1] < 2: # Need at least 2 samples and 2 features
            ax.text(0.5, 0.5, f"Not enough data for Group {group_val}\after filtering for available columns", ha='center', va='center', wrap=True)
            ax.set_title(f"Group {group_val} - Insufficient Data")
            ax.axis('off')
            continue

        corr_matrix = group_df_subset.corr()
        
        if corr_matrix.empty or corr_matrix.isnull().all().all():
            ax.text(0.5, 0.5, f"Correlation matrix is empty or all NaN\nfor Group {group_val}", ha='center', va='center', wrap=True)
            ax.set_title(f"Group {group_val} - Correlation Failed")
            ax.axis('off')
            continue

        current_heatmap_artist = sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, vmin=-1, vmax=1, cbar=False)
        if i == 0 or mappable is None: # Store mappable from the first valid plot, or if previous was invalid
            if current_heatmap_artist.collections:
                mappable = current_heatmap_artist.collections[0]
            # else: mappable remains None or its previous valid value

        ax.set_title(f"Correlation Heatmap - Group {group_val}")
        # Corrected tick_params: ha is not a direct parameter for tick_params
        # For x-axis tick label rotation and alignment, it's better to use plt.setp or ax.set_xticklabels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), rotation=0)
        # ax.tick_params(axis='x', rotation=45) # This would work if ha is not needed
        # ax.tick_params(axis='y', rotation=0)


    # Add a single colorbar for the entire figure if any heatmaps were drawn
    if mappable:
        # Adjust rect in tight_layout to make space for colorbar
        # The colorbar is added to the figure, not specific axes array.
        cbar = fig.colorbar(mappable, ax=axes_flat[:num_groups].tolist(), shrink=0.8, aspect=30, pad=0.05)
        cbar.set_label('Correlation Coefficient')


    # Hide any unused subplots
    for j in range(num_groups, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.tight_layout(rect=[0, 0, 0.95, 1]) # rect might need adjustment depending on figure size and colorbar
    
    script_dir = Path(__file__).parent
    output_plot_dir = script_dir / "plots"
    output_plot_dir.mkdir(parents=True, exist_ok=True)
    # Use a general name for the figure containing multiple heatmaps
    safe_main_title = plot_main_title.replace(" ", "_").replace("/", "-").lower()
    filename = f"{safe_main_title}.png"
    save_path = output_plot_dir / filename
    try:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    except Exception as e_save:
        print(f"Error saving plot {save_path}: {e_save}")
        
    plt.show()

def main():
    """
    Main function to demonstrate loading and analyzing Knud Jepsen data.
    """
    # --- Configuration ---
    # Adjust this path if your script is not in the Proactive-thesis root
    # or if Data/knudjepsen is located elsewhere relative to the script.
    # Assuming the script is run from a location where this relative path is valid.
    knudjepsen_data_path = Path("../../Data/knudjepsen") # Relative to the script's location
    
    # Attempt to make the path more robust, assuming script is within Proactive-thesis
    # and Proactive-thesis is the workspace root.
    workspace_root = Path(__file__).resolve().parents[3] # Adjust index based on actual depth
    knudjepsen_data_path_abs = workspace_root / "Data" / "knudjepsen"

    print(f"Attempting to load data from: {knudjepsen_data_path_abs}")

    # --- 1. Load Data ---
    df_knudjepsen, loaded_file_columns = load_knudjepsen_csvs(knudjepsen_data_path_abs, time_col='timestamp')

    if df_knudjepsen.empty:
        print("Failed to load data or data is empty. Exiting.")
        if loaded_file_columns:
            print("\n--- Column Presence Matrix (based on successfully read headers before full processing) ---")
            all_cols_from_map = set()
            for cols in loaded_file_columns.values():
                all_cols_from_map.update(cols)
            sorted_all_cols = sorted(list(all_cols_from_map))
            
            presence_data = []
            file_names_index = sorted(list(loaded_file_columns.keys()))
            for fname in file_names_index:
                row = [1 if col_name in loaded_file_columns.get(fname, set()) else 0 for col_name in sorted_all_cols]
                presence_data.append(row)
            
            if presence_data:
                column_presence_df = pd.DataFrame(presence_data, index=file_names_index, columns=sorted_all_cols)
                print(column_presence_df)
                print("\nHint: You can visualize the above matrix as a heatmap, e.g., using seaborn:")
                print("import seaborn as sns; import matplotlib.pyplot as plt; sns.heatmap(column_presence_df, annot=True, cbar=False); plt.show()")
                # Save to CSV
                script_dir = Path(__file__).parent
                output_dir = script_dir / "plots" # Save in the same plots directory
                output_dir.mkdir(parents=True, exist_ok=True)
                csv_save_path = output_dir / "column_presence_matrix.csv"
                try:
                    column_presence_df.to_csv(csv_save_path)
                    print(f"Column presence matrix saved to: {csv_save_path}")
                except Exception as e_csv_save:
                    print(f"Error saving column presence matrix to CSV: {e_csv_save}")
        return

    print("\nSuccessfully loaded data. DataFrame info:")
    df_knudjepsen.info()
    print("\nFirst 5 rows of the loaded data:")
    print(df_knudjepsen.head())
    print("\nUnique values in 'status' like columns (if any found by loader):")
    for col in df_knudjepsen.columns:
        if 'status' in col.lower() and pd.api.types.is_integer_dtype(df_knudjepsen[col]):
            print(f"Column '{col}': {df_knudjepsen[col].unique()}")

    # --- Display Column Presence/Absence Matrix ---
    if loaded_file_columns:
        print("\n--- Column Presence Matrix (based on successfully read headers before full processing) ---")
        all_cols_from_map = set()
        for cols in loaded_file_columns.values():
            all_cols_from_map.update(cols)
        sorted_all_cols = sorted(list(all_cols_from_map))
        
        presence_data = []
        file_names_index = sorted(list(loaded_file_columns.keys()))

        for fname in file_names_index:
            # Get the set of columns for the file, default to empty set if fname not found (should not happen)
            cols_in_file = loaded_file_columns.get(fname, set())
            row = [1 if col_name in cols_in_file else 0 for col_name in sorted_all_cols]
            presence_data.append(row)
        
        if presence_data:
            column_presence_df = pd.DataFrame(presence_data, index=file_names_index, columns=sorted_all_cols)
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
                print(column_presence_df)
            print("\nHint: You can visualize the above matrix as a heatmap, e.g., using seaborn:")
            print("import seaborn as sns; import matplotlib.pyplot as plt; sns.heatmap(column_presence_df, annot=True, cbar=False, linewidths=.5); plt.show()")
            
            # Save to CSV
            script_dir = Path(__file__).parent
            output_dir = script_dir / "plots" # Save in the same plots directory
            output_dir.mkdir(parents=True, exist_ok=True)
            csv_save_path = output_dir / "column_presence_matrix.csv"
            try:
                column_presence_df.to_csv(csv_save_path)
                print(f"Column presence matrix saved to: {csv_save_path}")
            except Exception as e_csv_save:
                print(f"Error saving column presence matrix to CSV: {e_csv_save}")
        else:
            print("Could not generate column presence matrix (no file column data found).")
    else:
        print("No file column map data loaded to generate presence matrix.")


    # --- 2. Data Preparation for Visualization ---
    # For some plots, we might need a 'group' column.
    # Let's try to infer or create a placeholder 'group' column.
    # This is a **placeholder** and might need actual logic based on your data.
    # Example: if filenames imply groups or if there's a column already.
    # For now, let's assume all data is one group or try to find a 'group' column.
    
    group_column_name = 'source_group' # Use the source_group generated during loading
    if group_column_name not in df_knudjepsen.columns:
        # This should ideally not happen if load_knudjepsen_csvs correctly adds it
        print(f"Warning: '{group_column_name}' not found. Adding a dummy group column.")
        df_knudjepsen['default_group_id'] = 'default_group' 
        group_column_name = 'default_group_id' # Fallback to a new dummy name
        # df_knudjepsen[group_column_name] = 'default_group' 
        # print(f"\nAdded a dummy '{group_column_name}' column for demonstration purposes.")
    else:
        print(f"\nUsing '{group_column_name}' derived from filenames for grouping visualizations.")

    # Identify light status and intensity columns (heuristic based on names)
    # These heuristics might need adjustment based on actual column names in your CSVs.
    identified_light_status_cols = [
        col for col in df_knudjepsen.columns 
        if ('status' in col.lower() or 'state' in col.lower()) and ('målt' in col.lower() or 'light' in col.lower()) # More specific to 'målt status' like columns
           and pd.api.types.is_integer_dtype(df_knudjepsen[col]) # Check if it's integer after processing
           and df_knudjepsen[col].dropna().isin([0, 1]).all()
    ]
    # For intensity, we need to be careful. If 'mål xxx' are intensities, their names need to reflect that
    # or we need a different heuristic. Let's assume intensity columns might have 'intensity', 'level', or 'mål' (measure)
    # and are numeric but NOT purely binary (0/1) unless explicitly named as intensity.
    identified_light_intensity_cols = [
        col for col in df_knudjepsen.columns 
        if ('intensity' in col.lower() or 'level' in col.lower() or ('mål' in col.lower() and 'light' in col.lower()) ) # Example: 'mål light_intensity'
           and pd.api.types.is_numeric_dtype(df_knudjepsen[col])
           and not (pd.api.types.is_integer_dtype(df_knudjepsen[col]) and df_knudjepsen[col].dropna().isin([0,1]).all() and 'status' not in col.lower()) # Avoid re-classifying processed status cols unless name implies intensity
    ]
    
    other_sensor_cols = [
        col for col in df_knudjepsen.columns 
        if col not in identified_light_status_cols + identified_light_intensity_cols + [group_column_name]
           and pd.api.types.is_numeric_dtype(df_knudjepsen[col]) # Only numeric sensors for correlation
    ][:5] # Limit to a few sensor columns for clarity in correlation plots

    print(f"\nIdentified light status columns: {identified_light_status_cols}")
    print(f"Identified light intensity columns: {identified_light_intensity_cols}")
    print(f"Identified other sensor columns for correlation: {other_sensor_cols}")

    # --- 3. Visualize Data ---
    # Note: These calls assume the columns are correctly identified.
    # You might need to adjust column names based on your actual data.

    # Example 1: Plot hourly light profile (status)
    if identified_light_status_cols:
        print("\nPlotting hourly light status profile...")
        try:
            plot_hourly_light_profile_by_group(
                df_knudjepsen, 
                group_col=group_column_name,
                light_status_cols=identified_light_status_cols
            )
        except Exception as e:
            print(f"Error plotting hourly light status profile: {e}")
    elif identified_light_intensity_cols: # Fallback to intensity if no status cols
        print("\nPlotting hourly light intensity profile...")
        try:
            plot_hourly_light_profile_by_group(
                df_knudjepsen, 
                group_col=group_column_name,
                light_intensity_cols=identified_light_intensity_cols
            )
        except Exception as e:
            print(f"Error plotting hourly light intensity profile: {e}")
    else:
        print("\nSkipping hourly light profile plot: No suitable light status/intensity columns identified.")

    # Example 2: Plot sensor values by group and light state
    # Try to pick columns that are more likely to co-exist for a meaningful boxplot
    sensor_for_boxplot = None
    light_col_for_boxplot = None

    # Prioritize combinations known to exist in some files, e.g., from LAMPEGRP
    if 'målt status' in df_knudjepsen.columns and 'mål temp afd  Mid' in df_knudjepsen.columns:
        light_col_for_boxplot = 'målt status'
        sensor_for_boxplot = 'mål temp afd  Mid'
        print(f"Selected primary pair for boxplot: sensor='{sensor_for_boxplot}', light='{light_col_for_boxplot}'")
    elif 'målt status' in df_knudjepsen.columns and 'CO2 målt' in df_knudjepsen.columns: # Another LAMPEGRP option
        light_col_for_boxplot = 'målt status'
        sensor_for_boxplot = 'CO2 målt'
        print(f"Selected secondary pair for boxplot: sensor='{sensor_for_boxplot}', light='{light_col_for_boxplot}'")
    # Fallback for files like NO3-NO4_belysningsgrp if the above aren't suitable for them
    # This requires identified_light_status_cols (e.g. målt status_1) and other_sensor_cols (e.g. mål FD_1)
    # to be populated and relevant to the *same* source_group primarily.
    elif identified_light_status_cols and other_sensor_cols:
        # Check if the first of each are likely to co-exist in at least one group
        # This is a simple heuristic; a more robust check would involve checking non-NaN counts per group for the pair.
        potential_light_col = identified_light_status_cols[0]
        potential_sensor_col = other_sensor_cols[0]
        # Simple check: Do these columns exist?
        if potential_light_col in df_knudjepsen.columns and potential_sensor_col in df_knudjepsen.columns:
            light_col_for_boxplot = potential_light_col
            sensor_for_boxplot = potential_sensor_col
            print(f"Selected fallback pair for boxplot: sensor='{sensor_for_boxplot}', light='{light_col_for_boxplot}'")

    if light_col_for_boxplot and sensor_for_boxplot:
        print(f"\nAttempting to plot '{sensor_for_boxplot}' by '{group_column_name}' and light state ('{light_col_for_boxplot}')...")
        try:
            plot_sensor_values_by_group_and_light_state(
                df_knudjepsen,
                sensor_col=sensor_for_boxplot,
                primary_light_status_col=light_col_for_boxplot,
                group_col=group_column_name
            )
        except Exception as e:
            print(f"Error plotting sensor values by group and light state: {e}")
    else:
        print("\nSkipping sensor values by light state plot: Could not find a suitable co-existing pair of sensor/light columns.")

    # Example 3: Plot total daily light ON duration
    if identified_light_status_cols:
        print("\nPlotting daily total light ON duration...")
        try:
            # Infer sample interval if possible, otherwise it defaults to sum of signals
            plot_daily_total_light_on_duration_per_group(
                df_knudjepsen,
                light_status_cols=identified_light_status_cols,
                group_col=group_column_name
            )
        except Exception as e:
            print(f"Error plotting daily total light ON duration: {e}")
    else:
        print("\nSkipping daily light ON duration plot: No light status columns identified.")
        
    # Example 4: Plot correlation heatmap
    # Use identified light status or intensity columns, and other sensor columns
    cols_for_corr = []
    if identified_light_status_cols:
        cols_for_corr.extend(identified_light_status_cols)
    elif identified_light_intensity_cols: # If no status, use intensity
        cols_for_corr.extend(identified_light_intensity_cols)
        
    if other_sensor_cols:
        cols_for_corr.extend(other_sensor_cols)

    if len(cols_for_corr) >= 2 and group_column_name in df_knudjepsen.columns:
        # Determine a sensible subset of light_cols and sensor_cols for the heatmap function
        # The function expects separate light_cols and sensor_cols.
        # Let's use one type of light col (status first, then intensity) and other sensors.
        
        heatmap_light_cols = []
        if identified_light_status_cols:
            heatmap_light_cols = identified_light_status_cols
        elif identified_light_intensity_cols:
            heatmap_light_cols = identified_light_intensity_cols
            
        heatmap_sensor_cols = other_sensor_cols
        
        if heatmap_light_cols and heatmap_sensor_cols:
            print("\nPlotting light-sensor correlation heatmap...")
            try:
                plot_light_sensor_correlation_by_group(
                    df_knudjepsen,
                    light_cols=heatmap_light_cols, # Can be status or intensity
                    sensor_cols=heatmap_sensor_cols, # Other numeric sensors
                    group_col=group_column_name
                )
            except Exception as e:
                print(f"Error plotting light-sensor correlation heatmap: {e}")
        elif len(cols_for_corr) >=2 : # Fallback if only one type of column identified (e.g. multiple light status)
             print("\nPlotting general correlation heatmap (not enough distinct light/sensor types for specific function call)...")
             # This case might need a more general correlation plot or adjustments to plot_light_sensor_correlation_by_group
             # For now, we'll try with what we have, assuming light_cols can be a mix if sensor_cols is empty
             temp_light_cols_for_heatmap = cols_for_corr[:len(cols_for_corr)//2] if len(cols_for_corr) > 1 else cols_for_corr
             temp_sensor_cols_for_heatmap = cols_for_corr[len(cols_for_corr)//2:] if len(cols_for_corr) > 1 else []
             if temp_light_cols_for_heatmap: # ensure not empty
                try:
                    plot_light_sensor_correlation_by_group(
                        df_knudjepsen,
                        light_cols=temp_light_cols_for_heatmap, 
                        sensor_cols=temp_sensor_cols_for_heatmap, 
                        group_col=group_column_name
                    )
                except Exception as e:
                    print(f"Error plotting general correlation heatmap: {e}")
             else:
                print("\nSkipping correlation heatmap: Not enough columns for a meaningful correlation plot between distinct light/sensor types.")

    else:
        print("\nSkipping correlation heatmap: Not enough columns or group column missing.")

    print("\n--- Analysis Script Finished ---")

if __name__ == "__main__":
    main()
