import pathlib
import pandas as pd
import pyarrow.parquet as pq # For efficiently reading schema
import pprint
from datetime import timedelta # Will be used for aggregate stats

def analyze_parquet_files(parquet_dir_path):
    parquet_dir = pathlib.Path(parquet_dir_path)
    files = sorted(parquet_dir.glob("*.parquet"))

    if not files:
        print(f"No parquet files found in {parquet_dir}")
        return

    # First pass: collect all unique era_ column names from all files' schemas
    all_discovered_era_column_names = set()
    for f_path in files:
        try:
            pq_file = pq.ParquetFile(f_path)
            for col_name in pq_file.schema.names:
                if col_name.startswith("era_"):
                    all_discovered_era_column_names.add(col_name)
        except Exception as e:
            print(f"Warning: Could not read schema from {f_path.name}: {e}")
    
    sorted_all_discovered_era_column_names = sorted(list(all_discovered_era_column_names))

    results = []
    for f_path in files:
        try:
            df = pd.read_parquet(f_path)
        except Exception as e:
            print(f"Error reading full parquet file {f_path.name}, skipping: {e}")
            results.append({"file": f_path.name, "error": str(e)}) # Add error entry
            continue

        # --- 1️⃣ basic facts ---
        n_rows = len(df)
        t_min, t_max = (df["time"].min(), df["time"].max()) if "time" in df.columns and not df.empty else (pd.NaT, pd.NaT)

        # --- 2️⃣ duplicates ---
        n_dupes = df["time"].duplicated().sum() if "time" in df.columns and not df.empty else 0

        # --- 3️⃣ NULL era labels (for columns PRESENT in THIS file) ---
        present_era_cols_in_this_file = [c for c in df.columns if c.startswith("era_")]
        n_null_for_present_cols = 0
        if present_era_cols_in_this_file and not df.empty:
            valid_present_era_cols = [col for col in present_era_cols_in_this_file if col in df.columns]
            if valid_present_era_cols:
                n_null_for_present_cols = df[valid_present_era_cols].isna().any(axis=1).sum()

        current_file_summary = {
            "file": f_path.name,
            "rows": n_rows,
            "from": t_min,
            "to": t_max,
            "dup_timestamps": n_dupes,
            "rows_missing_era_labels_in_present_columns": n_null_for_present_cols,
        }

        # --- 4️⃣ Era Label Details (for ALL discovered era columns) ---
        for master_era_col_name in sorted_all_discovered_era_column_names:
            count_key = f"{master_era_col_name}_unique_count"
            ids_key = f"{master_era_col_name}_ids"

            if master_era_col_name in df.columns and not df.empty:
                if not df[master_era_col_name].isna().all():
                    try:
                        unique_ids_series = df[master_era_col_name].dropna().unique()
                        try:
                            # Attempt to sort; handle potential TypeError for unorderable types
                            unique_ids = sorted(list(unique_ids_series))
                        except TypeError:
                            unique_ids = list(unique_ids_series) # Keep as list, but unsorted
                        current_file_summary[count_key] = len(unique_ids)
                        current_file_summary[ids_key] = unique_ids
                    except Exception as e_uniq: # Catch errors during unique ID processing
                        current_file_summary[count_key] = "Error"
                        current_file_summary[ids_key] = f"Error: {str(e_uniq)[:50]}" # Truncate error msg
                else: # Column exists but is all NaNs
                    current_file_summary[count_key] = 0
                    current_file_summary[ids_key] = []
            else: # This master era column is NOT in the current df's columns (or df is empty)
                current_file_summary[count_key] = 0
                current_file_summary[ids_key] = []


        # --- 5️⃣ Era Segment Details (for columns PRESENT in THIS file) ---
        era_segments_details_for_file = {}
        if "time" in df.columns and not df.empty:
            for era_col in present_era_cols_in_this_file: # Iterate only over era columns present in this file
                if era_col in df.columns and not df[era_col].isna().all(): # Ensure column exists and is not all NaN
                    segments_for_this_era_col = []
                    era_series = df[era_col]
                    time_series = df["time"]

                    # Determine change points: where era_id changes, or where NaN status changes
                    change_points = (era_series != era_series.shift()) | \
                                    (era_series.isna() != era_series.shift().isna())
                    # Get indices of change points, defaulting first change to index 0
                    change_indices = change_points.index[change_points].tolist()
                    
                    # Define segment boundaries including start and end of DataFrame
                    segment_boundaries = sorted(list(set([df.index[0]] + change_indices + [df.index[-1] + 1])))

                    for i in range(len(segment_boundaries) - 1):
                        start_row_idx_loc = segment_boundaries[i] # This is an index label
                        # End row is one before the start of the next segment boundary
                        # The actual end index label for slicing/loc needs to be handled carefully
                        # If segment_boundaries[i+1] is (len(df) or df.index[-1]+1), then true_end_idx_loc is df.index[-1]
                        # Otherwise, it's segment_boundaries[i+1] - 1 (if index is sequential int) or previous index label
                        
                        # Find the actual index label for the end of the segment
                        if segment_boundaries[i+1] == df.index[-1] + 1: # Covers end of DataFrame
                            end_row_idx_loc = df.index[-1]
                        else:
                            # Get the position of the next boundary start, then get the index label before it
                            next_boundary_pos = df.index.get_loc(segment_boundaries[i+1])
                            end_row_idx_loc = df.index[next_boundary_pos - 1]

                        if df.index.get_loc(start_row_idx_loc) > df.index.get_loc(end_row_idx_loc):
                            continue

                        segment_id = era_series.loc[start_row_idx_loc]

                        if pd.isna(segment_id):
                            continue # Skip segments that are defined by NaN values
                        
                        segment_start_time = time_series.loc[start_row_idx_loc]
                        segment_end_time = time_series.loc[end_row_idx_loc]
                        # Number of rows: loc is inclusive for end_row_idx_loc
                        num_rows = df.loc[start_row_idx_loc:end_row_idx_loc].shape[0]
                        
                        segments_for_this_era_col.append({
                            "id": segment_id,
                            "start_time": segment_start_time,
                            "end_time": segment_end_time,
                            "duration": segment_end_time - segment_start_time,
                            "rows": num_rows
                        })
                    
                    if segments_for_this_era_col: # Actual non-NaN segments were found
                        era_segments_details_for_file[era_col] = segments_for_this_era_col

                        # --- Calculate aggregate stats for this era_col ---
                        num_segments = len(segments_for_this_era_col)
                        durations = [seg['duration'] for seg in segments_for_this_era_col]
                        
                        min_duration = min(durations) if durations else pd.NaT
                        max_duration = max(durations) if durations else pd.NaT
                        total_duration_val = sum(durations, timedelta()) if durations else timedelta()
                        avg_duration = total_duration_val / num_segments if num_segments > 0 else pd.NaT
                        
                        current_file_summary[f"{era_col}_num_segments"] = num_segments
                        current_file_summary[f"{era_col}_min_duration"] = min_duration
                        current_file_summary[f"{era_col}_max_duration"] = max_duration
                        current_file_summary[f"{era_col}_avg_duration"] = avg_duration
                        current_file_summary[f"{era_col}_total_segment_duration"] = total_duration_val
                    else: # No non-NaN segments found for this era_col (it was empty or all NaNs effectively)
                        current_file_summary[f"{era_col}_num_segments"] = 0
                        current_file_summary[f"{era_col}_min_duration"] = pd.NaT
                        current_file_summary[f"{era_col}_max_duration"] = pd.NaT
                        current_file_summary[f"{era_col}_avg_duration"] = pd.NaT
                        current_file_summary[f"{era_col}_total_segment_duration"] = timedelta()
        
        current_file_summary["era_segments_detail"] = era_segments_details_for_file
        
        results.append(current_file_summary)

        # --- Print detailed segment breakdown for a specific target file ---
        TARGET_FILE_FOR_DETAILS = "docker_debug_run_single_signal_dli_sum_era_labels_levelA.parquet"
        if f_path.name == TARGET_FILE_FOR_DETAILS and "era_segments_detail" in current_file_summary:
            print(f"\n--- Detailed Era Segments for {f_path.name} ---")
            pprint.pprint(current_file_summary["era_segments_detail"])
            print("---------------------------------------------------\n")

    if results:
        summary_df = pd.DataFrame(results).set_index("file")
        # Ensure list column doesn't get truncated too much in display
        pd.set_option('display.max_colwidth', None)
        # Optional: control max rows printed if summary is very long
        # pd.set_option('display.max_rows', 100)
        print(summary_df)
    else:
        if files: # Files were found, but all had errors during processing
             print("All files encountered errors during processing. No summary to display.")
        # If no files were found initially, that's handled at the start.

if __name__ == "__main__":
    # The script expects the path to be hardcoded or passed differently if it were a CLI tool.
    # For this context, using the previously set path directly.
    parquet_dir_path = r"d:\GitKraken\Proactive-thesis\DataIngestion\feature_extraction\data\processed\era_labels"
    analyze_parquet_files(parquet_dir_path)
