import pandas as pd
import json
import sys
from pathlib import Path # Added for robust path construction

def main():
    if len(sys.argv) != 3:
        print("Usage: python auto_era_config_gen.py <input_level_c_parquet_path> <output_json_path>")
        sys.exit(1)

    era_file_path = Path(sys.argv[1])
    output_json_path = Path(sys.argv[2])

    if not era_file_path.exists():
        print(f"Error: Input Parquet file not found at {era_file_path}")
        sys.exit(1)

    print(f"Loading Level C era labels from: {era_file_path}")
    df = pd.read_parquet(era_file_path)

    # Ensure 'time' column is datetime and sort
    if 'time' not in df.columns:
        print("Error: 'time' column not found in Parquet file.")
        sys.exit(1)
    
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values("time").set_index('time') # Set time as index for easier processing

    if 'era_level_C' not in df.columns:
        print("Error: 'era_level_C' column not found in Parquet file.")
        sys.exit(1)

    print("Detecting era changes based on 'era_level_C'...")
    # Create a boolean series where True indicates a change in 'era_level_C'
    # Fill NaN for the first element's diff (which is always True for a change)
    df["era_change"] = df["era_level_C"].diff().ne(0)
    df.loc[df.index[0], "era_change"] = True # The very first record starts a new era segment

    # Identify start and end timestamps of each era segment
    eras = []
    current_era_start_time = None

    for timestamp, row in df.iterrows():
        if row["era_change"]:
            if current_era_start_time is not None:
                # End previous era just before the current change
                # The previous timestamp is df.index[df.index.get_loc(timestamp)-1]
                previous_timestamp = df.index[df.index.get_loc(timestamp)-1]
                eras.append((current_era_start_time, previous_timestamp))
            current_era_start_time = timestamp
            
    # Add the last era segment
    if current_era_start_time is not None:
        eras.append((current_era_start_time, df.index[-1]))
    
    print(f"Detected {len(eras)} dynamic eras.")

    auto_defs = {}
    # Determine the common base path from the input file to guess the original segment identifier
    # e.g. if era_file_path is /app/data/processed/MegaEra1_era_labels_levelC.parquet -> MegaEra1
    # This is a heuristic. A more robust way would be to pass the suffix explicitly.
    # For now, we'll use a generic prefix.
    # Or, better, let's derive it from the input filename if it follows the pattern <suffix>_era_labels_levelC.parquet
    input_filename_stem = era_file_path.stem # e.g., "MySegment_era_labels_levelC"
    suffix_match = input_filename_stem.replace("_era_labels_levelC", "")
    era_name_prefix = f"AutoEra_C_{suffix_match}" if suffix_match and suffix_match != input_filename_stem else "AutoEra_C"


    for i, (start_time, end_time) in enumerate(eras):
        # Ensure start_time and end_time are pandas Timestamps for isoformat
        start_iso = pd.Timestamp(start_time).isoformat()
        end_iso = pd.Timestamp(end_time).isoformat()
        
        era_key = f"{era_name_prefix}_Seg{i+1}"
        auto_defs[era_key] = {
            "description": f"Auto-detected Level-C era segment {i+1} from {suffix_match}",
            "db_table": "public.sensor_data_merged", # Default, might need to be configurable
            "start_date": start_iso,
            "end_date": end_iso,
            "target_frequency": "5T", # Default, make configurable if needed
            "outlier_rules_ref": "default_outlier_rules", # Default
            "imputation_rules_ref": "default_imputation_rules", # Default, plan suggested era2, make configurable
            "boolean_columns_to_int": ["co2_status"], # Example, make configurable
            "dead_columns_to_drop": [], # Default empty
            "era_feature_file": f"{suffix_match}_era_labels_levelC.parquet" # Point to the source of this auto-era
        }
    
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump({"auto_era_definitions": auto_defs}, f, indent=2)
    
    print(f"Successfully generated auto era definitions at: {output_json_path}")
    print(f"Summary: {len(auto_defs)} era definitions created.")

if __name__ == "__main__":
    main() 