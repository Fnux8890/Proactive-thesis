#!/usr/bin/env -S uv run --isolated
# /// script
# dependencies = ["pandas", "pyarrow"]
# ///

import pandas as pd
from pathlib import Path
import argparse # Import argparse

# Define the path to the output directory relative to this script
script_dir = Path(__file__).parent
output_dir = script_dir / 'output'

# Remove hardcoded filename
# parquet_file_name = 'your_file_name.parquet' # <-- Replace this with your actual file name
# parquet_file_path = output_dir / parquet_file_name

def read_parquet_file(file_path: Path) -> pd.DataFrame | None:
    """
    Reads a Parquet file into a pandas DataFrame.

    Args:
        file_path: The path to the Parquet file.

    Returns:
        A pandas DataFrame containing the data, or None if the file is not found
        or an error occurs.
    """
    if not file_path.is_file():
        print(f"Error: File not found at {file_path}")
        return None
    try:
        df = pd.read_parquet(file_path)
        print(f"Successfully read {file_path}")
        return df
    except Exception as e:
        print(f"Error reading Parquet file {file_path}: {e}")
        return None

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Read a Parquet file from the 'output' directory.")
    parser.add_argument("filename", help="The name of the Parquet file in the 'output' directory.")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Construct the full path to the Parquet file
    parquet_file_path = output_dir / args.filename

    # Read the Parquet file
    dataframe = read_parquet_file(parquet_file_path)

    # Display the first few rows of the DataFrame if read successfully
    if dataframe is not None:
        print("\nFirst 5 rows of the DataFrame:")
        print(dataframe.head())

        print("\nDataFrame Info:")
        dataframe.info()
