#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10, <3.13"
# dependencies = [
#   "pandas",
#   "ydata-profiling",
#   "setuptools",
# ]
# ///

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
from ydata_profiling import ProfileReport


def analyze_data(file_path: Path):
    """
    Analyzes a CSV or JSON data file and generates a profile report.

    Args:
        file_path: Path to the data file (CSV or JSON).
    """
    if not file_path.is_file():
        print(f"Error: File not found at {file_path}", file=sys.stderr)
        sys.exit(1)

    file_suffix = file_path.suffix.lower()
    df = None

    try:
        print(f"Reading file: {file_path}...")
        if file_suffix == ".csv":
            df = pd.read_csv(file_path)
        elif file_suffix == ".json":
            # Attempt to read line-delimited JSON first
            try:
                df = pd.read_json(file_path, lines=True)
            except ValueError:
                # If not line-delimited, try reading as a standard JSON array/object
                print("Failed to read as line-delimited JSON, trying standard JSON...")
                with open(file_path, 'r') as f:
                    data = json.load(f)
                # Handle potential nested structures or different JSON formats
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    # Might need more specific handling depending on JSON structure
                    df = pd.DataFrame([data])
                else:
                    raise ValueError("Unsupported JSON structure")
        else:
            print(f"Error: Unsupported file type '{file_suffix}'. Please provide a .csv or .json file.", file=sys.stderr)
            sys.exit(1)

        print("Successfully read data. Generating profile report...")

        # Generate the profile report
        profile = ProfileReport(df, title=f"Profiling Report for {file_path.name}", explorative=True)

        # Save the report to an HTML file
        output_filename = file_path.stem + "_profile.html"
        output_path = file_path.parent / output_filename
        profile.to_file(output_path)

        print(f"Successfully generated report: {output_path}")

    except pd.errors.EmptyDataError:
        print(f"Error: The file {file_path} is empty.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during processing: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a data profiling report for a CSV or JSON file.")
    parser.add_argument("file", help="Path to the input data file (.csv or .json)")
    args = parser.parse_args()

    # Resolve the input file path relative to the script's directory
    script_dir = Path(__file__).resolve().parent
    input_path = (script_dir / Path(args.file)).resolve()

    analyze_data(input_path) 