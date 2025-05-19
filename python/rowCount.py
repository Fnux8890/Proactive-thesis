#!/usr/bin/env -S uv run --script python
# /// script
# dependencies = [
#     "psycopg[binary]"
# ]
# ///
"""
Counts the total number of rows in CSV and JSON files within the ../Data directory,
and also queries the row count for the 'sensor_data' table in the TimescaleDB.

For CSV files, it counts the number of rows excluding the header.
For JSON files, it assumes the top-level structure is a list of objects
and counts the number of objects in the list.
It also handles newline-delimited JSON (JSONL) by counting lines.
"""

import csv
import json
import os
import sys
from pathlib import Path
from typing import Union, Optional
import psycopg # Import directly now

# --- File Counting Functions ---

def count_csv_rows(file_path: Path) -> int:
    """Counts rows in a CSV file, excluding the header."""
    try:
        with file_path.open('r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            # Skip header if it exists
            try:
                next(reader)
                count = sum(1 for _ in reader)
            except StopIteration:
                count = 0 # File was empty or only had a header
            return count
    except FileNotFoundError:
        print(f"Error: File not found {file_path}", file=sys.stderr)
        return 0
    except csv.Error as e:
        print(f"Error reading CSV {file_path}: {e}", file=sys.stderr)
        return 0
    except Exception as e:
        print(f"An unexpected error occurred with {file_path}: {e}", file=sys.stderr)
        return 0

def count_json_rows(file_path: Path) -> int:
    """
    Counts rows in a JSON file.
    Assumes either a top-level list or newline-delimited JSON (JSONL).
    """
    try:
        with file_path.open('r', encoding='utf-8', errors='ignore') as f:
            first_char = f.read(1)
            f.seek(0) # Reset file pointer

            if first_char == '[': # Assume list of objects
                data = json.load(f)
                if isinstance(data, list):
                    return len(data)
                else:
                    print(f"Warning: JSON file {file_path} does not contain a top-level list.", file=sys.stderr)
                    return 0
            elif first_char == '{': # Assume JSONL (or single object)
                # Count lines for JSONL
                count = sum(1 for line in f if line.strip())
                # Check if it might have been a single object misinterpreted as JSONL
                if count == 1:
                     try:
                         f.seek(0)
                         data = json.load(f)
                         if isinstance(data, dict) and len(data) > 0:
                             print(f"Warning: JSON file {file_path} seems to be a single object, interpreting as 1 row.", file=sys.stderr)
                             # Decide if a single object counts as 1 row or 0. Let's say 1 for now.
                             return 1
                     except json.JSONDecodeError:
                         pass # It was likely malformed or genuinely JSONL with one line
                return count
            else: # Empty or unknown format
                 print(f"Warning: Could not determine JSON format for {file_path}. Counting as 0 rows.", file=sys.stderr)
                 return 0

    except FileNotFoundError:
        print(f"Error: File not found {file_path}", file=sys.stderr)
        return 0
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON {file_path}: {e}", file=sys.stderr)
        # Attempt to count lines as fallback for potential JSONL with errors
        try:
            with file_path.open('r', encoding='utf-8', errors='ignore') as f_fallback:
                 count = sum(1 for line in f_fallback if line.strip())
                 if count > 0:
                     print(f"Info: Fallback - counted {count} lines in potentially malformed JSON/JSONL file {file_path}.", file=sys.stderr)
                     return count
                 else:
                     return 0
        except Exception as fallback_e:
             print(f"Error during fallback line count for {file_path}: {fallback_e}", file=sys.stderr)
             return 0

    except Exception as e:
        print(f"An unexpected error occurred with {file_path}: {e}", file=sys.stderr)
        return 0

# --- Database Functions ---

def get_db_connection() -> Optional[psycopg.Connection]:
    """Establishes a connection to the TimescaleDB database."""
    # No longer need psycopg check here, uv ensures it's installed

    # TODO: Consider using environment variables for credentials
    db_host = "localhost" # Connect via mapped port on host
    db_user = "postgres"
    db_pass = "postgres"
    db_name = "postgres" # Default db, change if needed

    conn_string = f"host={db_host} dbname={db_name} user={db_user} password={db_pass}"

    try:
        conn = psycopg.connect(conn_string)
        print("\nSuccessfully connected to the database.")
        return conn
    except psycopg.OperationalError as e:
        print(f"\nError connecting to the database: {e}", file=sys.stderr)
        print("Please ensure the database container is running and accessible.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"\nAn unexpected error occurred during DB connection: {e}", file=sys.stderr)
        return None

def get_db_table_row_count(conn: psycopg.Connection, table_name: str) -> Optional[int]:
    """Queries the database for the row count of a specific table."""
    if conn is None:
        return None

    try:
        with conn.cursor() as cur:
            # Use sql module for safe quoting
            query = psycopg.sql.SQL("SELECT COUNT(*) FROM {}").format(psycopg.sql.Identifier(table_name))
            cur.execute(query)
            result = cur.fetchone()
            if result:
                return result[0]
            else:
                return 0 # Should not happen with COUNT(*), but good practice
    except psycopg.Error as e:
        print(f"Error querying row count for table '{table_name}': {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"An unexpected error occurred during row count query for '{table_name}': {e}", file=sys.stderr)
        return None

# --- Main Execution ---  

def main():
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "Data"

    if not data_dir.is_dir():
        print(f"Error: Data directory not found at {data_dir}", file=sys.stderr)
        sys.exit(1)

    total_csv_rows = 0
    total_json_rows = 0
    csv_files_count = 0
    json_files_count = 0

    print(f"Scanning directory: {data_dir}")

    # Use rglob to find files recursively
    files = list(data_dir.rglob('*.*')) # Get all files first

    # --- File Counting Loop ---
    for file_path in files:
        if file_path.is_file():
            # Skip files with the double extension .csv.json
            if file_path.name.lower().endswith('.csv.json'):
                print(f"Skipping file with double extension: {file_path.relative_to(data_dir)}")
                continue

            if file_path.suffix.lower() == '.csv':
                print(f"Processing CSV: {file_path.relative_to(data_dir)}...")
                rows = count_csv_rows(file_path)
                total_csv_rows += rows
                csv_files_count += 1
                print(f"  Rows (excluding header): {rows}")
            elif file_path.suffix.lower() == '.json':
                print(f"Processing JSON: {file_path.relative_to(data_dir)}...")
                rows = count_json_rows(file_path)
                total_json_rows += rows
                json_files_count += 1
                print(f"  Rows (estimated): {rows}")

    # --- File Summary --- 
    print("\n--- File Summary ---")
    print(f"Total CSV files processed: {csv_files_count}")
    print(f"Total rows in CSV files (excluding headers): {total_csv_rows}")
    print(f"Total JSON files processed: {json_files_count}")
    print(f"Total estimated rows in JSON files: {total_json_rows}")
    print(f"Grand total estimated rows (CSV + JSON): {total_csv_rows + total_json_rows}")

    # --- Database Querying ---
    db_row_count: Optional[int] = None # Initialize for comparison scope
    if psycopg is None:
        print("\nSkipping database query because psycopg library is not available.", file=sys.stderr)
    else:
        db_conn = get_db_connection()
        if db_conn:
            table_to_query = "sensor_data"
            print(f"\nQuerying database for row count in table '{table_to_query}'...")
            db_row_count = get_db_table_row_count(db_conn, table_to_query)

            print("\n--- Database Summary ---")
            if db_row_count is not None:
                print(f"Total rows in database table '{table_to_query}': {db_row_count}")
            else:
                print(f"Could not retrieve row count for table '{table_to_query}'.")

            # Close the connection
            db_conn.close()
            print("Database connection closed.")
        else:
             print("\nSkipping database query due to connection failure.", file=sys.stderr)

    # --- Comparison ---
    print("\n--- Comparison ---")
    total_file_rows = total_csv_rows + total_json_rows
    if db_row_count is not None:
        print(f"Total rows found in local files (CSV+JSON): {total_file_rows}")
        print(f"Total rows found in database table 'sensor_data': {db_row_count}")
        if total_file_rows == db_row_count:
            print("Row counts match! ✅")
        else:
            difference = abs(total_file_rows - db_row_count)
            print(f"Row counts DO NOT match. Difference: {difference} ❌")
            # Note: This is a simple comparison. It assumes all counted files were intended for the sensor_data table.
            # Discrepancies could arise from files not yet ingested, different counting methods (e.g., JSON), or ingestion errors.
    else:
        print("Cannot compare counts as database row count was not retrieved.")

if __name__ == "__main__":
    main()
