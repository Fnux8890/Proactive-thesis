import csv
from pathlib import Path
from datetime import datetime, timezone
import logging
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
# Assuming the script runs from Jupyter/data_analysis
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = (SCRIPT_DIR / '../../Data').resolve()
OUTPUT_FILE = SCRIPT_DIR / 'metadata_output.txt'
SUBDIRS_TO_SCAN = ['aarslev', 'knudjepsen']
DELIMITER = ';'
# Common names for timestamp columns based on previous Java code context
TIMESTAMP_COLUMN_NAMES = ['timest', 'timestamp']
# Threshold to consider a number a potential Unix timestamp (e.g., > year 2000)
TIMESTAMP_THRESHOLD = 946684800 # 2000-01-01 UTC

def is_potential_unix_timestamp(value: str) -> bool:
    """Checks if a string represents a potential Unix timestamp (integer > threshold)."""
    try:
        # Check if it's a large integer, potentially with decimals to ignore
        ts = int(float(value))
        return ts > TIMESTAMP_THRESHOLD
    except (ValueError, TypeError):
        return False

def parse_timestamp(value: str) -> Optional[int]:
    """Safely parses a string into an integer timestamp."""
    try:
        # Handle potential float strings from CSVs
        return int(float(value))
    except (ValueError, TypeError):
        return None

def format_timestamp(timestamp: Optional[int]) -> str:
    """Formats an integer timestamp into an ISO 8601 string in UTC."""
    if timestamp is None:
        return "N/A"
    try:
        return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
    except (ValueError, OSError): # Handle invalid timestamp values
        return "Invalid Timestamp"

def find_timestamp_column(header: List[str], first_data_row: Optional[List[str]]) -> Optional[int]:
    """
    Identifies the timestamp column index based on common names or data format.
    """
    # Try finding by common names first
    for i, col_name in enumerate(header):
        if col_name.lower() in TIMESTAMP_COLUMN_NAMES:
            # Basic check if the first data row looks like a timestamp in this column
            if first_data_row and len(first_data_row) > i and is_potential_unix_timestamp(first_data_row[i]):
                 logging.debug(f"Found timestamp column by name: '{col_name}' at index {i}")
                 return i
            else:
                 logging.debug(f"Column name '{col_name}' matches but data doesn't look like timestamp.")

    # If not found by name, check the first column if it looks like a timestamp
    if first_data_row and len(first_data_row) > 0 and is_potential_unix_timestamp(first_data_row[0]):
         logging.debug(f"Found potential timestamp column by data format in first column (index 0)")
         return 0

    logging.debug("No timestamp column identified.")
    return None

def extract_metadata_from_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Extracts metadata from a single CSV file.

    Args:
        file_path: Path object pointing to the CSV file.

    Returns:
        A dictionary containing metadata, or None if an error occurs.
    """
    metadata: Dict[str, Any] = {
        "file_path": str(file_path.relative_to(DATA_DIR.parent)), # Relative to project root
        "row_count": 0,
        "column_count": 0,
        "headers": [],
        "min_timestamp": None,
        "max_timestamp": None,
        "timestamp_column_index": None,
        "error": None
    }
    min_ts: Optional[int] = None
    max_ts: Optional[int] = None

    try:
        with file_path.open('r', encoding='utf-8', errors='replace') as csvfile:
            # Handle potential large fields
            csv.field_size_limit(1024 * 1024 * 10) # 10 MB limit

            reader = csv.reader(csvfile, delimiter=DELIMITER)

            # Read header
            try:
                header = next(reader)
                metadata["headers"] = [h.strip() for h in header]
                metadata["column_count"] = len(header)
            except StopIteration:
                logging.warning(f"File is empty or has no header: {file_path}")
                metadata["error"] = "Empty or no header"
                return metadata
            except csv.Error as e:
                 logging.error(f"CSV reading error in header for {file_path}: {e}")
                 metadata["error"] = f"CSV header read error: {e}"
                 return metadata

            # Read first data row to help identify timestamp column
            first_data_row: Optional[List[str]] = None
            try:
                 first_data_row = next(reader)
                 metadata["row_count"] = 1 # Start counting
            except StopIteration:
                 logging.info(f"File has header but no data rows: {file_path}")
                 # Keep going to report header info, but timestamps will be N/A
                 pass
            except csv.Error as e:
                 logging.error(f"CSV reading error in first data row for {file_path}: {e}")
                 metadata["error"] = f"CSV first data row read error: {e}"
                 return metadata # Cannot determine timestamp column reliably

            # Identify timestamp column
            ts_col_idx = find_timestamp_column(metadata["headers"], first_data_row)
            metadata["timestamp_column_index"] = ts_col_idx

            # Process first data row if it exists and timestamp column identified
            if first_data_row and ts_col_idx is not None and len(first_data_row) > ts_col_idx:
                 ts = parse_timestamp(first_data_row[ts_col_idx])
                 if ts is not None:
                     min_ts = ts
                     max_ts = ts

            # Process remaining rows
            row_num = 2 # Start from row 2 (1-based index)
            for row in reader:
                metadata["row_count"] += 1
                row_num += 1
                if ts_col_idx is not None and len(row) > ts_col_idx:
                    current_ts = parse_timestamp(row[ts_col_idx])
                    if current_ts is not None:
                        if min_ts is None or current_ts < min_ts:
                            min_ts = current_ts
                        if max_ts is None or current_ts > max_ts:
                            max_ts = current_ts
                elif ts_col_idx is not None and len(row) <= ts_col_idx:
                     logging.warning(f"Row {row_num} in {file_path} is shorter than expected timestamp column index {ts_col_idx}.")
                # Handle rows with different column counts than header
                if len(row) != metadata["column_count"]:
                    logging.warning(f"Row {row_num} in {file_path} has {len(row)} columns, expected {metadata['column_count']}.")


            metadata["min_timestamp"] = min_ts
            metadata["max_timestamp"] = max_ts
            logging.info(f"Successfully processed: {file_path.name}")

    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        metadata["error"] = "File not found"
    except PermissionError:
        logging.error(f"Permission denied for file: {file_path}")
        metadata["error"] = "Permission denied"
    except UnicodeDecodeError as e:
        logging.error(f"Encoding error in file {file_path}: {e}")
        metadata["error"] = f"Encoding error: {e}"
    except Exception as e:
        logging.exception(f"An unexpected error occurred processing {file_path}: {e}")
        metadata["error"] = f"Unexpected error: {e}"

    return metadata

def main() -> None:
    """
    Main function to scan directories, extract metadata, and write to output file.
    """
    logging.info(f"Starting metadata extraction from: {DATA_DIR}")
    logging.info(f"Scanning subdirectories: {", ".join(SUBDIRS_TO_SCAN)}")
    logging.info(f"Output will be saved to: {OUTPUT_FILE}")

    all_metadata: List[Dict[str, Any]] = []
    file_count = 0

    for subdir_name in SUBDIRS_TO_SCAN:
        subdir_path = DATA_DIR / subdir_name
        if not subdir_path.is_dir():
            logging.warning(f"Subdirectory not found: {subdir_path}")
            continue

        logging.info(f"Scanning directory: {subdir_path}...")
        for file_path in subdir_path.rglob('*.csv'):
             if file_path.is_file():
                 file_count += 1
                 logging.debug(f"Processing file: {file_path}")
                 metadata = extract_metadata_from_file(file_path)
                 if metadata:
                     all_metadata.append(metadata)

    logging.info(f"Finished scanning. Found {file_count} CSV files. Extracted metadata for {len(all_metadata)} files.")

    # Write metadata to output file
    try:
        with OUTPUT_FILE.open('w', encoding='utf-8') as outfile:
            outfile.write(f"Metadata Extraction Report - {datetime.now(timezone.utc).isoformat()}\n")
            outfile.write(f"Scanned Directory: {DATA_DIR}\n")
            outfile.write(f"Subdirectories Scanned: {", ".join(SUBDIRS_TO_SCAN)}\n")
            outfile.write(f"Total CSV Files Found: {file_count}\n")
            outfile.write("=" * 80 + "\n\n")

            for i, meta in enumerate(all_metadata):
                outfile.write(f"--- File {i+1} ---\n")
                outfile.write(f"Path: {meta['file_path']}\n")
                if meta['error']:
                    outfile.write(f"Status: ERROR - {meta['error']}\n")
                else:
                    outfile.write(f"Status: OK\n")
                    outfile.write(f"Data Rows: {meta['row_count']}\n")
                    outfile.write(f"Columns: {meta['column_count']}\n")
                    outfile.write(f"Headers: {", ".join(meta['headers'])}\n")
                    outfile.write(f"Timestamp Column Index: {'N/A' if meta['timestamp_column_index'] is None else meta['timestamp_column_index']}\n")
                    outfile.write(f"Min Timestamp: {format_timestamp(meta['min_timestamp'])}\n")
                    outfile.write(f"Max Timestamp: {format_timestamp(meta['max_timestamp'])}\n")
                outfile.write("\n")

        logging.info(f"Metadata successfully written to {OUTPUT_FILE}")

    except IOError as e:
        logging.exception(f"Failed to write output file {OUTPUT_FILE}: {e}")

if __name__ == "__main__":
    main()
