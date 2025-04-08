"""Configuration settings for the data pipeline."""

import os
from pathlib import Path

# --- Directory Paths ---
# Get paths from environment variables set in docker-compose.yml
# Provide default paths for local development (adjust as needed)
DATA_SOURCE_PATH = Path(os.getenv('DATA_SOURCE_PATH', 'data/'))
OUTPUT_PATH = Path(os.getenv('OUTPUT_PATH', 'cleaned_data/'))

# --- Database Configuration (Example) ---
# Get from environment variable if set
DATABASE_URL = os.getenv('DATABASE_URL') # e.g., "postgresql://user:pass@host:port/db"

# --- File Handling ---
SUPPORTED_FILE_TYPES = [".csv", ".json"]
CSV_READ_PARAMS = {
    # 'sep': ';',       # Removed specific separator - let pandas infer
    'header': 0,      # Assume the first row is the header
    'on_bad_lines': 'warn', # Warn about rows with incorrect number of columns
    # Add other pd.read_csv parameters as needed, e.g.:
    # 'encoding': 'utf-8', 
    # 'low_memory': False,
}
JSON_READ_PARAMS = {
    "lines": True # Assume JSON Lines format
    # Add other pd.read_json parameters as needed
}

# --- Cleaning Parameters (Examples) ---
# Define columns expected to be numeric or datetime *after renaming*
# IMPORTANT: Adjust these lists based on the target column names used in DataCleaner._rename_columns
NUMERIC_COLUMNS = [
    "temperature", 
    "external_temperature", 
    "radiation",
    "co2_measured",
    "curtain_1_position",
    "curtain_2_position",
    # "light_status", # This might be numeric (0/1) or text - check data
    "humidity_deficit",
    "pipe_1_temp",
    "pipe_2_temp",
    # Original example columns - keep if needed, map if they come from elsewhere
    "sensor_value", # Keeping example from original schema
    "humidity"      # Keeping example from original schema
]
DATETIME_COLUMNS = ["timestamp"] # Assuming 'Unnamed: 0' maps to 'timestamp'

# Define columns *after renaming* to identify duplicates
# IMPORTANT: Adjust this list based on target column names
DUPLICATE_SUBSET_COLS = ["timestamp"] # Example: Just use timestamp for now

# Define the target schema for the database table
# IMPORTANT: Adjust column names and types to match your data AND the renaming map!
TARGET_SCHEMA = {
    # Target columns based on example rename mapping
    "timestamp": "TIMESTAMP WITH TIME ZONE NOT NULL",
    "temperature": "DOUBLE PRECISION",
    "external_temperature": "DOUBLE PRECISION",
    "radiation": "DOUBLE PRECISION", # Or INTEGER if always whole numbers
    "co2_measured": "DOUBLE PRECISION", # Or INTEGER
    "curtain_1_position": "DOUBLE PRECISION",
    "curtain_2_position": "DOUBLE PRECISION",
    "light_status": "INTEGER", # Assuming 0/1 after cleaning?
    "humidity_deficit": "DOUBLE PRECISION",
    "pipe_1_temp": "DOUBLE PRECISION",
    "pipe_2_temp": "DOUBLE PRECISION",
    # Original example columns - keep if needed, map if they come from elsewhere
    "sensor_id": "TEXT", 
    "sensor_value": "DOUBLE PRECISION",
    "humidity": "DOUBLE PRECISION",
    "location": "TEXT",
    "measurement_unit": "TEXT",
}

# --- Output Parameters ---
SAVE_FORMAT = "db" # Changed from "parquet"
TARGET_TABLE_NAME = "sensor_data" # Define a single target table
DB_SAVE_METHOD = "append" # Changed back to append for normal operation
DB_CHUNK_SIZE = 10000      # Insert data in batches
TIMESCALEDB_TIME_COLUMN = "timestamp" # IMPORTANT: Change this to your actual time column name!
# PARQUET_ENGINE = "pyarrow" # No longer primary, but keep if needed elsewhere

print("--- Configuration Loaded ---")
print(f"Data Source: {DATA_SOURCE_PATH.resolve()}")
print(f"Output Path: {OUTPUT_PATH.resolve()}")
print(f"Database URL Set: {bool(DATABASE_URL)}")
print(f"Save Format: {SAVE_FORMAT}")
print(f"Target DB Table: {TARGET_TABLE_NAME}")
print(f"DB Save Method: {DB_SAVE_METHOD}")
print(f"TimescaleDB Time Column: {TIMESCALEDB_TIME_COLUMN}")
print("-------------------------") 