import os

# --- Output Configuration ---
# Format for saving cleaned data: "parquet", "csv", or "db"
SAVE_FORMAT = os.getenv("SAVE_FORMAT", "parquet").lower()

# Parquet engine ('pyarrow' is recommended, 'fastparquet' is an alternative)
PARQUET_ENGINE = 'pyarrow'

# --- Database Configuration (if SAVE_FORMAT is "db") ---
TARGET_TABLE_NAME = "sensor_readings" # Name of the table to save data into

# Schema definition for the target database table.
# **MUST** match the columns produced by the cleaners (long format).
# Adjust types based on your specific database (e.g., TIMESTAMPTZ for PostgreSQL)
TARGET_SCHEMA = {
    "timestamp": "TIMESTAMP WITH TIME ZONE", # Assuming timestamp is timezone-aware
    "value": "DOUBLE PRECISION",
    "measurement": "TEXT",
    "unit": "TEXT",
    "location": "TEXT", # Extracted location (e.g., 'celle5', 'no3')
    "source_file": "TEXT", # Original filename
    "source_identifier": "TEXT", # Unique ID based on source/location/group
    "uuid": "TEXT", # Optional: UUID from JSON source
    # Add other columns if your cleaners produce them
}

# Column name used for creating TimescaleDB hypertable
# MUST exist as a key in TARGET_SCHEMA with a compatible type (e.g., TIMESTAMP, TIMESTAMPTZ)
TIMESCALEDB_TIME_COLUMN = "timestamp"

# Chunk size for writing to database with df.to_sql
DB_CHUNK_SIZE = 10000 