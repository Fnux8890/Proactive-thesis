#!/usr/bin/env -S uv run --isolated python
# /// script
# dependencies = [
#     "pandas",
#     # Use psycopg[binary] for C extensions, change if pure Python needed
#     "psycopg[binary]", 
#     "numpy", # Needed for describe(include=np.number)
#     "pydantic" # Needed by config_loader.py to load config properly
# ]
# ///

""" 
Script to perform initial database exploration for sensor_data and 
    sensor_data_merged tables using classes from the db_exploration module.

Relies on configuration loaded by config_loader.py and database interactions
handled by explorer.py.
"""

import sys
import time

try:
    from config_loader import get_database_config
    from explorer import DatabaseExplorer
except ImportError as e:
    print(f"Error: Could not import necessary modules. {e}")
    print("Ensure this script is run from the 'db_exploration' directory or that paths are correct.")
    sys.exit(1)

def main():
    """Loads config, connects to DB, runs exploration, and closes connection."""
    print("Starting database exploration script (using uv run)...")

    # 1. Load Configuration
    db_config = get_database_config()
    if not db_config:
        print("Failed to load database configuration. Exiting.")
        return

    # 2. Initialize Explorer (which connects automatically)
    explorer = None # Initialize explorer to None
    try:
        explorer = DatabaseExplorer(db_config)
    except Exception as e:
        print(f"Error during DatabaseExplorer initialization: {e}")
        # Attempt to clean up potential partial connection if explorer object was created
        if explorer:
            explorer.close_connection()
        return # Exit if initialization fails

    # 3. Check if connection was successful during init
    if not explorer.conn or explorer.conn.closed:
        print("Database connection failed during initialization. Cannot proceed.")
        # Explorer's __del__ should handle cleanup, but good practice:
        explorer.close_connection()
        return

    # print("Waiting 10 seconds for DB init scripts to potentially finish...")
    # time.sleep(10) # Removed - Retry logic moved to explorer.py

    # 4. Explore tables & run checks
    try:
        print("\n=== Exploring sensor_data ===")
        explorer.explore_table('sensor_data')

        print("\n=== Exploring sensor_data_merged ===")
        explorer.explore_table('sensor_data_merged')

        print("\n=== Running Quality Checks ===")
        explorer.check_merged_data_quality()

    except Exception as e:
        print(f"\n!!! An error occurred during exploration: {e} !!!")
    finally:
        # 6. Ensure connection is closed regardless of errors during exploration
        print("\nEnsuring database connection is closed...")
        if explorer:
            explorer.close_connection()

    print("\nDatabase exploration script finished.")

if __name__ == "__main__":
    main() 