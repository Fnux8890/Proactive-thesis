#!/usr/bin/env -S uv run --isolated
# /// script
# dependencies = ["requests", "pandas", "SQLAlchemy", "psycopg2-binary"] # Corrected: only external packages
# ///

import requests
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from pathlib import Path
import os
import time as py_time # To avoid conflict with 'time' column
import json # Added for fetch_open_meteo_data if it needs to print full JSON on error

# --- Database Utilities Import ---
# Assuming db_utils.py is in the same directory or accessible via PYTHONPATH
try:
    from db_utils import SQLAlchemyPostgresConnector
except ImportError:
    # Fallback if running in a context where db_utils isn't directly in path,
    # though for consistent Docker execution, it should be.
    print("Warning: Could not import SQLAlchemyPostgresConnector from db_utils.py.")
    print("Ensure db_utils.py is in the same directory or PYTHONPATH.")
    # Define a dummy class or exit if it's critical and not found
    class SQLAlchemyPostgresConnector:
        def __init__(self, *args, **kwargs):
            self.engine = None
            print("ERROR: db_utils.SQLAlchemyPostgresConnector not available.")
        def dispose(self): # Add dispose method to dummy
            pass


# --- Configuration ---
QUEENS_LAT: float = 56.16
QUEENS_LONG: float = 10.20

# Define desired hourly variables from Open-Meteo
# Refer to: https://open-meteo.com/en/docs/historical-weather-api
HOURLY_WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain", 
    "snowfall",
    "weathercode", # Weather condition as a WMO code
    "pressure_msl", # Mean sea level pressure in hPa
    "surface_pressure", # Surface pressure in hPa
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "shortwave_radiation", # Global Horizontal Irradiance (GHI) in W/m^2
    "direct_normal_irradiance",
    "diffuse_radiation",   # Diffuse Horizontal Irradiance (DHI)
    "wind_speed_10m",
    "wind_direction_10m"
]

# Date ranges for fetching data (cover your Eras with some buffer if needed)
# Era1: 2013-12-01 to 2014-08-27
# Era2: 2015-09-07 to 2016-09-06
# Fetching a broader range to ensure full coverage for both eras.
# Open-Meteo can handle multi-year requests.
HISTORICAL_START_DATE = "2013-11-01" 
HISTORICAL_END_DATE = "2016-10-31" 

OPEN_METEO_ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"
DB_TABLE_NAME = "external_weather_aarhus"
OUTPUT_DIR = Path(__file__).parent / "output" # Define an output directory for reports
WEATHER_REPORT_FILENAME = "fetch_external_weather_report.txt"

# Database connection (reuse environment variables if defined, or use defaults)
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost") # Default to localhost for local script runs
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
# DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}" # Will be handled by connector

def create_weather_table_if_not_exists(engine, table_name: str):
    """Creates the weather data table in the database if it doesn't exist."""
    # Define columns based on HOURLY_WEATHER_VARS, plus a time primary key
    # All weather variables will be REAL, time will be TIMESTAMPTZ
    cols_sql = ", ".join([f'{var} REAL' for var in HOURLY_WEATHER_VARS])
    
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS public.{table_name} (
        time TIMESTAMPTZ PRIMARY KEY,
        {cols_sql}
    );
    """
    try:
        with engine.begin() as connection: # Use begin for auto-commit/rollback
            connection.execute(text(create_table_sql))
        print(f"Table '{table_name}' ensured to exist with up-to-date columns based on HOURLY_WEATHER_VARS.")
        return True
    except SQLAlchemyError as e:
        print(f"Error creating/ensuring table '{table_name}': {e}")
        return False

def fetch_open_meteo_data(start_date: str, end_date: str, latitude: float, longitude: float, hourly_vars: list) -> pd.DataFrame:
    """Fetches historical weather data from Open-Meteo API for a given period and location."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(hourly_vars),
        "timezone": "GMT" # Request data in UTC for consistency
    }
    print(f"Fetching weather data from {start_date} to {end_date} for lat={latitude}, lon={longitude} with variables: {hourly_vars}")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(OPEN_METEO_ARCHIVE_API_URL, params=params, timeout=60) # Increased timeout
            print(f"API Request URL: {response.url}") # Log the exact URL called
            response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
            
            data = response.json()
            
            if 'hourly' not in data or 'time' not in data['hourly']:
                print(f"Warning: 'hourly' data or 'time' not found in API response for {start_date}-{end_date}. Skipping.")
                print(f"Response JSON: {data}")
                return pd.DataFrame()

            df = pd.DataFrame(data['hourly'])
            df['time'] = pd.to_datetime(df['time'], utc=True) # Ensure time is datetime and UTC
            
            print(f"Successfully fetched {len(df)} hourly records for {start_date} to {end_date}.")
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"API Request failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                py_time.sleep(5 * (attempt + 1)) # Exponential backoff
            else:
                print(f"Failed to fetch data after {max_retries} attempts for period {start_date}-{end_date}.")
                return pd.DataFrame() # Return empty df on persistent failure
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from API response (attempt {attempt + 1}/{max_retries}): {e}")
            print(f"Response text: {response.text[:500]}") # Log part of the response
            if attempt < max_retries - 1:
                py_time.sleep(5 * (attempt + 1))
            else:
                print(f"Failed to decode JSON after {max_retries} attempts for period {start_date}-{end_date}.")
                return pd.DataFrame()
    return pd.DataFrame() # Should be unreachable if retries fail

def save_data_to_db(df: pd.DataFrame, engine, table_name: str) -> bool:
    """Appends data from DataFrame to the specified database table."""
    if df.empty:
        print("DataFrame is empty. Nothing to save to database.")
        return False

    try:
        # For appending, ensure no PK conflicts. If 'time' is PK and unique, this is fine.
        # For re-runs, a more complex upsert or delete-then-insert might be needed if data can overlap.
        # For initial population, 'append' is fine if script is run once or for distinct date ranges.
        df.to_sql(table_name, engine, if_exists='append', index=False, schema='public', method='multi', chunksize=1000)
        print(f"Successfully appended {len(df)} rows to table '{table_name}'.")
        return True
    except SQLAlchemyError as e:
        print(f"Error saving data to table '{table_name}': {e}")
        # Consider logging failed data or attempting individual row inserts if critical
        return False
    except Exception as e: # Catch any other unexpected error during to_sql
        print(f"An unexpected error occurred while saving to table '{table_name}': {e}")
        return False

def generate_weather_report(report_items: list, output_dir: Path, filename: str):
    """Generates a simple text summary report for weather data fetching."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / filename
    print(f"\nGenerating weather summary report at: {report_path}")
    try:
        with open(report_path, 'w') as f:
            for item_type, content in report_items:
                f.write(f"--- {item_type} ---\n")
                if isinstance(content, pd.DataFrame):
                    f.write(content.to_string() + "\n\n")
                elif isinstance(content, pd.Series):
                    f.write(content.to_string() + "\n\n")
                elif isinstance(content, dict):
                    for key, val in content.items():
                        f.write(f"{key}: {val}\n")
                    f.write("\n")
                else:
                    f.write(str(content) + "\n\n")
        print(f"Weather summary report saved successfully to {report_path}")
    except Exception as e:
        print(f"Error saving weather summary report: {e}")

def main():
    """Main script to fetch historical weather data and store it in the database."""
    print("--- Starting External Weather Data Ingestion ---")
    report_data = []
    report_data.append(("Script Start Time", pd.Timestamp.now(tz='UTC').isoformat()))
    report_data.append(("Fetching Parameters", {
        "Latitude": QUEENS_LAT, "Longitude": QUEENS_LONG,
        "Start Date": HISTORICAL_START_DATE, "End Date": HISTORICAL_END_DATE,
        "Hourly Variables": ", ".join(HOURLY_WEATHER_VARS)
    }))
    
    db_connector = None
    engine = None # Explicitly define engine, will be set from connector
    table_created_successfully = False
    data_saved_successfully = False
    weather_df_for_report = pd.DataFrame() # For report sample

    try:
        # engine = get_db_engine() # OLD WAY
        print(f"Attempting to connect to DB: User={DB_USER}, Host={DB_HOST}, Port={DB_PORT}, DBName={DB_NAME}")
        db_connector = SQLAlchemyPostgresConnector(
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            db_name=DB_NAME
        )
        if not db_connector.engine: # Check if connection failed in connector's __init__
            raise ConnectionError("Failed to create database engine via SQLAlchemyPostgresConnector.")
        
        engine = db_connector.engine # Use the engine from the connector
        print("Database engine obtained from connector.")
        
        # --- Drop table before creating --- START ---
        print(f"Attempting to drop table '{DB_TABLE_NAME}' if it exists for a clean run...")
        try:
            with engine.begin() as connection:
                connection.execute(text(f"DROP TABLE IF EXISTS public.{DB_TABLE_NAME};"))
            print(f"Table '{DB_TABLE_NAME}' dropped successfully or did not exist.")
            report_data.append(("Table Drop Status", "Success or Did Not Exist"))
        except SQLAlchemyError as e:
            print(f"Error dropping table '{DB_TABLE_NAME}': {e}. Proceeding to create.")
            report_data.append(("Table Drop Status", f"Error: {e}"))
        # --- Drop table before creating --- END ---
        
        table_created_successfully = create_weather_table_if_not_exists(engine, DB_TABLE_NAME)
        report_data.append(("Table Creation Status", "Success" if table_created_successfully else "Failed (check logs)"))
        
        table_exists_check = False
        if engine: # Check if engine was successfully created/obtained
            with engine.connect() as conn_check:
                table_exists_check = engine.dialect.has_table(conn_check, DB_TABLE_NAME, schema='public')

        if table_exists_check:
            print(f"Fetching all historical data from {HISTORICAL_START_DATE} to {HISTORICAL_END_DATE}.")
            weather_df = fetch_open_meteo_data(
                start_date=HISTORICAL_START_DATE,
                end_date=HISTORICAL_END_DATE,
                latitude=QUEENS_LAT,
                longitude=QUEENS_LONG,
                hourly_vars=HOURLY_WEATHER_VARS
            )
            report_data.append(("Fetched Records Count", len(weather_df)))
            
            if not weather_df.empty:
                weather_df_for_report = weather_df.copy()
                data_saved_successfully = save_data_to_db(weather_df, engine, DB_TABLE_NAME)
                report_data.append(("Data Save to DB Status", "Success" if data_saved_successfully else "Failed"))
                report_data.append(("Sample of Fetched Data (Head)", weather_df_for_report.head()))
                report_data.append(("Basic Stats of Fetched Data (Describe)", weather_df_for_report.describe()))
            else:
                print("No weather data fetched. Database will not be updated.")
                report_data.append(("Data Fetching Outcome", "No data fetched."))
        elif engine: # Only print critical error if engine was fine but table is still missing
            print(f"Critical Error: Table '{DB_TABLE_NAME}' does not exist and could not be verified after creation attempt. Cannot proceed.")
            report_data.append(("DB Table Status", f"Table '{DB_TABLE_NAME}' missing and creation/verification failed."))
            
    except Exception as e:
        print(f"An error occurred in the main execution: {e}")
        report_data.append(("Overall Execution Error", str(e)))
    finally:
        if db_connector and db_connector.engine: # Dispose engine via connector
            db_connector.engine.dispose()
            print("Database engine disposed.")
    
    report_data.append(("Script End Time", pd.Timestamp.now(tz='UTC').isoformat()))
    generate_weather_report(report_data, OUTPUT_DIR, WEATHER_REPORT_FILENAME)
    print("--- External Weather Data Ingestion Finished ---")

if __name__ == "__main__":
    main()
