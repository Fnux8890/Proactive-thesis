#!/usr/bin/env -S uv run --isolated
# /// script
# dependencies = ["requests", "pandas", "SQLAlchemy", "psycopg2-binary"]
# ///

import json
import os
import time as py_time  # To avoid conflict with 'time' column
from pathlib import Path

import pandas as pd
import requests
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

# --- Database Utilities Import ---
try:
    from db_utils import SQLAlchemyPostgresConnector
except ImportError:
    print("Warning: Could not import SQLAlchemyPostgresConnector from db_utils.py.")

    class SQLAlchemyPostgresConnector:
        def __init__(self):
            self.engine = None
            print("ERROR: db_utils.SQLAlchemyPostgresConnector not available.")

        def dispose(self):
            pass


# --- Configuration ---
# IMPORTANT: Verify these dataset and column names from energidataservice.dk
EDS_API_BASE_URL = "https://api.energidataservice.dk/dataset"
# Common dataset name for spot prices, but needs verification for historical data format.
# Examples: Elspotprices, ElectricityProdexDeclaration, etc.
EDS_DATASET_NAME = "Elspotprices"  # <<< YOU MUST VERIFY THIS DATASET NAME

# Expected column names from the API (VERIFY THESE!)
# Energi Data Service often uses HourUTC or HourDK for time, PriceArea, SpotPriceDKK or SpotPriceEUR
API_TIME_COLUMN = "HourUTC"  # Or "HourDK"
API_PRICE_AREA_COLUMN = "PriceArea"
API_SPOT_PRICE_COLUMN = "SpotPriceDKK"  # Or "SpotPriceEUR"

PRICE_AREAS = ["DK1", "DK2"]  # Denmark West and East

# Date ranges for fetching data (cover your Eras with some buffer if needed)
HISTORICAL_START_DATE = "2013-11-01"
HISTORICAL_END_DATE = "2016-10-31"

DB_TABLE_NAME = "external_energy_prices_dk"
OUTPUT_DIR = Path(__file__).parent / "output"
ENERGY_REPORT_FILENAME = "fetch_energy_prices_report.txt"

DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")  # Default to localhost for local script runs
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
# DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}" # Handled by connector


def create_energy_price_table_if_not_exists(engine, table_name: str):
    """Creates the energy price data table in the database if it doesn't exist."""
    pk_time_col = API_TIME_COLUMN
    pk_area_col = API_PRICE_AREA_COLUMN
    create_table_sql_refined = f"""
    CREATE TABLE IF NOT EXISTS public.{table_name} (
        \"{pk_time_col}\" TIMESTAMPTZ NOT NULL,
        \"{pk_area_col}\" VARCHAR(10) NOT NULL,
        \"{API_SPOT_PRICE_COLUMN}\" REAL,
        PRIMARY KEY (\"{pk_time_col}\", \"{pk_area_col}\")
    );
    """

    try:
        with engine.begin() as connection:
            connection.execute(text(create_table_sql_refined))
        print(f"Table '{table_name}' ensured to exist for energy prices.")
        return True
    except SQLAlchemyError as e:
        print(f"Error creating/ensuring table '{table_name}': {e}")
        return False


def fetch_energi_data_service_prices(
    start_date: str, end_date: str, price_areas: list[str], dataset_name: str
) -> pd.DataFrame:
    """Fetches spot prices from Energi Data Service API."""

    # Energi Data Service API usually needs dates in YYYY-MM-DD or YYYY-MM-DDTHH:mm format.
    # The API returns data up to, but not including, the end date/time.
    # If fetching full days, ensure end_date is the day AFTER the last day you want data for.
    # For simplicity, assuming start_date and end_date are inclusive for the query here,
    # but the API itself might treat 'end' exclusively.

    # Construct filter string for price areas
    filter_str = f'{{"{API_PRICE_AREA_COLUMN}":{json.dumps(price_areas)}}}'

    # Columns to select
    columns_str = f"{API_TIME_COLUMN},{API_PRICE_AREA_COLUMN},{API_SPOT_PRICE_COLUMN}"

    api_url = f"{EDS_API_BASE_URL}/{dataset_name}"
    params = {
        "start": start_date,
        "end": end_date,  # API typically excludes the end date, adjust if needed by adding 1 day
        "filter": filter_str,
        "columns": columns_str,
        "sort": f"{API_TIME_COLUMN} ASC, {API_PRICE_AREA_COLUMN} ASC",  # Ensure consistent order
        "limit": 0,  # Attempt to get all records for the range
    }

    print(
        f"Fetching energy prices from {start_date} to {end_date} for areas {price_areas} from dataset {dataset_name}."
    )
    print(f"Requesting URL (params separate): {api_url}")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(api_url, params=params, timeout=60)
            print(f"Full API Request URL: {response.url}")  # Log the exact URL
            response.raise_for_status()
            data = response.json()

            # More robust check for records:
            if (
                not isinstance(data, dict)
                or "records" not in data
                or not isinstance(data["records"], list)
            ):
                print(
                    f"Warning: 'records' key not found, not a list, or main response is not a dict for {start_date}-{end_date}, areas {price_areas}."
                )
                print(
                    f"Response JSON: {json.dumps(data, indent=2)[:1000]}"
                )  # Log part of the response
                return pd.DataFrame()

            records = data["records"]
            if not records:  # Handles case where 'records' is an empty list
                print(
                    f"Warning: No 'records' found in API response for {start_date}-{end_date}, areas {price_areas}."
                )
                print(f"Response JSON: {data}")  # Log the full response for debugging
                return pd.DataFrame()

            df = pd.DataFrame(records)
            print(
                f"Successfully fetched {len(df)} records for {start_date}-{end_date}, areas {price_areas}."
            )

            # Data Cleaning and Type Conversion
            if API_TIME_COLUMN not in df.columns:
                print(
                    f"Error: Expected time column '{API_TIME_COLUMN}' not in fetched data. Columns: {df.columns}"
                )
                return pd.DataFrame()

            # Convert time column to datetime and standardize to UTC
            # API time is typically Danish time, need to localize then convert to UTC
            try:
                df[API_TIME_COLUMN] = pd.to_datetime(df[API_TIME_COLUMN])
                if df[API_TIME_COLUMN].dt.tz is None:  # If timezone naive
                    # ASSUMES API returns local Danish time
                    # Handle DST transitions explicitly by converting non-existent times to NaT
                    df[API_TIME_COLUMN] = df[API_TIME_COLUMN].dt.tz_localize(
                        "Europe/Copenhagen",
                        ambiguous="infer",
                        nonexistent="NaT",  # Convert invalid times during DST to NaT
                    )
                df[API_TIME_COLUMN] = df[API_TIME_COLUMN].dt.tz_convert("UTC")

                # Drop rows where time conversion might have resulted in NaT
                original_len = len(df)
                df.dropna(subset=[API_TIME_COLUMN], inplace=True)
                if len(df) < original_len:
                    print(
                        f"  Dropped {original_len - len(df)} rows due to NaT in time column after DST handling."
                    )

            except Exception as e:
                print(
                    f"Error converting time column '{API_TIME_COLUMN}': {e}. Skipping this batch."
                )
                return pd.DataFrame()

            # Ensure price column is numeric
            if API_SPOT_PRICE_COLUMN in df.columns:
                df[API_SPOT_PRICE_COLUMN] = pd.to_numeric(
                    df[API_SPOT_PRICE_COLUMN], errors="coerce"
                )
            else:
                print(f"Warning: Spot price column '{API_SPOT_PRICE_COLUMN}' not found.")

            return df

        except requests.exceptions.RequestException as e:
            print(f"API Request failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                py_time.sleep(5 * (attempt + 1))
            else:
                print(f"Failed to fetch data after {max_retries} attempts.")
                return pd.DataFrame()
        except json.JSONDecodeError as e:
            print(
                f"Error decoding JSON from API response (attempt {attempt + 1}/{max_retries}): {e}"
            )
            print(f"Response text: {response.text[:500]}")
            if attempt < max_retries - 1:
                py_time.sleep(5 * (attempt + 1))
            else:
                print(f"Failed to decode JSON after {max_retries} attempts.")
                return pd.DataFrame()
    return pd.DataFrame()


def save_data_to_db(df: pd.DataFrame, engine, table_name: str) -> bool:
    if df.empty:
        print("DataFrame is empty. Nothing to save to database.")
        return False
    try:
        # If table has composite PK (time, PriceArea), to_sql with 'append' might error on duplicates.
        # For initial load, this is fine if table is empty or data is new.
        # For robust re-runs, an upsert strategy would be needed or pre-delete for the date range.
        df.to_sql(
            table_name,
            engine,
            if_exists="append",
            index=False,
            schema="public",
            method="multi",
            chunksize=1000,
        )
        print(f"Successfully appended {len(df)} rows to table '{table_name}'.")
        return True
    except SQLAlchemyError as e:
        print(f"Error saving data to table '{table_name}': {e}")
        # This might catch PK violations if re-running for same time/area.
        print(
            "This might be due to duplicate primary keys if data for this period/area already exists."
        )
        return False
    except Exception as e:
        print(f"An unexpected error occurred while saving to table '{table_name}': {e}")
        return False


def generate_energy_report(report_items: list, output_dir: Path, filename: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / filename
    print(f"\nGenerating energy price summary report at: {report_path}")
    try:
        with open(report_path, "w") as f:
            for item_type, content in report_items:
                f.write(f"--- {item_type} ---\n")
                if isinstance(content, pd.DataFrame | pd.Series):
                    f.write(content.to_string() + "\n\n")
                elif isinstance(content, dict):
                    for key, val in content.items():
                        f.write(f"{key}: {val}\n")
                    f.write("\n")
                else:
                    f.write(str(content) + "\n\n")
        print(f"Energy price summary report saved successfully to {report_path}")
    except Exception as e:
        print(f"Error saving energy price summary report: {e}")


def main():
    print("--- Starting External Energy Price Data Ingestion ---")
    report_data = []
    report_data.append(("Script Start Time", pd.Timestamp.now(tz="UTC").isoformat()))
    report_data.append(
        (
            "Fetching Parameters",
            {
                "Price Areas": ", ".join(PRICE_AREAS),
                "Start Date": HISTORICAL_START_DATE,
                "End Date": HISTORICAL_END_DATE,
                "Dataset": EDS_DATASET_NAME,
            },
        )
    )

    db_connector = None
    engine = None  # Explicitly define engine
    table_created_successfully = False
    data_saved_successfully = False
    energy_df_for_report = pd.DataFrame()

    try:
        print(
            f"Attempting to connect to DB: User={DB_USER}, Host={DB_HOST}, Port={DB_PORT}, DBName={DB_NAME}"
        )
        db_connector = SQLAlchemyPostgresConnector(
            user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT, db_name=DB_NAME
        )
        if not db_connector.engine:
            raise ConnectionError(
                "Failed to create database engine via SQLAlchemyPostgresConnector."
            )

        engine = db_connector.engine
        print("Database engine obtained from connector.")

        print(f"Attempting to drop table '{DB_TABLE_NAME}' if it exists for a clean run...")
        try:
            with engine.begin() as connection:
                connection.execute(text(f"DROP TABLE IF EXISTS public.{DB_TABLE_NAME};"))
            print(f"Table '{DB_TABLE_NAME}' dropped successfully or did not exist.")
            report_data.append(("Table Drop Status", "Success or Did Not Exist"))
        except SQLAlchemyError as e_drop:
            print(f"Error dropping table '{DB_TABLE_NAME}': {e_drop}. Proceeding to create.")
            report_data.append(("Table Drop Status", f"Error: {e_drop}"))

        table_created_successfully = create_energy_price_table_if_not_exists(engine, DB_TABLE_NAME)
        report_data.append(
            (
                "Table Creation Status",
                "Success" if table_created_successfully else "Failed (check logs)",
            )
        )

        table_exists_check = False
        if engine:
            with engine.connect() as conn_check:
                table_exists_check = engine.dialect.has_table(
                    conn_check, DB_TABLE_NAME, schema="public"
                )

        if table_exists_check:
            # Fetching data in yearly chunks might be more robust for very long periods if API has issues
            # For now, attempting the whole range based on Open-Meteo experience
            print(f"Fetching energy prices from {HISTORICAL_START_DATE} to {HISTORICAL_END_DATE}.")
            energy_df = fetch_energi_data_service_prices(
                start_date=HISTORICAL_START_DATE,
                end_date=HISTORICAL_END_DATE,
                price_areas=PRICE_AREAS,
                dataset_name=EDS_DATASET_NAME,
            )
            report_data.append(("Fetched Records Count", len(energy_df)))

            if not energy_df.empty:
                energy_df_for_report = energy_df.copy()
                data_saved_successfully = save_data_to_db(energy_df, engine, DB_TABLE_NAME)
                report_data.append(
                    ("Data Save to DB Status", "Success" if data_saved_successfully else "Failed")
                )
                report_data.append(("Sample of Fetched Data (Head)", energy_df_for_report.head()))
                report_data.append(
                    ("Basic Stats of Fetched Data (Describe)", energy_df_for_report.describe())
                )
            else:
                print("No energy price data fetched. Database will not be updated.")
                report_data.append(("Data Fetching Outcome", "No data fetched."))
        elif engine:
            print(
                f"Critical Error: Table '{DB_TABLE_NAME}' does not exist and could not be verified. Cannot proceed."
            )
            report_data.append(
                (
                    "DB Table Status",
                    f"Table '{DB_TABLE_NAME}' missing and creation/verification failed.",
                )
            )

    except Exception as e:
        print(f"An error occurred in the main energy price execution: {e}")
        report_data.append(("Overall Execution Error", str(e)))
    finally:
        if db_connector and db_connector.engine:  # Dispose engine via connector
            db_connector.engine.dispose()
            print("Database engine disposed.")

    report_data.append(("Script End Time", pd.Timestamp.now(tz="UTC").isoformat()))
    generate_energy_report(report_data, OUTPUT_DIR, ENERGY_REPORT_FILENAME)
    print("--- External Energy Price Data Ingestion Finished ---")


if __name__ == "__main__":
    main()
