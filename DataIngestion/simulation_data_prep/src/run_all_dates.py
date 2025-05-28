#!/usr/bin/env python
# DataIngestion/simulation_data_prep/src/run_all_dates.py
import os
import asyncio
import logging
import argparse # <-- Import argparse
from datetime import date, timedelta, datetime

import psycopg
from psycopg.rows import tuple_row

# Workaround for Prefect 3.x internal initialization issue (see: https://linen.prefect.io/t/23222690/)
from prefect import flow # Import flow early to trigger necessary initializations

from prefect.client.orchestration import get_client
from prefect.states import Scheduled # Import the Scheduled state
# REMOVED: from prefect.deployments import Deployment # Obsolete in Prefect 3+

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Database Connection (reuse or adapt from db_connector.py) ---
def get_db_connection_sync():
    """Establishes a synchronous connection."""
    try:
        # Use DB_HOST_LOCAL explicitly if needed for running outside compose network sometimes
        # Or rely on docker-compose service name 'db' if running within compose
        host = os.getenv('DB_HOST', 'db') # Default to 'db' service name
        port = os.getenv('DB_PORT', '5432')
        user = os.getenv('DB_USER')
        password = os.getenv('DB_PASSWORD')
        dbname = os.getenv('DB_NAME')

        if not all([user, password, dbname]):
            missing_vars = []
            if not user: missing_vars.append("DB_USER")
            if not password: missing_vars.append("DB_PASSWORD")
            if not dbname: missing_vars.append("DB_NAME")
            logger.error(f"Database connection failed: Missing environment variables: {missing_vars}")
            return None

        conn_str = f"host={host} port={port} user={user} password={password} dbname={dbname}"
        logger.debug(f"Attempting DB connection with string: host={host} port={port} user={user} dbname={dbname} password=***")

        conn = psycopg.connect(conn_str)
        logger.info("Sync DB connection established.")
        return conn
    except psycopg.OperationalError as e:
        logger.error(f"Sync DB connection failed (OperationalError): {e}")
        return None
    except Exception as e:
        logger.exception(f"An unexpected error occurred during sync DB connection: {e}")
        return None

# --- Get Date Range --- #
def get_date_range(conn):
    """Queries sensor_data for min and max dates."""
    if not conn or conn.closed:
        logger.error("Cannot get date range: DB connection is closed.")
        return None, None
    try:
        with conn.cursor(row_factory=tuple_row) as cur:
            # Ensure the time column is quoted if it's case-sensitive or reserved
            cur.execute('SELECT MIN("time")::date, MAX("time")::date FROM sensor_data')
            result = cur.fetchone()
            if result and result[0] is not None and result[1] is not None:
                min_date, max_date = result
                # Convert to date objects if they are not already
                if isinstance(min_date, datetime):
                    min_date = min_date.date()
                if isinstance(max_date, datetime):
                    max_date = max_date.date()
                logger.info(f"Date range found: {min_date} to {max_date}")
                return min_date, max_date
            else:
                logger.warning("Could not determine date range from sensor_data (no data or nulls?).")
                return None, None
    except Exception as e:
        logger.error(f"Error querying date range: {e}")
        return None, None
    finally:
         if conn and not conn.closed:
             conn.close()
             logger.info("Sync DB connection closed.")


# --- Main Orchestration Logic --- #
async def submit_daily_flow_runs(start_date: date, end_date: date, deployment_name: str):
    """Finds deployment and triggers flow runs for each date in the range."""
    async with get_client() as client:
        try:
            # Prefect 3+ way to get deployment ID by name (format: "flow-name/deployment-name")
            logger.info(f"Attempting to read deployment: {deployment_name}")
            deployment = await client.read_deployment_by_name(deployment_name)
            # Correctly load the deployment using the format "flow_name/deployment_name"
            # flow_name = deployment_name.split('/')[0]  # No longer needed with read_deployment_by_name
            # dep_name_only = deployment_name.split('/')[1] # No longer needed
            # deployment = await Deployment.load(name=dep_name_only) # DEPRECATED Prefect 2 method
            deployment_id = deployment.id
            logger.info(f"Found deployment ID: {deployment_id} for '{deployment_name}'")
        except Exception as e:
            logger.error(f"Failed to load deployment '{deployment_name}': {e}. Ensure it exists and name format is 'flow_name/deployment_name'.")
            return

        current_date = start_date
        submitted_count = 0
        failed_count = 0
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            parameters = {"run_date_str": date_str}
            try:
                logger.info(f"Submitting run for {date_str} with deployment ID {deployment_id}...")
                run = await client.create_flow_run_from_deployment(
                    deployment_id=deployment_id,
                    parameters=parameters,
                    state=Scheduled() # Explicitly provide the initial state
                )
                logger.info(f"Submitted flow run for {date_str}: ID {run.id}")
                submitted_count += 1
                # Optional: Add a small delay
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"Failed to submit flow run for {date_str}: {e}")
                failed_count += 1

            current_date += timedelta(days=1)
        logger.info(f"Submission loop finished. Submitted: {submitted_count}, Failed: {failed_count}")

# --- Script Entry Point --- #
if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Trigger Prefect flow runs for a date range or a single date.")
    parser.add_argument(
        "--date",
        type=str,
        help="Optional: Specific date to run for (YYYY-MM-DD). Overrides DB range lookup."
    )
    args = parser.parse_args()

    min_sensor_date = None
    max_sensor_date = None

    if args.date:
        # --- Use Specific Date from Argument ---
        try:
            parsed_date = datetime.strptime(args.date, "%Y-%m-%d").date()
            min_sensor_date = parsed_date
            max_sensor_date = parsed_date
            logger.info(f"Processing single date specified via argument: {args.date}")
        except ValueError:
            logger.error(f"Invalid date format for --date argument: '{args.date}'. Please use YYYY-MM-DD.")
            exit(1)
    else:
        # --- Get Date Range from DB (Original Logic) ---
        logger.info("No specific date provided, determining date range from database...")
        db_conn = get_db_connection_sync()
        if not db_conn:
            exit(1)
        min_sensor_date, max_sensor_date = get_date_range(db_conn) # db connection closed inside function

    # --- Proceed if we have a valid date range/single date ---
    if not min_sensor_date or not max_sensor_date:
        logger.error("Exiting: Could not determine date range or parse provided date.")
        exit(1)

    # --- Hardcoded Date Range for Testing (Keep commented out or remove) ---
    # min_sensor_date = date(2013, 12, 21) # Use earliest date
    # max_sensor_date = date(2013, 12, 23) # Limit to three days
    # logger.info(f"Using hardcoded date range for testing: {min_sensor_date} to {max_sensor_date}")
    # --- End Hardcoded Date Range ---

    # 3. Define Target Deployment
    target_deployment_name = "main-feature-flow/feature-etl-deployment"

    # 4. Trigger Flow Runs Asynchronously
    logger.info(f"Starting submission loop for dates {min_sensor_date} to {max_sensor_date}")
    try:
        asyncio.run(submit_daily_flow_runs(min_sensor_date, max_sensor_date, target_deployment_name))
    except RuntimeError as e:
        if "no event loop" in str(e):
             logger.info("Detected potential nested asyncio loop. Trying async submission directly.")
             # In some environments (like certain task runners), a loop might already exist
             loop = asyncio.get_event_loop()
             loop.run_until_complete(submit_daily_flow_runs(min_sensor_date, max_sensor_date, target_deployment_name))
        else:
             raise e
    logger.info("Finished submitting all flow runs.") 