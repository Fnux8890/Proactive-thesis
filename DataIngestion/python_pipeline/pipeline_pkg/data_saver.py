"""Handles saving cleaned DataFrames to specified formats."""

import pandas as pd
import logging
from pathlib import Path
from typing import Optional
from sqlalchemy import create_engine, text # Import text for executing raw SQL
from sqlalchemy.exc import OperationalError, ProgrammingError

# Ensure relative import
from . import config # Corrected relative import

# Use the setup_logger relatively
# try: # Removed try/except block for missing import
#    from .utils.logger import setup_logger # Corrected relative import path
#    logger = setup_logger(__name__)
# except ImportError:
#    logger = logging.getLogger(__name__)
    # Basic config if logger util fails
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use standard logging directly

# Ensure relative import - Assuming config is now structured under a config directory
# Example: from .config import pipeline_config
# We need to know the exact structure/name of the config file containing saver settings.
# For now, let's assume pipeline_config holds these variables.
from .config import pipeline_config

class DataSaver:
    """Saves DataFrames to Parquet, CSV, or a database."""

    def __init__(self, output_format: str, output_path: Path, db_connection_string: Optional[str] = None):
        """Initializes the DataSaver and ensures DB schema if needed.

        Args:
            output_format: The format to save in ("parquet", "csv", "db").
            output_path: The base directory Path object for saving files (used for non-DB formats).
            db_connection_string: Database connection URL (required if output_format is "db").
        """
        self.output_format = output_format.lower()
        self.output_path = output_path
        self.db_connection_string = db_connection_string
        self.db_engine = None

        if self.output_format == "db":
            if not self.db_connection_string:
                logger.error("Database connection string (DATABASE_URL) is missing but SAVE_FORMAT is 'db'.")
                raise ValueError("Database connection string is required for 'db' output format.")
            try:
                self.db_engine = create_engine(self.db_connection_string)
                # Use constants from the imported config module
                self._ensure_db_schema(pipeline_config.TARGET_TABLE_NAME,
                                       pipeline_config.TARGET_SCHEMA,
                                       pipeline_config.TIMESCALEDB_TIME_COLUMN)
            except ImportError:
                logger.error("SQLAlchemy or a DB driver (e.g., psycopg2-binary) is required for database operations.")
                raise
            except OperationalError as e:
                logger.error(f"Failed to connect to database. Check connection string and DB status: {e}")
                raise
            except Exception as e:
                logger.error(f"Failed to create database engine: {e}", exc_info=True)
                raise

        logger.info(f"DataSaver initialized for format: {self.output_format}")

    def _ensure_db_schema(self, table_name: str, schema: dict, time_column: str):
        """Checks if the target table and hypertable exist, creates them if not."""
        if not self.db_engine:
            logger.error("Database engine not available for schema setup.")
            return
        
        if not table_name or not time_column or not schema:
            logger.error("TARGET_TABLE_NAME, TIMESCALEDB_TIME_COLUMN, or TARGET_SCHEMA missing in config. Cannot ensure schema.")
            return
        if time_column not in schema:
             logger.error(f"Configured time column '{time_column}' not found in TARGET_SCHEMA definition. Cannot ensure schema.")
             return

        try:
            with self.db_engine.connect() as connection:
                 with connection.begin(): # Use a transaction
                    # Check if table exists
                    table_exists_sql = text(
                        "SELECT EXISTS (SELECT FROM information_schema.tables "
                        "WHERE table_schema = 'public' AND table_name = :table)"
                    )
                    table_exists = connection.execute(table_exists_sql, {"table": table_name}).scalar()

                    if table_exists:
                        logger.info(f"Table '{table_name}' already exists. Skipping creation.")
                        # Even if table exists, ensure it's a hypertable (in case created manually)
                        self._create_hypertable_if_not_exists(connection, table_name, time_column)
                        return

                    # Table does not exist, create it
                    logger.info(f"Table '{table_name}' does not exist. Creating table...")
                    column_defs = [f'"{col}" {dtype}' for col, dtype in schema.items()]
                    create_table_sql = text(f'CREATE TABLE public."{table_name}" ({ ", ".join(column_defs) });')
                    logger.debug(f"Executing CREATE TABLE: {create_table_sql}")
                    connection.execute(create_table_sql)
                    logger.info(f"Successfully created table '{table_name}'.")

                    # Immediately create hypertable after creating the base table
                    self._create_hypertable_if_not_exists(connection, table_name, time_column, is_new_table=True)
        
        except Exception as e:
            logger.error(f"Failed to ensure database schema for table '{table_name}': {e}", exc_info=True)
            raise # Re-raise the exception to stop the pipeline if setup fails

    def _create_hypertable_if_not_exists(self, connection, table_name: str, time_column: str, is_new_table: bool = False):
        """Helper function to create the hypertable (expects to be called within a transaction)."""
        try:
            # Check if it's already a hypertable (skip check if we know it's a new table)
            if not is_new_table:
                check_sql = text(
                    "SELECT 1 FROM timescaledb_information.hypertables "
                    "WHERE hypertable_schema = 'public' AND hypertable_name = :table"
                )
                is_hypertable = connection.execute(check_sql, {"table": table_name}).scalar()
                if is_hypertable == 1:
                    logger.debug(f"Table '{table_name}' is already a hypertable.")
                    return

            # Create hypertable
            logger.info(f"Attempting to create hypertable '{table_name}' on column '{time_column}'.")
            create_sql = text("SELECT create_hypertable(:table, :time_col, if_not_exists => TRUE);")
            connection.execute(create_sql, {"table": table_name, "time_col": time_column})
            logger.info(f"Successfully created or ensured hypertable '{table_name}' on column '{time_column}'.")

        except ProgrammingError as e:
            if "relation \"timescaledb_information.hypertables\" does not exist" in str(e):
                logger.error(f"TimescaleDB extension might not be enabled. Cannot create hypertable.", exc_info=True)
            elif "column \"{time_column}\" does not exist" in str(e):
                logger.error(f"Time column '{time_column}' specified in config not found in table '{table_name}'. Cannot create hypertable.", exc_info=True)
            else:
                 logger.error(f"Database programming error during hypertable creation for '{table_name}': {e}", exc_info=True)
            raise # Re-raise error to stop pipeline if hypertable creation fails critically
        except Exception as e:
            logger.error(f"Unexpected error during hypertable creation for '{table_name}': {e}", exc_info=True)
            raise

    def save_data(self, df: pd.DataFrame, original_filename_with_ext: str):
        """Saves the DataFrame based on the configured output format.

        Args:
            df: The cleaned pandas DataFrame to save.
            original_filename_with_ext: The full name (including extension) of the original file.
        """
        if df.empty:
            logger.warning(f"DataFrame from {original_filename_with_ext} is empty. Skipping save.")
            return

        # Get the stem (filename without extension) for output naming
        original_stem = Path(original_filename_with_ext).stem

        try:
            if self.output_format == "parquet":
                self.output_path.mkdir(parents=True, exist_ok=True)
                # Use stem for cleaner output name
                output_filename = self.output_path / f"cleaned_{original_stem}.parquet"
                df.to_parquet(output_filename, index=False, engine=pipeline_config.PARQUET_ENGINE)
                logger.info(f"Saved cleaned data to Parquet: {output_filename}")

            elif self.output_format == "csv":
                self.output_path.mkdir(parents=True, exist_ok=True)
                 # Use stem for cleaner output name
                output_filename = self.output_path / f"cleaned_{original_stem}.csv"
                df.to_csv(output_filename, index=False)
                logger.info(f"Saved cleaned data to CSV: {output_filename}")

            elif self.output_format == "db":
                if not self.db_engine:
                    logger.error("Database engine not initialized. Cannot save to DB.")
                    return

                table_name = pipeline_config.TARGET_TABLE_NAME
                logger.info(f"Attempting to save data from {original_filename_with_ext} to database table: '{table_name}'")

                # --- Debugging: Log DataFrame info before saving ---
                logger.debug(f"DataFrame columns and types before save for {original_stem}:")
                # Use buffer to capture df.info output for logging
                from io import StringIO
                buffer = StringIO()
                df.info(buf=buffer, verbose=True, show_counts=True)
                logger.debug(buffer.getvalue())
                # Log head only if DataFrame is not excessively wide
                # --- Comment out verbose head logging ---
                # if df.shape[1] < 20:
                #      logger.debug(f"DataFrame head (first 5 rows) for {original_stem}:\\n{df.head().to_string()}")
                # else:
                #      logger.debug(f"DataFrame head (first 5 rows, limited cols) for {original_stem}:\\n{df.head().iloc[:, :20].to_string()}")
                # --- End Comment out ---
                # --- End Debugging ---

                # Now we can always use 'append' because setup ensures table exists
                df.to_sql(
                    name=table_name,
                    con=self.db_engine,
                    if_exists='append', # Always append now
                    index=False,
                    chunksize=pipeline_config.DB_CHUNK_SIZE,
                    method='multi'
                )
                # --- Add log right after successful to_sql call ---
                logger.debug(f"df.to_sql call completed for {original_filename_with_ext}. Rows attempted: {len(df)}")
                # --- End log ---
                logger.info(f"Successfully saved {len(df)} rows from {original_filename_with_ext} to table '{table_name}' using method 'append'.")

            else:
                logger.error(f"Unsupported output format in config: {self.output_format}")

        except ImportError as e:
             logger.error(f"Missing dependency for format '{self.output_format}': {e}. Please install required libraries (e.g., pyarrow, sqlalchemy, psycopg2-binary).", exc_info=True)
        except OperationalError as e:
            logger.error(f"Database operation failed for {original_stem} (table: {pipeline_config.TARGET_TABLE_NAME}): {e}")
            # Add more specific error handling if needed (e.g., connection refused)
        except ProgrammingError as e:
             logger.error(f"Database programming error (e.g., table/column mismatch) for {original_stem} (table: {pipeline_config.TARGET_TABLE_NAME}): {e}")
        except Exception as e:
            logger.error(f"Error saving data for {original_stem} (format: {self.output_format}, table: {pipeline_config.TARGET_TABLE_NAME}): {e}", exc_info=True) 