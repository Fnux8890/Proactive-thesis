"""Main script to run the data cleaning pipeline."""

import logging
import time
import os
from pathlib import Path
import argparse # Added for command-line arguments
import json # Added for status tracking

# Adjusted imports for new structure
from .config import file_discovery as fd_config # Assuming config is now in config/ subdir
from .config import pipeline_config # Assuming pipeline config like paths, output format
from .loaders.data_loader import DataLoader
from .cleaners import get_cleaner, BaseCleaner # Import factory and base class
from .data_saver import DataSaver
from .file_discovery import FileFinder

# --- Custom Filter to Suppress SQLAlchemy INFO logs ---
class SQLAlchemyInfoFilter(logging.Filter):
    def filter(self, record):
        # Suppress INFO level messages from loggers starting with 'sqlalchemy.'
        is_sqlalchemy_info = record.name.startswith('sqlalchemy.') and record.levelno == logging.INFO
        return not is_sqlalchemy_info
# --- End Custom Filter ---

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s' # Added filename/lineno
)
logger = logging.getLogger(__name__)

# --- Apply the custom filter ---
# Apply filter to the root logger's handlers
# This ensures it affects logs processed by the basicConfig setup
for handler in logging.root.handlers:
    handler.addFilter(SQLAlchemyInfoFilter())
# --- End Apply Filter ---

# Set higher level for SQLAlchemy engine logging to avoid verbose SQL statements
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO) # Or WARNING

# Status file path
STATUS_FILE_PATH = Path('./pipeline_status.json')

class Pipeline:
    """Orchestrates the data cleaning pipeline process using dynamic loading and cleaning."""

    def __init__(self):
        """Initializes the pipeline components."""
        logger.info("Initializing pipeline...")
        # Use environment variables with defaults for paths
        self.data_source_path = Path(os.getenv('DATA_SOURCE_PATH', '../../Data/'))
        self.output_path = Path(os.getenv('OUTPUT_DATA_PATH', './cleaned_data/'))
        db_url = os.getenv('DATABASE_URL', 'sqlite:///data/processed_data.sqlite') # Example default

        logger.info(f"Data source path: {self.data_source_path.resolve()}")
        logger.info(f"Output path: {self.output_path.resolve()}")

        self.finder = FileFinder(fd_config.SUPPORTED_FILE_TYPES) # Assuming supported types are in config
        self.loader = DataLoader() # DataLoader uses format_spec now
        self.saver = DataSaver(
            output_format=pipeline_config.SAVE_FORMAT,
            output_path=self.output_path,
            db_connection_string=db_url
        )
        self.file_statuses = self._load_status()
        logger.info("Pipeline initialized.")

    def _load_status(self) -> dict:
        """Loads the existing status file, or returns an empty dict."""
        if STATUS_FILE_PATH.exists():
            try:
                with open(STATUS_FILE_PATH, 'r') as f:
                    logger.info(f"Loading existing status from {STATUS_FILE_PATH}")
                    return json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON from {STATUS_FILE_PATH}. Starting with empty status.")
                return {}
            except Exception as e:
                logger.error(f"Error loading status file {STATUS_FILE_PATH}: {e}")
                return {}
        else:
            logger.info("No existing status file found. Starting fresh.")
            return {}

    def _save_status(self):
        """Saves the current file statuses to the JSON file."""
        try:
            with open(STATUS_FILE_PATH, 'w') as f:
                json.dump(self.file_statuses, f, indent=4)
            logger.info(f"Pipeline status saved to {STATUS_FILE_PATH}")
        except Exception as e:
            logger.error(f"Error saving status file {STATUS_FILE_PATH}: {e}")

    def run(self, single_file_path_str: str = None):
        """Executes the full data cleaning pipeline, optionally for a single file."""
        start_time = time.time()
        logger.info("=== Starting Pipeline Run ===")

        files_to_process = []
        if single_file_path_str:
            # Single file mode
            single_file_path = Path(single_file_path_str)
            if single_file_path.is_file():
                logger.info(f"Running in single-file mode for: {single_file_path}")
                # Check if it's within the data source path for consistency (optional but good)
                try:
                    # Attempt to make it relative to data source path
                    # This helps ensure format spec lookup works as expected
                    relative_path = single_file_path.resolve().relative_to(self.data_source_path.resolve())
                    files_to_process = [self.data_source_path.resolve() / relative_path]
                    logger.debug(f"Processing absolute path: {files_to_process[0]}")
                except ValueError:
                     logger.error(f"Specified file {single_file_path} is not within the data source directory {self.data_source_path}. Cannot process.")
                     files_to_process = [] # Ensure it's empty
            else:
                logger.error(f"Specified file does not exist or is not a file: {single_file_path}")
                files_to_process = []
        else:
            # Normal discovery mode
            logger.info(f"Discovering files in: {self.data_source_path}")
            files_to_process = self.finder.find_files(self.data_source_path)

        total_files = len(files_to_process)
        processed_count = 0
        failed_count = 0

        if not files_to_process:
            logger.warning(f"No files found or specified to process.")
            self._log_summary(start_time, total_files, processed_count, failed_count)
            self._save_status() # Save status even if no files processed
            return

        # 2. Process each file
        for file_path in files_to_process:
            file_name = file_path.name # Use simple name as key for status
            logger.info(f"--- Processing file: {file_name} ({file_path}) ---")
            status = "fail" # Default status to fail unless explicitly passed
            error_message = "Unknown processing error" # Default error

            try:
                # 2a. Load data using the spec-aware loader
                load_result = self.loader.load_file(file_path)

                if load_result is None:
                    error_message = "Loading failed or no format spec found."
                    logger.warning(f"Skipping {file_name}: {error_message}")
                    failed_count += 1
                    self.file_statuses[file_name] = {"status": status, "reason": error_message}
                    continue

                raw_df, format_spec = load_result
                logger.info(f"Successfully loaded {file_name} using spec '{format_spec['matched_pattern']}'")

                if raw_df is None or raw_df.empty:
                     error_message = "Loaded DataFrame is empty."
                     logger.warning(f"Skipping {file_name}: {error_message}")
                     failed_count += 1
                     self.file_statuses[file_name] = {"status": status, "reason": error_message}
                     continue

                # 2b. Get the appropriate cleaner and clean data
                cleaner_name = format_spec.get('cleaner')
                cleaned_df = None

                if cleaner_name:
                    try:
                        cleaner: BaseCleaner = get_cleaner(cleaner_name)
                        logger.info(f"Using cleaner: {cleaner_name}")
                        cleaned_df = cleaner.clean(raw_df, format_spec) # Pass df and spec
                        if cleaned_df is None or cleaned_df.empty:
                             error_message = f"Cleaning with {cleaner_name} resulted in empty DataFrame."
                             logger.warning(f"{error_message} Skipping save for {file_name}.")
                             failed_count += 1
                             self.file_statuses[file_name] = {"status": status, "reason": error_message}
                             continue
                        else:
                             logger.info(f"Successfully cleaned {file_name}. Shape: {cleaned_df.shape}")
                    except ValueError as e:
                         error_message = f"Could not get cleaner '{cleaner_name}': {e}"
                         logger.error(f"{error_message} for {file_name}")
                         failed_count += 1
                         self.file_statuses[file_name] = {"status": status, "reason": error_message}
                         continue # Skip to next file if cleaner not found
                    except Exception as e:
                        error_message = f"Error during cleaning of {file_name} with {cleaner_name}: {e}"
                        logger.error(error_message, exc_info=True)
                        failed_count += 1
                        self.file_statuses[file_name] = {"status": status, "reason": error_message}
                        continue # Skip to next file on cleaning error
                else:
                    error_message = "No cleaner specified in format_spec."
                    logger.warning(f"{error_message} Skipping cleaning and saving for {file_name}.")
                    failed_count += 1
                    self.file_statuses[file_name] = {"status": status, "reason": error_message}
                    continue

                # 2c. Save cleaned data
                self.saver.save_data(cleaned_df, file_name) # Saver determines output filename
                processed_count += 1
                status = "pass" # Mark as passed only if saved successfully
                error_message = "" # Clear error message on success

            except Exception as e:
                error_message = f"Unhandled exception during processing: {e}"
                logger.error(f"Unhandled exception for {file_name}: {e}", exc_info=True)
                failed_count += 1
            finally:
                # Update status for the file
                self.file_statuses[file_name] = {"status": status, "reason": error_message if status == 'fail' else None}
                logger.info(f"--- Finished processing file: {file_name} (Status: {status}) ---")
                # Optionally save status after each file for resilience
                # self._save_status()

        # Save final status at the end
        self._save_status()
        self._log_summary(start_time, total_files, processed_count, failed_count)

    def _log_summary(self, start_time, total_files, processed_count, failed_count):
        """Logs the summary of the pipeline run."""
        end_time = time.time()
        duration = end_time - start_time
        logger.info("=== Pipeline Run Summary ===")
        logger.info(f"Source Directory: {self.data_source_path.resolve()}")
        logger.info(f"Output Directory/Format: {self.output_path.resolve()} / {pipeline_config.SAVE_FORMAT}")
        logger.info(f"Status File: {STATUS_FILE_PATH.resolve()}")
        logger.info(f"Total files considered: {total_files}")
        logger.info(f"Successfully processed and saved: {processed_count}")
        logger.info(f"Failed/Skipped: {failed_count}")
        logger.info(f"Total execution time: {duration:.2f} seconds")

        # Log status details from the file
        passed_files = [f for f, data in self.file_statuses.items() if data['status'] == 'pass']
        failed_files = {f: data['reason'] for f, data in self.file_statuses.items() if data['status'] == 'fail'}
        logger.info(f"Files passing in last run: {len(passed_files)}")
        logger.info(f"Files failing in last run: {len(failed_files)}")
        if failed_files:
            logger.warning("Failed files and reasons:")
            for fname, reason in failed_files.items():
                logger.warning(f"  - {fname}: {reason}")

        # If saving to database, try to query and log which files were successfully saved
        if pipeline_config.SAVE_FORMAT.lower() == "db" and hasattr(self.saver, 'db_engine'):
            try:
                from sqlalchemy import text

                logger.info("=== Database Storage Summary ===")
                with self.saver.db_engine.connect() as connection:
                    # Get total count
                    count_result = connection.execute(text("SELECT COUNT(*) FROM sensor_readings")).scalar()
                    logger.info(f"Total rows in database: {count_result}")

                    # Get files saved and their counts
                    file_counts = connection.execute(text(
                        "SELECT source_file, COUNT(*) FROM sensor_readings GROUP BY source_file ORDER BY COUNT(*) DESC"
                    )).fetchall()

                    logger.info("Files saved to database and their row counts:")
                    for file_name, count in file_counts:
                        logger.info(f"  {file_name}: {count} rows")

                    # Log any missing files (comparing against files that *should* have passed)
                    # Consider comparing against `passed_files` list for more accuracy
                    saved_db_files = [row[0] for row in file_counts]
                    missing_db_files = [f for f in passed_files if f not in saved_db_files]

                    if missing_db_files:
                        logger.warning("Files marked as 'pass' but not found in DB summary:")
                        for file in missing_db_files:
                            logger.warning(f"  {file}")
            except Exception as e:
                logger.error(f"Error querying database for summary: {e}")

        logger.info("=============================")

# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the data processing pipeline.")
    parser.add_argument(
        "--file",
        type=str,
        help="Specify a single file path to process instead of discovering all files."
    )
    args = parser.parse_args()

    # Assuming config modules are set up correctly
    try:
        from .config import file_discovery as fd_config
        from .config import pipeline_config
        from .config import format_specs # Import to ensure it's loaded/checked
    except ImportError as e:
        logger.error(f"Failed to import configuration modules: {e}")
        logger.error("Ensure config files (pipeline_config.py, file_discovery.py, format_specs.py) exist in pipeline_pkg/config/")
        exit(1)

    pipeline = Pipeline()
    pipeline.run(single_file_path_str=args.file) 