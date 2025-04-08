"""Main script to run the data cleaning pipeline."""

import logging
import time
import os
from pathlib import Path

# Adjusted imports for new structure
from .config import file_discovery as fd_config # Assuming config is now in config/ subdir
from .config import pipeline_config # Assuming pipeline config like paths, output format
from .loaders.data_loader import DataLoader
from .cleaners import get_cleaner, BaseCleaner # Import factory and base class
from .data_saver import DataSaver
from .file_discovery import FileFinder

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        logger.info("Pipeline initialized.")

    def run(self):
        """Executes the full data cleaning pipeline."""
        start_time = time.time()
        logger.info("=== Starting Pipeline Run ===")

        # 1. Find files
        files_to_process = self.finder.find_files(self.data_source_path)
        total_files = len(files_to_process)
        processed_count = 0
        failed_count = 0

        if not files_to_process:
            logger.warning(f"No files found to process in {self.data_source_path}")
            self._log_summary(start_time, total_files, processed_count, failed_count)
            return

        # 2. Process each file
        for file_path in files_to_process:
            file_name = file_path.name
            logger.info(f"--- Processing file: {file_name} ---")

            try:
                # 2a. Load data using the spec-aware loader
                load_result = self.loader.load_file(file_path)

                if load_result is None:
                    logger.warning(f"Skipping {file_name}: Loading failed or no format spec found.")
                    failed_count += 1
                    continue

                raw_df, format_spec = load_result
                logger.info(f"Successfully loaded {file_name} using spec '{format_spec['matched_pattern']}'")

                if raw_df is None or raw_df.empty:
                     logger.warning(f"Skipping {file_name}: Loaded DataFrame is empty.")
                     failed_count += 1
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
                             logger.warning(f"Cleaning with {cleaner_name} resulted in empty DataFrame for {file_name}. Skipping save.")
                             failed_count += 1
                             continue
                        else:
                             logger.info(f"Successfully cleaned {file_name}. Shape: {cleaned_df.shape}")
                    except ValueError as e:
                         logger.error(f"Could not get cleaner '{cleaner_name}' for {file_name}: {e}")
                         failed_count += 1
                         continue # Skip to next file if cleaner not found
                    except Exception as e:
                        logger.error(f"Error during cleaning of {file_name} with {cleaner_name}: {e}", exc_info=True)
                        failed_count += 1
                        continue # Skip to next file on cleaning error
                else:
                    logger.warning(f"No cleaner specified in format_spec for {file_name}. Skipping cleaning and saving.")
                    # Decide if we should save raw data or skip entirely. Skipping for now.
                    failed_count += 1
                    continue

                # 2c. Save cleaned data
                self.saver.save_data(cleaned_df, file_name) # Saver determines output filename
                processed_count += 1

            except Exception as e:
                logger.error(f"Unhandled exception during processing of {file_name}: {e}", exc_info=True)
                failed_count += 1
            finally:
                logger.info(f"--- Finished processing file: {file_name} ---")


        self._log_summary(start_time, total_files, processed_count, failed_count)

    def _log_summary(self, start_time, total_files, processed_count, failed_count):
        """Logs the summary of the pipeline run."""
        end_time = time.time()
        duration = end_time - start_time
        logger.info("=== Pipeline Run Summary ===")
        logger.info(f"Source Directory: {self.data_source_path.resolve()}")
        logger.info(f"Output Directory: {self.output_path.resolve()}")
        logger.info(f"Total files found: {total_files}")
        logger.info(f"Successfully processed and saved: {processed_count}")
        logger.info(f"Failed/Skipped: {failed_count}")
        logger.info(f"Total execution time: {duration:.2f} seconds")
        logger.info("===========================")

# --- Main execution block ---
if __name__ == "__main__":
    # Assuming config modules are set up correctly
    # Load configurations (adjust imports/paths as needed)
    try:
        from .config import file_discovery as fd_config
        from .config import pipeline_config
        from .config import format_specs # Import to ensure it's loaded/checked
    except ImportError as e:
        logger.error(f"Failed to import configuration modules: {e}")
        logger.error("Ensure config files (pipeline_config.py, file_discovery.py, format_specs.py) exist in pipeline_pkg/config/")
        exit(1)

    pipeline = Pipeline()
    pipeline.run() 