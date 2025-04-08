"""Handles loading data from different file formats into Pandas DataFrames."""

import pandas as pd
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class DataLoader:
    """Loads data from CSV or JSON files."""

    def __init__(self, csv_params: Dict[str, Any], json_params: Dict[str, Any]):
        """Initializes the DataLoader with parameters for reading files.

        Args:
            csv_params: Dictionary of parameters for pandas.read_csv.
            json_params: Dictionary of parameters for pandas.read_json.
        """
        self.csv_params = csv_params
        self.json_params = json_params
        logger.info("DataLoader initialized.")

    def load_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Loads a single file based on its extension.

        Args:
            file_path: The Path object of the file to load.

        Returns:
            A pandas DataFrame if loading is successful, otherwise None.
        """
        file_extension = file_path.suffix.lower()
        logger.debug(f"Attempting to load file: {file_path} with extension {file_extension}")

        try:
            if file_extension == '.csv':
                df = None
                # First attempt: Try with semicolon separator
                semicolon_params = {**self.csv_params, 'sep': ';'}
                try:
                    logger.debug(f"Attempting to load CSV {file_path.name} with separator ';'")
                    df = pd.read_csv(file_path, **semicolon_params)
                    logger.debug(f"Successfully loaded CSV {file_path.name} with separator ';'. Shape: {df.shape}")
                except (pd.errors.ParserError, UnicodeDecodeError) as e1:
                    logger.warning(f"Failed to load CSV {file_path.name} with sep=';' ({type(e1).__name__}: {e1}). Trying default inference.")
                    # Second attempt: Try with default inference (remove sep)
                    inference_params = self.csv_params.copy()
                    inference_params.pop('sep', None) # Ensure sep is not specified
                    try:
                        logger.debug(f"Attempting to load CSV {file_path.name} with inferred separator")
                        df = pd.read_csv(file_path, **inference_params)
                        logger.debug(f"Successfully loaded CSV {file_path.name} with inferred separator. Shape: {df.shape}")
                    except (pd.errors.ParserError, UnicodeDecodeError) as e2:
                        logger.error(f"Failed to load CSV {file_path.name} with both sep=';' and inferred separator ({type(e2).__name__}: {e2})")
                        df = None # Ensure df is None if second attempt fails
                    except Exception as e3:
                        logger.error(f"Unexpected error during second CSV load attempt for {file_path.name}: {e3}", exc_info=True)
                        df = None
                except Exception as e4: # Catch other errors during first attempt
                    logger.error(f"Unexpected error during first CSV load attempt (sep=';') for {file_path.name}: {e4}", exc_info=True)
                    df = None
                
                # After attempts, log final details if successful
                if df is not None:
                    logger.debug(f"CSV {file_path.name} columns: {df.columns.tolist()}")
                    logger.debug(f"CSV {file_path.name} dtypes after load:\\n{df.dtypes}")
                return df # Return df if loaded, None otherwise
                
            elif file_extension == '.json':
                # Try loading as JSON Lines first based on common use case
                if self.json_params.get('lines', False):
                    try:
                        df = pd.read_json(file_path, **self.json_params)
                        logger.info(f"Successfully loaded JSON Lines: {file_path.name}")
                        # --- Debugging: Log JSON DataFrame info after loading ---
                        if df is not None:
                            logger.info(f"JSON DataFrame columns and types after load for {file_path.name}:")
                            df.info(verbose=True, show_counts=True)
                            logger.info(f"JSON DataFrame head (first 5 rows) after load for {file_path.name}:\n{df.head().to_string()}")
                        # --- End Debugging ---
                        logger.debug(f"Returning DataFrame from JSON Lines load for {file_path.name}")
                        return df
                    except ValueError as e:
                        logger.warning(f"Failed to load {file_path.name} as JSON Lines ({e}), trying standard JSON.")
                        # Fallback to standard JSON loading if lines=True fails or isn't set
                        # Create a copy of params without 'lines' if it exists
                        standard_json_params = self.json_params.copy()
                        standard_json_params.pop('lines', None)
                        df = pd.read_json(file_path, **standard_json_params)
                        logger.info(f"Successfully loaded standard JSON: {file_path.name}")
                        # Add debug logging here too if fallback is used
                        if df is not None:
                            logger.info(f"JSON DataFrame columns and types after fallback load for {file_path.name}:")
                            df.info(verbose=True, show_counts=True)
                            logger.info(f"JSON DataFrame head (first 5 rows) after fallback load for {file_path.name}:\n{df.head().to_string()}")
                        logger.debug(f"Returning DataFrame from JSON fallback load for {file_path.name}")
                        return df
                else:
                    df = pd.read_json(file_path, **self.json_params)
                    logger.info(f"Successfully loaded standard JSON: {file_path.name}")
                    # Add debug logging here too
                    if df is not None:
                        logger.info(f"JSON DataFrame columns and types after standard load for {file_path.name}:")
                        df.info(verbose=True, show_counts=True)
                        logger.info(f"JSON DataFrame head (first 5 rows) after standard load for {file_path.name}:\n{df.head().to_string()}")
                    logger.debug(f"Returning DataFrame from standard JSON load for {file_path.name}")
                    return df

            # Add elif blocks here for other supported types (e.g., .xlsx, .parquet)
            # elif file_extension == '.xlsx':
            #     df = pd.read_excel(file_path, engine='openpyxl') # Requires openpyxl
            #     logger.info(f"Successfully loaded Excel: {file_path.name}")
            #     return df

            else:
                logger.warning(f"Unsupported file type: {file_extension} for file {file_path.name}")
                logger.debug(f"Returning None for unsupported file {file_path.name}")
                return None

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format in {file_path.name}: {e}")
            logger.debug(f"Returning None after JSONDecodeError for {file_path.name}")
            return None
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            logger.debug(f"Returning None after FileNotFoundError for {file_path.name}")
            return None
        except Exception as e: # Catch other potential exceptions during load
            logger.error(f"Error loading {file_path.name}: {e}", exc_info=True)
            logger.debug(f"Returning None after generic Exception for {file_path.name}")
            return None 