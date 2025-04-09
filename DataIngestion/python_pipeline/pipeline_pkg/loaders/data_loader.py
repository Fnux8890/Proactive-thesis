import pandas as pd
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# Assuming get_format_spec is correctly placed relative to this file
# Adjust the import path based on your final project structure
from ..config.format_specs import get_format_spec

logger = logging.getLogger(__name__)

class DataLoader:
    """Loads data from files using format specifications."""

    def __init__(self):
        """Initializes the DataLoader."""
        logger.info("DataLoader initialized.")

    def load_file(self, file_path: Path) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
        """Loads a single file based on its matching format specification.

        Args:
            file_path: The Path object of the file to load.

        Returns:
            A tuple containing (pandas DataFrame, format_spec) if loading is successful,
            otherwise None.
        """
        logger.debug(f"Attempting to find format spec for: {file_path}")
        format_spec = get_format_spec(file_path)

        if not format_spec:
            logger.warning(f"No format specification found for file: {file_path}. Skipping.")
            return None

        file_type = format_spec.get("type")
        loader_params = format_spec.get("loader_params", {})
        file_name = file_path.name # For logging

        logger.info(f"Attempting to load {file_type} file: {file_name} using spec: {format_spec['matched_pattern']}")
        logger.debug(f"Loader params: {loader_params}")

        try:
            df: Optional[pd.DataFrame] = None
            if file_type == 'csv':
                read_params = {**loader_params} # Start with a copy
                
                # Special handling for Knudjepsen files - force timestamp column as string
                if format_spec.get("timestamp_handling") == "knudjepsen_manual":
                    logger.info(f"Using special handling for Knudjepsen file: {file_name}")
                    # Force the first column (index 0) to be loaded as string
                    # and disable pandas' automatic date parsing
                    read_params.setdefault('dtype', {})[0] = str
                    read_params['parse_dates'] = False
                    logger.debug(f"Modified CSV loading parameters for Knudjepsen file: {read_params}")

                # Check if header is specified as a list (for MultiIndex)
                header_spec = loader_params.get('header')
                if isinstance(header_spec, list):
                    # Ensure the header parameter is correctly set for MultiIndex
                    read_params['header'] = header_spec # Make sure the list is passed
                    logger.debug(f"Using MultiIndex header spec: {header_spec}")
                # No specific 'else' needed, default header=0 or value from spec works

                # Handle potential DtypeWarning specified in loader_params
                if 'low_memory' not in read_params:
                     read_params['low_memory'] = False # Default to avoid mixed types if not specified

                # --- Add explicit logging for read_params before calling read_csv ---
                logger.info(f"Calling pd.read_csv for {file_name} with params: {read_params}")
                # --- End logging ---
                df = pd.read_csv(file_path, **read_params)
                logger.debug(f"Successfully loaded CSV {file_name}. Shape: {df.shape}")
                logger.debug(f"Columns loaded: {df.columns.tolist()}")
                # If MultiIndex, log levels
                if isinstance(df.columns, pd.MultiIndex):
                    logger.debug(f"MultiIndex levels loaded: {df.columns.levels}")


            elif file_type == 'json':
                # JSON loading might need more specific logic based on structure
                # specified in format_spec (e.g., orient, lines).
                # The cleaner will handle complex parsing of raw JSON objects if needed.
                # For now, try a generic read_json with specified params.
                # If 'orient' is None in spec, implies cleaner will parse raw structure.
                if loader_params.get("orient") is not None:
                    df = pd.read_json(file_path, **loader_params)
                    logger.debug(f"Successfully loaded JSON {file_name} using pandas with orient='{loader_params.get('orient')}'. Shape: {df.shape}")
                    logger.debug(f"Columns loaded: {df.columns.tolist()}")
                else:
                    # Load raw JSON for cleaner to parse
                    logger.info(f"Loading raw JSON structure from {file_name} for cleaner (orient=None specified).")
                    with open(file_path, 'r', encoding=loader_params.get('encoding', 'utf-8')) as f:
                        raw_data = json.load(f)
                    # Always wrap the raw structure in a DataFrame with a specific column name
                    # The cleaner is responsible for parsing this raw data structure.
                    df = pd.DataFrame([{'raw_json_data': raw_data}])
                    logger.debug(f"Loaded raw JSON {file_name} into DataFrame wrapper column 'raw_json_data'. Shape: {df.shape}")


            else:
                logger.warning(f"Unsupported file type '{file_type}' specified in format_spec for {file_name}")
                return None

            # Log details after successful load
            if df is not None:
                logger.debug(f"DataFrame info after loading {file_name}:")
                # Use buffer to capture df.info output for logging
                from io import StringIO
                buffer = StringIO()
                df.info(buf=buffer, verbose=True, show_counts=True)
                logger.debug(buffer.getvalue())
                # Log head only if DataFrame is not excessively wide
                if df.shape[1] < 20:
                     logger.debug(f"DataFrame head (first 5 rows) for {file_name}:\\n{df.head().to_string()}")
                else:
                     logger.debug(f"DataFrame head (first 5 rows, limited cols) for {file_name}:\\n{df.head().iloc[:, :20].to_string()}")

                # Return the DataFrame and the spec used to load it
                return df, format_spec
            else:
                 logger.error(f"Loading resulted in None DataFrame for {file_name}")
                 return None


        except FileNotFoundError:
            logger.error(f"File not found during load attempt: {file_path}")
            return None
        except pd.errors.ParserError as e:
            logger.error(f"Pandas parsing error for {file_name} ({file_type}): {e}")
            return None
        except UnicodeDecodeError as e:
            logger.error(f"Encoding error loading {file_name} with {loader_params.get('encoding', 'default')}: {e}")
            return None
        except json.JSONDecodeError as e:
             logger.error(f"Invalid JSON format in {file_name}: {e}")
             return None
        except Exception as e:
            logger.error(f"Unexpected error loading {file_name}: {e}", exc_info=True)
            return None 