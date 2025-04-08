import pandas as pd
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any
from pathlib import Path
import re

logger = logging.getLogger(__name__)

class BaseCleaner(ABC):
    """Abstract base class for data cleaning operations."""

    @abstractmethod
    def clean(self, df: pd.DataFrame, format_spec: Dict[str, Any]) -> pd.DataFrame:
        """Cleans the input DataFrame according to the format specification.

        Args:
            df: The raw pandas DataFrame loaded from the file.
            format_spec: The dictionary containing format and metadata details
                         for the file, obtained from get_format_spec.

        Returns:
            A cleaned pandas DataFrame, ready for saving.
        """
        pass

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes column names to snake_case."""
        new_columns = {}
        for col in df.columns:
            if isinstance(col, tuple):
                # Handle MultiIndex columns - join with underscore
                new_col = '_'.join(map(str, col)).strip()
            else:
                new_col = str(col).strip()

            # Basic standardization: lower, replace space/special chars with _, remove trailing _
            new_col = new_col.lower()
            new_col = re.sub(r'[^\w\s-]', '', new_col) # Remove non-word/space/hyphen
            new_col = re.sub(r'[\s-]+|–', '_', new_col) # Replace space/hyphen/en-dash with _
            new_col = re.sub(r'_+', '_', new_col) # Collapse multiple underscores
            new_col = new_col.strip('_')
            new_columns[col] = new_col

        df = df.rename(columns=new_columns)
        logger.debug(f"Standardized columns: {df.columns.tolist()}")
        return df

    def _handle_timestamp(self, df: pd.DataFrame, format_spec: Dict[str, Any]) -> pd.DataFrame:
        """Parses timestamp columns based on format_spec."""
        timestamp_info = format_spec.get('timestamp_info')
        if not timestamp_info:
            logger.warning(f"No timestamp_info found in spec for {format_spec.get('full_path')}")
            return df

        df = df.copy()
        timestamp_col_name = 'timestamp' # Standard name

        try:
            if isinstance(timestamp_info[0], tuple): # Separate date and time columns
                (date_col, date_fmt), (time_col, time_fmt) = timestamp_info
                # Combine date and time strings before parsing
                # Handle potential missing values before string concatenation
                date_str = df[date_col].astype(str)
                time_str = df[time_col].astype(str)
                datetime_str = date_str + ' ' + time_str
                full_fmt = date_fmt + ' ' + time_fmt
                df[timestamp_col_name] = pd.to_datetime(datetime_str, format=full_fmt, errors='coerce')
                # Drop original date/time columns
                df = df.drop(columns=[date_col, time_col])
            else: # Single timestamp column
                col_name, fmt = timestamp_info
                # Handle MultiIndex column selection if needed
                if col_name == '' and isinstance(df.columns, pd.MultiIndex):
                     # Assume it's the first level of the first column for Knudjepsen
                     # This might need adjustment based on actual MultiIndex structure
                     ts_col_identifier = df.columns[0]
                     logger.debug(f"Using MultiIndex column {ts_col_identifier} as timestamp source.")
                elif col_name == '' and not isinstance(df.columns, pd.MultiIndex):
                    # If single index and col_name is empty, assume index or first col?
                    # Let's assume first column if unnamed, check if index is timestamp later
                     ts_col_identifier = df.columns[0]
                     logger.warning(f"Timestamp column name empty, assuming first column: {ts_col_identifier}")
                else:
                    ts_col_identifier = col_name

                if fmt == 'unix_ms':
                    # Convert milliseconds to datetime
                    df[timestamp_col_name] = pd.to_datetime(df[ts_col_identifier], unit='ms', errors='coerce')
                elif fmt == 'unix_s':
                     # Convert seconds to datetime
                    df[timestamp_col_name] = pd.to_datetime(df[ts_col_identifier], unit='s', errors='coerce')
                else:
                    # Parse with specified format string
                    df[timestamp_col_name] = pd.to_datetime(df[ts_col_identifier], format=fmt, errors='coerce')

                # Drop original timestamp column if it's not the new one and exists
                if ts_col_identifier != timestamp_col_name and ts_col_identifier in df.columns:
                    df = df.drop(columns=[ts_col_identifier])

            # Handle potential timezone information if available in spec
            # (Assuming timezone info might be added later, e.g., from JSON Properties)
            tz = format_spec.get('timezone', 'Europe/Copenhagen') # Default to Copenhagen
            if tz and pd.api.types.is_datetime64_any_dtype(df[timestamp_col_name]):
                 # Localize if naive, convert if aware but different
                 if df[timestamp_col_name].dt.tz is None:
                     df[timestamp_col_name] = df[timestamp_col_name].dt.tz_localize(tz, ambiguous='infer', nonexistent='shift_forward')
                     logger.debug(f"Localized timestamp to {tz}")
                 elif str(df[timestamp_col_name].dt.tz) != tz:
                     df[timestamp_col_name] = df[timestamp_col_name].dt.tz_convert(tz)
                     logger.debug(f"Converted timestamp timezone to {tz}")

            # Drop rows where timestamp parsing failed
            original_count = len(df)
            df = df.dropna(subset=[timestamp_col_name])
            dropped_count = original_count - len(df)
            if dropped_count > 0:
                logger.warning(f"Dropped {dropped_count} rows due to NaT timestamps for {format_spec.get('full_path')}")

            logger.info(f"Successfully processed timestamp column '{timestamp_col_name}'")
            # Set timestamp as index if desired (consider adding as config option)
            # df = df.set_index(timestamp_col_name)

        except KeyError as e:
            logger.error(f"Timestamp column '{e}' not found in DataFrame for {format_spec.get('full_path')}. Spec: {timestamp_info}")
            # Return original df or raise? Returning df for now.
        except Exception as e:
            logger.error(f"Error processing timestamp for {format_spec.get('full_path')}: {e}", exc_info=True)
            # Return original df

        return df

    def _convert_numeric(self, df: pd.DataFrame, cols_to_convert: list, decimal: str = '.') -> pd.DataFrame:
        """Converts specified columns to numeric, handling errors and decimal separator."""
        df = df.copy()
        for col in cols_to_convert:
            if col in df.columns:
                original_dtype = df[col].dtype
                # Handle potential non-string types before replace
                if pd.api.types.is_object_dtype(original_dtype):
                    # Replace decimal separator if needed
                    if decimal == ',':
                        df[col] = df[col].astype(str).str.replace('.', '', regex=False) # Remove thousand sep
                        df[col] = df[col].astype(str).str.replace(',', '.', regex=False) # Replace decimal comma
                    else:
                         # Remove potential commas used as thousand separators
                         df[col] = df[col].astype(str).str.replace(',', '', regex=False)

                    # Attempt conversion to numeric
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                elif pd.api.types.is_numeric_dtype(original_dtype):
                    logger.debug(f"Column '{col}' is already numeric ({original_dtype}). Skipping conversion.")
                    continue # Already numeric
                else:
                     # Attempt direct conversion for other types (e.g., bool, int that became obj)
                     df[col] = pd.to_numeric(df[col], errors='coerce')

                # Log conversion result
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    logger.warning(f"Column '{col}': {nan_count} values failed numeric conversion (became NaN).")
                logger.debug(f"Converted column '{col}' from {original_dtype} to {df[col].dtype}.")
            else:
                logger.warning(f"Numeric conversion skipped: Column '{col}' not found.")
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handles missing values (NaN/NaT) in the DataFrame."""
        # Simple strategy: fill numeric with median, object/category with 'Unknown'
        # More sophisticated strategies could be implemented based on spec
        df = df.copy()
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].isnull().any():
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    logger.debug(f"Filled NaNs in numeric column '{col}' with median ({median_val}).")
            elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                 if df[col].isnull().any():
                    df[col] = df[col].fillna('Unknown')
                    logger.debug(f"Filled NaNs in object/category column '{col}' with 'Unknown'.")
            # Keep NaT in timestamp column - already handled during parsing

        return df

    def _add_metadata(self, df: pd.DataFrame, format_spec: Dict[str, Any]) -> pd.DataFrame:
        """Adds metadata columns based on the format specification."""
        df = df.copy()
        file_path = format_spec.get('full_path')
        if not file_path:
            logger.warning("Cannot add metadata, 'full_path' missing from format_spec.")
            return df

        # Extract source ID components from spec or path patterns
        base_source_id = format_spec.get('base_source_id', 'unknown_source')
        location = format_spec.get('location')
        measurement_group = format_spec.get('measurement_group')
        source_id_detail = None # Extracted from path, e.g., 'celle5' or 'september2014'
        path_str = file_path.as_posix()

        if 'source_id_pattern' in format_spec:
            match = re.search(format_spec['source_id_pattern'], path_str)
            if match:
                source_id_detail = match.group(1) # Assume first capture group
                logger.debug(f"Extracted source detail '{source_id_detail}' using pattern {format_spec['source_id_pattern']}")

        # Construct a unique source identifier
        source_parts = [base_source_id]
        if location:
            source_parts.append(str(location)) # Ensure string
        if source_id_detail:
             source_parts.append(source_id_detail)
        elif measurement_group: # Use measurement group if detail not extracted
             source_parts.append(measurement_group)

        # Add source_file name for traceability
        df['source_file'] = file_path.name
        df['source_identifier'] = '_'.join(source_parts).lower()

        logger.debug(f"Added metadata: source_file='{file_path.name}', source_identifier='{df['source_identifier'].iloc[0]}'")

        return df

    def _reshape_to_long(self, df: pd.DataFrame, id_vars: list, var_name: str = 'measurement', value_name: str = 'value') -> pd.DataFrame:
        """Reshapes the DataFrame from wide to long format using melt."""
        if not all(col in df.columns for col in id_vars):
            logger.error(f"Cannot melt: One or more id_vars ({id_vars}) not found in columns {df.columns.tolist()}")
            return df # Return original if id_vars are missing

        value_vars = [col for col in df.columns if col not in id_vars]
        if not value_vars:
             logger.warning("No value variables found for melting. DataFrame might already be long or id_vars incorrect.")
             return df

        try:
            df_long = pd.melt(df, id_vars=id_vars, value_vars=value_vars, var_name=var_name, value_name=value_name)
            logger.info(f"Reshaped DataFrame to long format. Id vars: {id_vars}, Value vars: {len(value_vars)}")
            logger.debug(f"Long format columns: {df_long.columns.tolist()}")
            return df_long
        except Exception as e:
            logger.error(f"Error melting DataFrame: {e}", exc_info=True)
            return df # Return original on error 