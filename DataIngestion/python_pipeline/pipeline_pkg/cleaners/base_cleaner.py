import pandas as pd
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any
from pathlib import Path
import re
import unicodedata # Import unicodedata for normalization

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
        """Standardizes column names to snake_case and ensures ASCII compatibility."""
        new_columns = {}
        for col in df.columns:
            if isinstance(col, tuple):
                # Handle MultiIndex columns - join with underscore
                new_col = '_'.join(map(str, col)).strip()
            else:
                new_col = str(col).strip()

            # Normalize Unicode characters (e.g., å -> a, é -> e, ß -> ss)
            # NFKD decomposes characters, then encode/decode removes non-ASCII
            new_col = unicodedata.normalize('NFKD', new_col)
            new_col = new_col.encode('ascii', 'ignore').decode('ascii')

            # Basic standardization: lower, replace space/special chars with _, remove trailing _
            new_col = new_col.lower()
            
            # Enhanced replacement of problematic special characters
            # Replace / with '_per_' for division/units
            new_col = new_col.replace('/', '_per_')
            # Replace % with '_pct_' for percentage
            new_col = new_col.replace('%', '_pct')
            # Replace other common special chars
            new_col = new_col.replace('&', '_and_')
            new_col = new_col.replace('+', '_plus_')
            # NOTE: Hyphen/minus might be handled by regex below, check carefully
            # new_col = new_col.replace('-', '_minus_')
            new_col = new_col.replace('(', '_')
            new_col = new_col.replace(')', '_')
            new_col = new_col.replace('[', '_')
            new_col = new_col.replace(']', '_')
            
            # Remove any remaining non-alphanumeric characters (allow underscore)
            # Keep hyphens ONLY if they are part of a word (e.g. source-id), but replace standalone ones later?
            # Simplified regex: Keep word characters (letters, numbers, underscore)
            new_col = re.sub(r'[\W_]+(?<!^)', '_', new_col) # Replace non-word chars (except start) with _
            # Original regexes for comparison:
            # new_col = re.sub(r'[^\w\s-]', '', new_col) # Remove non-word/space/hyphen
            # new_col = re.sub(r'[\s-]+|–', '_', new_col) # Replace space/hyphen/en-dash with _
            
            # Collapse multiple underscores and strip leading/trailing underscores
            new_col = re.sub(r'_+', '_', new_col)
            new_col = new_col.strip('_')
            
            # Ensure column name is not empty after standardization
            if not new_col:
                # Create a generic name if standardization results in empty string
                original_repr = str(col).replace(' ', '_') # Basic representation of original
                new_col = f"unnamed_col_{original_repr[:20]}" # Limit length
                logger.warning(f"Standardization resulted in empty column name for original '{col}'. Renaming to '{new_col}'.")
            
            new_columns[col] = new_col

        df = df.rename(columns=new_columns)
        logger.debug(f"Standardized columns (ASCII): {df.columns.tolist()}")
        return df

    def _handle_timestamp(self, df: pd.DataFrame, format_spec: Dict[str, Any]) -> pd.DataFrame:
        """Parses timestamp columns based on format_spec."""
        # --- Add Entry Logging ---
        file_name_for_log = format_spec.get('full_path', Path('unknown_file')).name
        logger.debug(f"Entering _handle_timestamp for file: {file_name_for_log}")
        logger.debug(f"Received format_spec['timestamp_info']: {format_spec.get('timestamp_info')}")
        # --- End Entry Logging ---

        # Special handling for Knudjepsen files
        if format_spec.get("timestamp_handling") == "knudjepsen_manual":
            logger.info(f"Applying custom timestamp parsing for Knudjepsen file: {file_name_for_log}")
            return self._parse_knudjepsen_timestamp(df, format_spec)

        timestamp_info = format_spec.get('timestamp_info')
        if not timestamp_info:
            logger.warning(f"No timestamp_info found in spec for {format_spec.get('full_path')}")
            return df

        df = df.copy()
        timestamp_col_name = 'timestamp' # Standard name

        try:
            if isinstance(timestamp_info[0], tuple): # Separate date and time columns
                (date_col, date_fmt), (time_col, time_fmt) = timestamp_info

                # --- Find actual column names (handle potential pre-standardization) ---
                actual_date_col = date_col if date_col in df.columns else date_col.lower()
                actual_time_col = time_col if time_col in df.columns else time_col.lower()

                if actual_date_col not in df.columns:
                    raise KeyError(f"Date column '{date_col}' (or '{actual_date_col}') not found after potential cleaning.")
                if actual_time_col not in df.columns:
                    raise KeyError(f"Time column '{time_col}' (or '{actual_time_col}') not found after potential cleaning.")
                # --- End Find ---

                # Combine date and time strings before parsing
                date_str = df[actual_date_col].astype(str)
                time_str = df[actual_time_col].astype(str)
                datetime_str = date_str + ' ' + time_str
                full_fmt = date_fmt + ' ' + time_fmt
                df[timestamp_col_name] = pd.to_datetime(datetime_str, format=full_fmt, errors='coerce')
                # Drop original date/time columns
                df = df.drop(columns=[actual_date_col, actual_time_col])
            else: # Single timestamp column
                col_name, fmt = timestamp_info
                # Handle MultiIndex column selection if needed
                if col_name == '' and isinstance(df.columns, pd.MultiIndex):
                     # Assume it's the first level of the first column for Knudjepsen
                     # For MultiIndex, we might need to use a tuple key to access the data
                     # Get the first column identifier (likely a tuple for MultiIndex)
                     ts_col_identifier = df.columns[0]
                     logger.debug(f"Using MultiIndex column {ts_col_identifier} as timestamp source.")
                     
                     # ADDITIONAL: Try to directly access the timestamp column using its position 
                     # rather than the complex identifier if we're parsing timestamps in Knudjepsen files
                     file_path = format_spec.get('full_path')
                     file_name = file_path.name if hasattr(file_path, 'name') else str(file_path)
                     
                     if 'LAMPEGRP' in file_name or 'lampegrp' in file_name or 'belysningsgrp' in file_name or 'extra' in file_name:
                         logger.info(f"KNUDJEPSEN SPECIAL HANDLING: Accessing timestamp in first column for {file_name}")
                         
                         # For these specific files, we know the first column contains the date in dd-mm-yyyy HH:MM:SS format
                         try:
                             # Complete manual approach - skip pandas' accessor and go straight to the raw data
                             # Force the format parsing
                             first_col_values = df.iloc[:, 0].copy()
                             logger.debug(f"First column values (first 5): {first_col_values.head().to_string()}")
                             
                             # Try parsing directly with explicit formatting
                             df[timestamp_col_name] = pd.to_datetime(
                                 first_col_values, 
                                 format=fmt,  # Should be 'dd-mm-yyyy HH:MM:SS'
                                 errors='coerce'
                             )
                             
                             # If timestamp column was successfully created, use it
                             if df[timestamp_col_name].notna().any():
                                 logger.info(f"Successfully created timestamp column using direct parsing")
                                 # Skip the standard parsing below
                                 if ts_col_identifier != timestamp_col_name and ts_col_identifier in df.columns:
                                     df = df.drop(columns=[ts_col_identifier])
                                 # Skip to timezone handling
                                 continue_to_tz = True
                             else:
                                 logger.warning(f"Direct timestamp parsing created only NaT values, falling back to standard method")
                                 # Continue with standard parsing below
                                 continue_to_tz = False
                         except Exception as e:
                             logger.error(f"Error in special Knudjepsen timestamp handling: {e}")
                             # Fall back to standard parsing
                elif col_name == '' and not isinstance(df.columns, pd.MultiIndex):
                    # If single index and col_name is empty, assume index or first col?
                    # Let's assume first column if unnamed, check if index is timestamp later
                     ts_col_identifier = df.columns[0]
                     logger.warning(f"Timestamp column name empty, assuming first column: {ts_col_identifier}")
                else:
                    ts_col_identifier = col_name
                
                # --- Add Logging ---
                logger.debug(f"Determined timestamp column identifier: {ts_col_identifier} (Type: {type(ts_col_identifier)})")
                # --- End Logging ---

                # Initialize flag for skipping to timezone handling
                continue_to_tz = False

                if fmt == 'unix_ms':
                    # Convert milliseconds to datetime
                    df[timestamp_col_name] = pd.to_datetime(df[ts_col_identifier], unit='ms', errors='coerce')
                elif fmt == 'unix_s':
                     # Convert seconds to datetime
                    df[timestamp_col_name] = pd.to_datetime(df[ts_col_identifier], unit='s', errors='coerce')
                else:
                    # Parse with specified format string
                    # --- Add Debugging ---
                    if isinstance(ts_col_identifier, tuple):
                        logger.debug(f"Attempting to parse timestamp column: {ts_col_identifier}")
                    try:
                        target_series = df[ts_col_identifier]
                        logger.debug(f"Timestamp column '{ts_col_identifier}' dtype: {target_series.dtype}")
                        logger.debug(f"Timestamp column '{ts_col_identifier}' head(5):\n{target_series.head().to_string()}")
                    except Exception as log_err:
                        logger.error(f"Error accessing timestamp series '{ts_col_identifier}' for logging: {log_err}")
                    # --- End Debugging ---
                    df[timestamp_col_name] = pd.to_datetime(df[ts_col_identifier], format=fmt, errors='coerce')

                # Skip to timezone handling if we already parsed timestamps
                if continue_to_tz:
                    logger.debug("Skipping to timezone handling due to direct parsing success")
                # Otherwise drop original timestamp column if needed 
                elif ts_col_identifier != timestamp_col_name and ts_col_identifier in df.columns:
                    df = df.drop(columns=[ts_col_identifier])

            # Handle potential timezone information if available in spec
            # (Assuming timezone info might be added later, e.g., from JSON Properties)
            tz = format_spec.get('timezone', 'Europe/Copenhagen') # Default to Copenhagen
            if tz and pd.api.types.is_datetime64_any_dtype(df[timestamp_col_name]):
                 # Localize if naive, convert if aware but different
                 if df[timestamp_col_name].dt.tz is None:
                     try:
                         # Try normal localization first
                         df[timestamp_col_name] = df[timestamp_col_name].dt.tz_localize(
                             tz, ambiguous='NaT', nonexistent='shift_forward'
                         )
                         logger.debug(f"Localized timestamp to {tz} (with ambiguous=NaT)")
                     except Exception as tz_error:
                         # Handle localization errors by using a safer approach
                         logger.warning(f"Error during timezone localization: {tz_error}. Using alternative method.")
                         
                         # Convert timestamps to UTC first (no DST ambiguity) then to desired timezone
                         df[timestamp_col_name] = pd.to_datetime(df[timestamp_col_name])
                         # Create temp UTC timestamps
                         df['timestamp_utc'] = df[timestamp_col_name].dt.tz_localize('UTC')
                         # Convert from UTC to target timezone
                         df[timestamp_col_name] = df['timestamp_utc'].dt.tz_convert(tz)
                         # Drop temp column
                         df.drop('timestamp_utc', axis=1, inplace=True)
                         logger.debug(f"Used UTC-first conversion method to handle timezone {tz}")
                         
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
                # Defensive check - ensure we're accessing a Series, not a DataFrame
                try:
                    selected_col = df[col]
                    
                    # Handle the case where the column returns a DataFrame instead of a Series
                    if isinstance(selected_col, pd.DataFrame):
                        logger.warning(f"Column '{col}' returned a DataFrame instead of a Series. Attempting to fix.")
                        # Get the first column of the returned DataFrame as our Series
                        if not selected_col.empty and len(selected_col.columns) > 0:
                            first_sub_col = selected_col.columns[0]
                            selected_col = selected_col[first_sub_col]
                            logger.info(f"Using first sub-column '{first_sub_col}' from the DataFrame returned by '{col}'")
                            # Replace the original column with this series
                            df[col] = selected_col
                        else:
                            logger.error(f"Cannot convert column '{col}' - returned an empty DataFrame.")
                            continue
                            
                    # Now proceed with numeric conversion on the Series
                    original_dtype = selected_col.dtype
                    
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
                except Exception as e:
                    logger.error(f"Error converting column '{col}' to numeric: {e}")
                    continue
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
                    try:
                        # Try median first (traditional approach)
                        if df[col].count() > 0:  # Only if we have non-NA values
                            median_val = df[col].median()
                            df[col] = df[col].fillna(median_val)
                            logger.debug(f"Filled NaNs in numeric column '{col}' with median ({median_val}).")
                        else:
                            # For empty columns or all NaN, try forward/backward fill
                            logger.warning(f"Column '{col}' has all NaN values. Using forward/backward fill.")
                            # First try forward fill
                            df[col] = df[col].ffill()
                            # Then backward fill any remaining NaNs
                            df[col] = df[col].bfill()
                            
                            # If still have NaNs (all NaNs case), use 0 as last resort
                            if df[col].isnull().any():
                                logger.warning(f"Forward/backward fill failed for '{col}'. Filling with 0.")
                                df[col] = df[col].fillna(0)
                    except Exception as e:
                        logger.error(f"Error calculating median for column '{col}': {e}")
                        # Use forward/backward fill as fallback
                        logger.info(f"Using forward/backward fill for column '{col}' due to median calculation error.")
                        # Try forward fill first
                        df[col] = df[col].ffill()
                        # Then backward fill
                        df[col] = df[col].bfill()
                        # If still have NaNs, use 0
                        if df[col].isnull().any():
                            df[col] = df[col].fillna(0)
                            logger.debug(f"Filled remaining NaNs in column '{col}' with 0.")
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

        # Only log if DataFrame is not empty
        if not df.empty:
            logger.debug(f"Added metadata: source_file='{file_path.name}', source_identifier='{df['source_identifier'].iloc[0]}'")
        else:
            logger.debug(f"Added metadata columns to empty DataFrame for {file_path.name}")

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

    def _parse_knudjepsen_timestamp(self, df: pd.DataFrame, format_spec: Dict[str, Any]) -> pd.DataFrame:
        """Custom timestamp parser for Knudjepsen files.
        
        Accesses the first column directly by position, ignoring potentially complex column names,
        and applies manual datetime parsing with the specified format.
        """
        df = df.copy()
        timestamp_col_name = 'timestamp'  # Standard name for output column
        
        # Get format string from spec
        _, fmt = format_spec.get('timestamp_info', ('', 'dd-mm-yyyy HH:MM:SS'))
        logger.debug(f"Using timestamp format: {fmt}")
        
        try:
            first_col_values = df.iloc[:, 0].copy()
            parsed_timestamps = pd.Series(index=df.index, dtype='datetime64[ns]')
            failed_indices = []
            failed_values = []

            # --- Attempt 1: Direct parsing with format ---
            logger.debug(f"Attempt 1: Direct parsing with format: {fmt}")
            parsed_timestamps = pd.to_datetime(first_col_values, format=fmt, errors='coerce')
            failed_mask = parsed_timestamps.isna()
            num_failed_attempt1 = failed_mask.sum()
            logger.debug(f"Attempt 1 results: {len(df) - num_failed_attempt1} timestamps parsed successfully, {num_failed_attempt1} failed initially.")

            if num_failed_attempt1 > 0:
                # --- Attempt 2: Clean strings first for those that failed --- 
                failed_indices_attempt1 = df.index[failed_mask]
                values_to_retry = first_col_values.loc[failed_indices_attempt1]
                logger.debug(f"Attempt 2: Cleaning {num_failed_attempt1} strings first...")
                
                if not pd.api.types.is_string_dtype(values_to_retry.dtype) and not pd.api.types.is_object_dtype(values_to_retry.dtype):
                    logger.warning(f"Timestamp values to retry dtype is {values_to_retry.dtype}, converting to string.")
                    values_to_retry = values_to_retry.astype(str)
                
                cleaned_values = values_to_retry.str.strip('"\' ').str.strip() # More aggressive strip
                parsed_attempt2 = pd.to_datetime(cleaned_values, format=fmt, errors='coerce')
                # Update the main series with successfully parsed values from attempt 2
                parsed_timestamps.loc[parsed_attempt2.notna()] = parsed_attempt2[parsed_attempt2.notna()]
                
                failed_mask_attempt2 = parsed_timestamps.isna() # Re-evaluate failures
                num_failed_attempt2 = failed_mask_attempt2.sum()
                logger.debug(f"Attempt 2 results: {num_failed_attempt1 - num_failed_attempt2} additional timestamps parsed, {num_failed_attempt2} still failed.")

                if num_failed_attempt2 > 0:
                     # --- Attempt 3: Use dateutil's flexible parser for remaining failures ---
                    failed_indices_attempt2 = df.index[failed_mask_attempt2]
                    values_to_retry_dateutil = first_col_values.loc[failed_indices_attempt2]
                    logger.debug(f"Attempt 3: Using dateutil parser for {num_failed_attempt2} remaining failures...")
                    from dateutil import parser
                    
                    def safe_parse(val):
                        try:
                            if pd.isna(val) or str(val).strip() == '': return pd.NaT
                            # Try parsing, potentially with dayfirst=True if formats are ambiguous like d/m/y
                            # For dd-mm-yyyy HH:MM:SS, dayfirst isn't strictly needed but doesn't hurt
                            return parser.parse(str(val), dayfirst=True) 
                        except Exception: # Catch broad errors during parsing
                            return pd.NaT
                    
                    parsed_attempt3 = values_to_retry_dateutil.apply(safe_parse)
                    parsed_timestamps.loc[parsed_attempt3.notna()] = parsed_attempt3[parsed_attempt3.notna()]

            # --- Final check and Logging of failed values ---
            final_failed_mask = parsed_timestamps.isna()
            num_failed_final = final_failed_mask.sum()
            if num_failed_final > 0:
                failed_indices = df.index[final_failed_mask]
                failed_values = first_col_values.loc[failed_indices].tolist()
                # Log only a sample of failed values to avoid huge logs
                sample_size = min(20, len(failed_values))
                logger.warning(f"{num_failed_final} values could not be parsed as timestamps.")
                logger.warning(f"Sample of failed values (first {sample_size}): {failed_values[:sample_size]}")

            # Assign the successfully parsed timestamps to the DataFrame column
            df[timestamp_col_name] = parsed_timestamps
            # Handle timezone localization
            tz = format_spec.get('timezone', 'Europe/Copenhagen')  # Default to Copenhagen
            if tz and pd.api.types.is_datetime64_any_dtype(df[timestamp_col_name]):
                if df[timestamp_col_name].dt.tz is None:
                    try:
                        # Try normal localization first
                        df[timestamp_col_name] = df[timestamp_col_name].dt.tz_localize(
                            tz, ambiguous='NaT', nonexistent='shift_forward'
                        )
                        logger.debug(f"Localized timestamp to {tz} (with ambiguous=NaT)")
                    except Exception as tz_error:
                        # Handle localization errors by using a safer approach
                        logger.warning(f"Error during timezone localization: {tz_error}. Using alternative method.")
                        
                        # Convert timestamps to UTC first (no DST ambiguity) then to desired timezone
                        df[timestamp_col_name] = pd.to_datetime(df[timestamp_col_name])
                        # Create temp UTC timestamps
                        df['timestamp_utc'] = df[timestamp_col_name].dt.tz_localize('UTC')
                        # Convert from UTC to target timezone
                        df[timestamp_col_name] = df['timestamp_utc'].dt.tz_convert(tz)
                        # Drop temp column
                        df.drop('timestamp_utc', axis=1, inplace=True)
                        logger.debug(f"Used UTC-first conversion method to handle timezone {tz}")
                        
                elif str(df[timestamp_col_name].dt.tz) != tz:
                    df[timestamp_col_name] = df[timestamp_col_name].dt.tz_convert(tz)
                    logger.debug(f"Converted timestamp timezone to {tz}")
            
            # Drop rows where timestamp parsing failed (using the final mask)
            original_count = len(df)
            df = df.dropna(subset=[timestamp_col_name])
            dropped_count = original_count - len(df)
            
            if dropped_count > 0:
                # This log now reflects the final count after all attempts
                logger.warning(f"Dropped {dropped_count} rows due to NaT timestamps (after {num_failed_final} parsing failures)")

            # Final success message
            logger.info(f"Successfully parsed timestamps for Knudjepsen file. {len(df)} valid rows retained.")
            return df
            
        except Exception as e:
            logger.error(f"Error in custom Knudjepsen timestamp handling: {e}", exc_info=True)
            # Return empty DataFrame on failure
            return pd.DataFrame() 