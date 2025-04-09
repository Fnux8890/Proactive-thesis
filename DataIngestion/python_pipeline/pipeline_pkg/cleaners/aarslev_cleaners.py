import pandas as pd
import logging
import re
import json
from .base_cleaner import BaseCleaner
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

class AarslevSimpleCsvCleaner(BaseCleaner):
    """Cleaner for simple Aarslev CSVs (weather, data_jan_feb, winter)."""
    def clean(self, df: pd.DataFrame, format_spec: Dict[str, Any]) -> pd.DataFrame:
        logger.info(f"Starting cleaning for Aarslev Simple CSV: {format_spec.get('full_path')}")
        # 1. Standardize columns
        df = self._standardize_columns(df)

        # 2. Handle timestamp (unix_ms)
        df = self._handle_timestamp(df, format_spec)
        if 'timestamp' not in df.columns:
            logger.error("Timestamp handling failed.")
            return pd.DataFrame()

        # 3. Identify value columns
        value_cols_spec = format_spec.get('value_columns', [])
        # Standardize the column names from the spec
        std_value_cols_spec = [self._standardize_columns(pd.DataFrame(columns=[c])).columns[0] for c in value_cols_spec]
        # Keep only the standardized spec names that exist in the standardized DataFrame columns
        value_cols = [col for col in std_value_cols_spec if col in df.columns]

        # 4. Convert numeric
        decimal_sep = format_spec.get('loader_params', {}).get('decimal', '.')
        df = self._convert_numeric(df, value_cols, decimal=decimal_sep)

        # 5. Handle missing values
        df = self._handle_missing_values(df)

        # 6. Add metadata
        df = self._add_metadata(df, format_spec)

        # 7. Reshape
        id_vars = ['timestamp', 'source_file', 'source_identifier']
        df_long = self._reshape_to_long(df, id_vars=id_vars)

        # Add units if specified separately (optional, could be merged elsewhere)
        # units = format_spec.get('units', {})
        # if units and 'measurement' in df_long.columns:
        #     std_units = {self._standardize_columns(pd.DataFrame(columns=[k]))[k]: v for k, v in units.items()}
        #     df_long['unit'] = df_long['measurement'].map(std_units)

        logger.info(f"Finished cleaning for Aarslev Simple CSV. Shape: {df_long.shape}")
        return df_long

class AarslevMortenCsvCleaner(BaseCleaner):
    """Cleaner for MortenSDUData CSV files."""
    def clean(self, df: pd.DataFrame, format_spec: Dict[str, Any]) -> pd.DataFrame:
        logger.info(f"Starting cleaning for Aarslev Morten CSV: {format_spec.get('full_path')}")

        # 1. Handle timestamp ('start' column) - BEFORE standardization
        df = self._handle_timestamp(df, format_spec)
        if 'timestamp' not in df.columns:
            logger.error("Timestamp handling failed.")
            return pd.DataFrame()

        # 2. Standardize columns (now that timestamp is handled and potentially dropped)
        df = self._standardize_columns(df)

        # 3. Extract Measurement and Unit from column names using regex
        #    (Ensure regex patterns target standardized names if necessary,
        #     or adjust standardization if it breaks patterns)
        value_info = format_spec.get('value_info', {})
        regex_pattern = value_info.get('regex_pattern')
        name_group = value_info.get('column_name_group', 1)
        unit_group = value_info.get('unit_group', 2)

        if not regex_pattern:
            logger.error("Missing 'regex_pattern' in value_info for AarslevMortenCsvCleaner.")
            return pd.DataFrame()

        rename_map = {}
        units_map = {}
        value_cols_new = []
        original_value_cols = [col for col in df.columns if col not in ['timestamp', 'end']] # Exclude timestamp and 'end' column

        for col in original_value_cols:
            match = re.match(regex_pattern, col)
            if match:
                try:
                    measurement_name = match.group(name_group).strip()
                    unit = match.group(unit_group).strip()
                    # Standardize measurement name
                    std_measurement_name = self._standardize_columns(pd.DataFrame(columns=[measurement_name])).columns[0]
                    rename_map[col] = std_measurement_name
                    units_map[std_measurement_name] = unit
                    value_cols_new.append(std_measurement_name)
                except IndexError:
                    logger.warning(f"Regex groups not found for column '{col}' using pattern '{regex_pattern}'. Skipping.")
            else:
                logger.warning(f"Column '{col}' did not match regex pattern '{regex_pattern}'. Keeping original name.")
                # Keep original standardized name if no match
                value_cols_new.append(col)
                rename_map[col] = col # Keep original name in map

        df = df.rename(columns=rename_map)
        # Keep only the identified/renamed value columns + timestamp
        cols_to_keep = ['timestamp'] + value_cols_new
        df = df[[col for col in cols_to_keep if col in df.columns]] # Ensure columns exist

        # 4. Convert numeric
        decimal_sep = format_spec.get('loader_params', {}).get('decimal', '.')
        df = self._convert_numeric(df, value_cols_new, decimal=decimal_sep)

        # 5. Handle missing values
        df = self._handle_missing_values(df)

        # 6. Add metadata
        df = self._add_metadata(df, format_spec)

        # 7. Reshape
        id_vars = ['timestamp', 'source_file', 'source_identifier']
        df_long = self._reshape_to_long(df, id_vars=id_vars)

        # Add units from extracted map
        if units_map and 'measurement' in df_long.columns:
            df_long['unit'] = df_long['measurement'].map(units_map)
            df_long['unit'] = df_long['unit'].fillna('Unknown')
            logger.debug("Added units from column name regex.")

        # Add location from column name regex
        location_pattern = format_spec.get('location_pattern')
        if location_pattern and 'measurement' in df_long.columns:
            # Need to map based on the *original* measurement name before standardization?
            # This is tricky. Let's try applying pattern to the *standardized* name.
            def extract_location(meas_name):
                # Find the original column name that maps to this std name
                orig_col = next((k for k, v in rename_map.items() if v == meas_name), None)
                if orig_col:
                     match = re.match(location_pattern, orig_col)
                     if match: return match.group(1).strip().lower()
                # Fallback: try pattern on std name itself
                match = re.match(location_pattern, meas_name)
                if match: return match.group(1).strip().lower()
                return 'unknown_location'

            df_long['location'] = df_long['measurement'].apply(extract_location)
            logger.debug(f"Added location based on pattern: {location_pattern}")

        logger.info(f"Finished cleaning for Aarslev Morten CSV. Shape: {df_long.shape}")
        return df_long

class AarslevCelleCsvCleaner(BaseCleaner):
    """Cleaner for the Celle output CSV files with quoted headers."""
    def clean(self, df: pd.DataFrame, format_spec: Dict[str, Any]) -> pd.DataFrame:
        logger.info(f"Starting cleaning for Aarslev Celle CSV: {format_spec.get('full_path')}")
        # 1. Clean and Standardize Column Names (remove quotes, apply regex)
        value_info = format_spec.get('value_info', {})
        regex_pattern = value_info.get('regex_pattern') # e.g., r'"(Celle \d+: .*)"+'
        name_group = value_info.get('column_name_group', 1)

        if not regex_pattern:
            logger.error("Missing 'regex_pattern' in value_info for AarslevCelleCsvCleaner.")
            return pd.DataFrame()

        rename_map = {}
        original_cols = list(df.columns)

        for col in original_cols:
            col_str = str(col).strip()
            match = re.match(regex_pattern, col_str)
            if match:
                try:
                    extracted_name = match.group(name_group).strip()
                    # Apply base standardization (snake_case)
                    std_name = self._standardize_columns(pd.DataFrame(columns=[extracted_name])).columns[0]
                    rename_map[col] = std_name
                    logger.debug(f"Renamed column '{col}' to '{std_name}'")
                except IndexError:
                     logger.warning(f"Regex group {name_group} not found for column '{col}' using pattern '{regex_pattern}'. Standardizing original.")
                     std_name = self._standardize_columns(pd.DataFrame(columns=[col_str])).columns[0]
                     rename_map[col] = std_name
            elif col_str in ('Date', 'Time'): # Keep date/time for timestamp handling
                rename_map[col] = col_str.lower()
            else:
                 logger.warning(f"Column '{col}' did not match pattern '{regex_pattern}'. Standardizing original name.")
                 std_name = self._standardize_columns(pd.DataFrame(columns=[col_str])).columns[0]
                 rename_map[col] = std_name

        df = df.rename(columns=rename_map)
        logger.debug(f"Columns after initial cleaning/rename: {df.columns.tolist()}")

        # 2. Handle timestamp (combined Date and Time)
        # Ensure 'date' and 'time' columns exist after renaming
        if 'date' not in df.columns or 'time' not in df.columns:
             logger.error("Could not find 'date' or 'time' columns after renaming. Check regex/column names.")
             return pd.DataFrame()

        df = self._handle_timestamp(df, format_spec)
        if 'timestamp' not in df.columns:
            logger.error("Timestamp handling failed.")
            return pd.DataFrame()

        # 3. Identify value columns (all except timestamp)
        value_cols = [col for col in df.columns if col != 'timestamp']

        # 4. Convert numeric
        # Important: Meta says decimal="," - use this from spec
        decimal_sep = format_spec.get('loader_params', {}).get('decimal', '.')
        logger.info(f"Using decimal separator: '{decimal_sep}' for numeric conversion.")
        df = self._convert_numeric(df, value_cols, decimal=decimal_sep)

        # 5. Handle missing values
        df = self._handle_missing_values(df)

        # 6. Add metadata (extracts 'celleX' using source_id_pattern)
        df = self._add_metadata(df, format_spec)

        # 7. Reshape
        id_vars = ['timestamp', 'source_file', 'source_identifier']
        df_long = self._reshape_to_long(df, id_vars=id_vars)

        # 8. Add Units using unit_map_pattern
        unit_map_pattern = format_spec.get('unit_map_pattern', {})
        if unit_map_pattern and 'measurement' in df_long.columns:
            # Store a mapping of extracted original names directly during initial column processing
            # and store it as a class attribute
            
            # We need the original extracted name (before standardization) to match patterns
            # Create a reverse map from standardized name to original extracted name
            reverse_rename_map = {v: k for k, v in rename_map.items() if k not in ('Date', 'Time')}
            original_extracted_names = {}
            for std_name, orig_col in reverse_rename_map.items():
                 match = re.match(regex_pattern, str(orig_col).strip())
                 if match:
                     try: 
                         extracted = match.group(name_group).strip()
                         original_extracted_names[std_name] = extracted
                         logger.debug(f"Mapped standardized '{std_name}' back to extracted '{extracted}'")
                     except IndexError: 
                         pass

            def get_unit(std_meas_name):
                orig_extracted = original_extracted_names.get(std_meas_name)
                if orig_extracted:
                    for pattern, unit in unit_map_pattern.items():
                        if re.match(pattern, orig_extracted):
                            logger.debug(f"Matched unit '{unit}' for measurement '{std_meas_name}' using pattern '{pattern}'")
                            return unit
                logger.warning(f"No unit pattern matched for measurement: '{std_meas_name}' (Original: '{orig_extracted}')")
                return 'Unknown' # Default if no pattern matches

            df_long['unit'] = df_long['measurement'].apply(get_unit)
            logger.debug("Applied units based on unit_map_pattern.")

        # 9. Add Location using location_pattern
        location_pattern = format_spec.get('location_pattern')
        if location_pattern and 'measurement' in df_long.columns:
             # Similar to units, match against original extracted name
             def extract_location(std_meas_name):
                 orig_extracted = original_extracted_names.get(std_meas_name)
                 if orig_extracted:
                     match = re.match(location_pattern, orig_extracted)
                     if match: 
                         location = match.group(1).strip().lower()
                         logger.debug(f"Matched location '{location}' for measurement '{std_meas_name}' using pattern '{location_pattern}'")
                         return location
                 logger.warning(f"No location pattern matched for measurement: '{std_meas_name}' (Original: '{orig_extracted}')")
                 return 'unknown_location'

             df_long['location'] = df_long['measurement'].apply(extract_location)
             logger.debug(f"Added location based on pattern: {location_pattern}")


        logger.info(f"Finished cleaning for Aarslev Celle CSV. Shape: {df_long.shape}")
        return df_long

class AarslevUuidHeaderCsvCleaner(BaseCleaner):
    """Cleaner for CSVs with UUIDs in the header."""
    def clean(self, df: pd.DataFrame, format_spec: Dict[str, Any]) -> pd.DataFrame:
        logger.info(f"Starting cleaning for Aarslev UUID Header CSV: {format_spec.get('full_path')}")
        # 1. Map UUIDs to measurement names and units
        uuid_map = format_spec.get('uuid_map')
        if not uuid_map:
            logger.error("Missing 'uuid_map' in format_spec for AarslevUuidHeaderCsvCleaner.")
            return pd.DataFrame()

        rename_map = {}
        units_map = {}
        value_cols_new = []

        for col in df.columns:
            col_str = str(col).strip()
            if col_str in uuid_map:
                measurement_name, unit = uuid_map[col_str]
                # Standardize measurement name
                std_measurement_name = self._standardize_columns(pd.DataFrame(columns=[measurement_name])).columns[0]
                rename_map[col] = std_measurement_name
                units_map[std_measurement_name] = unit
                value_cols_new.append(std_measurement_name)
                logger.debug(f"Mapped UUID '{col_str}' to '{std_measurement_name}' (Unit: {unit})")
            elif col_str.lower() == 'timestamp': # Keep timestamp column
                 rename_map[col] = 'timestamp' # Ensure lowercase
            else:
                logger.warning(f"Column/UUID '{col_str}' not found in uuid_map. Standardizing name.")
                std_name = self._standardize_columns(pd.DataFrame(columns=[col_str])).columns[0]
                rename_map[col] = std_name
                # Keep it? Assume it might be needed, but won't have unit.
                if std_name != 'timestamp':
                    value_cols_new.append(std_name)

        df = df.rename(columns=rename_map)
        cols_to_keep = ['timestamp'] + value_cols_new
        df = df[[col for col in cols_to_keep if col in df.columns]] # Ensure columns exist

        # 2. Handle timestamp (unix_ms)
        df = self._handle_timestamp(df, format_spec)
        if 'timestamp' not in df.columns:
            logger.error("Timestamp handling failed.")
            return pd.DataFrame()

        # 3. Convert numeric
        decimal_sep = format_spec.get('loader_params', {}).get('decimal', '.')
        df = self._convert_numeric(df, value_cols_new, decimal=decimal_sep)

        # 4. Handle missing values
        df = self._handle_missing_values(df)

        # 5. Add metadata
        df = self._add_metadata(df, format_spec)

        # 6. Reshape
        id_vars = ['timestamp', 'source_file', 'source_identifier']
        df_long = self._reshape_to_long(df, id_vars=id_vars)

        # Add units from map
        if units_map and 'measurement' in df_long.columns:
            df_long['unit'] = df_long['measurement'].map(units_map)
            df_long['unit'] = df_long['unit'].fillna('Unknown')
            logger.debug("Added units from UUID map.")

        logger.info(f"Finished cleaning for Aarslev UUID Header CSV. Shape: {df_long.shape}")
        return df_long

# --- JSON Cleaners ---

class AarslevStreamListJsonCleaner(BaseCleaner):
    """Cleaner for Aarslev JSON format: List of stream dictionaries."""
    def clean(self, df: pd.DataFrame, format_spec: Dict[str, Any]) -> pd.DataFrame:
        logger.info(f"Starting cleaning for Aarslev Stream List JSON: {format_spec.get('full_path')}")
        # Expects df to contain raw JSON data if loader didn't use orient
        # Or potentially a normalized structure if json_normalize worked.

        if 'raw_json_data' in df.columns and len(df) == 1:
            raw_data = df['raw_json_data'].iloc[0]
            if not isinstance(raw_data, list):
                logger.error("Expected raw_json_data to be a list of streams.")
                return pd.DataFrame()
            logger.debug("Processing raw JSON list structure.")
        elif isinstance(df, pd.DataFrame) and not df.empty:
             # Assume json_normalize might have worked partially, try to reconstruct
             # This path is less ideal, requires assumptions about normalization.
             # Or, perhaps the loader *was* able to load it directly.
             # Let's proceed assuming df *is* the list of streams (e.g., loaded with orient='records')
             # OR requires iterating rows if normalized.
             # For simplicity, let's REQUIRE raw_json_data for this cleaner.
            logger.error("Expected DataFrame with single 'raw_json_data' column. Input DF structure not supported.")
            # If loader passed raw_data directly (not wrapped): clean(self, raw_data: List[Dict], ...)
            return pd.DataFrame()
        else:
             logger.error("Input data for AarslevStreamListJsonCleaner is empty or invalid.")
             return pd.DataFrame()

        all_readings = []
        for stream in raw_data:
            try:
                uuid = stream.get('uuid')
                properties = stream.get('Properties', {})
                unit = properties.get('UnitofMeasure', 'Unknown')
                # ReadingType? properties.get('ReadingType')
                metadata = stream.get('Metadata', {})
                # Use source name from Metadata if available, else use file-based one
                source_name = metadata.get('SourceName')
                timezone = properties.get('Timezone', 'Europe/Copenhagen') # Get timezone per stream

                readings = stream.get('Readings', [])
                if not readings:
                    continue

                # Create DataFrame for this stream's readings
                stream_df = pd.DataFrame(readings, columns=['timestamp_ms', 'value'])
                stream_df['uuid'] = uuid
                stream_df['measurement'] = uuid # Use UUID as initial measurement identifier
                stream_df['unit'] = unit
                stream_df['source_name_meta'] = source_name
                stream_df['timezone_meta'] = timezone

                all_readings.append(stream_df)

            except Exception as e:
                logger.error(f"Error processing stream: {uuid if 'uuid' in locals() else 'Unknown UUID'}: {e}", exc_info=True)
                continue

        if not all_readings:
            logger.warning("No valid readings found in JSON streams.")
            return pd.DataFrame()

        # Combine all stream DataFrames
        df_combined = pd.concat(all_readings, ignore_index=True)
        logger.info(f"Combined {len(all_readings)} streams into DataFrame. Shape: {df_combined.shape}")

        # 1. Handle timestamp (unix_ms)
        df_combined['timestamp'] = pd.to_datetime(df_combined['timestamp_ms'], unit='ms', errors='coerce')
        # Handle timezone - apply timezone found in metadata
        # This assumes all streams in a file *should* have the same timezone, or we handle per row
        # Group by timezone and apply? Simpler: apply first found timezone or default.
        first_tz = df_combined['timezone_meta'].dropna().unique()
        tz_to_apply = first_tz[0] if len(first_tz) > 0 else 'Europe/Copenhagen'

        if pd.api.types.is_datetime64_any_dtype(df_combined['timestamp']):
             if df_combined['timestamp'].dt.tz is None:
                 df_combined['timestamp'] = df_combined['timestamp'].dt.tz_localize(tz_to_apply, ambiguous='infer', nonexistent='shift_forward')
                 logger.debug(f"Localized timestamp to {tz_to_apply}")
             else: # If already localized (e.g., by unit='ms'), convert if needed
                 df_combined['timestamp'] = df_combined['timestamp'].dt.tz_convert(tz_to_apply)
                 logger.debug(f"Converted timestamp timezone to {tz_to_apply}")
        
        # Drop rows with invalid timestamps
        original_count = len(df_combined)
        df_combined = df_combined.dropna(subset=['timestamp'])
        dropped_count = original_count - len(df_combined)
        if dropped_count > 0:
            logger.warning(f"Dropped {dropped_count} rows due to NaT timestamps during JSON processing.")

        if 'timestamp' not in df_combined.columns or df_combined['timestamp'].isnull().all():
             logger.error("Timestamp handling failed for combined JSON data.")
             return pd.DataFrame()

        # 2. Convert value column to numeric
        # Use a generic approach as ReadingType might vary
        df_combined['value'] = pd.to_numeric(df_combined['value'], errors='coerce')
        nan_count = df_combined['value'].isna().sum()
        if nan_count > 0:
             logger.warning(f"'value' column: {nan_count} values failed numeric conversion (became NaN).")

        # 3. Handle missing values (value column)
        # Fill based on median *per measurement group*?
        df_combined['value'] = df_combined.groupby('measurement')['value'].transform(lambda x: x.fillna(x.median()))
        # If all values in a group were NaN, median is NaN, fill remaining with 0? Or global median?
        df_combined['value'] = df_combined['value'].fillna(0) # Fallback fill
        logger.debug("Filled NaNs in 'value' column using median per measurement group (UUID).")

        # 4. Add metadata (source_file, source_identifier)
        df_combined = self._add_metadata(df_combined, format_spec)

        # 5. Select and rename final columns
        # Keep: timestamp, value, measurement (UUID), unit, source_file, source_identifier
        final_cols = ['timestamp', 'value', 'measurement', 'unit', 'source_file', 'source_identifier']
        df_final = df_combined[[col for col in final_cols if col in df_combined.columns]]

        # Data is already in long format
        logger.info(f"Finished cleaning for Aarslev Stream List JSON. Shape: {df_final.shape}")
        return df_final

class AarslevStreamDictJsonCleaner(BaseCleaner):
    """Cleaner for Aarslev JSON format: Dictionary of stream dictionaries keyed by path."""
    def clean(self, df: pd.DataFrame, format_spec: Dict[str, Any]) -> pd.DataFrame:
        logger.info(f"Starting cleaning for Aarslev Stream Dict JSON: {format_spec.get('full_path')}")
        # Similar expectation as List cleaner: prefers raw JSON data.
        if 'raw_json_data' in df.columns and len(df) == 1:
            raw_data = df['raw_json_data'].iloc[0]
            if not isinstance(raw_data, dict):
                logger.error("Expected raw_json_data to be a dictionary of streams.")
                return pd.DataFrame()
            logger.debug("Processing raw JSON dictionary structure.")
        else:
             logger.error("Input data for AarslevStreamDictJsonCleaner is empty, invalid, or not raw JSON.")
             return pd.DataFrame()

        all_readings = []
        for path_key, stream in raw_data.items():
            try:
                if not isinstance(stream, dict): # Skip if value is not a stream dict
                    logger.warning(f"Skipping non-dictionary item found at key '{path_key}'")
                    continue

                uuid = stream.get('uuid')
                properties = stream.get('Properties', {})
                unit = properties.get('UnitofMeasure', 'Unknown')
                timezone = properties.get('Timezone', 'Europe/Copenhagen')
                metadata = stream.get('Metadata', {})
                source_name = metadata.get('SourceName')

                readings = stream.get('Readings', [])
                if not readings:
                    continue

                # Create DataFrame for this stream
                stream_df = pd.DataFrame(readings, columns=['timestamp_ms', 'value'])
                stream_df['uuid'] = uuid
                stream_df['measurement_path'] = path_key # Use the dict key as measurement identifier
                stream_df['unit'] = unit
                stream_df['source_name_meta'] = source_name
                stream_df['timezone_meta'] = timezone

                all_readings.append(stream_df)

            except Exception as e:
                logger.error(f"Error processing stream with key '{path_key}': {e}", exc_info=True)
                continue

        if not all_readings:
            logger.warning("No valid readings found in JSON stream dictionary.")
            return pd.DataFrame()

        # Combine all stream DataFrames
        df_combined = pd.concat(all_readings, ignore_index=True)
        logger.info(f"Combined {len(all_readings)} streams from dict. Shape: {df_combined.shape}")

        # 1. Handle timestamp (unix_ms) & Timezone (similar to List cleaner)
        df_combined['timestamp'] = pd.to_datetime(df_combined['timestamp_ms'], unit='ms', errors='coerce')
        first_tz = df_combined['timezone_meta'].dropna().unique()
        tz_to_apply = first_tz[0] if len(first_tz) > 0 else 'Europe/Copenhagen'
        if pd.api.types.is_datetime64_any_dtype(df_combined['timestamp']):
             if df_combined['timestamp'].dt.tz is None:
                 df_combined['timestamp'] = df_combined['timestamp'].dt.tz_localize(tz_to_apply, ambiguous='infer', nonexistent='shift_forward')
                 logger.debug(f"Localized timestamp to {tz_to_apply}")
             else:
                 df_combined['timestamp'] = df_combined['timestamp'].dt.tz_convert(tz_to_apply)
                 logger.debug(f"Converted timestamp timezone to {tz_to_apply}")
        original_count = len(df_combined)
        df_combined = df_combined.dropna(subset=['timestamp'])
        dropped_count = original_count - len(df_combined)
        if dropped_count > 0: logger.warning(f"Dropped {dropped_count} rows due to NaT timestamps.")
        if 'timestamp' not in df_combined.columns or df_combined['timestamp'].isnull().all():
             logger.error("Timestamp handling failed for combined JSON data.")
             return pd.DataFrame()

        # 2. Convert value column to numeric
        df_combined['value'] = pd.to_numeric(df_combined['value'], errors='coerce')
        nan_count = df_combined['value'].isna().sum()
        if nan_count > 0: logger.warning(f"'value' column: {nan_count} values failed numeric conversion.")

        # 3. Handle missing values (value column, grouped by path)
        df_combined['value'] = df_combined.groupby('measurement_path')['value'].transform(lambda x: x.fillna(x.median()))
        df_combined['value'] = df_combined['value'].fillna(0) # Fallback
        logger.debug("Filled NaNs in 'value' column using median per measurement path.")

        # 4. Add metadata (base source_id, file name)
        df_combined = self._add_metadata(df_combined, format_spec)

        # 5. Add Location based on measurement_path and location_pattern
        location_pattern = format_spec.get('location_pattern') # e.g., r"/(Cell\d+)/.*"
        if location_pattern and 'measurement_path' in df_combined.columns:
            def extract_location(path):
                 match = re.match(location_pattern, path)
                 if match: return match.group(1).strip().lower()
                 return 'unknown_location'
            df_combined['location'] = df_combined['measurement_path'].apply(extract_location)
            logger.debug(f"Added location based on pattern: {location_pattern}")
        else:
             df_combined['location'] = 'unknown' # Default if no pattern

        # 6. Standardize measurement_path to create 'measurement' column?
        # Example: "/Cell5/air_temperature" -> "cell5_air_temperature"
        df_combined['measurement'] = df_combined['measurement_path'].apply(
            lambda x: self._standardize_columns(pd.DataFrame(columns=[x.strip('/').replace('/', '_')])).columns[0]
        )
        logger.debug("Created standardized 'measurement' column from path.")

        # 7. Select and rename final columns
        # Keep: timestamp, value, measurement, unit, location, source_file, source_identifier, uuid
        final_cols = ['timestamp', 'value', 'measurement', 'unit', 'location', 'source_file', 'source_identifier', 'uuid']
        df_final = df_combined[[col for col in final_cols if col in df_combined.columns]]

        logger.info(f"Finished cleaning for Aarslev Stream Dict JSON. Shape: {df_final.shape}")
        return df_final 