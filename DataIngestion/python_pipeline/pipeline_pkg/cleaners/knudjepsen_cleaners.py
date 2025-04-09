import pandas as pd
import logging
from .base_cleaner import BaseCleaner
from typing import Dict, Any
import re
from ..config import pipeline_config
import unicodedata

logger = logging.getLogger(__name__)

class KnudjepsenMultiHeaderCsvCleaner(BaseCleaner):
    def clean(self, df: pd.DataFrame, format_spec: Dict[str, Any]) -> pd.DataFrame:
        logger.info(f"Starting cleaning for Knudjepsen MultiHeader CSV: {format_spec.get('full_path')}")
        # 1. Handle multi-index columns (assuming loader created them)
        if isinstance(df.columns, pd.MultiIndex):
            # Combine levels, e.g., ('mål temp afd  Mid', '°C', 'Source1') -> 'mal_temp_afd_mid_c_source1'
            # Customize joining logic if needed
            df.columns = ['_'.join(map(str, col)).strip() for col in df.columns.values]
            df = self._standardize_columns(df) # Apply snake_case standardization
        else:
            logger.warning("Expected MultiIndex columns, but got single index. Proceeding with standardization.")
            df = self._standardize_columns(df)

        # --- Store original first column name BEFORE timestamp handling ---
        # This is likely the original, unparsed timestamp column
        original_ts_col_name = df.columns[0] 
        logger.debug(f"Identified original timestamp column name for potential drop: {original_ts_col_name}")
        # --- End Store ---

        # 2. Handle timestamp
        df = self._handle_timestamp(df, format_spec)
        if 'timestamp' not in df.columns:
             logger.error("Timestamp handling failed. Cannot proceed without timestamp.")
             return pd.DataFrame() # Return empty if timestamp fails
        else:
            # --- Drop the original timestamp column if it still exists and is different ---
            if original_ts_col_name in df.columns and original_ts_col_name != 'timestamp':
                logger.info(f"Dropping original timestamp column: {original_ts_col_name}")
                df = df.drop(columns=[original_ts_col_name])
            # --- End Drop ---

        # 3. Identify value columns (needs refinement based on actual column names after step 1)
        # Assuming standardization makes columns predictable based on value_columns spec
        value_cols_spec = format_spec.get('value_columns', [])
        value_cols = []
        for vc in value_cols_spec:
            if isinstance(vc, tuple) and len(vc) > 0:
                # If spec gives a tuple, assume first element is the name to standardize
                original_name = vc[0]
                std_name = self._standardize_columns(pd.DataFrame(columns=[original_name])).columns[0]
                value_cols.append(std_name) # Add the standardized name
            elif isinstance(vc, str):
                # If spec gives a string, standardize it
                std_name = self._standardize_columns(pd.DataFrame(columns=[vc])).columns[0]
                value_cols.append(std_name)
            else:
                 logger.warning(f"Skipping invalid value column spec item: {vc}")

        # Keep only value columns that actually exist in the standardized DataFrame
        value_cols = [col for col in value_cols if col in df.columns]

        # 4. Convert numeric columns
        decimal_sep = format_spec.get('loader_params', {}).get('decimal', '.')
        df = self._convert_numeric(df, value_cols, decimal=decimal_sep)

        # 5. Handle missing values
        df = self._handle_missing_values(df)

        # 6. Add metadata
        df = self._add_metadata(df, format_spec)

        # 7. Reshape to long format
        id_vars = ['timestamp', 'source_file', 'source_identifier'] # Standard ID vars
        df_long = self._reshape_to_long(df, id_vars=id_vars)

        logger.info(f"Finished cleaning for Knudjepsen MultiHeader CSV. Shape: {df_long.shape}")
        return df_long

class KnudjepsenExtraCsvCleaner(BaseCleaner):
    def clean(self, df: pd.DataFrame, format_spec: Dict[str, Any]) -> pd.DataFrame:
        logger.info(f"Starting cleaning for Knudjepsen Extra CSV: {format_spec.get('full_path')}")
        # Specific logic for NO3NO4.extra.csv
        # 1. Handle multi-index (measurement, unit, location_suffix)
        if isinstance(df.columns, pd.MultiIndex) and len(df.columns.levels) >= 3:
            # --- Add Logging ---
            logger.debug(f"Inside cleaner: df.columns type: {type(df.columns)}")
            logger.debug(f"Inside cleaner: df.columns: {df.columns}")
            logger.debug(f"Inside cleaner: df.columns.values: {df.columns.values}")
            # --- End Logging ---
            # Custom column naming: measurement_unit_location
            new_cols = []
            level_map = format_spec.get('value_info', {}).get('level_map', {0: 'meas', 1: 'unit', 2: 'loc'})
            loc_prefix = format_spec.get('value_info', {}).get('location_prefix', 'Afd')
            # Create a mapping from original position to safe column name
            col_name_map = {}
            
            for i, col_tuple in enumerate(df.columns.values):
                logger.debug(f"Processing raw column tuple {i}: {col_tuple}") # Log tuple
                # Handle timestamp column specially (first column, likely unnamed at level 0)
                if col_tuple[0] == '' and col_tuple[1] == '' and col_tuple[2] == '':
                     new_cols.append('timestamp_orig') # Rename placeholder
                     col_name_map[tuple(range(len(col_tuple)))] = 'timestamp_orig'
                     logger.debug(f"  -> Identified as timestamp_orig")
                     continue
                
                # FIXED: Use level positions directly, not the string values from level_map
                # Get level positions from map keys, defaulting to positions 0, 1, 2
                meas_pos = int(list(level_map.keys())[0]) if level_map else 0
                unit_pos = int(list(level_map.keys())[1]) if len(level_map) > 1 else 1
                loc_pos = int(list(level_map.keys())[2]) if len(level_map) > 2 else 2
                
                # Use positions to access tuple elements
                meas = str(col_tuple[meas_pos]).strip()
                unit = str(col_tuple[unit_pos]).strip()
                loc_suffix = str(col_tuple[loc_pos]).strip().replace(loc_prefix, '').strip()
                logger.debug(f"  -> Extracted parts: meas='{meas}', unit='{unit}', loc_suffix='{loc_suffix}'")

                # Check for unnamed columns - similar to our fix for BelysningCsvCleaner
                is_unnamed_col = False
                
                # Check if any part is empty
                if meas == '' or unit == '' or loc_suffix == '':
                    is_unnamed_col = True
                # Check if any part contains 'unnamed' (case insensitive)
                elif any('unnamed' in part.lower() for part in [meas, unit, loc_suffix]):
                    is_unnamed_col = True
                # Check for pandas auto-generated unnamed patterns
                elif any(any(pattern in part.lower() for pattern in ['unnamed:', 'untitled']) 
                        for part in [meas, unit, loc_suffix]):
                    is_unnamed_col = True
                
                if is_unnamed_col:
                    # This is a helper/index column that should be skipped
                    new_cols.append(f"_skip_col_{len(new_cols)}")
                    logger.debug(f"  -> Marked as skip: {col_tuple}")
                    continue
                
                # Standardize parts
                # --- Log before standardizing meas ---
                logger.debug(f"  -> Standardizing meas: '{meas}'")
                std_meas = self._standardize_columns(pd.DataFrame(columns=[meas])).columns[0]
                logger.debug(f"  -> Standardized meas: '{std_meas}'")
                # --- End Log ---
                
                # Clean up the unit and loc_suffix to avoid problematic characters
                # Replace problematic characters in unit - be even more aggressive with replacement
                unit_cleaned = unit.replace('%', 'pct').replace('/', 'per').replace(' ', '').strip()
                loc_suffix_cleaned = loc_suffix.replace('/', 'per').replace('%', 'pct').strip()
                logger.debug(f"  -> Cleaned parts: unit='{unit_cleaned}', loc_suffix='{loc_suffix_cleaned}'")

                # Ensure column name is valid for pandas access - no special chars at all
                safe_col_name_pre_regex = f"{std_meas}_loc{loc_suffix_cleaned}"
                logger.debug(f"  -> Generated name pre-regex: '{safe_col_name_pre_regex}'")
                # Remove any remaining special characters
                safe_col_name_post_regex = re.sub(r'[^\w_]', '', safe_col_name_pre_regex)
                logger.debug(f"  -> Generated name post-regex: '{safe_col_name_post_regex}'")
                
                new_cols.append(safe_col_name_post_regex)
                # Store mapping from original column tuple to new name
                col_name_map[col_tuple] = safe_col_name_post_regex
            
            # --- Log the final list of generated column names ---
            logger.debug(f"Final list of generated column names: {new_cols}")
            # --- End Log ---

            # Set the new column names
            df.columns = new_cols
            
            # Drop any columns we marked for skipping
            skip_cols = [col for col in df.columns if col.startswith('_skip_col_')]
            if skip_cols:
                logger.info(f"Dropping {len(skip_cols)} helper/unnamed columns: {skip_cols}")
                df = df.drop(columns=skip_cols)
                
            df = df.rename(columns={'timestamp_orig': ''}) # Use empty name for timestamp logic
        else:
            logger.error("Expected 3-level MultiIndex columns for Knudjepsen Extra CSV. Cannot process structure.")
            return pd.DataFrame()

        # 2. Handle timestamp
        df = self._handle_timestamp(df, format_spec)
        if 'timestamp' not in df.columns:
             logger.error("Timestamp handling failed.")
             return pd.DataFrame()

        # 3. Identify value columns (all except timestamp now)
        value_cols = [col for col in df.columns if col != 'timestamp']

        # 4. Convert numeric - with improved error handling
        decimal_sep = format_spec.get('loader_params', {}).get('decimal', '.')
        for col in value_cols:
            if col in df.columns:
                try:
                    # First check if column is actually a Series
                    selected_col = df[col]
                    if isinstance(selected_col, pd.DataFrame):
                        logger.warning(f"Column '{col}' is a DataFrame, not a Series. Skipping numeric conversion.")
                        # Keep the column but don't try to convert it
                        continue
                        
                    # Handle decimal separator if needed
                    if pd.api.types.is_object_dtype(selected_col.dtype):
                        if decimal == ',':
                            df[col] = df[col].astype(str).str.replace('.', '', regex=False)
                            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
                        else:
                            df[col] = df[col].astype(str).str.replace(',', '', regex=False)
                    
                    # Convert to numeric
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    logger.debug(f"Converted column '{col}' to numeric type {df[col].dtype}")
                except Exception as e:
                    logger.error(f"Error converting column '{col}' to numeric: {e}")
                    # Leave column as-is if conversion fails
        
        # 5. Handle missing values
        df = self._handle_missing_values(df)

        # 6. Add metadata
        df = self._add_metadata(df, format_spec)

        # 7. Reshape
        id_vars = ['timestamp', 'source_file', 'source_identifier']
        df_long = self._reshape_to_long(df, id_vars=id_vars)

        logger.info(f"Finished cleaning for Knudjepsen Extra CSV. Shape: {df_long.shape}")
        return df_long

class KnudjepsenBelysningCsvCleaner(BaseCleaner):
    def clean(self, df: pd.DataFrame, format_spec: Dict[str, Any]) -> pd.DataFrame:
        logger.info(f"Starting cleaning for Knudjepsen Belysning CSV: {format_spec.get('full_path')}")
        # Specific logic for NO3-NO4_belysningsgrp.csv
        # 1. Handle multi-index (measurement_base, group)
        if isinstance(df.columns, pd.MultiIndex) and len(df.columns.levels) >= 2:
            # --- Add Logging ---
            logger.debug(f"Inside cleaner: df.columns type: {type(df.columns)}")
            logger.debug(f"Inside cleaner: df.columns: {df.columns}")
            logger.debug(f"Inside cleaner: df.columns.values: {df.columns.values}")
            # --- End Logging ---
            new_cols = []
            level_map = format_spec.get('value_info', {}).get('level_map', {0: 'meas', 1: 'group'})
            group_prefix = format_spec.get('value_info', {}).get('group_prefix', 'LAMPGRP')
            
            for col_tuple in df.columns.values:
                # Fix: Better detection for timestamp column (empty values at both levels)
                if col_tuple[0] == '' and col_tuple[1] == '': # Timestamp
                    new_cols.append('timestamp_orig')
                    continue
                
                # FIXED: Use level positions directly, not the string values from level_map
                # Get level positions from map keys, defaulting to positions 0, 1
                meas_pos = int(list(level_map.keys())[0]) if level_map else 0
                group_pos = int(list(level_map.keys())[1]) if len(level_map) > 1 else 1
                
                # Use positions to access tuple elements
                meas_base = str(col_tuple[meas_pos]).strip()
                group = str(col_tuple[group_pos]).strip().replace(group_prefix, '').strip()
                
                # Fix: Improved detection for unnamed columns with more patterns
                # Check for common unnamed column patterns in pandas
                is_unnamed_col = False
                
                # Check if either part is empty
                if meas_base == '' or group == '':
                    is_unnamed_col = True
                # Check if either part contains 'unnamed' (case insensitive)
                elif 'unnamed' in meas_base.lower() or 'unnamed' in group.lower():
                    is_unnamed_col = True
                # Check for pandas auto-generated unnamed patterns like 'Unnamed: 0_level_1'
                elif any(pattern in meas_base.lower() for pattern in ['unnamed:', 'untitled']) or \
                     any(pattern in group.lower() for pattern in ['unnamed:', 'untitled']):
                    is_unnamed_col = True
                
                if is_unnamed_col:
                    # This is a helper/index column that should be skipped
                    new_cols.append(f"_skip_col_{len(new_cols)}")
                    logger.debug(f"Marking column as skip: {col_tuple}")
                    continue
                
                # Standardize and clean parts to avoid problematic characters
                meas_base_std = self._standardize_columns(pd.DataFrame(columns=[meas_base])).columns[0]
                # Clean up group to avoid problematic characters
                group_cleaned = group.replace('/', 'per').replace('%', 'pct').replace(' ', '_').strip()
                
                new_col_name = f"{meas_base_std}_grp{group_cleaned}" # e.g., malt_status_grp1
                new_cols.append(new_col_name)
            
            df.columns = new_cols
            # Fix: Rename the timestamp column but now drop any skipped columns
            skip_cols = [col for col in df.columns if col.startswith('_skip_col_')]
            df = df.rename(columns={'timestamp_orig': ''})
            if skip_cols:
                logger.info(f"Dropping {len(skip_cols)} helper/unnamed columns: {skip_cols}")
                df = df.drop(columns=skip_cols)
        else:
            logger.error("Expected 2-level MultiIndex columns for Knudjepsen Belysning CSV. Cannot process.")
            return pd.DataFrame()

        # 2. Handle timestamp
        df = self._handle_timestamp(df, format_spec)
        if 'timestamp' not in df.columns:
             logger.error("Timestamp handling failed.")
             return pd.DataFrame()

        # 3. Identify value columns (all except timestamp)
        value_cols = [col for col in df.columns if col != 'timestamp']

        # 4. Convert numeric (these are likely integer/boolean status flags)
        # Use convert_numeric, it should handle integers correctly
        decimal_sep = format_spec.get('loader_params', {}).get('decimal', '.')
        df = self._convert_numeric(df, value_cols, decimal=decimal_sep)
        # Optional: Convert to integer type if appropriate after conversion
        for col in value_cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col].dtype):
                 # Check if conversion to nullable Int64 is possible
                 try:
                     df[col] = df[col].astype('Int64')
                     logger.debug(f"Converted column {col} to Int64.")
                 except Exception as e:
                     logger.debug(f"Could not convert {col} to Int64: {e}")

        # 5. Handle missing values (maybe fill with 0 or a specific status code?)
        df = self._handle_missing_values(df) # Default handles numeric with median - adjust if needed

        # 6. Add metadata
        df = self._add_metadata(df, format_spec)

        # 7. Reshape
        id_vars = ['timestamp', 'source_file', 'source_identifier']
        df_long = self._reshape_to_long(df, id_vars=id_vars)

        logger.info(f"Finished cleaning for Knudjepsen Belysning CSV. Shape: {df_long.shape}")
        return df_long 

class KnudjepsenSourceUnitCleaner(BaseCleaner):
    """Cleaner for Knudjepsen files with Measurement, Source, Unit header levels."""
    def clean(self, df: pd.DataFrame, format_spec: Dict[str, Any]) -> pd.DataFrame:
        logger.info(f"Starting cleaning with KnudjepsenSourceUnitCleaner for: {format_spec.get('full_path')}")
        
        if not isinstance(df.columns, pd.MultiIndex) or len(df.columns.levels) < 3:
            logger.error("Expected 3-level MultiIndex columns (Measurement, Source, Unit). Cannot process.")
            return pd.DataFrame()

        # --- 1. Parse MultiIndex and Create Intermediate Wide Format --- 
        parsed_data = {}
        level_map = format_spec.get('value_info', {}).get('level_map', {0: 'meas', 1: 'source', 2: 'unit'})
        location_regex = format_spec.get('value_info', {}).get('source_location_regex')

        # Identify positions from level_map
        meas_pos = int(list(level_map.keys())[list(level_map.values()).index('measurement')]) if 'measurement' in level_map.values() else 0
        source_pos = int(list(level_map.keys())[list(level_map.values()).index('source')]) if 'source' in level_map.values() else 1
        unit_pos = int(list(level_map.keys())[list(level_map.values()).index('unit')]) if 'unit' in level_map.values() else 2
        logger.debug(f"Using level positions: meas={meas_pos}, source={source_pos}, unit={unit_pos}")

        temp_col_names = []
        original_ts_col_index = -1

        for i, col_tuple in enumerate(df.columns.values):
            logger.debug(f"Processing raw column tuple {i}: {col_tuple}")
            
            # Identify Timestamp Column by all empty levels
            if all(str(level).strip() == '' for level in col_tuple):
                temp_col_names.append('timestamp_orig')
                original_ts_col_index = i
                logger.debug("  -> Identified as timestamp_orig")
                continue

            # Extract parts based on identified positions
            meas = str(col_tuple[meas_pos]).strip()
            source = str(col_tuple[source_pos]).strip()
            unit_raw = str(col_tuple[unit_pos]).strip() # Get raw unit
            
            # --- Normalize Unit --- 
            unit_normalized = unicodedata.normalize('NFKD', unit_raw)
            unit_ascii = unit_normalized.encode('ascii', 'ignore').decode('ascii').strip()
            # Basic replacements for common symbols if needed after normalization
            unit_ascii = unit_ascii.replace('%', 'pct').replace('/', 'per') 
            # Log the transformation
            if unit_raw != unit_ascii:
                logger.debug(f"  -> Normalized unit: '{unit_raw}' -> '{unit_ascii}'")
            else:
                logger.debug(f"  -> Unit: '{unit_ascii}' (already ASCII or no change)")
            # --- End Normalize Unit ---
            
            logger.debug(f"  -> Extracted parts: meas='{meas}', source='{source}', unit='{unit_ascii}' (normalized)")

            # Basic check for unnamed/skip columns
            if not meas or meas.lower().startswith('unnamed'):
                temp_col_names.append(f"_skip_col_{i}")
                logger.debug("  -> Marked as skip (unnamed measurement)")
                continue

            # Standardize measurement name
            std_meas = self._standardize_columns(pd.DataFrame(columns=[meas])).columns[0]
            logger.debug(f"  -> Standardized meas: '{std_meas}'")

            # Extract location from source if regex provided
            location_suffix = ''
            if location_regex:
                match = re.search(location_regex, source)
                if match:
                    location_suffix = match.group(1).strip()
                    logger.debug(f"  -> Extracted location suffix: '{location_suffix}'")
            
            # Create a unique temporary column name for the wide format
            # Include index to prevent duplicates even if logic fails
            temp_col_name = f"{std_meas}_loc{location_suffix}_{i}"
            temp_col_names.append(temp_col_name)
            
            # Store parsed info for melt later
            parsed_data[temp_col_name] = {
                'measurement': std_meas,
                'unit': unit_ascii, # Store the normalized ASCII unit
                'location_detail': location_suffix
            }
            logger.debug(f"  -> Temp name: '{temp_col_name}', Parsed: {parsed_data[temp_col_name]}")

        # Rename columns in the DataFrame to temporary unique names
        df.columns = temp_col_names

        # Drop skipped columns
        skip_cols = [col for col in df.columns if col.startswith('_skip_col_')]
        if skip_cols:
            logger.info(f"Dropping {len(skip_cols)} helper/unnamed columns: {skip_cols}")
            df = df.drop(columns=skip_cols)
        
        # Rename original timestamp column to empty string for handle_timestamp
        if original_ts_col_index != -1 and 'timestamp_orig' in df.columns:
             df = df.rename(columns={'timestamp_orig': ''}) 

        # --- 2. Handle Timestamp --- 
        df = self._handle_timestamp(df, format_spec)
        if 'timestamp' not in df.columns:
             logger.error("Timestamp handling failed.")
             return pd.DataFrame()
        
        # Original timestamp column ('') should have been dropped by _handle_timestamp if successful
        if '' in df.columns:
             logger.warning("Original timestamp column ('') still exists after _handle_timestamp. Dropping.")
             df = df.drop(columns=[''])

        # --- 3. Convert Numeric --- 
        value_cols_temp = list(parsed_data.keys()) # Use the temp names we created
        decimal_sep = format_spec.get('loader_params', {}).get('decimal', ',') # Default to comma for these files
        df = self._convert_numeric(df, value_cols_temp, decimal=decimal_sep)
        
        # --- 4. Handle Missing Values --- 
        df = self._handle_missing_values(df)

        # --- 5. Add File-Level Metadata --- 
        df = self._add_metadata(df, format_spec) # Adds source_file, source_identifier

        # --- 6. Reshape to Long Format --- 
        id_vars = ['timestamp', 'source_file', 'source_identifier'] 
        value_vars = [col for col in df.columns if col not in id_vars]
        
        if not value_vars:
            logger.error("No value variables found after processing. Cannot reshape.")
            return pd.DataFrame()
        
        logger.info(f"Melting DataFrame. ID vars: {id_vars}, Value vars: {value_vars}")
        df_long = pd.melt(df, 
                          id_vars=id_vars, 
                          value_vars=value_vars, 
                          var_name='temp_col_name', 
                          value_name='value')
        
        # --- 7. Map Parsed Data (Measurement, Unit, Location) --- 
        # Create mapping functions based on the parsed_data dictionary
        def get_info(temp_name, key):
            return parsed_data.get(temp_name, {}).get(key, None)

        df_long['measurement'] = df_long['temp_col_name'].apply(get_info, key='measurement')
        df_long['unit'] = df_long['temp_col_name'].apply(get_info, key='unit')
        df_long['location'] = df_long['temp_col_name'].apply(get_info, key='location_detail') # Use detail as location
        
        # Combine base location from spec with extracted detail if needed, or just use detail
        base_location = format_spec.get('location', '')
        if base_location:
            # Example combination: NO3_3, NO4_4. Adjust logic as needed.
            df_long['location'] = base_location + "_" + df_long['location'].fillna('').astype(str)
            df_long['location'] = df_long['location'].str.strip('_')
        
        df_long = df_long.drop(columns=['temp_col_name']) # Drop the temporary column name
        
        # --- 8. Final Checks and Return --- 
        # Ensure required columns exist for the database schema
        required_db_cols = list(pipeline_config.TARGET_SCHEMA.keys())
        for col in required_db_cols:
            if col not in df_long.columns:
                logger.warning(f"Required DB column '{col}' missing after cleaning. Adding empty column.")
                df_long[col] = None # Add as None or appropriate default
                
        # Reorder columns to match schema potentially (optional, good practice)
        # Find common columns and order
        cols_in_order = [col for col in required_db_cols if col in df_long.columns]
        # Add any extra columns produced by cleaner (shouldn't be many)
        extra_cols = [col for col in df_long.columns if col not in cols_in_order]
        df_long = df_long[cols_in_order + extra_cols]

        logger.info(f"Finished cleaning with KnudjepsenSourceUnitCleaner. Shape: {df_long.shape}")
        return df_long 