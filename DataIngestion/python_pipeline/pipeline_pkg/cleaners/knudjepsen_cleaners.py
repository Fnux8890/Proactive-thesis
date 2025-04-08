import pandas as pd
import logging
from .base_cleaner import BaseCleaner
from typing import Dict, Any

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

        # 2. Handle timestamp
        df = self._handle_timestamp(df, format_spec)
        if 'timestamp' not in df.columns:
             logger.error("Timestamp handling failed. Cannot proceed without timestamp.")
             return pd.DataFrame() # Return empty if timestamp fails

        # 3. Identify value columns (needs refinement based on actual column names after step 1)
        # Assuming standardization makes columns predictable based on value_columns spec
        value_cols_spec = format_spec.get('value_columns', [])
        # This mapping needs to be robust against the standardized names
        # Placeholder: Assume standardized names are derived directly
        value_cols = [self._standardize_columns(pd.DataFrame(columns=[vc[0]]))[vc[0]] for vc in value_cols_spec if isinstance(vc, tuple)]
        # Filter out any cols that might not exist after standardization/loading issues
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
            # Custom column naming: measurement_unit_location
            new_cols = []
            level_map = format_spec.get('value_info', {}).get('level_map', {0: 'meas', 1: 'unit', 2: 'loc'})
            loc_prefix = format_spec.get('value_info', {}).get('location_prefix', 'Afd')
            
            for col_tuple in df.columns.values:
                # Handle timestamp column specially (first column, likely unnamed at level 0)
                if col_tuple[0] == '' and col_tuple[1] == '' and col_tuple[2] == '':
                     new_cols.append('timestamp_orig') # Rename placeholder
                     continue
                
                meas = str(col_tuple[level_map.get(0, 0)]).strip()
                unit = str(col_tuple[level_map.get(1, 1)]).strip()
                loc_suffix = str(col_tuple[level_map.get(2, 2)]).strip().replace(loc_prefix, '').strip()
                
                # Standardize parts
                meas = self._standardize_columns(pd.DataFrame(columns=[meas])).columns[0]
                unit_cleaned = unit.replace('%', 'perc').replace('/', '_').replace(' ', '') # Basic unit clean
                
                new_col_name = f"{meas}_loc{loc_suffix}" # e.g., mal_fd_loc3
                new_cols.append(new_col_name)
                # Store unit info? Could add to spec or use later
                # format_spec.setdefault('_units', {})[new_col_name] = unit
            
            df.columns = new_cols
            df = df.rename(columns={'timestamp_orig': ''}) # Use empty name for timestamp logic
            # No need for _standardize_columns here as we did it manually

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

        logger.info(f"Finished cleaning for Knudjepsen Extra CSV. Shape: {df_long.shape}")
        return df_long

class KnudjepsenBelysningCsvCleaner(BaseCleaner):
    def clean(self, df: pd.DataFrame, format_spec: Dict[str, Any]) -> pd.DataFrame:
        logger.info(f"Starting cleaning for Knudjepsen Belysning CSV: {format_spec.get('full_path')}")
        # Specific logic for NO3-NO4_belysningsgrp.csv
        # 1. Handle multi-index (measurement_base, group)
        if isinstance(df.columns, pd.MultiIndex) and len(df.columns.levels) >= 2:
            new_cols = []
            level_map = format_spec.get('value_info', {}).get('level_map', {0: 'meas', 1: 'group'})
            group_prefix = format_spec.get('value_info', {}).get('group_prefix', 'LAMPGRP')
            
            for col_tuple in df.columns.values:
                if col_tuple[0] == '' and col_tuple[1] == '': # Timestamp
                    new_cols.append('timestamp_orig')
                    continue
                
                meas_base = str(col_tuple[level_map.get(0, 0)]).strip()
                group = str(col_tuple[level_map.get(1, 1)]).strip().replace(group_prefix, '').strip()
                
                meas_base_std = self._standardize_columns(pd.DataFrame(columns=[meas_base])).columns[0]
                new_col_name = f"{meas_base_std}_grp{group}" # e.g., malt_status_grp1
                new_cols.append(new_col_name)
            
            df.columns = new_cols
            df = df.rename(columns={'timestamp_orig': ''})
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