import pandas as pd
# import polars as pl # Uncomment if you plan to use Polars
from typing import List, Dict, Any

class OutlierHandler:
    def __init__(self, rules: List[Dict[str, Any]]):
        self.rules = rules
        print(f"OutlierHandler initialized with {len(rules)} rules.")

    def clip_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Starting outlier clipping...")
        df_processed = df.copy()
        for rule in self.rules:
            col = rule['column']
            if col in df_processed.columns:
                min_val = rule.get('min_value')
                max_val = rule.get('max_value')
                if rule.get('clip', False):
                    print(f"Clipping outliers in column '{col}' to [{min_val}, {max_val}]")
                    df_processed[col] = df_processed[col].clip(lower=min_val, upper=max_val)
            else:
                print(f"Warning: Column '{col}' for outlier rule not found in DataFrame.")
        print("Outlier clipping completed.")
        return df_processed

class ImputationHandler:
    def __init__(self, rules: List[Dict[str, Any]]):
        self.rules = rules
        print(f"ImputationHandler initialized with {len(rules)} rules.")

    def impute_data(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Starting data imputation...")
        df_processed = df.copy()
        for rule in self.rules:
            col = rule['column']
            strategy = rule.get('strategy')
            if col in df_processed.columns:
                print(f"Applying imputation for column '{col}' using strategy '{strategy}'")
                if strategy == 'forward_fill':
                    limit = rule.get('limit')
                    df_processed[col] = df_processed[col].ffill(limit=limit)
                elif strategy == 'backward_fill' or strategy == 'bfill':
                    limit = rule.get('limit')
                    df_processed[col] = df_processed[col].bfill(limit=limit)
                elif strategy == 'linear':
                    limit_val = rule.get('limit')
                    if pd.api.types.is_numeric_dtype(df_processed[col]):
                        df_processed[col] = df_processed[col].interpolate(method='linear', limit_direction='both', limit=limit_val)
                    else:
                        print(f"    Warning: Column '{col}' is not numeric (dtype: {df_processed[col].dtype}). Skipping linear interpolation.")
                elif strategy == 'mean':
                    if pd.api.types.is_numeric_dtype(df_processed[col]):
                        df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
                    else:
                        print(f"    Warning: Column '{col}' is not numeric. Skipping mean imputation.")
                elif strategy == 'median':
                    if pd.api.types.is_numeric_dtype(df_processed[col]):
                        df_processed[col] = df_processed[col].fillna(df_processed[col].median())
                    else:
                        print(f"    Warning: Column '{col}' is not numeric. Skipping median imputation.")
                else:
                    print(f"Warning: Unknown imputation strategy '{strategy}' for column '{col}'. Skipping.")
            else:
                print(f"Warning: Column '{col}' for imputation rule not found in DataFrame.")
        print("Data imputation completed.")
        return df_processed

class DataSegmenter:
    def __init__(self, era_config: Dict[str, Any], common_config: Dict[str, Any] = None):
        self.era_config = era_config
        self.common_config = common_config if common_config is not None else {}
        
        # Try to get time_col from era_config first, then common_config, then default
        self.time_col = self.era_config.get(
            'time_col',
            self.common_config.get('time_col', 'timestamp') 
        )
        
        # Get segmentation specific config from era_config, default to 24 hours gap
        segmentation_settings = self.era_config.get('segmentation', {})
        self.min_gap_for_new_segment = pd.Timedelta(
            segmentation_settings.get('min_gap_hours', 24), 
            unit='h'
        )
        print(f"DataSegmenter initialized. Time column: '{self.time_col}'. Min gap for new segment: {self.min_gap_for_new_segment}")

    def segment_by_availability(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        print("Starting data segmentation by availability...")
        
        # Check if time_col is a regular column or the index
        if self.time_col in df.columns:
            df_sorted = df.sort_values(by=self.time_col).copy()
            time_data = pd.to_datetime(df_sorted[self.time_col])
        elif df.index.name == self.time_col:
            # If time_col is the index, ensure it's a DatetimeIndex and use it
            if not isinstance(df.index, pd.DatetimeIndex):
                print(f"Warning: Index '{self.time_col}' is not a DatetimeIndex. Attempting conversion.")
                try:
                    df.index = pd.to_datetime(df.index, utc=True) # Assuming UTC from previous steps
                except Exception as e:
                    print(f"Error converting index '{self.time_col}' to DatetimeIndex: {e}. Returning single segment.")
                    return [df]
            df_sorted = df.sort_index().copy() # Sort by index if it's time
            time_data = df_sorted.index
        else:
            print(f"Warning: Time column or index '{self.time_col}' not found. Returning single segment.")
            return [df]
        
        if df_sorted.empty:
            print("Input DataFrame is empty. Returning empty list of segments.")
            return []

        segments = []
        current_segment_start_idx = 0
        for i in range(1, len(df_sorted)):
            # Access elements directly if time_data is an Index, or via .iloc if it's a Series
            if isinstance(time_data, pd.Index): # Covers DatetimeIndex
                current_time = time_data[i]
                previous_time = time_data[i-1]
            else: # It's a Series
                current_time = time_data.iloc[i]
                previous_time = time_data.iloc[i-1]
            
            time_diff = current_time - previous_time
            if time_diff > self.min_gap_for_new_segment:
                segments.append(df_sorted.iloc[current_segment_start_idx:i])
                current_segment_start_idx = i
        
        # Add the last segment
        segments.append(df_sorted.iloc[current_segment_start_idx:])
        
        print(f"Data segmentation completed. Found {len(segments)} segments.")
        for i, seg_df in enumerate(segments):
            if not seg_df.empty:
                # Access time data correctly whether it's a column or index
                # For printing start/end, ensure we are using the time_data series/index directly
                # which has already been assured to be datetime by this point.
                idx_for_print = seg_df.index if isinstance(seg_df.index, pd.DatetimeIndex) and seg_df.index.name == self.time_col else pd.to_datetime(seg_df[self.time_col])
                start_time = idx_for_print[0]
                end_time = idx_for_print[-1]
                print(f"  Segment {i+1}: {len(seg_df)} rows, from {start_time} to {end_time}")
            else:
                print(f"  Segment {i+1}: Empty")
        return [s for s in segments if not s.empty] # Filter out empty segments

# Example Usage (for testing this module directly)
if __name__ == '__main__':
    # Sample DataFrame
    data = {
        'timestamp': pd.to_datetime([
            '2023-01-01 00:00:00', '2023-01-01 01:00:00', '2023-01-01 02:00:00', # Segment 1
            '2023-01-03 03:00:00', '2023-01-03 04:00:00',                 # Segment 2 (after >24h gap)
            '2023-01-03 04:05:00'                                      # Still Segment 2
        ]),
        'temperature': [10, 11, None, 100, 15, 16],
        'humidity': [50, 52, 51, 50, -10, 53]
    }
    sample_df = pd.DataFrame(data)

    # Sample Config
    sample_config = {
        'time_col': 'timestamp',
        'preprocessing': {
            'outlier_rules': [
                {'column': 'temperature', 'min_value': 0, 'max_value': 50, 'clip': True},
                {'column': 'humidity', 'min_value': 0, 'max_value': 100, 'clip': True}
            ],
            'imputation_rules': [
                {'column': 'temperature', 'strategy': 'linear'},
                {'column': 'humidity', 'strategy': 'forward_fill', 'limit': 1}
            ]
        },
        'segmentation': {
            'min_gap_hours': 24
        }
    }

    print("--- Testing OutlierHandler ---")
    outlier_handler = OutlierHandler(sample_config['preprocessing']['outlier_rules'])
    df_after_outliers = outlier_handler.clip_outliers(sample_df.copy())
    print("DataFrame after outlier handling:")
    print(df_after_outliers)

    print("\n--- Testing ImputationHandler ---")
    imputation_handler = ImputationHandler(sample_config['preprocessing']['imputation_rules'])
    df_after_imputation = imputation_handler.impute_data(df_after_outliers.copy())
    print("DataFrame after imputation:")
    print(df_after_imputation)

    print("\n--- Testing DataSegmenter ---")
    # Pass both era_config (as sample_config) and a common_config example
    segmenter = DataSegmenter(era_config=sample_config, common_config={'time_col': 'timestamp'})
    segments = segmenter.segment_by_availability(df_after_imputation.copy())
    for i, seg in enumerate(segments):
        print(f"\nSegment {i+1}:")
        print(seg) 