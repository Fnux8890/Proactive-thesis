#!/usr/bin/env python3
"""
GPU-Accelerated Sparse Feature Extraction

This module provides GPU-accelerated feature extraction specifically designed
for sparse time series data (91.3% missing values). It complements the Rust
sparse_features.rs module by handling computationally intensive operations.
"""

import json
import sys
import logging
from typing import Dict, List, Any, Optional, Tuple
import traceback
import numpy as np

# GPU imports with fallback
try:
    import cudf
    import cupy as cp
    from cupy import ndarray as cp_array
    GPU_AVAILABLE = True
except ImportError:
    import pandas as pd
    import numpy as np
    GPU_AVAILABLE = False
    cp_array = np.ndarray

# Configure logging to stderr
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


class SparseGPUFeatureExtractor:
    """GPU-accelerated feature extraction for sparse greenhouse data."""
    
    def __init__(self, use_gpu: bool = None):
        """Initialize the sparse feature extractor."""
        if use_gpu is None:
            use_gpu = GPU_AVAILABLE
        
        self.use_gpu = use_gpu and GPU_AVAILABLE
        logger.info(f"Initialized sparse feature extractor (GPU: {self.use_gpu})")
    
    def extract_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract sparse-aware features from greenhouse sensor data.
        
        Args:
            data: Dictionary containing:
                - timestamps: List of ISO format timestamps
                - sensors: Dict of sensor_name -> List of Optional[float] values
                - energy_prices: Optional list of (timestamp, price) tuples
                - window_configs: Optional window configurations
                
        Returns:
            Dictionary containing extracted sparse features
        """
        try:
            # Parse input data
            timestamps = data.get('timestamps', [])
            sensors = data.get('sensors', {})
            energy_prices = data.get('energy_prices', [])
            window_configs = data.get('window_configs', {
                'gap_analysis': [60, 180, 360],  # minutes
                'event_detection': [30, 120],     # minutes
                'pattern_windows': [1440, 10080]  # day, week in minutes
            })
            
            if not timestamps or not sensors:
                raise ValueError("Missing required data: timestamps and sensors")
            
            # Convert to appropriate DataFrame type
            df = self._create_dataframe(timestamps, sensors)
            
            # Extract features for each sensor
            all_features = {}
            
            # 1. Coverage and gap analysis (GPU-accelerated)
            coverage_features = self._extract_coverage_features(df, sensors)
            all_features.update(coverage_features)
            
            # 2. Event-based features (GPU-accelerated)
            event_features = self._extract_event_features(df, sensors, window_configs['event_detection'])
            all_features.update(event_features)
            
            # 3. Pattern analysis (GPU-accelerated)
            pattern_features = self._extract_pattern_features(df, sensors, window_configs['pattern_windows'])
            all_features.update(pattern_features)
            
            # 4. Greenhouse-specific features
            greenhouse_features = self._extract_greenhouse_features(df, sensors, energy_prices)
            all_features.update(greenhouse_features)
            
            # 5. Multi-sensor correlations (sparse-aware)
            correlation_features = self._extract_correlation_features(df, sensors)
            all_features.update(correlation_features)
            
            return {
                'status': 'success',
                'features': all_features,
                'metadata': {
                    'num_samples': len(df),
                    'num_features': len(all_features),
                    'gpu_used': self.use_gpu,
                    'average_coverage': np.mean([v for k, v in all_features.items() if k.endswith('_coverage')])
                }
            }
            
        except Exception as e:
            logger.error(f"Sparse feature extraction failed: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _create_dataframe(self, timestamps: List[str], sensors: Dict[str, List[Optional[float]]]):
        """Create GPU or CPU dataframe from sparse data."""
        if self.use_gpu:
            # Convert timestamps
            df_data = {'timestamp': pd.to_datetime(timestamps)}
            
            # Handle sparse data - replace None with NaN
            for sensor_name, values in sensors.items():
                df_data[sensor_name] = [float(v) if v is not None else np.nan for v in values]
            
            df = cudf.DataFrame(df_data)
            df = df.set_index('timestamp').sort_index()
        else:
            df_data = {'timestamp': pd.to_datetime(timestamps)}
            
            for sensor_name, values in sensors.items():
                df_data[sensor_name] = [float(v) if v is not None else np.nan for v in values]
            
            df = pd.DataFrame(df_data)
            df = df.set_index('timestamp').sort_index()
        
        return df
    
    def _extract_coverage_features(self, df, sensors: Dict[str, List]) -> Dict[str, float]:
        """Extract coverage and gap analysis features using GPU."""
        features = {}
        
        for sensor_name in sensors.keys():
            if sensor_name not in df.columns:
                continue
            
            series = df[sensor_name]
            
            if self.use_gpu:
                # GPU-accelerated coverage calculation
                not_null = ~series.isna()
                coverage = float(not_null.sum() / len(series))
                
                # Gap analysis on GPU
                # Create a binary mask and find transitions
                mask = not_null.astype(cp.int32)
                diff = cp.diff(mask.values)
                
                # Gap starts (1 -> 0 transitions)
                gap_starts = cp.where(diff == -1)[0] + 1
                # Gap ends (0 -> 1 transitions)
                gap_ends = cp.where(diff == 1)[0] + 1
                
                # Handle edge cases
                if len(mask) > 0:
                    if mask.iloc[0] == 0:
                        gap_starts = cp.concatenate([cp.array([0]), gap_starts])
                    if mask.iloc[-1] == 0:
                        gap_ends = cp.concatenate([gap_ends, cp.array([len(mask)])])
                
                # Calculate gap lengths in samples
                if len(gap_starts) > 0 and len(gap_ends) > 0:
                    gap_lengths = gap_ends - gap_starts
                    
                    # Convert to time if we have regular sampling
                    if len(df) > 1:
                        time_diff = (df.index[1] - df.index[0]).total_seconds() / 60  # minutes
                        gap_lengths_minutes = gap_lengths * time_diff
                        
                        features[f'{sensor_name}_longest_gap_minutes'] = float(cp.max(gap_lengths_minutes))
                        features[f'{sensor_name}_mean_gap_minutes'] = float(cp.mean(gap_lengths_minutes))
                        features[f'{sensor_name}_num_gaps'] = len(gap_starts)
                    else:
                        features[f'{sensor_name}_longest_gap_samples'] = float(cp.max(gap_lengths))
                        features[f'{sensor_name}_mean_gap_samples'] = float(cp.mean(gap_lengths))
                        features[f'{sensor_name}_num_gaps'] = len(gap_starts)
                else:
                    features[f'{sensor_name}_longest_gap_minutes'] = 0.0
                    features[f'{sensor_name}_mean_gap_minutes'] = 0.0
                    features[f'{sensor_name}_num_gaps'] = 0
                
            else:
                # CPU fallback
                not_null = ~series.isna()
                coverage = float(not_null.sum() / len(series))
                
                # Gap analysis
                mask = not_null.astype(int).values
                diff = np.diff(mask)
                
                gap_starts = np.where(diff == -1)[0] + 1
                gap_ends = np.where(diff == 1)[0] + 1
                
                if len(mask) > 0:
                    if mask[0] == 0:
                        gap_starts = np.concatenate([[0], gap_starts])
                    if mask[-1] == 0:
                        gap_ends = np.concatenate([gap_ends, [len(mask)]])
                
                if len(gap_starts) > 0 and len(gap_ends) > 0:
                    gap_lengths = gap_ends - gap_starts
                    
                    if len(df) > 1:
                        time_diff = (df.index[1] - df.index[0]).total_seconds() / 60
                        gap_lengths_minutes = gap_lengths * time_diff
                        
                        features[f'{sensor_name}_longest_gap_minutes'] = float(np.max(gap_lengths_minutes))
                        features[f'{sensor_name}_mean_gap_minutes'] = float(np.mean(gap_lengths_minutes))
                        features[f'{sensor_name}_num_gaps'] = len(gap_starts)
                    else:
                        features[f'{sensor_name}_longest_gap_samples'] = float(np.max(gap_lengths))
                        features[f'{sensor_name}_mean_gap_samples'] = float(np.mean(gap_lengths))
                        features[f'{sensor_name}_num_gaps'] = len(gap_starts)
                else:
                    features[f'{sensor_name}_longest_gap_minutes'] = 0.0
                    features[f'{sensor_name}_mean_gap_minutes'] = 0.0
                    features[f'{sensor_name}_num_gaps'] = 0
            
            features[f'{sensor_name}_coverage'] = coverage
        
        return features
    
    def _extract_event_features(self, df, sensors: Dict[str, List], windows: List[int]) -> Dict[str, float]:
        """Extract event-based features optimized for sparse data."""
        features = {}
        
        for sensor_name in sensors.keys():
            if sensor_name not in df.columns:
                continue
            
            series = df[sensor_name]
            
            # Only process if we have data
            valid_data = series.dropna()
            if len(valid_data) < 2:
                continue
            
            if self.use_gpu:
                # Calculate statistics on available data
                mean_val = float(valid_data.mean())
                std_val = float(valid_data.std())
                
                # Detect events: changes, extremes, crossings
                diff = valid_data.diff().dropna()
                
                # Large changes (> 2 std)
                if std_val > 0:
                    large_changes = (cp.abs(diff.values) > 2 * std_val).sum()
                    features[f'{sensor_name}_large_changes'] = int(large_changes)
                
                # Zero crossings (around mean)
                centered = valid_data - mean_val
                sign_changes = (cp.diff(cp.sign(centered.values)) != 0).sum()
                features[f'{sensor_name}_mean_crossings'] = int(sign_changes)
                
                # Extreme values (using percentiles)
                p95 = float(valid_data.quantile(0.95))
                p5 = float(valid_data.quantile(0.05))
                extreme_high = (valid_data > p95).sum()
                extreme_low = (valid_data < p5).sum()
                features[f'{sensor_name}_extreme_high_count'] = int(extreme_high)
                features[f'{sensor_name}_extreme_low_count'] = int(extreme_low)
                
            else:
                # CPU implementation
                mean_val = float(valid_data.mean())
                std_val = float(valid_data.std())
                
                diff = valid_data.diff().dropna()
                
                if std_val > 0:
                    large_changes = (np.abs(diff.values) > 2 * std_val).sum()
                    features[f'{sensor_name}_large_changes'] = int(large_changes)
                
                centered = valid_data - mean_val
                sign_changes = (np.diff(np.sign(centered.values)) != 0).sum()
                features[f'{sensor_name}_mean_crossings'] = int(sign_changes)
                
                p95 = float(valid_data.quantile(0.95))
                p5 = float(valid_data.quantile(0.05))
                extreme_high = (valid_data > p95).sum()
                extreme_low = (valid_data < p5).sum()
                features[f'{sensor_name}_extreme_high_count'] = int(extreme_high)
                features[f'{sensor_name}_extreme_low_count'] = int(extreme_low)
        
        return features
    
    def _extract_pattern_features(self, df, sensors: Dict[str, List], windows: List[int]) -> Dict[str, float]:
        """Extract temporal pattern features for sparse data."""
        features = {}
        
        for sensor_name in sensors.keys():
            if sensor_name not in df.columns:
                continue
            
            series = df[sensor_name]
            
            # Hour of day patterns
            if self.use_gpu:
                not_null = ~series.isna()
                if not_null.sum() > 0:
                    # Extract hour from index
                    hours = df.index.hour
                    
                    # Coverage by hour
                    hourly_coverage = {}
                    for h in range(24):
                        mask = hours == h
                        if mask.sum() > 0:
                            hourly_coverage[h] = float((not_null & mask).sum() / mask.sum())
                    
                    # Peak coverage hours
                    if hourly_coverage:
                        peak_hour = max(hourly_coverage, key=hourly_coverage.get)
                        features[f'{sensor_name}_peak_coverage_hour'] = peak_hour
                        features[f'{sensor_name}_peak_coverage_ratio'] = hourly_coverage[peak_hour]
                    
                    # Day vs night coverage (6am-6pm as day)
                    day_mask = (hours >= 6) & (hours < 18)
                    night_mask = ~day_mask
                    
                    day_coverage = float((not_null & day_mask).sum() / day_mask.sum()) if day_mask.sum() > 0 else 0
                    night_coverage = float((not_null & night_mask).sum() / night_mask.sum()) if night_mask.sum() > 0 else 0
                    
                    features[f'{sensor_name}_day_coverage'] = day_coverage
                    features[f'{sensor_name}_night_coverage'] = night_coverage
                    features[f'{sensor_name}_day_night_ratio'] = day_coverage / night_coverage if night_coverage > 0 else 0
                    
            else:
                # CPU implementation
                not_null = ~series.isna()
                if not_null.sum() > 0:
                    hours = df.index.hour
                    
                    hourly_coverage = {}
                    for h in range(24):
                        mask = hours == h
                        if mask.sum() > 0:
                            hourly_coverage[h] = float((not_null & mask).sum() / mask.sum())
                    
                    if hourly_coverage:
                        peak_hour = max(hourly_coverage, key=hourly_coverage.get)
                        features[f'{sensor_name}_peak_coverage_hour'] = peak_hour
                        features[f'{sensor_name}_peak_coverage_ratio'] = hourly_coverage[peak_hour]
                    
                    day_mask = (hours >= 6) & (hours < 18)
                    night_mask = ~day_mask
                    
                    day_coverage = float((not_null & day_mask).sum() / day_mask.sum()) if day_mask.sum() > 0 else 0
                    night_coverage = float((not_null & night_mask).sum() / night_mask.sum()) if night_mask.sum() > 0 else 0
                    
                    features[f'{sensor_name}_day_coverage'] = day_coverage
                    features[f'{sensor_name}_night_coverage'] = night_coverage
                    features[f'{sensor_name}_day_night_ratio'] = day_coverage / night_coverage if night_coverage > 0 else 0
        
        return features
    
    def _extract_greenhouse_features(self, df, sensors: Dict[str, List], energy_prices: List[Tuple]) -> Dict[str, float]:
        """Extract greenhouse-specific features for sparse data."""
        features = {}
        
        # Lamp usage patterns (binary sensors)
        lamp_sensors = [s for s in sensors.keys() if 'lamp' in s.lower() and 'status' in s.lower()]
        if lamp_sensors:
            total_on_hours = 0.0
            total_switches = 0
            
            for lamp in lamp_sensors:
                if lamp in df.columns:
                    series = df[lamp].dropna()
                    if len(series) > 1:
                        # Count switches
                        binary = (series > 0.5).astype(int)
                        switches = (binary.diff().abs() > 0).sum()
                        total_switches += int(switches)
                        
                        # Estimate on hours
                        on_ratio = float((binary > 0).sum() / len(binary))
                        if len(df) > 1:
                            total_hours = (df.index[-1] - df.index[0]).total_seconds() / 3600
                            total_on_hours += on_ratio * total_hours
            
            features['total_lamp_on_hours'] = total_on_hours
            features['total_lamp_switches'] = total_switches
            
            # Lamp efficiency (if DLI data available)
            if 'dli_sum' in df.columns and total_on_hours > 0:
                dli_series = df['dli_sum'].dropna()
                if len(dli_series) > 0:
                    total_dli = float(dli_series.sum())
                    features['lamp_efficiency_dli_per_hour'] = total_dli / total_on_hours
        
        # Temperature control features
        if 'air_temp_c' in df.columns and 'heating_setpoint_c' in df.columns:
            temp = df['air_temp_c'].dropna()
            setpoint = df['heating_setpoint_c'].dropna()
            
            # Find overlapping valid data
            valid_idx = temp.index.intersection(setpoint.index)
            if len(valid_idx) > 0:
                temp_valid = temp[valid_idx]
                setpoint_valid = setpoint[valid_idx]
                
                # Heating needed (temp < setpoint)
                heating_needed = (temp_valid < setpoint_valid).sum()
                features['heating_needed_ratio'] = float(heating_needed / len(valid_idx))
                
                # Temperature deviation
                deviation = (temp_valid - setpoint_valid).abs()
                features['mean_temp_deviation'] = float(deviation.mean())
                features['max_temp_deviation'] = float(deviation.max())
        
        # VPD stress analysis
        if 'vpd_hpa' in df.columns:
            vpd = df['vpd_hpa'].dropna()
            if len(vpd) > 0:
                # Optimal VPD range for most plants: 0.4-1.6 kPa
                stress_low = (vpd < 0.4).sum()
                stress_high = (vpd > 1.6).sum()
                features['vpd_stress_low_ratio'] = float(stress_low / len(vpd))
                features['vpd_stress_high_ratio'] = float(stress_high / len(vpd))
        
        # Energy cost features (if prices provided)
        if energy_prices and lamp_sensors:
            # Simple peak hour analysis
            features['energy_price_correlation'] = 0.0  # Placeholder for more complex analysis
        
        return features
    
    def _extract_correlation_features(self, df, sensors: Dict[str, List]) -> Dict[str, float]:
        """Extract multi-sensor correlation features accounting for sparsity."""
        features = {}
        
        # Key sensor pairs for greenhouse control
        sensor_pairs = [
            ('air_temp_c', 'relative_humidity_percent'),
            ('light_intensity_umol', 'air_temp_c'),
            ('co2_measured_ppm', 'vpd_hpa'),
            ('heating_setpoint_c', 'air_temp_c')
        ]
        
        for sensor1, sensor2 in sensor_pairs:
            if sensor1 in df.columns and sensor2 in df.columns:
                # Find overlapping non-null values
                s1 = df[sensor1]
                s2 = df[sensor2]
                
                # Create mask for valid pairs
                valid_mask = ~(s1.isna() | s2.isna())
                valid_count = valid_mask.sum()
                
                if valid_count > 10:  # Need minimum samples for correlation
                    s1_valid = s1[valid_mask]
                    s2_valid = s2[valid_mask]
                    
                    if self.use_gpu:
                        # GPU correlation calculation
                        corr = float(s1_valid.corr(s2_valid))
                        features[f'{sensor1}_{sensor2}_correlation'] = corr if not cp.isnan(corr) else 0.0
                    else:
                        # CPU correlation
                        corr = float(s1_valid.corr(s2_valid))
                        features[f'{sensor1}_{sensor2}_correlation'] = corr if not np.isnan(corr) else 0.0
                    
                    features[f'{sensor1}_{sensor2}_overlap_ratio'] = float(valid_count / len(df))
        
        return features


def main():
    """Main entry point for the sparse GPU feature extraction service."""
    try:
        # Read JSON input from stdin
        input_data = json.load(sys.stdin)
        
        # Create extractor
        use_gpu = input_data.get('use_gpu', True)
        extractor = SparseGPUFeatureExtractor(use_gpu=use_gpu)
        
        # Extract features
        result = extractor.extract_features(input_data)
        
        # Write JSON output to stdout
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.flush()
        
    except Exception as e:
        # Return error as JSON
        error_result = {
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        json.dump(error_result, sys.stdout, indent=2)
        sys.stdout.flush()
        sys.exit(1)


if __name__ == '__main__':
    main()