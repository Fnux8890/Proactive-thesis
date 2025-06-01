#!/usr/bin/env python3
"""
Minimal GPU Feature Extraction Service

This script provides a simple interface for Rust to call Python GPU feature extraction.
It reads data from stdin (JSON), processes it on GPU, and returns results to stdout.
"""

import json
import sys
import logging
from typing import Dict, List, Any
import traceback

# Conditionally import GPU libraries
try:
    import cudf
    GPU_AVAILABLE = True
except ImportError:
    import pandas as pd
    GPU_AVAILABLE = False

# Configure logging to stderr so it doesn't interfere with stdout communication
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)

logger = logging.getLogger(__name__)


class MinimalGPUFeatureExtractor:
    """Minimal feature extractor with GPU acceleration."""
    
    def __init__(self, use_gpu: bool = None):
        """Initialize the extractor."""
        if use_gpu is None:
            use_gpu = GPU_AVAILABLE
        
        self.use_gpu = use_gpu and GPU_AVAILABLE
        logger.info(f"Initialized feature extractor (GPU: {self.use_gpu})")
    
    def extract_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features from the input data.
        
        Args:
            data: Dictionary containing:
                - timestamps: List of ISO format timestamps
                - sensors: Dict of sensor_name -> List of values
                - window_sizes: List of window sizes in minutes (optional)
                
        Returns:
            Dictionary containing extracted features
        """
        try:
            # Parse input data
            timestamps = data.get('timestamps', [])
            sensors = data.get('sensors', {})
            window_sizes = data.get('window_sizes', [30, 120])  # Default: 30min, 2h
            
            if not timestamps or not sensors:
                raise ValueError("Missing required data: timestamps and sensors")
            
            # Create DataFrame
            if self.use_gpu:
                df_data = {'timestamp': timestamps}
                df_data.update(sensors)
                df = cudf.DataFrame(df_data)
                df['timestamp'] = cudf.to_datetime(df['timestamp'])
                df = df.set_index('timestamp').sort_index()
            else:
                df_data = {'timestamp': pd.to_datetime(timestamps)}
                df_data.update(sensors)
                df = pd.DataFrame(df_data)
                df = df.set_index('timestamp').sort_index()
            
            # Extract features
            features = {}
            
            for sensor_name in sensors.keys():
                if sensor_name in df.columns:
                    sensor_features = self._extract_sensor_features(
                        df[sensor_name], 
                        sensor_name, 
                        window_sizes
                    )
                    features.update(sensor_features)
            
            # Add metadata
            result = {
                'status': 'success',
                'features': features,
                'metadata': {
                    'num_samples': len(df),
                    'num_features': len(features),
                    'gpu_used': self.use_gpu
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _extract_sensor_features(self, series, sensor_name: str, window_sizes: List[int]) -> Dict[str, float]:
        """Extract features for a single sensor."""
        features = {}
        
        # Basic statistics
        if self.use_gpu:
            features[f'{sensor_name}_mean'] = float(series.mean())
            features[f'{sensor_name}_std'] = float(series.std())
            features[f'{sensor_name}_min'] = float(series.min())
            features[f'{sensor_name}_max'] = float(series.max())
            
            # Percentiles
            for p in [25, 50, 75]:
                features[f'{sensor_name}_p{p}'] = float(series.quantile(p/100))
        else:
            features[f'{sensor_name}_mean'] = float(series.mean())
            features[f'{sensor_name}_std'] = float(series.std())
            features[f'{sensor_name}_min'] = float(series.min())
            features[f'{sensor_name}_max'] = float(series.max())
            
            # Percentiles
            for p in [25, 50, 75]:
                features[f'{sensor_name}_p{p}'] = float(series.quantile(p/100))
        
        # Rolling window features
        for window_minutes in window_sizes:
            # Estimate window size in samples (assuming regular sampling)
            if len(series) > 1:
                time_diff = series.index[1] - series.index[0]
                samples_per_minute = int(60 / time_diff.total_seconds())
                window = max(1, window_minutes * samples_per_minute)
            else:
                window = 1
            
            if window <= len(series):
                rolled = series.rolling(window=window, min_periods=1)
                
                features[f'{sensor_name}_rolling_mean_{window_minutes}m'] = float(rolled.mean().iloc[-1])
                features[f'{sensor_name}_rolling_std_{window_minutes}m'] = float(rolled.std().iloc[-1])
                features[f'{sensor_name}_rolling_min_{window_minutes}m'] = float(rolled.min().iloc[-1])
                features[f'{sensor_name}_rolling_max_{window_minutes}m'] = float(rolled.max().iloc[-1])
        
        # Change features
        if len(series) > 1:
            if self.use_gpu:
                diff = series.diff()
                features[f'{sensor_name}_mean_change'] = float(diff.mean())
                features[f'{sensor_name}_std_change'] = float(diff.std())
            else:
                diff = series.diff()
                features[f'{sensor_name}_mean_change'] = float(diff.mean())
                features[f'{sensor_name}_std_change'] = float(diff.std())
        
        return features


def main():
    """Main entry point for the feature extraction service."""
    try:
        # Read JSON input from stdin
        input_data = json.load(sys.stdin)
        
        # Create extractor
        use_gpu = input_data.get('use_gpu', True)
        extractor = MinimalGPUFeatureExtractor(use_gpu=use_gpu)
        
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