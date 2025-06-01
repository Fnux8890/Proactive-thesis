#!/usr/bin/env python3
"""
Minimal Feature Extraction Test (No External Dependencies)
Tests JSON stdin/stdout communication pattern for the Python bridge.
"""

import json
import sys
import logging
from typing import Dict, List, Any
import traceback

# Configure logging to stderr
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

def extract_basic_features(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract basic features without external dependencies.
    """
    try:
        # Parse input data
        timestamps = data.get('timestamps', [])
        sensors = data.get('sensors', {})
        
        if not timestamps or not sensors:
            raise ValueError("Missing required data: timestamps and sensors")
        
        # Extract basic features using only built-in Python
        features = {}
        
        for sensor_name, values in sensors.items():
            # Convert None values to skip them
            numeric_values = [v for v in values if v is not None]
            
            if numeric_values:
                # Basic statistics using built-in functions
                features[f'{sensor_name}_count'] = len(numeric_values)
                features[f'{sensor_name}_min'] = min(numeric_values)
                features[f'{sensor_name}_max'] = max(numeric_values)
                features[f'{sensor_name}_mean'] = sum(numeric_values) / len(numeric_values)
                
                # Coverage (non-null ratio)
                features[f'{sensor_name}_coverage'] = len(numeric_values) / len(values)
                
                # Range
                features[f'{sensor_name}_range'] = max(numeric_values) - min(numeric_values)
                
                # Simple variance calculation
                mean_val = features[f'{sensor_name}_mean']
                variance = sum((x - mean_val) ** 2 for x in numeric_values) / len(numeric_values)
                features[f'{sensor_name}_variance'] = variance
                features[f'{sensor_name}_std'] = variance ** 0.5
        
        # Cross-sensor features
        if 'air_temp_c' in sensors and 'relative_humidity_percent' in sensors:
            temp_values = [v for v in sensors['air_temp_c'] if v is not None]
            humidity_values = [v for v in sensors['relative_humidity_percent'] if v is not None]
            
            if temp_values and humidity_values:
                # Simple VPD calculation approximation
                avg_temp = sum(temp_values) / len(temp_values)
                avg_humidity = sum(humidity_values) / len(humidity_values)
                
                # Simplified VPD formula
                if avg_temp > 0 and avg_humidity > 0:
                    sat_vapor_pressure = 0.611 * (2.718281828 ** (17.27 * avg_temp / (avg_temp + 237.3)))
                    vpd = sat_vapor_pressure * (1 - avg_humidity / 100)
                    features['calculated_vpd_kpa'] = vpd
        
        return {
            'status': 'success',
            'features': features,
            'metadata': {
                'num_samples': len(timestamps),
                'num_features': len(features),
                'gpu_used': False
            }
        }
        
    except Exception as e:
        logger.error(f"Feature extraction failed: {str(e)}")
        return {
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def main():
    """Main entry point for the minimal feature extraction service."""
    try:
        # Read JSON input from stdin
        input_data = json.load(sys.stdin)
        
        # Extract features
        result = extract_basic_features(input_data)
        
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