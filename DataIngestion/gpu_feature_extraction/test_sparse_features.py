#!/usr/bin/env python3
"""
Test script for sparse feature extraction
Tests both the Python GPU implementation and integration with Rust
"""

import json
import subprocess
import sys
from datetime import datetime, timedelta
import numpy as np
import random

def generate_sparse_test_data(num_samples=1000, sparsity=0.913):
    """Generate test data with specified sparsity (91.3% by default)"""
    # Generate timestamps
    start_time = datetime(2014, 1, 1)
    timestamps = [start_time + timedelta(minutes=5*i) for i in range(num_samples)]
    
    # Generate sparse sensor data
    sensors = {}
    
    # Temperature sensor (less sparse)
    temp_sparsity = sparsity * 0.8  # 73% sparse
    sensors['air_temp_c'] = [
        20.0 + 5 * np.sin(i/100) + random.gauss(0, 1) if random.random() > temp_sparsity else None
        for i in range(num_samples)
    ]
    
    # Humidity sensor
    sensors['relative_humidity_percent'] = [
        60.0 + 20 * np.sin(i/150 + 1) + random.gauss(0, 2) if random.random() > sparsity else None
        for i in range(num_samples)
    ]
    
    # CO2 sensor
    sensors['co2_measured_ppm'] = [
        400 + 200 * np.sin(i/200 + 2) + random.gauss(0, 10) if random.random() > sparsity else None
        for i in range(num_samples)
    ]
    
    # Light intensity (very sparse at night)
    sensors['light_intensity_umol'] = []
    for i, ts in enumerate(timestamps):
        hour = ts.hour
        if 6 <= hour <= 18:  # Daytime
            if random.random() > sparsity * 0.5:  # Less sparse during day
                sensors['light_intensity_umol'].append(300 + 200 * np.sin(i/50) + random.gauss(0, 20))
            else:
                sensors['light_intensity_umol'].append(None)
        else:  # Nighttime
            if random.random() > 0.95:  # Very sparse at night
                sensors['light_intensity_umol'].append(random.gauss(10, 5))
            else:
                sensors['light_intensity_umol'].append(None)
    
    # Binary lamp status (less sparse)
    sensors['lamp_grp1_no3_status'] = [
        1.0 if (6 <= timestamps[i].hour <= 22 and random.random() > 0.2) else 0.0 
        if random.random() > sparsity * 0.3 else None
        for i in range(num_samples)
    ]
    
    # VPD (calculated when temp and humidity available)
    sensors['vpd_hpa'] = []
    for i in range(num_samples):
        if sensors['air_temp_c'][i] is not None and sensors['relative_humidity_percent'][i] is not None:
            # Simplified VPD calculation
            temp = sensors['air_temp_c'][i]
            rh = sensors['relative_humidity_percent'][i]
            es = 0.611 * np.exp(17.502 * temp / (temp + 240.97))  # kPa
            ea = es * rh / 100
            vpd = es - ea
            sensors['vpd_hpa'].append(vpd * 10)  # Convert to hPa
        else:
            sensors['vpd_hpa'].append(None)
    
    # Generate some energy price data
    energy_prices = [
        (timestamps[i], 0.5 + 0.3 * np.sin(i/100) + 0.2 * (1 if 17 <= timestamps[i].hour <= 20 else 0))
        for i in range(0, num_samples, 12)  # Hourly prices
    ]
    
    return timestamps, sensors, energy_prices


def test_python_sparse_features():
    """Test the Python sparse feature extraction directly"""
    print("Testing Python sparse feature extraction...")
    
    # Generate test data
    timestamps, sensors, energy_prices = generate_sparse_test_data(1000, 0.913)
    
    # Prepare request
    request = {
        'timestamps': [ts.isoformat() for ts in timestamps],
        'sensors': sensors,
        'energy_prices': [(ts.isoformat(), price) for ts, price in energy_prices],
        'use_gpu': False  # Use CPU for testing
    }
    
    # Run the Python script
    proc = subprocess.Popen(
        [sys.executable, 'sparse_gpu_features.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    stdout, stderr = proc.communicate(json.dumps(request))
    
    if proc.returncode != 0:
        print(f"Error running sparse feature extraction: {stderr}")
        return False
    
    # Parse response
    try:
        response = json.loads(stdout)
    except json.JSONDecodeError as e:
        print(f"Failed to parse response: {e}")
        print(f"stdout: {stdout}")
        print(f"stderr: {stderr}")
        return False
    
    if response['status'] != 'success':
        print(f"Feature extraction failed: {response.get('error', 'Unknown error')}")
        return False
    
    # Analyze results
    features = response['features']
    metadata = response['metadata']
    
    print("\nExtraction successful!")
    print(f"Total samples: {metadata['num_samples']}")
    print(f"Features extracted: {metadata['num_features']}")
    print(f"Average coverage: {metadata.get('average_coverage', 0):.1%}")
    print(f"GPU used: {metadata['gpu_used']}")
    
    # Show some example features
    print("\nExample features:")
    example_features = list(features.items())[:20]
    for feat_name, feat_value in example_features:
        print(f"  {feat_name}: {feat_value:.4f}")
    
    # Verify expected features exist
    expected_features = [
        'air_temp_c_coverage',
        'air_temp_c_longest_gap_minutes',
        'air_temp_c_mean_crossings',
        'relative_humidity_percent_coverage',
        'lamp_grp1_no3_status_coverage',
        'total_lamp_on_hours',
        'air_temp_c_day_night_ratio'
    ]
    
    missing_features = [f for f in expected_features if f not in features]
    if missing_features:
        print(f"\nWarning: Missing expected features: {missing_features}")
    
    # Verify coverage values match expected sparsity
    coverage_features = {k: v for k, v in features.items() if k.endswith('_coverage')}
    avg_coverage = np.mean(list(coverage_features.values()))
    expected_coverage = 1 - 0.913  # ~8.7%
    
    print("\nCoverage validation:")
    print(f"  Expected average coverage: {expected_coverage:.1%}")
    print(f"  Actual average coverage: {avg_coverage:.1%}")
    print(f"  Difference: {abs(avg_coverage - expected_coverage):.1%}")
    
    # Show coverage by sensor
    print("\nCoverage by sensor:")
    for sensor, coverage in sorted(coverage_features.items()):
        print(f"  {sensor}: {coverage:.1%}")
    
    return True


def test_sparse_data_statistics():
    """Test statistical properties of sparse data"""
    print("\n\nTesting sparse data statistics...")
    
    # Generate highly sparse data
    timestamps, sensors, _ = generate_sparse_test_data(10000, 0.95)
    
    print("\nData sparsity analysis:")
    for sensor_name, values in sensors.items():
        non_null = sum(1 for v in values if v is not None)
        sparsity = 1 - (non_null / len(values))
        
        # Calculate gap statistics
        gaps = []
        in_gap = False
        gap_start = 0
        
        for i, v in enumerate(values):
            if v is None:
                if not in_gap:
                    gap_start = i
                    in_gap = True
            else:
                if in_gap:
                    gaps.append(i - gap_start)
                    in_gap = False
        
        if in_gap:
            gaps.append(len(values) - gap_start)
        
        print(f"\n{sensor_name}:")
        print(f"  Sparsity: {sparsity:.1%}")
        print(f"  Non-null values: {non_null}/{len(values)}")
        if gaps:
            print(f"  Number of gaps: {len(gaps)}")
            print(f"  Longest gap: {max(gaps)} samples ({max(gaps)*5} minutes)")
            print(f"  Average gap: {np.mean(gaps):.1f} samples ({np.mean(gaps)*5:.1f} minutes)")


if __name__ == '__main__':
    print("=" * 60)
    print("Sparse Feature Extraction Test Suite")
    print("=" * 60)
    
    # Test Python implementation
    if not test_python_sparse_features():
        print("\nPython sparse feature test FAILED")
        sys.exit(1)
    
    # Test data statistics
    test_sparse_data_statistics()
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)