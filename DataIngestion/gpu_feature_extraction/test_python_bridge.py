#!/usr/bin/env python3
"""
Test the Python Bridge Communication
Tests that minimal_gpu_features.py and sparse_gpu_features.py work correctly
when called from Rust via JSON stdin/stdout.
"""

import json
import subprocess
import sys
from datetime import datetime, timedelta

def test_minimal_gpu_features():
    """Test minimal_gpu_features.py script."""
    print("Testing minimal_gpu_features.py...")
    
    # Prepare test data
    timestamps = []
    air_temp_data = []
    humidity_data = []
    
    # Generate 48 hours of hourly data
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    for i in range(48):
        timestamp = start_time + timedelta(hours=i)
        timestamps.append(timestamp.isoformat() + 'Z')
        air_temp_data.append(20.0 + 2.0 * (i % 24) / 24.0)  # Simple temperature pattern
        humidity_data.append(50.0 + 10.0 * ((i + 12) % 24) / 24.0)  # Humidity pattern
    
    test_data = {
        "timestamps": timestamps,
        "sensors": {
            "air_temp_c": air_temp_data,
            "relative_humidity_percent": humidity_data
        },
        "window_sizes": [30, 120],
        "use_gpu": False  # Test CPU fallback first
    }
    
    # Run the script
    try:
        proc = subprocess.run(
            [sys.executable, "/mnt/c/Users/fhj88/Documents/Github/Proactive-thesis/DataIngestion/gpu_feature_extraction/minimal_gpu_features.py"],
            input=json.dumps(test_data),
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if proc.returncode != 0:
            print(f"FAIL: Script returned error code {proc.returncode}")
            print(f"STDERR: {proc.stderr}")
            return False
        
        # Parse response
        try:
            response = json.loads(proc.stdout)
            
            if response.get('status') != 'success':
                print(f"FAIL: Response status is {response.get('status')}")
                print(f"Error: {response.get('error')}")
                return False
            
            features = response.get('features', {})
            metadata = response.get('metadata', {})
            
            print(f"SUCCESS: Generated {len(features)} features from {metadata.get('num_samples', 0)} samples")
            print(f"GPU used: {metadata.get('gpu_used', False)}")
            
            # Check for expected features
            expected_features = ['air_temp_c_mean', 'air_temp_c_std', 'relative_humidity_percent_mean']
            for feature in expected_features:
                if feature not in features:
                    print(f"WARN: Expected feature {feature} not found")
                else:
                    print(f"  {feature}: {features[feature]}")
            
            return True
            
        except json.JSONDecodeError as e:
            print(f"FAIL: Could not parse JSON response: {e}")
            print(f"STDOUT: {proc.stdout}")
            return False
            
    except subprocess.TimeoutExpired:
        print("FAIL: Script timed out after 30 seconds")
        return False
    except Exception as e:
        print(f"FAIL: Unexpected error: {e}")
        return False

def test_sparse_gpu_features():
    """Test sparse_gpu_features.py script."""
    print("\nTesting sparse_gpu_features.py...")
    
    # Prepare sparse test data (91.3% missing)
    timestamps = []
    air_temp_data = []
    humidity_data = []
    
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    for i in range(100):  # 100 samples
        timestamp = start_time + timedelta(minutes=i * 30)  # 30-minute intervals
        timestamps.append(timestamp.isoformat() + 'Z')
        
        # Create sparse data: only ~9% have values
        if i % 11 == 0:  # Every 11th sample has data
            air_temp_data.append(20.0 + 2.0 * (i % 24) / 24.0)
            humidity_data.append(50.0 + 10.0 * ((i + 12) % 24) / 24.0)
        else:
            air_temp_data.append(None)
            humidity_data.append(None)
    
    test_data = {
        "timestamps": timestamps,
        "sensors": {
            "air_temp_c": air_temp_data,
            "relative_humidity_percent": humidity_data
        },
        "energy_prices": [],
        "use_gpu": False  # Test CPU fallback first
    }
    
    # Run the script
    try:
        proc = subprocess.run(
            [sys.executable, "/mnt/c/Users/fhj88/Documents/Github/Proactive-thesis/DataIngestion/gpu_feature_extraction/sparse_gpu_features.py"],
            input=json.dumps(test_data),
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if proc.returncode != 0:
            print(f"FAIL: Script returned error code {proc.returncode}")
            print(f"STDERR: {proc.stderr}")
            return False
        
        # Parse response
        try:
            response = json.loads(proc.stdout)
            
            if response.get('status') != 'success':
                print(f"FAIL: Response status is {response.get('status')}")
                print(f"Error: {response.get('error')}")
                return False
            
            features = response.get('features', {})
            metadata = response.get('metadata', {})
            
            print(f"SUCCESS: Generated {len(features)} sparse features from {metadata.get('num_samples', 0)} samples")
            print(f"GPU used: {metadata.get('gpu_used', False)}")
            print(f"Average coverage: {metadata.get('average_coverage', 0):.3f}")
            
            # Check for expected sparse features
            expected_features = ['air_temp_c_coverage', 'air_temp_c_longest_gap_minutes', 'air_temp_c_num_gaps']
            for feature in expected_features:
                if feature not in features:
                    print(f"WARN: Expected sparse feature {feature} not found")
                else:
                    print(f"  {feature}: {features[feature]}")
            
            return True
            
        except json.JSONDecodeError as e:
            print(f"FAIL: Could not parse JSON response: {e}")
            print(f"STDOUT: {proc.stdout}")
            return False
            
    except subprocess.TimeoutExpired:
        print("FAIL: Script timed out after 30 seconds")
        return False
    except Exception as e:
        print(f"FAIL: Unexpected error: {e}")
        return False

if __name__ == '__main__':
    print("Testing Python GPU Feature Extraction Bridge")
    print("=" * 50)
    
    success1 = test_minimal_gpu_features()
    success2 = test_sparse_gpu_features()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("SUCCESS: Both Python scripts work correctly!")
        print("The Rust-Python bridge should work properly.")
        sys.exit(0)
    else:
        print("FAIL: One or more tests failed.")
        sys.exit(1)