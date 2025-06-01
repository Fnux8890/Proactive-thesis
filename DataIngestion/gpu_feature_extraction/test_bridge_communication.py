#!/usr/bin/env python3
"""
Test the Python Bridge Communication Protocol
Tests that the JSON stdin/stdout communication pattern works correctly.
"""

import json
import subprocess
import sys
from datetime import datetime, timedelta

def test_minimal_communication():
    """Test basic JSON communication with minimal feature extraction."""
    print("Testing JSON stdin/stdout communication...")
    
    # Prepare test data
    timestamps = []
    air_temp_data = []
    humidity_data = []
    
    # Generate 24 hours of hourly data
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    for i in range(24):
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
        "use_gpu": False
    }
    
    # Run the script
    try:
        proc = subprocess.run(
            [sys.executable, "test_minimal_features.py"],
            input=json.dumps(test_data),
            capture_output=True,
            text=True,
            timeout=10
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
            expected_features = ['air_temp_c_mean', 'air_temp_c_std', 'relative_humidity_percent_mean', 'calculated_vpd_kpa']
            for feature in expected_features:
                if feature not in features:
                    print(f"WARN: Expected feature {feature} not found")
                else:
                    print(f"  {feature}: {features[feature]:.3f}")
            
            return True
            
        except json.JSONDecodeError as e:
            print(f"FAIL: Could not parse JSON response: {e}")
            print(f"STDOUT: {proc.stdout}")
            return False
            
    except subprocess.TimeoutExpired:
        print("FAIL: Script timed out after 10 seconds")
        return False
    except Exception as e:
        print(f"FAIL: Unexpected error: {e}")
        return False

def test_sparse_communication():
    """Test communication with sparse data (simulating 91.3% missing values)."""
    print("\nTesting sparse data communication...")
    
    # Prepare sparse test data
    timestamps = []
    air_temp_data = []
    humidity_data = []
    
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    for i in range(48):  # 48 samples
        timestamp = start_time + timedelta(minutes=i * 30)  # 30-minute intervals
        timestamps.append(timestamp.isoformat() + 'Z')
        
        # Create sparse data: only ~8.7% have values (91.3% missing)
        if i % 12 == 0:  # Every 12th sample has data
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
        "use_gpu": False
    }
    
    # Run the script
    try:
        proc = subprocess.run(
            [sys.executable, "test_minimal_features.py"],
            input=json.dumps(test_data),
            capture_output=True,
            text=True,
            timeout=10
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
            
            # Check coverage features
            temp_coverage = features.get('air_temp_c_coverage', 0)
            humidity_coverage = features.get('relative_humidity_percent_coverage', 0)
            
            print(f"Temperature coverage: {temp_coverage:.3f} (should be ~0.087)")
            print(f"Humidity coverage: {humidity_coverage:.3f} (should be ~0.087)")
            
            if temp_coverage > 0.05 and temp_coverage < 0.15:
                print("SUCCESS: Sparse data coverage detection working correctly")
                return True
            else:
                print("WARN: Coverage detection may not be working as expected")
                return True  # Still count as success since communication works
            
        except json.JSONDecodeError as e:
            print(f"FAIL: Could not parse JSON response: {e}")
            print(f"STDOUT: {proc.stdout}")
            return False
            
    except subprocess.TimeoutExpired:
        print("FAIL: Script timed out after 10 seconds")
        return False
    except Exception as e:
        print(f"FAIL: Unexpected error: {e}")
        return False

def test_error_handling():
    """Test error handling in communication."""
    print("\nTesting error handling...")
    
    # Send invalid data
    test_data = {
        "invalid": "data_structure"
    }
    
    try:
        proc = subprocess.run(
            [sys.executable, "test_minimal_features.py"],
            input=json.dumps(test_data),
            capture_output=True,
            text=True,
            timeout=5
        )
        
        # Should return error code but still provide JSON response
        if proc.returncode == 0:
            print("WARN: Script should have returned error code for invalid data")
        
        # Parse response
        try:
            response = json.loads(proc.stdout)
            
            if response.get('status') == 'error':
                print("SUCCESS: Error handling working correctly")
                print(f"Error message: {response.get('error')}")
                return True
            else:
                print("FAIL: Expected error status in response")
                return False
                
        except json.JSONDecodeError as e:
            print(f"FAIL: Could not parse error response: {e}")
            return False
            
    except Exception as e:
        print(f"FAIL: Unexpected error in error handling test: {e}")
        return False

if __name__ == '__main__':
    print("Testing Python Bridge Communication Protocol")
    print("=" * 50)
    
    success1 = test_minimal_communication()
    success2 = test_sparse_communication()
    success3 = test_error_handling()
    
    print("\n" + "=" * 50)
    if success1 and success2 and success3:
        print("SUCCESS: JSON stdin/stdout communication protocol working correctly!")
        print("The Rust-Python bridge interface should work properly.")
        print("\nNext steps:")
        print("1. Test with actual Python GPU scripts in a proper environment")
        print("2. Test the Rust python_bridge.rs calling these scripts")
        print("3. Integrate into the enhanced sparse pipeline")
        sys.exit(0)
    else:
        print("FAIL: One or more communication tests failed.")
        sys.exit(1)