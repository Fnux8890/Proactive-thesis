#!/usr/bin/env python3
"""
Test script for Python GPU feature extraction
"""

import json
import subprocess
import sys

def test_gpu_features():
    """Test the GPU feature extraction with sample data."""
    
    # Sample test data
    test_data = {
        "timestamps": [
            "2024-01-01T00:00:00Z",
            "2024-01-01T01:00:00Z",
            "2024-01-01T02:00:00Z",
            "2024-01-01T03:00:00Z",
            "2024-01-01T04:00:00Z"
        ],
        "sensors": {
            "temperature": [20.5, 21.0, 21.5, 22.0, 21.8],
            "humidity": [65.0, 66.0, 64.0, 63.0, 64.5],
            "co2": [400.0, 410.0, 420.0, 415.0, 405.0]
        },
        "window_sizes": [30, 120],
        "use_gpu": True
    }
    
    # Test 1: Direct Python script call
    print("Test 1: Direct Python script call")
    try:
        # Run the script directly
        result = subprocess.run(
            [sys.executable, "minimal_gpu_features.py"],
            input=json.dumps(test_data),
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse the output
        output = json.loads(result.stdout)
        
        if output["status"] == "success":
            print("✓ Feature extraction successful")
            print(f"  - Features extracted: {output['metadata']['num_features']}")
            print(f"  - GPU used: {output['metadata']['gpu_used']}")
            print("  - Sample features:")
            for key, value in list(output['features'].items())[:5]:
                print(f"    - {key}: {value:.3f}")
        else:
            print("✗ Feature extraction failed")
            print(f"  - Error: {output.get('error', 'Unknown error')}")
            
    except subprocess.CalledProcessError as e:
        print(f"✗ Script execution failed: {e}")
        print(f"  - stderr: {e.stderr}")
    except json.JSONDecodeError as e:
        print(f"✗ Failed to parse output: {e}")
        print(f"  - stdout: {result.stdout}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    
    # Test 2: Docker container call (if image exists)
    print("\nTest 2: Docker container call")
    try:
        # Check if Docker image exists
        check_result = subprocess.run(
            ["docker", "images", "-q", "gpu-feature-python:latest"],
            capture_output=True,
            text=True
        )
        
        if check_result.stdout.strip():
            # Run the Docker container
            result = subprocess.run(
                [
                    "docker", "run", "--rm", "-i",
                    "--gpus", "all",
                    "gpu-feature-python:latest",
                    "python", "/app/minimal_gpu_features.py"
                ],
                input=json.dumps(test_data),
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse the output
            output = json.loads(result.stdout)
            
            if output["status"] == "success":
                print("✓ Docker feature extraction successful")
                print(f"  - Features extracted: {output['metadata']['num_features']}")
                print(f"  - GPU used: {output['metadata']['gpu_used']}")
            else:
                print("✗ Docker feature extraction failed")
                print(f"  - Error: {output.get('error', 'Unknown error')}")
        else:
            print("⚠ Docker image not found. Build it with:")
            print("  docker build -f Dockerfile.python-gpu -t gpu-feature-python:latest .")
            
    except Exception as e:
        print(f"✗ Docker test failed: {e}")

if __name__ == "__main__":
    test_gpu_features()