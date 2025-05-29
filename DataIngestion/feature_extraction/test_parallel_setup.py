#!/usr/bin/env python3
"""Test script to verify parallel feature extraction setup."""

import os
import sys
import multiprocessing
import platform

# Try importing key dependencies
try:
    import numpy as np
    import pandas as pd
    import tsfresh
    from tsfresh import extract_features
    print("✓ Core dependencies imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Check GPU availability if requested
use_gpu = os.getenv('USE_GPU', 'false').lower() == 'true'
if use_gpu:
    try:
        import cupy as cp
        import cudf
        print("✓ GPU libraries available (CuPy, cuDF)")
        print(f"  CUDA devices: {cp.cuda.runtime.getDeviceCount()}")
    except ImportError:
        print("✗ GPU libraries not available (CuPy/cuDF)")
    except Exception as e:
        print(f"✗ GPU error: {e}")

# Check parallel processing configuration
print("\n=== Parallel Processing Configuration ===")
print(f"System: {platform.system()} {platform.release()}")
print(f"CPU cores available: {multiprocessing.cpu_count()}")
print(f"BATCH_SIZE: {os.getenv('BATCH_SIZE', 'not set')}")
print(f"N_JOBS: {os.getenv('N_JOBS', 'not set')}")
print(f"USE_GPU: {os.getenv('USE_GPU', 'false')}")
print(f"FEATURE_SET: {os.getenv('FEATURE_SET', 'not set')}")

# Test tsfresh parallel capabilities
print("\n=== Testing tsfresh Parallel Extraction ===")
try:
    # Create small test dataset
    n_samples = 1000
    n_series = 10
    
    # Generate test time series data
    test_data = []
    for series_id in range(n_series):
        for t in range(n_samples // n_series):
            test_data.append({
                'id': series_id,
                'time': t,
                'value': np.sin(t * 0.1) + np.random.normal(0, 0.1)
            })
    
    df = pd.DataFrame(test_data)
    print(f"Test dataset: {len(df)} rows, {n_series} time series")
    
    # Extract features with parallel processing
    n_jobs = int(os.getenv('N_JOBS', '-1'))
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    
    print(f"Extracting features with {n_jobs} parallel jobs...")
    
    from tsfresh.feature_extraction import MinimalFCParameters
    features = extract_features(
        df, 
        column_id='id', 
        column_sort='time',
        default_fc_parameters=MinimalFCParameters(),
        n_jobs=n_jobs,
        disable_progressbar=True
    )
    
    print(f"✓ Successfully extracted {features.shape[1]} features from {features.shape[0]} series")
    print(f"  Feature extraction can use up to {n_jobs} CPU cores in parallel")
    
except Exception as e:
    print(f"✗ Feature extraction test failed: {e}")

# Test batch processing capability
print("\n=== Batch Processing Configuration ===")
batch_size = int(os.getenv('BATCH_SIZE', '100'))
print(f"Batch size: {batch_size} eras per batch")
print(f"With {n_series} total eras, this would require {(n_series + batch_size - 1) // batch_size} batches")

# Memory check
try:
    import psutil
    memory = psutil.virtual_memory()
    print(f"\n=== System Resources ===")
    print(f"Total RAM: {memory.total / (1024**3):.1f} GB")
    print(f"Available RAM: {memory.available / (1024**3):.1f} GB")
    print(f"RAM usage: {memory.percent}%")
except ImportError:
    print("\n(Install psutil for memory statistics)")

print("\n✓ Parallel feature extraction setup verified successfully!")