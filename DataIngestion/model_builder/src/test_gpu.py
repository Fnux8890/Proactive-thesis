#!/usr/bin/env python
"""Test GPU availability for model_builder container."""

import os
import subprocess
import sys

print("=" * 60)
print("GPU AVAILABILITY TEST")
print("=" * 60)

# Check environment variables
print("\n1. Environment Variables:")
print(f"   NVIDIA_VISIBLE_DEVICES: {os.getenv('NVIDIA_VISIBLE_DEVICES', 'Not set')}")
print(f"   CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"   NVIDIA_DRIVER_CAPABILITIES: {os.getenv('NVIDIA_DRIVER_CAPABILITIES', 'Not set')}")

# Check nvidia-smi
print("\n2. NVIDIA-SMI Output:")
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout)
    else:
        print("   nvidia-smi command failed")
        print(f"   Error: {result.stderr}")
except Exception as e:
    print(f"   nvidia-smi not available: {e}")

# Check PyTorch CUDA
print("\n3. PyTorch CUDA Check:")
try:
    import torch
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA device count: {torch.cuda.device_count()}")
        print(f"   Current device: {torch.cuda.current_device()}")
        print(f"   Device name: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
except Exception as e:
    print(f"   PyTorch CUDA check failed: {e}")

# Check LightGBM GPU support
print("\n4. LightGBM GPU Check:")
try:
    import lightgbm as lgb
    import numpy as np
    
    # Try to create a small GPU dataset
    X = np.random.rand(100, 10)
    y = np.random.rand(100)
    
    params = {
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'objective': 'regression',
        'verbose': -1
    }
    
    train_data = lgb.Dataset(X, label=y)
    model = lgb.train(params, train_data, num_boost_round=1)
    print("   ✓ LightGBM GPU support confirmed!")
    print(f"   Model trained successfully with {model.num_trees()} trees")
except Exception as e:
    print(f"   ✗ LightGBM GPU test failed: {e}")

print("\n" + "=" * 60)
print("GPU test completed!")
print("=" * 60)