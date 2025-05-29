#!/usr/bin/env python
"""Wrapper script to ensure GPU is available before running training."""

import subprocess
import sys
import os

def check_gpu():
    """Quick GPU check before starting training."""
    print("Checking GPU availability...")
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                               capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            gpu_name = result.stdout.strip()
            print(f"✓ GPU detected: {gpu_name}")
            return True
        else:
            print("✗ nvidia-smi failed - GPU may not be available")
            return False
    except Exception as e:
        print(f"✗ GPU check failed: {e}")
        return False

def main():
    """Run the training with GPU check."""
    # Check GPU
    gpu_available = check_gpu()
    
    if not gpu_available:
        print("\nWARNING: GPU not detected! Training will fall back to CPU.")
        print("Make sure Docker is configured with GPU support.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Aborting.")
            return 1
    
    # Run the actual training
    print("\nStarting model training...")
    cmd = [sys.executable, "-m", "src.training.train_all_objectives"] + sys.argv[1:]
    return subprocess.call(cmd)

if __name__ == "__main__":
    sys.exit(main())