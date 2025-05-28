#!/usr/bin/env python3
"""Fix specific Python syntax issues in the new files."""

from pathlib import Path


def fix_extract_features_gpu_enhanced():
    """Fix the broken imports in extract_features_gpu_enhanced.py"""
    filepath = Path("features/extract_features_gpu_enhanced.py")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Fix the broken import statement
    fixed_content = """\"\"\"Enhanced feature extraction with GPU acceleration around tsfresh.

This script orchestrates the entire pipeline:
1. GPU-accelerated data loading and preprocessing
2. tsfresh feature extraction (CPU, 600+ features)
3. GPU-accelerated feature selection
4. GPU-accelerated database writes
\"\"\"

import gc
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import (
    MinimalFCParameters,
    EfficientFCParameters,
    ComprehensiveFCParameters
)

from .gpu_preprocessing import GPUDataPreprocessor, GPUFeatureSelector, GPUMemoryManager
from .adapters import tsfresh_extract_features
from ..db_utils_optimized import SQLAlchemyPostgresConnector

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
"""
    
    # Add the rest of the file content
    lines = content.split('\n')
    # Find where the class definition starts
    class_start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('class GPUEnhancedFeatureExtractor'):
            class_start = i
            break
    
    # Append the rest of the content
    if class_start > 0:
        fixed_content += '\n'.join(lines[class_start:])
    
    with open(filepath, 'w') as f:
        f.write(fixed_content)
    
    print(f"Fixed {filepath}")


def fix_extract_features_enhanced():
    """Fix missing newline at end of file"""
    filepath = Path("feature/extract_features_enhanced.py")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    if not content.endswith('\n'):
        content += '\n'
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"Fixed {filepath}")


def fix_validate_pipeline():
    """Fix missing newline at end of file"""
    filepath = Path("validate_pipeline.py")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    if not content.endswith('\n'):
        content += '\n'
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"Fixed {filepath}")


def fix_gpu_preprocessing():
    """Fix missing newline at end of file"""
    filepath = Path("features/gpu_preprocessing.py")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    if not content.endswith('\n'):
        content += '\n'
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"Fixed {filepath}")


def main():
    """Fix all files"""
    print("Fixing Python files...")
    
    # Change to feature extraction directory
    import os
    os.chdir(Path(__file__).parent)
    
    fix_extract_features_gpu_enhanced()
    fix_extract_features_enhanced()
    fix_validate_pipeline()
    fix_gpu_preprocessing()
    
    print("\nAll files fixed!")


if __name__ == "__main__":
    main()