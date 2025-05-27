"""Enhanced feature extraction integrating GPU preprocessing with tsfresh.
This is a wrapper around the existing extract_features.py that adds:
1. GPU-accelerated preprocessing
2. Validation checks
3. Performance monitoring
"""

from pathlib import Path
import logging
import os
import sys
import time

from extract_features import main as original_main
from features.extract_features_gpu_enhanced import GPUEnhancedFeatureExtractor
from features.gpu_preprocessing import GPUDataPreprocessor, GPUFeatureSelector, GPUMemoryManager

logger = logging.getLogger(__name__)
def run_enhanced_extraction():
    """Run feature extraction with GPU enhancements."""
    # Check if we should use GPU acceleration
    use_gpu = os.getenv('USE_GPU', 'true').lower() == 'true'
    gpu_preprocess = os.getenv('GPU_PREPROCESS', 'true').lower() == 'true'
    if not use_gpu or not gpu_preprocess:
        logger.info("Running original feature extraction (GPU disabled)")
        return original_main()
    logger.info("Running GPU-enhanced feature extraction")
    # Initialize GPU components
    memory_manager = GPUMemoryManager()
    try:
        # Log initial memory state
        memory_manager.log_memory_usage()
        # Create enhanced extractor
        extractor = GPUEnhancedFeatureExtractor(
            use_gpu=True,
            feature_set=os.getenv('FEATURE_SET', 'efficient'),
            batch_size=int(os.getenv('BATCH_SIZE', '10000'))
        )
        # Validate pipeline
        if not extractor.validate_pipeline():
            logger.error("Pipeline validation failed")
            return 1
        # Load era definitions from environment or database
        era_definitions = load_era_definitions()
        if not era_definitions:
            logger.warning("No era definitions found, falling back to original extraction")
            return original_main()
        # Run GPU-enhanced extraction
        start_time = time.time()
        extractor.run_batch_extraction(
            era_definitions,
            output_table=os.getenv('FEATURES_TABLE', 'feature_data')
        )
        elapsed = time.time() - start_time
        logger.info(f"GPU-enhanced extraction completed in {elapsed/60:.1f} minutes")
        # Final memory report
        memory_manager.log_memory_usage()
        return 0
    except Exception as e:
        logger.error(f"GPU-enhanced extraction failed: {e}")
        logger.info("Falling back to original extraction")
        return original_main()
    finally:
        # Clean up GPU memory
        if memory_manager:
            memory_manager.clear_memory()
def load_era_definitions():
    """Load era definitions from database or file."""
    # This should match your era detection output format
    # For now, returning empty to trigger fallback
    return []
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Check if validation was requested
    if len(sys.argv) > 1 and sys.argv[1] == "--validate":
        from validate_pipeline import PipelineValidator
        validator = PipelineValidator()
        success = validator.run_all_checks()
        sys.exit(0 if success else 1)
    # Run extraction
    sys.exit(run_enhanced_extraction())
