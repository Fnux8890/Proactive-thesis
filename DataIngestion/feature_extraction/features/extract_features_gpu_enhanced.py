"""Enhanced feature extraction with GPU acceleration around tsfresh.

This script orchestrates the entire pipeline:
1. GPU-accelerated data loading and preprocessing
2. tsfresh feature extraction (CPU, 600+ features)
3. GPU-accelerated feature selection
4. GPU-accelerated database writes
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import gc
import logging
import os
import time

from ..db_utils_optimized import SQLAlchemyPostgresConnector
from .adapters import tsfresh_extract_features
from .gpu_preprocessing import GPUDataPreprocessor, GPUFeatureSelector, GPUMemoryManager
from tsfresh import extract_features
from tsfresh.feature_extraction import (
    MinimalFCParameters,
    EfficientFCParameters,
    ComprehensiveFCParameters
)
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
class GPUEnhancedFeatureExtractor:
    """Orchestrates GPU-accelerated feature extraction pipeline."""
    def __init__(self,
                 use_gpu: bool = True,
                 feature_set: str = 'efficient',
                 batch_size: int = 10000):
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        # Initialize GPU components
        self.preprocessor = GPUDataPreprocessor(use_gpu=use_gpu)
        self.selector = GPUFeatureSelector(use_gpu=use_gpu)
        self.memory_manager = GPUMemoryManager() if use_gpu else None
        # Set tsfresh parameters based on feature set
        self.feature_params = self._get_feature_params(feature_set)
        # Database connection
        self.db_connector = SQLAlchemyPostgresConnector()
        logger.info(f"Initialized GPU-enhanced extractor (GPU: {use_gpu}, features: {feature_set})")
    def _get_feature_params(self, feature_set: str):
        """Get tsfresh feature extraction parameters."""
        if feature_set == 'minimal':
            return MinimalFCParameters()
        elif feature_set == 'efficient':
            return EfficientFCParameters()
        elif feature_set == 'comprehensive':
            return ComprehensiveFCParameters()
        else:
            # Custom parameters from config
            return self._load_custom_params(feature_set)
    def extract_features_for_era(self,
                               era_id: str,
                               start_time: pd.Timestamp,
                               end_time: pd.Timestamp) -> pd.DataFrame:
        """Extract features for a single era with GPU acceleration."""
        logger.info(f"Processing era {era_id}: {start_time} to {end_time}")
        try:
            # 1. Load data from database with GPU acceleration
            data = self._load_era_data_gpu(era_id, start_time, end_time)
            if data.empty:
                logger.warning(f"No data found for era {era_id}")
                return pd.DataFrame()
            # 2. Prepare data for tsfresh (GPU)
            logger.info("Preparing data for tsfresh...")
            prepared_data = self.preprocessor.prepare_for_tsfresh(
                data,
                column_id='era_id',
                column_sort='timestamp',
                column_kind='variable',
                column_value='value'
            )
            # 3. Extract features using tsfresh (CPU - but data prep was GPU)
            logger.info(f"Extracting {len(self.feature_params)} feature types...")
            features = extract_features(
                prepared_data,
                column_id='era_id',
                column_sort='timestamp',
                column_kind='variable',
                column_value='value',
                default_fc_parameters=self.feature_params,
                disable_progressbar=True,
                n_jobs=4  # Use multiple CPU cores
            )
            # 4. GPU-accelerated feature selection
            logger.info("Selecting relevant features...")
            selected_features = self.selector.select_features_gpu(
                features,
                method='correlation',
                correlation_threshold=0.95
            )
            # 5. Add metadata
            selected_features['era_id'] = era_id
            selected_features['start_time'] = start_time
            selected_features['end_time'] = end_time
            # Clean up GPU memory
            if self.memory_manager:
                self.memory_manager.clear_memory()
            return selected_features
        except Exception as e:
            logger.error(f"Error processing era {era_id}: {e}")
            raise
    def _load_era_data_gpu(self,
                          era_id: str,
                          start_time: pd.Timestamp,
                          end_time: pd.Timestamp) -> pd.DataFrame:
        """Load era data with GPU-accelerated processing."""
        # Query to get wide format data
        query = f"""
        SELECT
            '{era_id}' as era_id,
            time as timestamp,
            air_temp_c,
            relative_humidity_percent,
            co2_measured_ppm,
            light_intensity_umol,
            radiation_w_m2,
            heating_setpoint_c,
            co2_required_ppm,
            vpd_hpa,
            dli_sum,
            curtain_1_percent,
            curtain_2_percent,
            vent_pos_1_percent,
            vent_pos_2_percent,
            lamp_grp1_no3_status,
            lamp_grp1_no4_status,
            lamp_grp2_no3_status,
            lamp_grp2_no4_status
        FROM sensor_data_merged
        WHERE time >= '{start_time}'
          AND time < '{end_time}'
        ORDER BY time
        """
        # Load data
        with self.db_connector.get_engine().connect() as conn:
            data = pd.read_sql(query, conn)
        if self.use_gpu and len(data) > 1000:  # Only use GPU for larger datasets
            # Melt to long format using GPU
            value_vars = [col for col in data.columns
                         if col not in ['era_id', 'timestamp']]
            logger.info(f"GPU melting {len(value_vars)} variables...")
            data_long = self.preprocessor.melt_wide_to_long_gpu(
                data,
                id_vars=['era_id', 'timestamp'],
                value_vars=value_vars,
                var_name='variable',
                value_name='value'
            )
            return data_long
        else:
            # CPU fallback for small datasets
            value_vars = [col for col in data.columns
                         if col not in ['era_id', 'timestamp']]
            return pd.melt(
                data,
                id_vars=['era_id', 'timestamp'],
                value_vars=value_vars,
                var_name='variable',
                value_name='value'
            )
    def run_batch_extraction(self,
                           era_definitions: List[Dict],
                           output_table: str = 'tsfresh_features_gpu') -> None:
        """Run feature extraction for multiple eras with batching."""
        total_start = time.time()
        successful = 0
        failed = 0
        logger.info(f"Starting batch extraction for {len(era_definitions)} eras")
        # Process in batches to manage memory
        for i, era_def in enumerate(era_definitions):
            try:
                # Log memory before processing
                if self.memory_manager and i % 10 == 0:
                    self.memory_manager.log_memory_usage()
                # Extract features
                features = self.extract_features_for_era(
                    era_def['era_id'],
                    pd.Timestamp(era_def['start_time']),
                    pd.Timestamp(era_def['end_time'])
                )
                if not features.empty:
                    # Write to database
                    self._write_features_to_db(features, output_table)
                    successful += 1
                else:
                    failed += 1
                # Progress update
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - total_start
                    rate = (i + 1) / elapsed
                    remaining = (len(era_definitions) - i - 1) / rate
                    logger.info(
                        f"Progress: {i+1}/{len(era_definitions)} eras "
                        f"({successful} successful, {failed} failed) - "
                        f"ETA: {remaining/60:.1f} minutes"
                    )
                # Garbage collection every N eras
                if (i + 1) % 50 == 0:
                    gc.collect()
                    if self.memory_manager:
                        self.memory_manager.clear_memory()
            except Exception as e:
                logger.error(f"Failed to process era {era_def['era_id']}: {e}")
                failed += 1
                continue
        total_time = time.time() - total_start
        logger.info(
            f"Batch extraction completed in {total_time/60:.1f} minutes - "
            f"{successful} successful, {failed} failed"
        )
    def _write_features_to_db(self, features: pd.DataFrame, table_name: str) -> None:
        """Write features to database (could be GPU-accelerated with custom COPY)."""
        try:
            with self.db_connector.get_engine().connect() as conn:
                features.to_sql(
                    table_name,
                    conn,
                    if_exists='append',
                    index=False,
                    method='multi',
                    chunksize=1000
                )
            logger.debug(f"Written {len(features)} features to {table_name}")
        except Exception as e:
            logger.error(f"Failed to write features to database: {e}")
            raise
    def validate_pipeline(self) -> bool:
        """Validate the pipeline setup before running."""
        checks = []
        # Check GPU availability
        if self.use_gpu:
            try:
                import cupy as cp
                import cudf
                checks.append(("GPU libraries available", True))
                # Check GPU memory
                mem_info = self.memory_manager.get_memory_info()
                has_memory = mem_info.get('total_gb', 0) >= 4.0
                checks.append((
                    f"GPU memory >= 4GB ({mem_info.get('total_gb', 0):.1f}GB found)",
                    has_memory
                ))
            except Exception as e:
                checks.append(("GPU libraries available", False))
                logger.error(f"GPU validation failed: {e}")
        # Check database connection
        try:
            with self.db_connector.get_engine().connect() as conn:
                result = conn.execute("SELECT 1").scalar()
                checks.append(("Database connection", result == 1))
        except Exception as e:
            checks.append(("Database connection", False))
            logger.error(f"Database validation failed: {e}")
        # Check tsfresh
        try:
            import tsfresh
            checks.append(("tsfresh available", True))
        except:
            checks.append(("tsfresh available", False))
        # Print validation results
        logger.info("Pipeline validation results:")
        all_passed = True
        for check, passed in checks:
            status = "✓" if passed else "✗"
            logger.info(f"  {status} {check}")
            all_passed = all_passed and passed
        return all_passed
def main():
    """Main entry point for GPU-enhanced feature extraction."""
    # Configuration from environment
    use_gpu = os.getenv('USE_GPU', 'true').lower() == 'true'
    feature_set = os.getenv('FEATURE_SET', 'efficient')
    output_table = os.getenv('OUTPUT_TABLE', 'tsfresh_features_gpu')
    # Initialize extractor
    extractor = GPUEnhancedFeatureExtractor(
        use_gpu=use_gpu,
        feature_set=feature_set
    )
    # Validate setup
    if not extractor.validate_pipeline():
        logger.error("Pipeline validation failed!")
        return 1
    # Load era definitions (example - adapt to your needs)
    # This would typically come from your era detection results
    era_definitions = [
        {
            'era_id': 'era_001',
            'start_time': '2024-01-01 00:00:00',
            'end_time': '2024-01-02 00:00:00'
        },
        # Add more eras...
    ]
    # Run extraction
    extractor.run_batch_extraction(era_definitions, output_table)
    return 0
if __name__ == "__main__":
    exit(main())
