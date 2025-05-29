"""
GPU-accelerated feature extraction worker
Handles large eras with GPU preprocessing and postprocessing
"""

"""GPU-accelerated feature extraction worker.

Handles large eras with GPU preprocessing and postprocessing.
"""

import logging
import os
from typing import Any, Dict

import cudf
import cupy as cp
import numpy as np
import pandas as pd
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

from worker_base import FeatureExtractionConfig, FeatureWorkerBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GPUFeatureWorker(FeatureWorkerBase):
    """GPU-accelerated feature extraction worker.
    
    Optimized for:
    - Large eras (>1M rows)
    - High-frequency continuous sensors
    - Parallel statistical computations
    - FFT and spectral features
    """
    
    def __init__(self, worker_id: str, **kwargs):
        super().__init__(worker_id, "GPU", **kwargs)
        
        # Set GPU device
        gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
        cp.cuda.Device(gpu_id).use()

        # Initialize GPU memory pool
        mempool = cp.get_default_memory_pool()
        mempool.set_limit(size=30 * 1024**3)  # 30GB limit

        # Determine feature extraction profiles
        self.efficient_columns = set(FeatureExtractionConfig.EFFICIENT_PROFILE_COLUMNS)
        self.gpu_suitable_columns = set(FeatureExtractionConfig.GPU_SUITABLE_COLUMNS)
        
        logger.info(f"GPU worker {worker_id} using GPU {gpu_id}")
    
    def gpu_preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """GPU-accelerated data preprocessing.
        
        Performs:
        - Missing value imputation
        - Rolling statistics computation
        - Data normalization
        
        Args:
            df: Input dataframe
            
        Returns:
            Preprocessed dataframe
        """
        # Convert to cuDF for GPU processing
        gdf = cudf.from_pandas(df)
        
        # Fill missing values using forward fill
        numeric_cols = [col for col in gdf.columns if col in FeatureExtractionConfig.FEATURE_COLUMNS]
        for col in numeric_cols:
            gdf[col] = gdf[col].fillna(method="ffill").fillna(method="bfill")
        
        # Compute rolling statistics on GPU for high-frequency sensors
        # Use different windows based on sampling frequency
        window_sizes = [12, 60, 288]  # 1hr, 5hr, 24hr (assuming 5-min samples)

        # Only compute rolling stats for GPU-suitable columns
        rolling_cols = [col for col in numeric_cols if col in self.gpu_suitable_columns]
        
        for window in window_sizes:
            for col in rolling_cols:
                # Rolling mean
                gdf[f"{col}_rolling_mean_{window}"] = (
                    gdf[col].rolling(window=window, min_periods=1).mean()
                )

                # Rolling std
                gdf[f"{col}_rolling_std_{window}"] = (
                    gdf[col].rolling(window=window, min_periods=1).std()
                )
        
        # Convert back to pandas
        df_processed = gdf.to_pandas()
        
        # Clear GPU memory
        del gdf
        cp._default_memory_pool.free_all_blocks()
        
        return df_processed
    
    def gpu_transform_to_long_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """GPU-accelerated transformation to long format for tsfresh"""
        # Use cuDF for fast melting
        gdf = cudf.from_pandas(df)
        
        # Select columns for melting
        id_vars = ['timestamp', 'compartment_id']
        value_vars = [col for col in df.columns if col not in id_vars]
        
        # Melt on GPU
        gdf_long = gdf.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name='variable',
            value_name='value'
        )
        
        # Add ID column for tsfresh
        gdf_long['id'] = gdf_long['compartment_id']
        
        # Convert back to pandas
        df_long = gdf_long.to_pandas()
        
        # Clear GPU memory
        del gdf, gdf_long
        cp._default_memory_pool.free_all_blocks()
        
        return df_long
    
    def gpu_feature_selection(self, features_df: pd.DataFrame, 
                            target_col: str = None) -> pd.DataFrame:
        """GPU-accelerated feature selection using correlation"""
        # Convert to CuPy array for GPU computation
        feature_values = cp.asarray(features_df.values)
        
        # Compute correlation matrix on GPU
        corr_matrix = cp.corrcoef(feature_values.T)
        
        # Find highly correlated features
        upper_tri = cp.triu(cp.ones_like(corr_matrix, dtype=bool), k=1)
        high_corr_pairs = cp.where(
            (cp.abs(corr_matrix) > 0.95) & upper_tri
        )
        
        # Remove highly correlated features
        features_to_drop = set()
        for i, j in zip(high_corr_pairs[0].get(), high_corr_pairs[1].get()):
            # Keep feature with lower mean absolute correlation
            col_i = features_df.columns[i]
            col_j = features_df.columns[j]
            
            mean_corr_i = cp.mean(cp.abs(corr_matrix[i, :])).get()
            mean_corr_j = cp.mean(cp.abs(corr_matrix[j, :])).get()
            
            if mean_corr_i > mean_corr_j:
                features_to_drop.add(col_i)
            else:
                features_to_drop.add(col_j)
        
        # Clear GPU memory
        del feature_values, corr_matrix
        cp._default_memory_pool.free_all_blocks()
        
        # Drop highly correlated features
        selected_features = features_df.drop(columns=list(features_to_drop))
        
        logger.info(f"Selected {len(selected_features.columns)} features "
                   f"(dropped {len(features_to_drop)} correlated features)")
        
        return selected_features
    
    def process_era(self, task_data: Dict) -> Dict[str, Any]:
        """Process a single era with GPU acceleration"""
        try:
            # Fetch era data
            df = self.fetch_era_data(
                task_data['era_id'],
                task_data['compartment_id'],
                task_data['start_time'],
                task_data['end_time']
            )
            
            if df.empty:
                return {
                    'status': 'skipped',
                    'features_extracted': 0,
                    'reason': 'No data found'
                }
            
            # GPU preprocessing
            logger.info(f"GPU preprocessing {len(df)} rows")
            df_processed = self.gpu_preprocess_data(df)
            
            # Transform to long format
            df_long = self.gpu_transform_to_long_format(df_processed)
            
            # Get feature parameters based on environment
            feature_params = FeatureExtractionConfig.get_feature_parameters()
            feature_set = FeatureExtractionConfig.get_feature_set()
            
            # Extract features using tsfresh (CPU-bound)
            logger.info(f"Extracting features with tsfresh ({feature_set} set)")
            features = extract_features(
                df_long,
                column_id='id',
                column_sort='timestamp',
                column_kind='variable',
                column_value='value',
                default_fc_parameters=feature_params,
                **FeatureExtractionConfig.TSFRESH_SETTINGS
            )
            
            # Impute missing values
            features = impute(features)
            
            # GPU-accelerated feature selection
            logger.info("GPU feature selection")
            selected_features = self.gpu_feature_selection(features)
            
            # Transform to long format for storage
            features_long = selected_features.reset_index().melt(
                id_vars=['index'],
                var_name='variable',
                value_name='value'
            )
            
            # Save features
            success = self.save_features(
                features_long,
                task_data['era_id'],
                task_data['compartment_id']
            )
            
            return {
                'status': 'success' if success else 'error',
                'features_extracted': len(features_long),
                'error': None if success else 'Failed to save features'
            }
            
        except Exception as e:
            logger.error(f"Error processing era: {str(e)}")
            return {
                'status': 'error',
                'features_extracted': 0,
                'error': str(e)
            }
        finally:
            # Clear GPU memory
            cp._default_memory_pool.free_all_blocks()


if __name__ == "__main__":
    worker_id = os.environ.get('WORKER_ID', 'gpu-0')
    worker = GPUFeatureWorker(worker_id)
    worker.run()