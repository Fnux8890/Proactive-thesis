"""GPU-accelerated preprocessing for tsfresh feature extraction.
This module handles all data preparation steps using CUDA/CuPy to maximize
performance before and after tsfresh processing.
"""

from typing import Dict, List, Optional, Tuple, Union
import logging
import time

import numpy as np
import pandas as pd

try:
    import cupy as cp
    import cudf
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    logging.warning("GPU libraries not available, will use CPU fallback")
logger = logging.getLogger(__name__)
class GPUDataPreprocessor:
    """Handles GPU-accelerated data preprocessing for tsfresh."""
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        if self.use_gpu:
            logger.info("GPU preprocessing enabled")
            # Set memory pool for better performance
            mempool = cp.get_default_memory_pool()
            mempool.set_limit(size=4 * 1024**3)  # 4GB limit
        else:
            logger.info("GPU not available, using CPU preprocessing")
    def prepare_for_tsfresh(self,
                          df: Union[pd.DataFrame, cudf.DataFrame],
                          column_id: str,
                          column_sort: str,
                          column_kind: str,
                          column_value: str) -> pd.DataFrame:
        """
        Prepare data for tsfresh using GPU acceleration.
        Steps:
        1. Sort data efficiently on GPU
        2. Handle missing values
        3. Normalize if needed
        4. Convert to long format required by tsfresh
        5. Return as pandas DataFrame (required by tsfresh)
        """
        start_time = time.time()
        if self.use_gpu and not isinstance(df, cudf.DataFrame):
            logger.info(f"Converting to cuDF DataFrame (shape: {df.shape})")
            df_gpu = cudf.from_pandas(df)
        elif self.use_gpu:
            df_gpu = df
        else:
            return self._cpu_prepare(df, column_id, column_sort, column_kind, column_value)
        # GPU operations
        logger.info("Sorting data on GPU...")
        df_gpu = df_gpu.sort_values([column_id, column_sort])
        # Handle missing values on GPU
        logger.info("Handling missing values on GPU...")
        df_gpu = self._gpu_handle_missing(df_gpu, column_value)
        # Ensure correct data types
        df_gpu[column_value] = df_gpu[column_value].astype('float32')
        # Convert back to pandas for tsfresh
        logger.info("Converting back to pandas for tsfresh...")
        df_pandas = df_gpu.to_pandas()
        prep_time = time.time() - start_time
        logger.info(f"GPU preprocessing completed in {prep_time:.2f}s")
        return df_pandas
    def _gpu_handle_missing(self, df: cudf.DataFrame, value_column: str) -> cudf.DataFrame:
        """Handle missing values using GPU operations."""
        # Forward fill missing values within each group
        # Note: cuDF supports fillna but group-wise operations might need custom kernels
        if df[value_column].isnull().any():
            df[value_column] = df[value_column].fillna(method='ffill')
            df[value_column] = df[value_column].fillna(method='bfill')
            # Fill any remaining with 0
            df[value_column] = df[value_column].fillna(0)
        return df
    def _cpu_prepare(self, df: pd.DataFrame,
                    column_id: str, column_sort: str,
                    column_kind: str, column_value: str) -> pd.DataFrame:
        """CPU fallback for data preparation."""
        logger.info("Using CPU preprocessing...")
        df = df.sort_values([column_id, column_sort])
        df[column_value] = df[column_value].fillna(method='ffill').fillna(method='bfill').fillna(0)
        return df
    def melt_wide_to_long_gpu(self,
                             df: Union[pd.DataFrame, cudf.DataFrame],
                             id_vars: List[str],
                             value_vars: List[str],
                             var_name: str = 'kind',
                             value_name: str = 'value') -> pd.DataFrame:
        """
        GPU-accelerated melting from wide to long format.
        This is often the most memory-intensive operation before tsfresh.
        """
        start_time = time.time()
        if self.use_gpu:
            if not isinstance(df, cudf.DataFrame):
                df_gpu = cudf.from_pandas(df)
            else:
                df_gpu = df
            logger.info(f"GPU melting {len(value_vars)} columns...")
            # cuDF melt is much faster than pandas for large datasets
            df_long = df_gpu.melt(
                id_vars=id_vars,
                value_vars=value_vars,
                var_name=var_name,
                value_name=value_name
            )
            # Remove rows where value is null
            df_long = df_long[df_long[value_name].notna()]
            # Convert to pandas for tsfresh
            result = df_long.to_pandas()
        else:
            logger.info("CPU melting...")
            result = pd.melt(
                df,
                id_vars=id_vars,
                value_vars=value_vars,
                var_name=var_name,
                value_name=value_name
            )
            result = result[result[value_name].notna()]
        melt_time = time.time() - start_time
        logger.info(f"Melting completed in {melt_time:.2f}s - {len(result):,} rows")
        return result
class GPUFeatureSelector:
    """GPU-accelerated feature selection after tsfresh extraction."""
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.scaler = cuStandardScaler() if self.use_gpu else None
    def select_features_gpu(self,
                           features_df: pd.DataFrame,
                           method: str = 'variance',
                           threshold: float = 0.01,
                           correlation_threshold: float = 0.95) -> pd.DataFrame:
        """
        GPU-accelerated feature selection.
        Methods:
        - variance: Remove low variance features
        - correlation: Remove highly correlated features
        """
        start_time = time.time()
        if not self.use_gpu:
            return self._cpu_select_features(features_df, method, threshold, correlation_threshold)
        # Convert to GPU
        features_gpu = cudf.from_pandas(features_df)
        # Get numeric columns only
        numeric_cols = features_gpu.select_dtypes(include=[np.number]).columns
        if method == 'variance':
            selected_features = self._gpu_variance_selection(
                features_gpu[numeric_cols], threshold
            )
        elif method == 'correlation':
            selected_features = self._gpu_correlation_selection(
                features_gpu[numeric_cols], correlation_threshold
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        # Filter and convert back
        result = features_gpu[selected_features].to_pandas()
        select_time = time.time() - start_time
        logger.info(
            f"GPU feature selection completed in {select_time:.2f}s - "
            f"Selected {len(selected_features)}/{len(numeric_cols)} features"
        )
        return result
    def _gpu_variance_selection(self, df: cudf.DataFrame, threshold: float) -> List[str]:
        """Remove low variance features using GPU."""
        variances = df.var()
        # Normalize by mean to get coefficient of variation
        means = df.mean()
        cv = variances / (means + 1e-10)
        # Select features with CV above threshold
        mask = cv > threshold
        return df.columns[mask.to_pandas()].tolist()
    def _gpu_correlation_selection(self, df: cudf.DataFrame, threshold: float) -> List[str]:
        """Remove highly correlated features using GPU."""
        # Calculate correlation matrix on GPU
        corr_matrix = df.corr()
        # Convert to numpy for processing (small enough after correlation)
        corr_np = corr_matrix.to_numpy()
        # Find features to drop
        upper_tri = np.triu(np.abs(corr_np), k=1)
        to_drop = set()
        for i in range(len(upper_tri)):
            for j in range(i+1, len(upper_tri)):
                if upper_tri[i, j] > threshold:
                    # Drop the feature with lower variance
                    var_i = df.iloc[:, i].var()
                    var_j = df.iloc[:, j].var()
                    if var_i < var_j:
                        to_drop.add(df.columns[i])
                    else:
                        to_drop.add(df.columns[j])
        return [col for col in df.columns if col not in to_drop]
    def _cpu_select_features(self, df: pd.DataFrame, method: str,
                           threshold: float, correlation_threshold: float) -> pd.DataFrame:
        """CPU fallback for feature selection."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if method == 'variance':
            variances = df[numeric_cols].var()
            means = df[numeric_cols].mean()
            cv = variances / (means + 1e-10)
            selected = numeric_cols[cv > threshold].tolist()
        else:  # correlation
            corr_matrix = df[numeric_cols].corr().abs()
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            to_drop = [column for column in upper_tri.columns
                      if any(upper_tri[column] > correlation_threshold)]
            selected = [col for col in numeric_cols if col not in to_drop]
        return df[selected]
class GPUMemoryManager:
    """Manages GPU memory for large-scale feature extraction."""
    def __init__(self):
        if GPU_AVAILABLE:
            self.mempool = cp.get_default_memory_pool()
            self.pinned_mempool = cp.get_default_pinned_memory_pool()
    def clear_memory(self):
        """Clear GPU memory pools."""
        if GPU_AVAILABLE:
            self.mempool.free_all_blocks()
            self.pinned_mempool.free_all_blocks()
            cp.cuda.Stream.null.synchronize()
    def get_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        if not GPU_AVAILABLE:
            return {"available": False}
        return {
            "used_bytes": self.mempool.used_bytes(),
            "total_bytes": self.mempool.total_bytes(),
            "used_gb": self.mempool.used_bytes() / (1024**3),
            "total_gb": self.mempool.total_bytes() / (1024**3),
        }
    def log_memory_usage(self):
        """Log current memory usage."""
        info = self.get_memory_info()
        if info.get("available", True):
            logger.info(
                f"GPU Memory: {info['used_gb']:.2f}GB / {info['total_gb']:.2f}GB "
                f"({info['used_bytes'] / info['total_bytes'] * 100:.1f}% used)"
            )
