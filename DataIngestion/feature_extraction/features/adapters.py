"""
Library boundary adapters for external libraries that only support pandas.

This module provides adapter functions that handle conversions between
cuDF and pandas DataFrames when interfacing with libraries that don't
support GPU dataframes.
"""

import time
import logging
from typing import Dict, Any, Optional, Union, List
import warnings

from ..backend import pd, DataFrame, USE_GPU, BACKEND_TYPE
# Note: ensure_backend is now available in backend module if needed
# from ..backend.backend import ensure_backend

logger = logging.getLogger(__name__)


def tsfresh_extract_features(
    df: DataFrame,
    column_id: Optional[str] = None,
    column_sort: Optional[str] = None,
    column_kind: Optional[str] = None,
    column_value: Optional[str] = None,
    default_fc_parameters: Optional[Dict[str, Any]] = None,
    kind_to_fc_parameters: Optional[Dict[str, Any]] = None,
    n_jobs: int = 1,
    show_warnings: bool = False,
    disable_progressbar: bool = True,
    impute_function: Optional[Any] = None,
    profile: bool = False,
    profiling_filename: Optional[str] = None,
    profiling_sorting: str = "cumulative",
    distributor: Optional[Any] = None
) -> DataFrame:
    """
    Extract features using tsfresh, handling GPU/CPU conversions.
    
    Input type: pandas DataFrame or cuDF DataFrame
    Output type: Same as input type
    
    This adapter handles the conversion to pandas (required by tsfresh)
    and back to the original backend type.
    
    Args:
        df: Input DataFrame (pandas or cuDF)
        column_id: Column with ID for aggregation
        column_sort: Column with sort order
        column_kind: Column with kind information
        column_value: Column with values
        default_fc_parameters: Feature calculation parameters
        kind_to_fc_parameters: Kind-specific parameters
        n_jobs: Number of parallel jobs
        show_warnings: Show tsfresh warnings
        disable_progressbar: Disable progress bar
        impute_function: Function for imputation
        profile: Enable profiling
        profiling_filename: Profiling output file
        profiling_sorting: Profiling sort method
        distributor: Distributor for parallel processing
        
    Returns:
        DataFrame with extracted features (same backend as input)
    """
    from tsfresh import extract_features
    
    # Track conversion overhead
    conversion_time = 0
    extraction_time = 0
    input_was_cudf = False
    
    # Handle GPU->CPU conversion if needed
    # Check if the input is actually a cuDF DataFrame
    if USE_GPU and hasattr(df, 'to_pandas'):
        input_was_cudf = True
        logger.info(f"Converting cuDF DataFrame to pandas for tsfresh (shape: {df.shape})")
        start = time.perf_counter()
        df_pandas = df.to_pandas()
        conversion_time = time.perf_counter() - start
        logger.debug(f"Conversion to pandas took {conversion_time:.2f}s")
    else:
        # Already pandas or doesn't have to_pandas method
        df_pandas = df
    
    # Suppress warnings if requested
    with warnings.catch_warnings():
        if not show_warnings:
            warnings.filterwarnings("ignore")
        
        # Extract features
        start = time.perf_counter()
        features_pandas = extract_features(
            df_pandas,
            column_id=column_id,
            column_sort=column_sort,
            column_kind=column_kind,
            column_value=column_value,
            default_fc_parameters=default_fc_parameters,
            kind_to_fc_parameters=kind_to_fc_parameters,
            n_jobs=n_jobs,
            show_warnings=show_warnings,
            disable_progressbar=disable_progressbar,
            impute_function=impute_function,
            profile=profile,
            profiling_filename=profiling_filename,
            profiling_sorting=profiling_sorting,
            distributor=distributor
        )
        extraction_time = time.perf_counter() - start
    
    logger.info(
        f"tsfresh extracted {features_pandas.shape[1]} features from "
        f"{df.shape[0]} rows in {extraction_time:.2f}s"
    )
    
    # Convert back to match input type
    if input_was_cudf:
        # Only convert back if input was cuDF
        import cudf
        start = time.perf_counter()
        features = cudf.DataFrame.from_pandas(features_pandas)
        back_conversion_time = time.perf_counter() - start
        logger.debug(f"Conversion back to cuDF took {back_conversion_time:.2f}s")
        logger.info(
            f"Total overhead for GPU<->CPU conversion: "
            f"{conversion_time + back_conversion_time:.2f}s"
        )
    else:
        # Keep as pandas if input was pandas
        features = features_pandas
    
    return features


def sklearn_fit_transform(
    estimator: Any,
    X: DataFrame,
    y: Optional[Union[DataFrame, Any]] = None,
    **fit_params
) -> Union[DataFrame, Any]:
    """
    Fit and transform using scikit-learn estimator, handling GPU/CPU conversions.
    
    Input type: pandas DataFrame or cuDF DataFrame
    Output type: Same as input type (or numpy array if estimator returns array)
    
    Args:
        estimator: Scikit-learn estimator with fit_transform method
        X: Input features
        y: Target values (optional)
        **fit_params: Additional parameters for fit
        
    Returns:
        Transformed data (same backend as input if possible)
    """
    import numpy as np
    
    # Check if estimator supports GPU
    gpu_capable = hasattr(estimator, 'device') and estimator.device == 'gpu'
    
    # Track input types
    X_is_cudf = hasattr(X, 'to_pandas')
    y_is_cudf = hasattr(y, 'to_pandas') if y is not None else False
    
    if X_is_cudf and not gpu_capable:
        # Need to convert cuDF to pandas for CPU estimator
        logger.debug(f"Converting cuDF to pandas for sklearn estimator: {type(estimator).__name__}")
        X_cpu = X.to_pandas()
        y_cpu = y.to_pandas() if y_is_cudf else y
        
        result = estimator.fit_transform(X_cpu, y_cpu, **fit_params)
        
        # Convert back to cuDF if input was cuDF and result is DataFrame
        if isinstance(result, pd.DataFrame):
            import cudf
            return cudf.DataFrame.from_pandas(result)
        else:
            # Return numpy array as-is
            return result
    else:
        # Can use directly (either estimator is GPU-capable or data is already pandas)
        return estimator.fit_transform(X, y, **fit_params)


def ensure_pandas(df: DataFrame) -> pd.DataFrame:
    """
    Ensure DataFrame is pandas, converting from cuDF if necessary.
    
    Use this sparingly and document why pandas is required.
    
    Args:
        df: Input DataFrame (pandas or cuDF)
        
    Returns:
        pandas DataFrame
    """
    if hasattr(df, 'to_pandas'):
        logger.warning(f"Explicit conversion to pandas requested (shape: {df.shape})")
        return df.to_pandas()
    # Already pandas or pandas-like
    return df