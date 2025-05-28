# data_cleaning.py
import polars as pl
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple

# Assuming config models are accessible or passed appropriately
# from .config import OutlierRule, ImputationStrategy 

logger = logging.getLogger(__name__)

# --- Outlier Identification Helpers (Polars) ---

def identify_iqr_outliers_pl(
    df: pl.DataFrame, 
    column: str, 
    factor: float = 1.5
) -> Tuple[pl.Expr, Optional[float], Optional[float]]:
    """Creates a Polars expression for identifying outliers using IQR.
    Returns the boolean expression and calculated lower/upper bounds.
    """
    q1_expr = pl.col(column).quantile(0.25)
    q3_expr = pl.col(column).quantile(0.75)
    
    # Calculate bounds within an aggregation context to handle potential missing quantiles
    bounds = df.select([
        q1_expr.alias("q1"),
        q3_expr.alias("q3")
    ])
    
    q1 = bounds["q1"][0]
    q3 = bounds["q3"][0]

    if q1 is None or q3 is None:
        logger.warning(f"IQR bounds calculation failed for '{column}' (likely due to insufficient non-null data). Returning no outliers.")
        return pl.lit(False), None, None # No outliers if bounds can't be calculated

    iqr_value = q3 - q1
    lower_bound = q1 - factor * iqr_value
    upper_bound = q3 + factor * iqr_value

    outlier_expr = (pl.col(column) < lower_bound) | (pl.col(column) > upper_bound)
    return outlier_expr, lower_bound, upper_bound

def identify_rolling_zscore_outliers_pl(
    column: str, 
    window_size_str: str, # Polars duration string, e.g., "1d", "12h"
    threshold: float = 3.0,
    min_periods: Optional[int] = 1,
    time_col: str = "time"
) -> pl.Expr:
    """Creates a Polars expression for identifying outliers using rolling Z-scores."""
    if not min_periods: # Default Polars rolling needs min_periods if window is duration
        min_periods = 1 # Default to 1 if not specified for time-based rolling
        
    rolling_mean_expr = pl.col(column).rolling(index_column=time_col, period=window_size_str, min_periods=min_periods).mean()
    rolling_std_expr = pl.col(column).rolling(index_column=time_col, period=window_size_str, min_periods=min_periods).std()

    # Handle potential zero standard deviation - replace with null before division
    rolling_std_safe_expr = pl.when(rolling_std_expr != 0).then(rolling_std_expr).otherwise(None)
    
    z_score_expr = (pl.col(column) - rolling_mean_expr) / rolling_std_safe_expr
    
    outlier_expr = z_score_expr.abs() > threshold
    # Fill nulls resulting from calculation (e.g., division by null std) with False (not an outlier)
    return outlier_expr.fill_null(False) 

# --- Outlier Handling Function (Polars) ---

def apply_outlier_treatment_pl(df: pl.DataFrame, outlier_configs: List[Dict[str, Any]], time_col: str = "time") -> pl.DataFrame:
    """Applies various outlier treatments to specified columns in a Polars DataFrame.
    
    Args:
        df: Input Polars DataFrame.
        outlier_configs: List of rule dictionaries, e.g., from DataProcessingConfig.outlier_detection.rules.
                         Each dict should contain 'column', 'method', 'params', 'handling_strategy', etc.
        time_col: Name of the time column, needed for rolling Z-score.

    Returns:
        Polars DataFrame with outliers handled.
    """
    if not outlier_configs:
        logger.info("No outlier rules provided, returning original DataFrame.")
        return df

    df_cleaned = df.lazy() # Operate lazily for efficiency

    for cfg in outlier_configs:
        col = cfg.get("column")
        method = cfg.get("method")
        params = cfg.get("params", {})
        strategy = cfg.get("handling_strategy", "to_nan")
        clip_bounds_cfg = cfg.get("clip_bounds") # e.g., [0, 100]

        if not col or not method:
            logger.warning(f"Skipping invalid outlier config: {cfg}")
            continue
        if col not in df.columns:
            logger.warning(f"Column '{col}' for outlier treatment not found in DataFrame. Skipping.")
            continue
            
        # Check if column is numeric-like before proceeding
        if not df[col].dtype.is_numeric():
            logger.warning(f"Column '{col}' is not numeric ({df[col].dtype}). Skipping outlier treatment.")
            continue
            
        logger.info(f"Applying outlier rule for column '{col}': method='{method}', strategy='{strategy}'")

        outlier_expr = pl.lit(False) # Default: no outliers
        lower_bound_iqr: Optional[float] = None
        upper_bound_iqr: Optional[float] = None

        try:
            if method == 'iqr':
                factor = params.get('factor', 1.5)
                outlier_expr_temp, lower_bound_iqr, upper_bound_iqr = identify_iqr_outliers_pl(df, col, factor)
                outlier_expr = outlier_expr_temp

            elif method == 'zscore_rolling':
                window_minutes = params.get('window_size', 1440) # Default to 1 day in minutes? Config needs clarity
                window_str = f"{window_minutes}m" # Assume window_size is in minutes
                threshold = params.get('threshold', 3.0)
                min_p = params.get('min_periods', 1)
                outlier_expr = identify_rolling_zscore_outliers_pl(col, window_str, threshold, min_p, time_col)
            
            elif method == 'domain':
                lower = params.get('lower_bound')
                upper = params.get('upper_bound')
                domain_expr = pl.lit(False)
                if lower is not None:
                    domain_expr = domain_expr | (pl.col(col) < lower)
                if upper is not None:
                    domain_expr = domain_expr | (pl.col(col) > upper)
                outlier_expr = domain_expr
            else:
                logger.warning(f"Unknown outlier detection method: '{method}' for column '{col}'. Skipping.")
                continue

            # Apply handling strategy within with_columns using the outlier expression
            if strategy == 'to_nan':
                df_cleaned = df_cleaned.with_columns(
                    pl.when(outlier_expr).then(None).otherwise(pl.col(col)).alias(col)
                )

            elif strategy == 'clip':
                lower_clip: Optional[float] = None
                upper_clip: Optional[float] = None

                if method == 'iqr' and lower_bound_iqr is not None and upper_bound_iqr is not None:
                    lower_clip = lower_bound_iqr
                    upper_clip = upper_bound_iqr
                elif method == 'domain':
                     lower_clip = params.get('lower_bound')
                     upper_clip = params.get('upper_bound')
                elif clip_bounds_cfg and isinstance(clip_bounds_cfg, (list, tuple)) and len(clip_bounds_cfg) == 2:
                    lower_clip = clip_bounds_cfg[0]
                    upper_clip = clip_bounds_cfg[1]
                else:
                    logger.warning(f"Clip strategy requested for '{col}' method '{method}', but valid bounds not determined (from IQR/Domain) or provided via 'clip_bounds'. Clipping skipped.")
                    continue # Skip clipping if no bounds

                # Validate bounds if both are set
                if lower_clip is not None and upper_clip is not None and lower_clip > upper_clip:
                    logger.warning(f"Clip bounds for '{col}' are reversed ({lower_clip}, {upper_clip}). Clipping skipped.")
                    continue
                
                logger.info(f"  -> Clipping '{col}' to bounds: lower={lower_clip}, upper={upper_clip}")
                df_cleaned = df_cleaned.with_columns(
                    pl.col(col).clip(lower_bound=lower_clip, upper_bound=upper_clip).alias(col)
                )
            else:
                 logger.warning(f"Unknown handling strategy: '{strategy}' for column '{col}'. Skipping handling.")

        except Exception as e:
             logger.exception(f"Error applying outlier rule {cfg} for column '{col}': {e}. Skipping rule.")

    return df_cleaned.collect() # Collect the result after applying all rules lazily


# --- Imputation Function (Polars) ---

def impute_missing_data_pl(
    df: pl.DataFrame, 
    imputation_strategies: Dict[str, Dict[str, Any]], # e.g., {"col_A": {"method": "linear"}, "col_B": {"method": "mean"}}
    default_strategy: Optional[Dict[str, Any]] = None, # e.g., {"method": "forward_fill"}
    time_col: str = "time"
) -> pl.DataFrame:
    """Imputes missing data in a Polars DataFrame based on column-specific or default strategies."""
    
    df_imputed_lazy = df.lazy()
    
    if default_strategy is None:
        default_strategy = {"method": "linear"} # Use linear as default
        logger.info(f"No default imputation strategy provided, using: {default_strategy}")

    processed_cols = set()

    for column, col_config in imputation_strategies.items():
        if column not in df.columns:
            logger.warning(f"Column '{column}' specified for imputation not found in DataFrame. Skipping.")
            continue
        
        if df[column].null_count() == 0: # Skip if no nulls
            processed_cols.add(column)
            continue

        method = col_config.get("method")
        params = {k: v for k, v in col_config.items() if k != 'method'} # e.g., limit
        
        if not method:
             logger.warning(f"No imputation method specified for column '{column}'. Using default: {default_strategy['method']}")
             method = default_strategy['method']
             params = {k: v for k, v in default_strategy.items() if k != 'method'}
            
        logger.debug(f"Preparing imputation for column '{column}' using method '{method}' with params {params}")
        
        imputation_expr = pl.col(column) # Start with original column expression
        
        try:
            if method == 'linear' or method == 'interpolate':
                 imputation_expr = imputation_expr.interpolate()
            elif method == 'time':
                 if df[time_col].dtype.is_temporal():
                     logger.warning("Polars interpolate assumes linear for time series gaps unless using 'over' context. Applying standard interpolate.")
                     imputation_expr = imputation_expr.interpolate()
                 else:
                     logger.warning(f"Cannot use 'time' interpolation logic for '{column}' as time column '{time_col}' is not temporal. Falling back to linear.")
                     imputation_expr = imputation_expr.interpolate()
            elif method == 'mean':
                 imputation_expr = imputation_expr.fill_null(pl.mean(column))
            elif method == 'median':
                 imputation_expr = imputation_expr.fill_null(pl.median(column))
            elif method == 'mode':
                 imputation_expr = imputation_expr.fill_null(pl.col(column).mode().first())
            elif method == 'ffill' or method == 'forward_fill':
                 limit = params.get('limit')
                 imputation_expr = imputation_expr.forward_fill(limit=limit)
            elif method == 'bfill' or method == 'backward_fill':
                 limit = params.get('limit')
                 imputation_expr = imputation_expr.backward_fill(limit=limit)
            elif method == 'zero' or method == 'constant_zero':
                  imputation_expr = imputation_expr.fill_null(0)
            else:
                 logger.warning(f"Unknown imputation method: '{method}' for column '{column}'. Column not imputed by this strategy.")
                 processed_cols.add(column)
                 continue

            df_imputed_lazy = df_imputed_lazy.with_columns(imputation_expr.alias(column))
            processed_cols.add(column)
            logger.info(f"Applied imputation method '{method}' to column '{column}'.")

        except Exception as e:
             logger.exception(f"Error applying imputation method '{method}' to column '{column}': {e}. Skipping imputation for this column.")
             processed_cols.add(column) 


    # Apply default strategy to remaining columns with nulls
    remaining_cols = [col for col in df.columns if col not in processed_cols and df[col].null_count() > 0]
    if remaining_cols:
        default_method = default_strategy.get('method', 'linear') 
        default_params = {k: v for k, v in default_strategy.items() if k != 'method'}
        logger.info(f"Applying default imputation strategy '{default_method}' to remaining columns with nulls: {remaining_cols}")

        for column in remaining_cols:
             logger.debug(f"Applying default imputation '{default_method}' to column '{column}'.")
             imputation_expr = pl.col(column)
             try:
                 if default_method == 'linear' or default_method == 'interpolate':
                      imputation_expr = imputation_expr.interpolate()
                 elif default_method == 'time':
                      if df[time_col].dtype.is_temporal():
                          imputation_expr = imputation_expr.interpolate()
                      else:
                           imputation_expr = imputation_expr.interpolate()
                 elif default_method == 'mean':
                      imputation_expr = imputation_expr.fill_null(pl.mean(column))
                 elif default_method == 'median':
                      imputation_expr = imputation_expr.fill_null(pl.median(column))
                 elif default_method == 'mode':
                      imputation_expr = imputation_expr.fill_null(pl.col(column).mode().first())
                 elif default_method == 'ffill' or default_method == 'forward_fill':
                      limit = default_params.get('limit')
                      imputation_expr = imputation_expr.forward_fill(limit=limit)
                 elif default_method == 'bfill' or default_method == 'backward_fill':
                      limit = default_params.get('limit')
                      imputation_expr = imputation_expr.backward_fill(limit=limit)
                 elif default_method == 'zero' or default_method == 'constant_zero':
                       imputation_expr = imputation_expr.fill_null(0)
                 else:
                      logger.warning(f"Unknown default imputation method: '{default_method}' for column '{column}'. Skipping.")
                      continue
                 
                 df_imputed_lazy = df_imputed_lazy.with_columns(imputation_expr.alias(column))

             except Exception as e:
                  logger.exception(f"Error applying default imputation method '{default_method}' to column '{column}': {e}. Skipping.")

    df_imputed = df_imputed_lazy.collect()

    # Final check for remaining NaNs
    final_nans = df_imputed.null_count().sum(axis=1)[0] # Sum across columns
    if final_nans > 0:
         cols_with_nans = df_imputed.select(pl.all().is_null().sum()).transpose(include_header=True, header_name="column", column_names=["null_count"]).filter(pl.col("null_count") > 0)
         logger.warning(f"NaNs remaining after all imputation attempts ({final_nans} total): \n{cols_with_nans}")
    else:
         logger.info("Imputation complete. No NaNs remaining.")

    return df_imputed 