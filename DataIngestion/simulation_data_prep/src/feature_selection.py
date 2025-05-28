# feature_selection.py
import pandas as pd
import numpy as np # For np.triu in inter-feature correlation example
# from sklearn.feature_selection import VarianceThreshold
# from sklearn.ensemble import RandomForestRegressor # For model-based selection example
# from . import config # To access selection thresholds

# TODO: Function select_by_variance(df, threshold) -> pd.DataFrame.
# Example:
# def select_by_variance(df_features: pd.DataFrame, threshold: float) -> list:
#     """Selects features with variance above a certain threshold."""
#     selector = VarianceThreshold(threshold=threshold)
#     try:
#         selector.fit(df_features)
#         return df_features.columns[selector.get_support(indices=True)].tolist()
#     except ValueError as e:
#         print(f"Error in VarianceThreshold: {e}. Check for NaNs or all-zero columns.")
#         return df_features.columns.tolist() # Return all columns if error

# TODO: Function select_by_correlation(df_features, series_target, threshold) -> list_of_selected_cols.
# This can be split into two parts: correlation with target, and inter-feature correlation.

# Example for correlation with target:
# def select_by_correlation_with_target(df_features: pd.DataFrame, target_series: pd.Series, threshold: float) -> list:
#     """Selects features based on their absolute correlation with the target variable."""
#     if target_series.name not in df_features.columns:
#         # Temporarily add target to df for corrwith, then remove its own correlation
#         temp_df = pd.concat([df_features, target_series], axis=1)
#         correlations = temp_df.corrwith(target_series).abs()
#         correlations = correlations.drop(target_series.name, errors='ignore') # remove target's self-correlation
#     else: # if target is already a column in df_features (e.g. before splitting X and y)
#         correlations = df_features.corrwith(target_series).abs()
#         correlations = correlations.drop(target_series.name, errors='ignore')
#
#     return correlations[correlations > threshold].index.tolist()

# Example for removing highly correlated features (multicollinearity):
# def remove_highly_correlated_features(df_features: pd.DataFrame, threshold: float) -> list:
#     """Removes one of two features that are correlated above a certain threshold."""
#     corr_matrix = df_features.corr().abs()
#     upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
#     to_drop = set() # Use a set to avoid duplicates
#     for column in upper.columns:
#         if any(upper[column] > threshold):
#             # Try to keep the one more correlated with the (potential) target, if info is available
#             # This part is heuristic or needs target correlation as input to make a better choice.
#             # For simplicity here, we just drop the current 'column' if it forms a pair.
#             # A more sophisticated approach would compare which of the pair has lower avg correlation with others,
#             # or higher correlation with a known target.
#             to_drop.add(column)
#     selected_cols = [col for col in df_features.columns if col not in to_drop]
#     return selected_cols

# TODO: Function select_by_model_importance(df_features, series_target, model_type='rf', top_n) -> list_of_selected_cols.
# Example:
# def select_by_model_importance(df_features: pd.DataFrame, target_series: pd.Series, model_type: str = 'rf', top_n: int = 20) -> list:
#     """Selects top N features based on importance from a model (e.g., RandomForest)."""
#     # Ensure no NaNs in data for model fitting, or use a model that handles them
#     df_features_filled = df_features.fillna(df_features.mean()) # Example: mean imputation
#     target_series_filled = target_series.fillna(target_series.mean())
#
#     if model_type == 'rf':
#         # Ensure RandomForestRegressor is imported
#         # from sklearn.ensemble import RandomForestRegressor
#         model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
#     # elif model_type == 'lightgbm': # Example for another model
#         # from lightgbm import LGBMRegressor
#         # model = LGBMRegressor(random_state=42, n_jobs=-1)
#     else:
#         raise ValueError(f"Unsupported model type: {model_type} for feature importance.")
#
#     try:
#         model.fit(df_features_filled, target_series_filled)
#         importances = pd.Series(model.feature_importances_, index=df_features_filled.columns)
#         return importances.nlargest(top_n).index.tolist()
#     except Exception as e:
#         print(f"Error during model-based feature selection: {e}")
#         # Fallback: return top N by absolute correlation with target if model fails
#         if not df_features_filled.empty:
#             return select_by_correlation_with_target(df_features_filled, target_series_filled, threshold=0.001)[:top_n]
#         return [] 