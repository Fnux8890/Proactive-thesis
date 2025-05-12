#!/usr/bin/env -S uv run --isolated
# /// script
# dependencies = ["pandas", "numpy", "scikit-learn", "matplotlib", "seaborn"]
# ///

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
import os

class ImputationHandler:
    """A class to handle various data imputation strategies."""

    def __init__(self):
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)

    def impute_mean(self, series: pd.Series) -> pd.Series:
        """Imputes missing values using the mean of the series."""
        return series.fillna(series.mean())

    def impute_median(self, series: pd.Series) -> pd.Series:
        """Imputes missing values using the median of the series."""
        return series.fillna(series.median())

    def impute_mode(self, series: pd.Series) -> pd.Series:
        """Imputes missing values using the mode of the series."""
        # Mode can return multiple values if they have the same frequency
        # We'll take the first one.
        mode_val = series.mode()
        return series.fillna(mode_val[0] if not mode_val.empty else np.nan)

    def impute_forward_fill(self, series: pd.Series) -> pd.Series:
        """Imputes missing values using forward fill (ffill)."""
        return series.ffill()

    def impute_backward_fill(self, series: pd.Series) -> pd.Series:
        """Imputes missing values using backward fill (bfill)."""
        return series.bfill()

    def impute_linear_interpolation(self, series: pd.Series) -> pd.Series:
        """Imputes missing values using linear interpolation."""
        return series.interpolate(method='linear')

    def impute_time_interpolation(self, series: pd.Series) -> pd.Series:
        """Imputes missing values using time-based interpolation (requires datetime index)."""
        if not isinstance(series.index, pd.DatetimeIndex):
            print("Warning: Time interpolation requires a DatetimeIndex. Falling back to linear.")
            return series.interpolate(method='linear')
        return series.interpolate(method='time')

    def impute_knn(self, df: pd.DataFrame, column_to_impute: str, n_neighbors: int = 5) -> pd.DataFrame:
        """Imputes missing values in a specific column of a DataFrame using KNNImputer."""
        if column_to_impute not in df.columns:
            raise ValueError(f"Column '{column_to_impute}' not found in DataFrame.")
        
        df_copy = df.copy()
        # KNNImputer works best with all numeric features for distance calculation
        # For simplicity, we'll use only numeric columns. A more robust approach might involve encoding non-numeric ones.
        numeric_cols = df_copy.select_dtypes(include=np.number).columns
        if not numeric_cols.any():
            print(f"Warning: No numeric columns found for KNN imputation of '{column_to_impute}'. Returning original.")
            return df_copy

        imputer = KNNImputer(n_neighbors=n_neighbors)
        df_copy[numeric_cols] = imputer.fit_transform(df_copy[numeric_cols])
        return df_copy
    
    def impute_simple(self, series: pd.Series, strategy: str = 'mean') -> pd.Series:
        """Imputes using sklearn's SimpleImputer (mean, median, most_frequent, constant)."""
        imputer = SimpleImputer(strategy=strategy)
        return pd.Series(imputer.fit_transform(series.to_frame())[:,0], index=series.index, name=series.name)


if __name__ == '__main__':
    print("Testing ImputationHandler class...")
    handler = ImputationHandler()

    # Create an output directory for images if it doesn't exist
    script_dir = os.path.dirname(__file__)
    output_image_dir = os.path.join(script_dir, "images")
    os.makedirs(output_image_dir, exist_ok=True)
    print(f"Saving plots to: {output_image_dir}")

    # Sample data with NaNs
    data = {
        'A': [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10],
        'B': [np.nan, 15, 16, 17, np.nan, 19, 20, np.nan, 22, 23],
        'C': [30, np.nan, np.nan, 33, 34, 35, 36, 37, 38, np.nan],
        'D_categorical': ['x', 'y', 'x', np.nan, 'y', 'z', 'x', 'y', np.nan, 'z']
    }
    # Using a date range more representative of the project data for the example
    df_sample = pd.DataFrame(data)
    df_sample.index = pd.date_range(start='2014-05-01', periods=len(df_sample), freq='D')

    print("\nOriginal DataFrame:")
    print(df_sample)

    # --- Test individual series methods ---
    print("\n--- Imputing Series 'A' ---")
    series_a = df_sample['A'].copy()
    
    imputed_methods = {
        "Mean": handler.impute_mean(series_a),
        "Median": handler.impute_median(series_a),
        "Mode (Series D)": handler.impute_mode(df_sample['D_categorical'].copy()), # Mode is better for categorical
        "Forward Fill": handler.impute_forward_fill(series_a),
        "Backward Fill": handler.impute_backward_fill(series_a),
        "Linear Interpolate": handler.impute_linear_interpolation(series_a),
        "Time Interpolate": handler.impute_time_interpolation(series_a),
        "SimpleImputer (median)": handler.impute_simple(series_a, strategy='median')
    }

    for name, imputed_series in imputed_methods.items():
        print(f"\n{name} Imputed Series A (or D for Mode):")
        print(imputed_series)

    # --- Visualize imputation for a time series (Series A) ---
    plt.figure(figsize=(15, 10))
    plt.plot(series_a.index, series_a, 'o-', label='Original Series A', color='black', alpha=0.7, ms=8)
    
    plot_colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink']
    for i, (name, imputed_series) in enumerate(imputed_methods.items()):
        if name.startswith("Mode") : continue # Don't plot mode of categorical here
        if name.startswith("SimpleImputer"): # Give it a distinct marker for clarity
             plt.plot(imputed_series.index, imputed_series, '.--', label=name, color=plot_colors[i % len(plot_colors)], alpha=0.9)
        else:
            plt.plot(imputed_series.index, imputed_series, 'x--', label=name, color=plot_colors[i % len(plot_colors)], alpha=0.7)
    
    plt.title("Comparison of Imputation Methods for Series A")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    
    plot_path_series_a = os.path.join(output_image_dir, "imputation_comparison_series_a.png")
    plt.savefig(plot_path_series_a)
    print(f"\nSeries A imputation comparison plot saved to {plot_path_series_a}")
    plt.show()
    plt.close()

    # --- Test KNN Imputation (on a copy of the DataFrame) ---
    print("\n--- KNN Imputation on DataFrame (column B) ---")
    df_for_knn = df_sample.copy()
    # KNN needs numeric for other columns too, let's ffill/bfill other NaNs for a cleaner KNN run
    # This is just for demonstration; in practice, handle other columns appropriately
    for col in df_for_knn.columns:
        if col != 'B' and df_for_knn[col].isnull().any() and pd.api.types.is_numeric_dtype(df_for_knn[col]):
            df_for_knn[col] = df_for_knn[col].ffill().bfill()
    
    df_knn_imputed = handler.impute_knn(df_for_knn.drop(columns=['D_categorical']), column_to_impute='B', n_neighbors=3)
    print("Original Series B:")
    print(df_sample['B'])
    print("KNN Imputed Series B:")
    print(df_knn_imputed['B'])

    # Visualize KNN imputation for Series B
    plt.figure(figsize=(12, 6))
    plt.plot(df_sample.index, df_sample['B'], 'o-', label='Original Series B', color='black', alpha=0.7, ms=8)
    plt.plot(df_knn_imputed.index, df_knn_imputed['B'], 'x--', label='KNN Imputed (k=3)', color='red')
    plt.title("KNN Imputation for Series B")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plot_path_knn = os.path.join(output_image_dir, "knn_imputation_series_b.png")
    plt.savefig(plot_path_knn)
    print(f"KNN imputation plot saved to {plot_path_knn}")
    plt.show()
    plt.close()

    print("\nImputation testing complete.") 