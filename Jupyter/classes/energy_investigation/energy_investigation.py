#!/usr/bin/env -S uv run --isolated
# /// script
# dependencies = ["pandas", "numpy", "matplotlib", "seaborn", "statsmodels", "ruptures", "pyarrow"]
# ///


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import ruptures as rpt
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Determine paths relative to the script file
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent # Proactive-thesis/

PROCESSED_DATA_DIR = PROJECT_ROOT / "DataIngestion" / "feature_extraction" / "data" / "processed"
OUTPUT_DIR = SCRIPT_DIR / "output_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_all_processed_data(eras: List[str], data_dir: Path = PROCESSED_DATA_DIR) -> pd.DataFrame:
    """
    Loads all processed Parquet segments from specified Eras and concatenates them.
    Focuses on 'spot_price_dkk_mwh' and ensures a sorted DatetimeIndex.
    """
    all_dfs = []
    for era_name in eras:
        era_path = data_dir / era_name
        if era_path.is_dir():
            for segment_file in era_path.glob("*.parquet"):
                try:
                    df_segment = pd.read_parquet(segment_file)
                    if "spot_price_dkk_mwh" in df_segment.columns:
                        # Ensure DatetimeIndex
                        if not isinstance(df_segment.index, pd.DatetimeIndex):
                            # Attempt to convert 'timestamp' or first column to DatetimeIndex
                            if 'timestamp' in df_segment.columns:
                                df_segment['timestamp'] = pd.to_datetime(df_segment['timestamp'])
                                df_segment = df_segment.set_index('timestamp')
                            elif df_segment.columns[0] == 'datetime': # A common name
                                df_segment[df_segment.columns[0]] = pd.to_datetime(df_segment[df_segment.columns[0]])
                                df_segment = df_segment.set_index(df_segment.columns[0])
                            else:
                                print(f"Warning: Could not automatically determine DatetimeIndex for {segment_file}. Skipping.")
                                continue

                        all_dfs.append(df_segment[["spot_price_dkk_mwh"]])
                    else:
                        print(f"Warning: 'spot_price_dkk_mwh' not found in {segment_file}")
                except Exception as e:
                    print(f"Error loading or processing {segment_file}: {e}")
        else:
            print(f"Warning: Era directory not found: {era_path}")

    if not all_dfs:
        print("No data loaded. Please check era names and data paths.")
        return pd.DataFrame(columns=['spot_price_dkk_mwh'])

    full_df = pd.concat(all_dfs)
    full_df = full_df.sort_index()
    
    # Handle duplicates that might arise from overlapping segments by taking the mean
    full_df = full_df.groupby(full_df.index).mean()

    # Create a full date range for the entire period and reindex
    # This helps in identifying implicit gaps if any segment was missing
    if not full_df.empty:
        # Assuming data is at least hourly. If it's 5-min, adjust accordingly.
        # preprocess.py output is 5-min. Energy price data is hourly and ffilled.
        complete_range = pd.date_range(start=full_df.index.min(), end=full_df.index.max(), freq='5min')
        full_df = full_df.reindex(complete_range)

    # Handle Missing Price Data
    full_df["spot_price_dkk_mwh"] = full_df["spot_price_dkk_mwh"].ffill().bfill()
    
    print(f"Loaded data from {full_df.index.min()} to {full_df.index.max()} with {len(full_df)} rows.")
    return full_df


def create_aggregated_series(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Creates aggregated time series for energy prices.
    """
    if "spot_price_dkk_mwh" not in df.columns or df.empty:
        print("Error: 'spot_price_dkk_mwh' not in DataFrame or DataFrame is empty.")
        return {}

    price_series = df["spot_price_dkk_mwh"]
    
    aggregated = {
        "price_hourly": price_series.resample('H').mean().ffill().bfill(), # Resample to hourly mean if original is finer
        "price_daily_mean": price_series.resample('D').mean(),
        "price_daily_max": price_series.resample('D').max(),
        "price_daily_min": price_series.resample('D').min(),
        "price_weekly_mean": price_series.resample('W-MON').mean(), # Weekly, starting Monday
        "price_monthly_mean": price_series.resample('MS').mean() # Monthly, start of month
    }
    return aggregated


def plot_overall_time_series(series_dict: Dict[str, pd.Series], output_dir: Path = OUTPUT_DIR):
    """Plots daily and monthly mean prices."""
    if not series_dict or "price_daily_mean" not in series_dict or "price_monthly_mean" not in series_dict:
        print("Error: Necessary series not found for overall time series plot.")
        return

    plt.figure(figsize=(15, 7))
    series_dict["price_daily_mean"].plot(label="Daily Mean Price", alpha=0.7)
    series_dict["price_monthly_mean"].plot(label="Monthly Mean Price", linewidth=2, color='red')
    plt.title("Energy Spot Price (DKK/MWh) - Daily and Monthly Mean")
    plt.xlabel("Date")
    plt.ylabel("Price (DKK/MWh)")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "overall_price_time_series.png")
    plt.close()
    print(f"Saved overall price time series plot to {output_dir / 'overall_price_time_series.png'}")


def plot_distribution_analysis(price_hourly: pd.Series, output_dir: Path = OUTPUT_DIR):
    """Plots histogram and density plot of hourly prices."""
    if price_hourly.empty:
        print("Error: Hourly price series is empty for distribution analysis.")
        return

    plt.figure(figsize=(12, 6))
    sns.histplot(price_hourly, kde=True, bins=50)
    plt.title("Distribution of Hourly Energy Prices")
    plt.xlabel("Price (DKK/MWh)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(output_dir / "price_distribution.png")
    plt.close()
    print(f"Saved price distribution plot to {output_dir / 'price_distribution.png'}")


def plot_box_plots(price_hourly: pd.Series, price_monthly_mean: pd.Series, output_dir: Path = OUTPUT_DIR):
    """Generates box plots by hour, day of week, and month."""
    if price_hourly.empty or price_monthly_mean.empty:
        print("Error: Price series empty for box plots.")
        return

    # By Hour of Day
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=price_hourly.index.hour, y=price_hourly)
    plt.title("Energy Price by Hour of Day")
    plt.xlabel("Hour of Day")
    plt.ylabel("Price (DKK/MWh)")
    plt.grid(True)
    plt.savefig(output_dir / "price_boxplot_hour_of_day.png")
    plt.close()

    # By Day of Week
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=price_hourly.index.dayofweek, y=price_hourly)
    plt.title("Energy Price by Day of Week")
    plt.xlabel("Day of Week (0=Mon, 6=Sun)")
    plt.ylabel("Price (DKK/MWh)")
    plt.grid(True)
    plt.savefig(output_dir / "price_boxplot_day_of_week.png")
    plt.close()

    # By Month
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=price_monthly_mean.index.month, y=price_monthly_mean)
    plt.title("Monthly Mean Energy Price by Month")
    plt.xlabel("Month")
    plt.ylabel("Price (DKK/MWh)")
    plt.grid(True)
    plt.savefig(output_dir / "price_boxplot_month.png")
    plt.close()
    print(f"Saved box plots to {output_dir}")


def plot_heatmaps(price_hourly: pd.Series, output_dir: Path = OUTPUT_DIR):
    """Generates heatmaps of average price."""
    if price_hourly.empty:
        print("Error: Hourly price series is empty for heatmaps.")
        return

    # Month vs HourOfDay
    try:
        heatmap_data_month_hour = price_hourly.groupby([price_hourly.index.month, price_hourly.index.hour]).mean().unstack()
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data_month_hour, cmap="viridis", annot=False) # Annot can be True for smaller grids
        plt.title("Average Energy Price by Month and Hour of Day")
        plt.xlabel("Hour of Day")
        plt.ylabel("Month")
        plt.savefig(output_dir / "price_heatmap_month_hour.png")
        plt.close()
    except Exception as e:
        print(f"Could not generate Month vs HourOfDay heatmap: {e}")


    # DayOfWeek vs HourOfDay
    try:
        heatmap_data_day_hour = price_hourly.groupby([price_hourly.index.dayofweek, price_hourly.index.hour]).mean().unstack()
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data_day_hour, cmap="viridis", annot=False)
        plt.title("Average Energy Price by Day of Week and Hour of Day")
        plt.xlabel("Hour of Day")
        plt.ylabel("Day of Week (0=Mon, 6=Sun)")
        plt.savefig(output_dir / "price_heatmap_day_hour.png")
        plt.close()
        print(f"Saved heatmaps to {output_dir}")
    except Exception as e:
        print(f"Could not generate DayOfWeek vs HourOfDay heatmap: {e}")


def perform_seasonal_decomposition(series: pd.Series, series_name: str, period: int, output_dir: Path = OUTPUT_DIR):
    """Performs and plots seasonal decomposition."""
    if series.isnull().all() or len(series) < 2 * period : # Check for sufficient data
        print(f"Warning: Not enough data or all NaN for {series_name} with period {period}. Skipping decomposition.")
        return

    try:
        # Fill NaNs that might exist from resampling before decomposition
        series_filled = series.ffill().bfill()
        if series_filled.isnull().all(): # Check again after fill
             print(f"Warning: Series {series_name} is still all NaN after fill. Skipping decomposition.")
             return

        result_add = seasonal_decompose(series_filled, model='additive', period=period, extrapolate_trend='freq')
        result_mul = seasonal_decompose(series_filled, model='multiplicative', period=period, extrapolate_trend='freq')

        fig, axes = plt.subplots(4, 2, figsize=(15, 12), sharex=True)
        
        axes[0,0].set_title(f"Additive Decomposition - {series_name}")
        result_add.observed.plot(ax=axes[0,0], legend=False)
        axes[0,0].set_ylabel('Observed')
        result_add.trend.plot(ax=axes[1,0], legend=False)
        axes[1,0].set_ylabel('Trend')
        result_add.seasonal.plot(ax=axes[2,0], legend=False)
        axes[2,0].set_ylabel('Seasonal')
        result_add.resid.plot(ax=axes[3,0], legend=False)
        axes[3,0].set_ylabel('Residual')

        axes[0,1].set_title(f"Multiplicative Decomposition - {series_name}")
        result_mul.observed.plot(ax=axes[0,1], legend=False)
        axes[0,1].set_ylabel('Observed')
        result_mul.trend.plot(ax=axes[1,1], legend=False)
        axes[1,1].set_ylabel('Trend')
        result_mul.seasonal.plot(ax=axes[2,1], legend=False)
        axes[2,1].set_ylabel('Seasonal')
        result_mul.resid.plot(ax=axes[3,1], legend=False)
        axes[3,1].set_ylabel('Residual')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"seasonal_decomposition_{series_name.replace(' ', '_')}.png")
        plt.close()
        print(f"Saved seasonal decomposition plot for {series_name} to {output_dir}")

    except Exception as e:
        print(f"Error during seasonal decomposition for {series_name}: {e}")


def plot_acf_pacf(series: pd.Series, series_name: str, lags: int, output_dir: Path = OUTPUT_DIR):
    """Plots ACF and PACF."""
    if series.isnull().all() or len(series) <= lags : # Check for sufficient data
        print(f"Warning: Not enough data or all NaN for {series_name} for ACF/PACF. Skipping.")
        return

    series_filled = series.ffill().bfill()
    if series_filled.isnull().all(): # Check again after fill
        print(f"Warning: Series {series_name} is still all NaN after fill for ACF/PACF. Skipping.")
        return

    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(211)
    plot_acf(series_filled, lags=lags, ax=ax1, title=f"Autocorrelation (ACF) - {series_name}")
    ax2 = plt.subplot(212)
    plot_pacf(series_filled, lags=lags, ax=ax2, title=f"Partial Autocorrelation (PACF) - {series_name}")
    plt.tight_layout()
    plt.savefig(output_dir / f"acf_pacf_{series_name.replace(' ', '_')}.png")
    plt.close()
    print(f"Saved ACF/PACF plot for {series_name} to {output_dir}")


def calculate_rolling_volatility(price_series: pd.Series, window_days: List[int]=[7, 30]) -> Dict[str, pd.Series]:
    """Calculates rolling standard deviation."""
    if price_series.empty:
        print("Error: Price series empty for volatility calculation.")
        return {}
    
    # Assuming price_series is at least daily. If hourly, window needs to be adjusted.
    # The plan mentions price_hourly.rolling(window=24*7), so input should be hourly or finer.
    # If price_series is daily, window is just number of days.
    # Let's assume input `price_series` is HOURLY as per the plan's example.
    
    volatility_series = {}
    for days in window_days:
        window_size = 24 * days # For hourly data
        key = f"price_volatility_{days}d"
        volatility_series[key] = price_series.rolling(window=window_size, min_periods=1).std()
    return volatility_series


def plot_rolling_volatility(volatility_dict: Dict[str, pd.Series], output_dir: Path = OUTPUT_DIR):
    """Plots rolling volatility."""
    if not volatility_dict:
        print("Error: No volatility series to plot.")
        return

    plt.figure(figsize=(15, 7))
    for key, series in volatility_dict.items():
        if not series.empty:
            series.plot(label=key.replace("_", " ").title())
    plt.title("Rolling Volatility of Hourly Energy Prices")
    plt.xlabel("Date")
    plt.ylabel("Standard Deviation (DKK/MWh)")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "rolling_volatility.png")
    plt.close()
    print(f"Saved rolling volatility plot to {output_dir}")


def identify_price_regimes(series: pd.Series, series_name: str, n_bkps: int = 5, model: str = "l2", output_dir: Path = OUTPUT_DIR) -> Optional[pd.Series]:
    """Identifies price regimes using changepoint detection."""
    if series.isnull().all() or len(series) < 20: # Need some data for detection
        print(f"Warning: Not enough data or all NaN for {series_name} for regime identification. Skipping.")
        return None
    
    series_filled = series.ffill().bfill()
    if series_filled.isnull().all():
        print(f"Warning: Series {series_name} is still all NaN after fill for regime identification. Skipping.")
        return None

    points = series_filled.values.reshape(-1, 1)
    
    try:
        algo = rpt.Pelt(model=model).fit(points)
        result = algo.predict(n_bkps=n_bkps) # number of changepoints
        
        # Create a series indicating the regime
        regime_series = pd.Series(0, index=series_filled.index, name=f"{series_name}_regime_id")
        current_regime = 0
        bkps = [0] + result # Add start and end
        if bkps[-1] == len(series_filled): # Remove last if it's the end of series
            bkps = bkps[:-1]

        for i in range(len(bkps) -1 ):
            start_idx = series_filled.index[bkps[i]]
            end_idx = series_filled.index[bkps[i+1]-1] # -1 because predict gives the point *after* the change
            regime_series.loc[start_idx:end_idx] = current_regime
            current_regime += 1
        # Handle the last segment
        if bkps[-1] < len(series_filled):
             regime_series.iloc[bkps[-1]:] = current_regime


        plt.figure(figsize=(15, 7))
        series_filled.plot(label=f"Original Series ({series_name})", alpha=0.7)
        
        colors = plt.cm.get_cmap('tab10', current_regime +1) # +1 because regime_id is 0-indexed
        for r_id in range(current_regime + 1):
            segment = series_filled[regime_series == r_id]
            if not segment.empty:
                plt.scatter(segment.index, segment.values, color=colors(r_id), label=f'Regime {r_id}', s=10)

        # Plot changepoints
        for cp_idx in result:
            if cp_idx < len(series_filled.index): # Ensure changepoint is within bounds
                plt.axvline(series_filled.index[cp_idx-1], color='red', linestyle='--', lw=1, label='Changepoint' if cp_idx == result[0] else None)

        plt.title(f"Price Regime Identification for {series_name} (n_bkps={n_bkps})")
        plt.xlabel("Date")
        plt.ylabel("Value")
        handles, labels = plt.gca().get_legend_handles_labels() # To avoid duplicate labels
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.grid(True)
        plt.savefig(output_dir / f"price_regimes_{series_name.replace(' ', '_')}.png")
        plt.close()
        print(f"Saved price regime plot for {series_name} to {output_dir}")
        return regime_series
    except Exception as e:
        print(f"Error during price regime identification for {series_name}: {e}")
        return None


def correlate_with_external_factors(price_series: pd.Series, external_data: pd.DataFrame, factors: List[str], output_dir: Path = OUTPUT_DIR):
    """
    Calculates and prints correlation between price_series (e.g., daily price) 
    and specified external factors from external_data.
    Assumes price_series and external_data are already aligned by DatetimeIndex and aggregated to the same frequency (e.g., daily).
    """
    if price_series.empty or external_data.empty:
        print("Error: Price series or external data is empty for correlation analysis.")
        return

    print("\n--- Correlation Analysis ---")
    for factor in factors:
        if factor in external_data.columns:
            # Ensure both series are aligned and drop NaNs from alignment
            aligned_price, aligned_factor = price_series.align(external_data[factor], join='inner')
            combined = pd.concat([aligned_price.rename("price"), aligned_factor.rename(factor)], axis=1).dropna()
            
            if len(combined) < 2:
                print(f"Not enough overlapping data points for correlation with {factor}.")
                continue

            correlation = combined["price"].corr(combined[factor])
            print(f"Correlation between {price_series.name or 'Price'} and {factor}: {correlation:.2f}")
            
            # Optional: Scatter plot
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=combined[factor], y=combined["price"])
            plt.title(f"Scatter Plot: {price_series.name or 'Price'} vs {factor}\nCorrelation: {correlation:.2f}")
            plt.xlabel(factor)
            plt.ylabel(price_series.name or 'Price')
            plt.grid(True)
            plt.savefig(output_dir / f"correlation_price_vs_{factor}.png")
            plt.close()
        else:
            print(f"Warning: Factor '{factor}' not found in external data.")
    print(f"Saved correlation plots (if any) to {output_dir}")


def engineer_features(price_series_dict: Dict[str, pd.Series], volatility_dict: Dict[str, pd.Series], regime_series: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Engineers new features based on the analysis.
    Returns a DataFrame with these features, indexed like the original fine-grained data.
    """
    print("\n--- Feature Engineering ---")
    if "price_hourly" not in price_series_dict or price_series_dict["price_hourly"].empty:
        print("Error: 'price_hourly' required for feature engineering base.")
        return pd.DataFrame()

    base_series = price_series_dict["price_hourly"] # Use hourly as base for index
    features_df = pd.DataFrame(index=base_series.index)

    # 1. Seasonal Flags (Example: Winter based on months)
    # This is a simple example. More sophisticated seasonality could come from decomposition.
    features_df['is_winter_price_season'] = base_series.index.month.isin([11, 12, 1, 2]).astype(int)

    # 2. Time-of-Day Flags (Example: Peak hours 8-10 and 17-19)
    # This should be based on box plots/heatmaps. Adjust hours as per your findings.
    peak_hours = list(range(8, 11)) + list(range(17, 20)) # Example
    features_df['is_peak_price_hour'] = base_series.index.hour.isin(peak_hours).astype(int)

    # 3. Price Volatility (already calculated, need to align and forward fill)
    for key, vol_series in volatility_dict.items():
        if not vol_series.empty:
            # Volatility is usually calculated on hourly, so index should match or be resample-able
            features_df[key] = vol_series.reindex(features_df.index).ffill().bfill()

    # 4. Lagged Prices (using hourly price for finer lags, or daily for daily lags)
    features_df['price_lag_1hour'] = base_series.shift(1)
    features_df['price_lag_24hour'] = base_series.shift(24) # Lag 1 day (if hourly)
    if "price_daily_mean" in price_series_dict:
        daily_prices = price_series_dict["price_daily_mean"].reindex(features_df.index, method='ffill')
        features_df['price_lag_1day_avg'] = daily_prices.shift(24) # This needs care if base_series is not daily

    # 5. Price Relative to Rolling Mean (e.g., monthly)
    if "price_monthly_mean" in price_series_dict:
        monthly_mean_ffilled = price_series_dict["price_monthly_mean"].reindex(features_df.index, method='ffill')
        features_df['price_deviation_from_monthly_mean'] = base_series - monthly_mean_ffilled
    
    # 6. Price Regime ID
    if regime_series is not None and not regime_series.empty:
        # Regime series might be daily, reindex and ffill to hourly
        features_df[regime_series.name] = regime_series.reindex(features_df.index, method='ffill')

    print(f"Engineered features: {features_df.columns.tolist()}")
    print(features_df.head())
    # features_df.to_csv(OUTPUT_DIR / "engineered_price_features.csv") # Optional: save
    return features_df.dropna(how='all') # Drop rows that are all NaN if any from shifts


def main():
    """Main function to run the energy price investigation."""
    # Specify eras to load data from
    # Example: list all subdirectories in PROCESSED_DATA_DIR
    eras = [d.name for d in PROCESSED_DATA_DIR.iterdir() if d.is_dir()]
    # Or specify manually: eras = ["april2014", "august2014", "your_era3_output"] 
    
    print(f"Found eras: {eras}")
    if not eras:
        print(f"No era subdirectories found in {PROCESSED_DATA_DIR}. Please check the path or create era data.")
        return

    # 1. Load and Prepare Data
    full_price_data = load_all_processed_data(eras=eras)
    if full_price_data.empty or "spot_price_dkk_mwh" not in full_price_data.columns:
        print("Failed to load or process energy price data. Exiting.")
        return
    
    aggregated_series = create_aggregated_series(full_price_data)
    if not aggregated_series:
        print("Failed to create aggregated series. Exiting.")
        return

    price_hourly = aggregated_series.get("price_hourly")
    price_daily_mean = aggregated_series.get("price_daily_mean")
    price_monthly_mean = aggregated_series.get("price_monthly_mean")

    if price_hourly is None or price_daily_mean is None or price_monthly_mean is None:
        print("One or more essential aggregated series are missing. Exiting.")
        return
    
    # 2. EDA & Visualization
    plot_overall_time_series(aggregated_series)
    plot_distribution_analysis(price_hourly)
    plot_box_plots(price_hourly, price_monthly_mean) # Using monthly for the month box plot
    plot_heatmaps(price_hourly)

    # 3. Seasonality and Trend Confirmation
    perform_seasonal_decomposition(price_daily_mean.dropna(), "Daily Mean Price", period=365) # Dropna for decomposition
    perform_seasonal_decomposition(price_monthly_mean.dropna(), "Monthly Mean Price", period=12)
    
    plot_acf_pacf(price_daily_mean.dropna(), "Daily Mean Price", lags=50) # Lags for daily, e.g. ~1.5 months
    plot_acf_pacf(price_monthly_mean.dropna(), "Monthly Mean Price", lags=24) # Lags for monthly, e.g. 2 years

    # 4. Volatility Analysis
    # Using hourly data for rolling volatility as per plan example
    volatility_dict = calculate_rolling_volatility(price_hourly, window_days=[7, 30, 90])
    plot_rolling_volatility(volatility_dict)

    # 5. Price Regime Identification (Optional Advanced)
    # Using daily mean prices for regime identification to smooth out noise
    regime_series_daily = identify_price_regimes(price_daily_mean.dropna(), "Daily Mean Price", n_bkps=5) # Trying 5 breakpoints

    # 6. Correlation with External Factors (Illustrative - requires external data loading)
    # This section is a placeholder. You would need to load your weather data similarly to price data.
    # For example, if you have `full_weather_data_daily_avg` DataFrame:
    # external_factors_to_correlate = ["outside_temp_c", "radiation_w_m2"]
    # print("\nNote: Correlation analysis with external factors is illustrative and needs actual weather data loading.")
    # # dummy_weather_data = pd.DataFrame(index=price_daily_mean.index, 
    # #                                   data={'outside_temp_c': np.random.rand(len(price_daily_mean))*20,
    # #                                         'radiation_w_m2': np.random.rand(len(price_daily_mean))*500})
    # # correlate_with_external_factors(price_daily_mean, dummy_weather_data, external_factors_to_correlate)


    # 7. Feature Engineering
    engineered_features_df = engineer_features(aggregated_series, volatility_dict, regime_series_daily)
    if not engineered_features_df.empty:
        print("\nSuccessfully engineered features.")
        # You might want to save these features or merge them back.
        # For example: engineered_features_df.to_parquet(OUTPUT_DIR / "engineered_energy_price_features.parquet")

    print("\nEnergy price investigation complete. Plots and results are in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
