# Chapter 2: Time Regularization and Resampling

This chapter details the critical process of transforming the initially cleaned time-series data (from Chapter 1) into a format with a regular, consistent time frequency. This step, often involving upsampling or downsampling, is essential for most time-series analysis techniques, feature engineering for sequence-based models (LSTMs, GANs), and reliable imputation of missing values.

## 2.1. Objectives

* Ensure every Era-specific dataset has a `DatetimeIndex` with a predefined, regular frequency (e.g., every 1 minute, 5 minutes, or 1 hour).
* Address sparse data by **upsampling** to the target frequency, which involves creating new time slots and initially populating corresponding sensor readings as `NaN` if no original data point exists at that exact new timestamp.
* Provide a mechanism for **downsampling** if the original data frequency is higher than required for modeling, using appropriate aggregation methods.
* Prepare the data for robust imputation (Chapter 4) by ensuring all expected time slots are present in the index.

## 2.2. Importance of a Regular Time Index

* **Model Requirements:** Many time-series models, including LSTMs and GANs like DoppelGANger that operate on fixed-length sequences, assume or perform best with data sampled at regular intervals. Irregular timestamps complicate sequence generation and pattern learning.
* **Feature Engineering:** Calculating lagged features, rolling window statistics, and other time-dependent features becomes straightforward and unambiguous with a regular time index.
* **Imputation:** Consistent time steps allow for more reliable application of imputation techniques like linear interpolation or forward/backward fill with defined limits.
* **Comparability:** Ensures that data points across different time series or segments are comparable based on their temporal position.

## 2.3. Resampling Strategy and Implementation (`preprocess.py` - `resample_data_for_era` function)

The `resample_data_for_era` function in `preprocess.py` is responsible for this step, driven by settings in `data_processing_config.json`.

* **Configuration (`data_processing_config.json`):
  * `era_definitions.[era_id].target_frequency`: A string specifying the target Pandas frequency alias for resampling (e.g., `"1T"` for 1 minute, `"5T"` for 5 minutes, `"1H"` for 1 hour). This is crucial for defining the desired regular grid.
  * `common_settings.time_col`: Specifies the name of the time column to be used for creating the `DatetimeIndex`.

* **Implementation Steps:**
    1. **Input:** Receives the DataFrame for a specific Era after initial cleaning and outlier treatment (output of Chapter 1 processes).
    2. **Pre-checks:** Verifies the DataFrame is not empty and that the specified `time_col` exists and is of a datetime type.
    3. **Set `DatetimeIndex`:**
        * The time column is set as the DataFrame's index using `df.set_index(time_col)`. This is a prerequisite for Pandas' `resample()` method.
        * Ensures the index is a `pd.DatetimeIndex` and UTC localized (this should have been handled in `sort_and_prepare_df`, but a check or re-conversion here can add robustness).
    4. **Perform Resampling using `.asfreq()`:**
        * The core operation is `df_resampled = df.resample(target_freq).asfreq()$.
        * `df.resample(target_freq)`: Groups data by the `target_frequency`.
        * `.asfreq()`: This method returns a new DataFrame with the resampled index, conforming to the new frequency.
            * **Upsampling Scenario:** If the `target_frequency` is higher than the original data's frequency (e.g., original data is every 5 minutes, target is 1 minute), `.asfreq()` will create new rows for the intermediate timestamps. The values for all sensor columns in these new rows will be `NaN` initially.
            * **Downsampling Scenario (with `.asfreq()`):** If the `target_frequency` is lower, `.asfreq()` will pick the value at the exact end of the new, coarser frequency bin if an original data point aligns perfectly. If not, it might also result in `NaN`. For proper downsampling with aggregation (mean, sum, etc.), `df.resample(target_freq).mean()` or `.agg()` would be used (see Section 2.4).
            * **No Change Scenario:** If the data is already at the `target_frequency` and regular, `.asfreq()` primarily ensures the index is perfectly regular and fills any implicit micro-gaps if they existed.
    5. **Logging:** Records the shape of the DataFrame before and after resampling, and the new time range of `df_resampled`.
    6. **Error Handling:** If resampling fails, it logs an error and returns the original DataFrame with its index reset to avoid downstream issues with a potentially misconfigured index.

## 2.4. Handling Upsampling vs. Downsampling (Detailed Considerations)

While `preprocess.py` currently focuses on `.asfreq()` which is ideal for upsampling to create a regular grid for later imputation, a complete preprocessing toolkit should consider explicit downsampling strategies if needed.

* **Upsampling (Current Focus for Sparse Data to GANs):**
  * **Goal:** To create a dense, regular time grid (e.g., every minute) even if original sensor readings are less frequent or irregular within an Era.
  * **Mechanism:** `df.resample(target_freq).asfreq()`.
  * **Outcome:** Introduces `NaN` values in the newly created time slots. These NaNs represent periods where no direct sensor reading was available at that exact new timestamp.
  * **Next Step:** These NaNs **must** be handled by robust imputation techniques (Chapter 4).
  * **`data_processing_config.json` Implication:** The `target_frequency` should be chosen based on the desired granularity for modeling and the finest reasonable interval that imputation can reliably fill.

* **Downsampling (If Original Data is Too Granular):**
  * **Goal:** To reduce data frequency and size, potentially smoothing noise, by aggregating data over coarser time intervals (e.g., from 1-second data to 1-minute data).
  * **Mechanism:** `df.resample(target_freq).agg(aggregation_rules)` or specific methods like `.mean()`, `.sum()`, `.first()`, `.last()`.
  * **Implementation Detail:** Requires defining aggregation rules per column in `data_processing_config.json` (e.g., `mean` for continuous sensors like temperature, `sum` for countable values like energy or light duration if it were pre-aggregated, `first` or `last` for status columns within the interval).

        ```json
        // Example in config for column-specific aggregations
        "era_definitions": {
            "Era1": {
                // ... other settings ...
                "aggregation_rules": {
                    "air_temp_c": "mean",
                    "rain_status": "max", // Max of 0/1 would indicate if it rained at all in interval
                    "co2_dosing_status": "last"
                }
            }
        }
        ```

  * The `resample_data_for_era` function would need to be enhanced to parse these rules and apply them using `.agg()`.
  * **Current Status:** The current `preprocess.py` does not implement this detailed aggregation for downsampling; it relies on `.asfreq()`. If downsampling with specific aggregation is a primary need, this function requires extension.

## 2.5. Outputs from this Chapter

* For each processed Era:
  * A Pandas DataFrame with a **regular `DatetimeIndex`** at the `target_frequency` specified in the configuration.
  * If upsampling occurred, this DataFrame will contain `NaN` values for the newly created time slots.
  * Updated entries in the summary report (`preprocessing_summary_report_{era_identifier}.txt`):
    * Target resampling frequency used.
    * Shape of the DataFrame before resampling.
    * Shape of the DataFrame after resampling.
    * The new, regularized time range of the data.

## 2.6. Considerations for Subsequent Steps & Toolkits

* **Input to Imputation (Chapter 4):** The resampled DataFrame, with its regular index and potentially many NaNs (from upsampling or original gaps now aligned to the grid), is the direct input for the imputation stage. The regularity of the index is crucial for effective imputation.
* **Data for Segmentation (Chapter 3):** While `DataSegmenter` in `processing_steps.py` can operate on non-regularly spaced data (it calculates time diffs), applying it *after* resampling ensures that the `min_gap_hours` for defining new segments is evaluated against a consistent time grid. This can lead to more predictable segmentation.
* **GAN / LSTM Input:** The goal of this step is to prepare data for windowing into fixed-length sequences. A regular time index is fundamental for this windowing process to be meaningful and consistent.
