# Chapter 4: Data Integrity Checks and Advanced Imputation

This chapter details the critical process of handling missing data (NaNs) within the continuous, resampled data segments generated in Chapter 3. Effective imputation is vital for preparing complete datasets required by LSTMs and GANs like DoppelGANger. It also briefly touches upon further data integrity checks that might be relevant at this stage.

## 4.1. Objectives

* Systematically impute missing values (`NaN`) within each data segment, using strategies appropriate for time-series data and sensor types.
* Ensure that imputation methods are configuration-driven, allowing for different strategies per column or Era if necessary.
* Minimize the introduction of artificial patterns or biases through imputation.
* Produce data segments that are as complete as possible for subsequent scaling and feature engineering.
* Discuss further data integrity checks post-imputation.

## 4.2. Nature and Origin of Missing Data at this Stage

By this point in the preprocessing pipeline, NaNs in a data segment can originate from:

1. **Upsampling (Chapter 2):** If the data was upsampled to a higher `target_frequency`, new time slots were created and filled with `NaN` for all sensor columns.
2. **Original Sensor Intermittency/Gaps (Chapter 1 & 3):** Even after selecting an Era and segmenting by major gaps, shorter periods of sensor malfunction or data transmission loss might result in NaNs that align with the regular time grid created by resampling.
3. **Outlier Treatment (Chapter 1):** If an outlier treatment strategy involved replacing outliers with `NaN` (though the current `OutlierHandler` uses clipping).

Understanding the likely origin helps in choosing appropriate imputation methods.

## 4.3. Imputation Strategies and Implementation (`preprocess.py` - `ImputationHandler` from `processing_steps.py`)

The `ImputationHandler` class is responsible for applying imputation rules defined in `data_processing_config.json` to each data segment.

* **Configuration (`data_processing_config.json`):
  * `era_definitions.[era_id].imputation_rules_ref`: Points to a specific set of imputation rules in the `preprocessing_rules` section (e.g., `"era1_imputation_rules"`).
  * `preprocessing_rules.[rule_set_name]`: A list of dictionaries, each defining a rule for a column:
    * `{"column": "column_name", "strategy": "method_name", "limit": X, ...}`
    * `"column"`: The target column.
    * `"strategy"`: The imputation method (e.g., `"forward_fill"`, `"linear"`, `"mean"`, `"median"`, `"bfill"`).
    * `"limit"` (optional): For methods like `ffill`, `bfill`, and `linear` (via `limit_direction`/`limit_area`), this restricts how many consecutive NaNs are filled.
* **Implementation (`ImputationHandler.impute_data` method):
    1. **Input:** Receives a single, continuous data segment (Pandas DataFrame) for a specific Era.
    2. **Iterate Through Rules:** For each rule defined in the configured rule set:
        * Identifies the target `column` and `strategy`.
        * Applies the specified Pandas imputation method:
            * **`forward_fill` (`ffill`):** Propagates the last valid observation forward. Useful for status columns or sensors where values tend to persist. A `limit` restricts how far it fills.
            * **`backward_fill` (`bfill`):** Propagates the next valid observation backward. Can fill leading NaNs or complement `ffill`. A `limit` applies.
            * **`linear` interpolation (`interpolate(method='linear')`):** Fills NaNs by drawing a straight line between the valid points before and after the gap. Suitable for continuous sensor data where a linear trend is plausible over short gaps. A `limit` can restrict the gap size for interpolation (via `limit_direction`, `limit_area`, `limit` arguments of `.interpolate()`).
            * **`mean` / `median`:** Replaces NaNs with the mean or median of the entire column (within the segment). Generally too simplistic for time-series as it ignores temporal context but can be a last resort for some cases if other methods fail or are inappropriate. These are applied segment-wise.
    3. **Order of Rules:** The order of rules in the config can matter if multiple rules target the same column or if general fallbacks are used after specific methods.
    4. **Logging:** Reports the imputation strategy applied to each column and can show NaN counts before and after.

* **Considerations for Trend-Based Imputation (Relating to "Augurs" concept):
  * The current `linear` interpolation is a simple form of trend-based imputation.
  * **Advanced Trend Imputation (Future Enhancement):** For longer gaps *within a segment* where a linear trend is insufficient, more sophisticated methods could be integrated:
    * Fit a more complex model (e.g., polynomial regression, LOESS, or a model from a toolkit like a conceptual "Augurs" if it offers trend fitting) to the non-missing parts of a sensor's data within the segment.
    * Use this fitted model to predict and fill the NaNs.
    * This would likely involve creating a new imputation strategy within `ImputationHandler` or a separate processing step.
  * **Changepoint-Aware Imputation:** If changepoint detection (Chapter 3) identified sub-regimes *within* a segment, imputation strategies could be adapted per sub-regime. The current `DataSegmenter` based on `min_gap_hours` aims to create segments that are already internally consistent to some degree.

* **Handling Remaining NaNs:**
  * Even after applying configured imputation rules (especially those with limits), some NaNs might persist if gaps are too large.
  * The `preprocess.py` script currently logs these (`nan_counts[nan_counts > 0]`).
  * **Decision Point for GANs/LSTMs:** These models generally require fully complete input sequences. Remaining NaNs after this imputation step must be addressed before windowing for these models:
        1. **Aggressive Fill (e.g., global `ffill` then `bfill` on the segment):** As a final pass, fill any remaining NaNs. This might be acceptable if very few remain.
        2. **Drop Rows/Sequences:** If key features still have NaNs, rows containing them might be dropped, or sequences formed from these parts might be excluded from training.
        3. **`_is_valid` Flags:** If imputation is aggressive, carrying forward `_is_valid` flags (generated before imputation based on original missingness) becomes even more important for the GAN to learn data reliability patterns.

## 4.4. Further Data Integrity Checks (Post-Imputation)

After imputation, it's good practice to perform some final checks on each segment:

* **Confirm No NaNs (for critical columns):** Verify that columns intended as direct input to models are now complete.
* **Plausibility of Imputed Values:** Briefly inspect time-series plots of imputed columns, especially where long gaps were filled by interpolation, to ensure no obviously artificial or unrealistic patterns were introduced.
* **Statistical Properties:** Compare basic statistics (mean, std, min, max) of imputed columns before and after imputation (on the non-missing data vs. the fully imputed series) to understand the impact of imputation. Significant shifts might indicate issues.
* **Data Type Consistency:** Ensure all columns have their expected final data types (e.g., floats for continuous sensors, integers for statuses).

## 4.5. Outputs from this Chapter

* For each input data segment from an Era:
  * A Pandas DataFrame segment with NaNs imputed according to the configured strategies.
  * Ideally, these segments should be fully populated for columns intended for direct model input, or a clear strategy for handling any remaining NaNs should be in place.
* Updated entries in the summary report (`preprocessing_summary_report_{era_identifier}.txt`):
  * For each segment processed:
    * Shape before and after imputation.
    * A summary of NaN counts per column *after* imputation, highlighting any remaining NaNs.

## 4.6. Considerations for Subsequent Steps & Toolkits

* **Ready for Scaling (Chapter 5):** The imputed data segments are now ready for numerical scaling/normalization.
* **Ready for Feature Engineering (Chapter 5):** Complete data segments are essential for calculating lagged features, rolling statistics, and other derived features without being skewed by NaNs.
* **Input for GAN Windowing:** The primary goal of imputation is to provide clean, complete segments from which fixed-length sequences can be extracted for training DoppelGANger or other sequence-based models.
* **Evaluation of Imputation Quality:** The impact of different imputation strategies on the final quality of synthetic data (if using GANs) or model performance (if using LSTMs for forecasting) is an important area for evaluation. More sophisticated imputation might be revisited if simpler methods prove insufficient.
