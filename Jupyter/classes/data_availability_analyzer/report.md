# Data Availability Analysis Report for `sensor_data_merged`

**Date:** October 26, 2023
**Source Script:** `Jupyter/classes/data_availability_analyzer/data_availability.py`

## 1. Introduction

This report summarizes the findings from an analysis of data availability within the `public.sensor_data_merged` table, covering the period from late 2013 to late 2016. The analysis involved generating plots for individual sensor columns to show periods of data presence versus absence (NaNs), and heatmaps to visualize missing data patterns across groups of sensors on a daily basis. Understanding data availability is crucial for effective feature calculation, selection, and extraction for the greenhouse simulation and optimization project.

## 2. Key Findings from Visualizations

The analysis of individual availability plots (e.g., `availability_air_temp_c.png`, `availability_co2_measured_ppm.png`, etc.) and grouped daily heatmaps (e.g., `heatmap_missing_group_1_D.png`, etc.) revealed several critical patterns:

### 2.1. Distinct Data Availability Eras and Major Gaps

Two primary periods of data collection, separated by a significant gap, were identified:

* **Era 1 (Approx. Early 2014 - Mid/Late August 2014):**
  * **Good Availability:** Core environmental sensors such as `air_temp_c`, `relative_humidity_percent`, `light_intensity_umol`, `radiation_w_m2`, and `rain_status` show consistent data presence with only minor, short dropouts.
  * **Patchy Availability:** `co2_measured_ppm` is present but exhibits some intermittency during this period.
  * **Limited Availability:** `heating_setpoint_c` (a potential target variable) is only available for a very short window (approx. June-August 2014) within this era.
  * **Largely Missing:** Most actuator statuses (vents, curtains, lamps) and specific measurements like pipe temperatures are largely missing during this first era.

* **Major Gap (Approx. Late August 2014 - Mid/Late September/October 2015):**
  * A period of roughly one year where most sensors show no data (consistently black on heatmaps, zero on availability plots).

* **Era 2 (Approx. Mid/Late October 2015 - End of Data, Late 2016):**
  * **Good Availability (Continuous):** `radiation_w_m2` becomes available again and is fairly continuous.
  * **Intermittent Availability:** Sensors like `co2_measured_ppm`, `pipe_temp_1_c`, `pipe_temp_2_c`, various `vent_..._afd..._percent` columns, and `lamp_grp*status` columns become active. However, while they might show daily presence on heatmaps, individual plots reveal **high intra-day intermittency** (frequent on/off spikes of data).
  * **Largely Missing:** Many core environmental sensors that were good in Era 1 (e.g., `air_temp_c`, `relative_humidity_percent`, `light_intensity_umol`) are mostly unavailable in this later era.
  * **AFD Humidity Metrics:** `humidity_deficit_afd3_g_m3`, `relative_humidity_afd3_percent`, `humidity_deficit_afd4_g_m3`, `relative_humidity_afd4_percent` become available and appear fairly consistent in this era.

### 2.2. Limited Overlap Between Key Sensor Groups Across Eras

A significant challenge is the lack of extended periods where *both* the core environmental sensors from Era 1 *and* the actuator/pipe/lamp/AFD-specific data from Era 2 are simultaneously and reliably available.

### 2.3. Scarcity of Key Variables

* **`heating_setpoint_c`:** Extremely limited availability, posing a major challenge if it's a primary target variable for models spanning the entire dataset duration.
* **Many "Dead" or Near-Dead Columns:** Several columns (e.g., `window_1_percent`, `window_2_percent`, `measured_status_bool`, and many from the latter part of the schema like `behov`, `status_str`, `timer_on/off`, `dli_sum`, `oenske_ekstra_lys`, `lampe_timer_on/off`, `value`) show little to no data across the entire dataset. Their utility is highly questionable.

### 2.4. Intermittency as a Major Challenge

For several sensors that do have data in Era 2, the high frequency of missing points *within* days (even if the day itself isn't entirely black on the heatmap) makes them difficult to use for features requiring continuous signals (e.g., hourly rolling averages, lags based on consistent sampling).

## 3. Implications for Feature Calculation, Selection, and Extraction

These findings have significant implications for the data processing pipeline (`src/`) and feature engineering strategy:

**3.1. Data Segmentation:**

* **Conclusion:** It's highly unlikely that the entire 2013-2016 dataset can be treated as a single, homogenous block for feature engineering and modeling due to the vast differences in sensor availability.
* **Action:** Consider processing the data in distinct segments (e.g., "Era 1" and "Era 2" identified above). This should be configurable in `DataProcessingConfig`.
  * Feature sets might need to be adapted per segment based on what data is reliably available.
  * Models might need to be trained separately for these segments or be made aware of the segment they are processing.

**3.2. Target Variable Strategy:**

* **Conclusion:** The scarcity of `heating_setpoint_c` severely restricts its use as a target variable for any model intended to cover the full dataset duration.
* **Action:**
  * Re-evaluate the primary target variable for the LSTM model. If `heating_setpoint_c` is critical, modeling efforts might have to be narrowly focused on its short window of availability.
  * Explore if other columns could serve as suitable targets or if a proxy can be engineered (though this is complex).

**3.3. Outlier Handling and Imputation Strategies (`src/data_cleaning.py`):

* **Conclusion:** Imputation strategies must be context-aware.
* **Action:**
  * **Large Gaps:** Avoid aggressive imputation (like linear interpolation) across year-long gaps. This would generate highly artificial data.
  * **Intermittent Data:** For sensors with high intra-day intermittency (e.g., `co2_measured_ppm` in Era 2), simple interpolation might still be problematic. Consider:
    * Forward-fill (`ffill`) with a short limit.
    * Using the data as sparse/event-based if imputation proves too unreliable.
    * Excluding these sensors from features requiring high continuity during their intermittent periods.
  * **Configuration:** Imputation strategies in `DataProcessingConfig` should be highly column-specific and potentially period-specific.

**3.4. Feature Engineering (`src/feature_engineering.py`, `src/feature_calculator.py`):

* **Conclusion:** Feature engineering must be adapted to data availability.
* **Action:**
  * **Availability-Aware Features:** Consider creating features that explicitly capture data availability, such as:
    * `[column_name]_is_available` (binary flag).
    * `[column_name]_time_since_last_valid`.
    * Count of valid points in a rolling window.
  * **Rolling Window Features:** Be cautious with `min_periods` when data is sparse. A rolling mean over very few actual points can be noisy.
  * **Lag Features:** Lags will be frequently NaN if the source data is intermittent. This needs to be handled (e.g., by imputation of the lag feature itself, or by only calculating lags where continuity allows).
  * **Segment-Specific Features:** If data is processed in segments, the set of derivable features might differ significantly between segments.

**3.5. Feature Selection (`src/feature_selection.py`):

* **Conclusion:** Feature selection methods will be affected by missingness.
* **Action:**
  * **High NaN Percentage:** Features with a very high percentage of NaNs (due to underlying data gaps) might be poor candidates for selection or may need robust handling by the selection algorithm.
  * Consider running feature selection independently on different data segments if the available feature sets vary greatly.

**3.6. Column Pruning:**

* **Conclusion:** Columns identified as almost entirely empty (e.g., `behov`, `status_str`, `oenske_ekstra_lys`, `value`, etc.) offer little to no informational value.
* **Action:** These columns should likely be excluded from the data loading process entirely (by not requesting them from the database in `retriever.py` or by dropping them early in `preprocess_pipeline.py` or `flow_main.py`) to save memory and processing time. The `POTENTIAL_NUMERIC_COLS` list and other column lists in configurations should be updated to reflect this.

## 4. Overall Recommendation

The primary recommendation is to **not treat the dataset as monolithic.** The data availability analysis strongly suggests a segmented approach to data processing and modeling. The period from **early 2014 to mid/late August 2014** appears to be the richest in terms of co-occurring core environmental sensor data and should perhaps be the focus for initial, comprehensive feature engineering and model development.

For other periods, a more selective approach to feature engineering will be necessary, focusing on the subset of sensors that *are* reliably available, and potentially incorporating features that describe the data intermittency itself.

This analysis is a critical input for refining the `DataProcessingConfig` and the logic within the `src/` pipeline modules to handle the realities of the available data effectively.
