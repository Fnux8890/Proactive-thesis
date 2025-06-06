**Software Engineering Report: Focused Data Processing for "Era 1"**

**Date:** October 27, 2023
**Project:** Proactive Thesis - Greenhouse Simulation Optimization
**Author:** AI Assistant (Gemini 2.5 Pro)
**Status:** Proposed

**1. Introduction & Goal**

This document outlines the software engineering tasks required to adapt the `DataIngestion/simulation_data_prep/` pipeline to focus on processing a specific high-quality data segment, designated "Era 1". This focused approach is based on previous data availability analysis (`report.md` and Section 3.1 below) which identified "Era 1 (approx. early 2014 - mid/late August 2014)" as the period with the most consistent and co-occurring core environmental sensor data. Visual analysis of outlier detection plots (Section 3.2 below) further informs the data cleaning strategy for this segment.

The primary goal of this focused effort is to produce a high-quality, coherent feature set from Era 1 to serve as the initial input for training and improving the LSTM-based surrogate model (`DataIngestion/model_builder/`). By isolating this "golden segment," we aim to simplify initial feature engineering, establish robust data cleaning and transformation processes informed by detailed data inspection, and provide a reliable baseline for model performance.

**2. Scope: "Era 1" Definition**

* **Name:** `Era1_CoreEnv` (as defined in `data_processing_config.json`)
* **Start Date:** 2014-01-01T00:00:00Z
* **End Date:** 2014-08-31T23:59:59Z
* **Description:** "Focus on core environmental sensors (temp, rh, light, rad, basic co2)"

All subsequent tasks will target the isolation, processing, and feature engineering of data exclusively within these date boundaries.

**3. Overall Strategy**

The strategy involves modifying the existing data processing pipeline to:

1. **Isolate Era 1 Data:** Ensure that only data within the defined Era 1 date range is extracted and processed.
2. **Verify Era 1 Sensor Quality & Characteristics:** Programmatically confirm the availability, quality, and outlier characteristics of individual sensor columns *within the isolated Era 1 data*, using insights from Sections 3.1 and 3.2.
3. **Refine Data Cleaning for Era 1:** Implement and tune outlier detection and imputation strategies specifically for Era 1 data, guided by visual analysis.
4. **Refine Feature Engineering for Era 1:** Adapt and thoroughly test feature calculations (`feature_calculator.py`, `feature_engineering.py`) using the cleaned characteristics and available sensors of Era 1.
5. **Unit Testing:** Maintain and expand unit tests (`test_feature_engineering.py`) to ensure the correctness of all data cleaning and feature calculations for Era 1.

**3.1. Visual Rationale: Insights from Data Availability Analysis (Based on `Jupyter/classes/data_availability_analyzer/`)**

The decision to focus initially on "Era 1" is heavily supported by the data availability visualizations generated by `data_availability.py` (outputs in `images/` and summarized in `report.md`). These visuals highlight several critical data characteristics:

* **Distinct Data Eras & Major Gaps:**
  * The plots (e.g., `availability_air_temp_c.png`, `availability_light_intensity_umol.png`) clearly show a period of relatively good data presence for core environmental sensors from roughly **early 2014 to mid/late August 2014 (Era 1)**.
  * Following this, there's a significant data gap spanning approximately one year (late August 2014 - mid/late Q3 2015), visible as a flat line at 0 (missing) in most individual availability plots and as large black blocks in the heatmaps (e.g., `heatmap_missing_group_1_D.png`).
  * A second period of data collection, "Era 2" (approx. mid/late Q3 2015 - late 2016), shows different sensor availability. While some sensors like `radiation_w_m2.png`, `pipe_temp_1_c.png`, and various actuator/lamp statuses (e.g., `availability_lamp_grp1_no3_status.png`, `availability_vent_lee_afd3_percent.png`) become active, many of the core environmental sensors from Era 1 become largely unavailable.

* **Limited Overlap of Key Sensor Groups:**
  * A crucial observation from comparing plots like `availability_air_temp_c.png` (good in Era 1) with `availability_pipe_temp_1_c.png` (only good in Era 2) is the lack of extended periods where *both comprehensive environmental data and detailed actuator/system data* are reliably co-occurring. This makes holistic modeling across the entire dataset extremely challenging without significant imputation or feature set compromises.

* **Variable Sensor Quality & Intermittency:**
  * **Era 1:** Sensors like `air_temp_c`, `light_intensity_umol`, `relative_humidity_percent`, and `rain_status` show fairly consistent data (solid blocks of '1' or near '1' in their availability plots during this time).
  * `co2_measured_ppm.png` shows more patchiness even within Era 1, indicating some intermittency.
  * **Era 2:** Many sensors that become active in Era 2, such as `co2_measured_ppm.png` (second active phase), `availability_pipe_temp_1_c.png`, and lamp/vent statuses, exhibit high intra-day intermittency. While they might appear on the daily heatmaps, their individual availability plots show frequent on/off spikes, making them difficult for continuous feature calculations (e.g., rolling averages without careful handling).

* **Scarcity of Specific Variables:**
  * `availability_heating_setpoint_c.png` illustrates extreme scarcity, with data only present for a very short window (June-August 2014), severely limiting its use as a primary target variable across wider timeframes.

* **Heatmap Overview:**
  * The heatmaps (`heatmap_missing_group_1_D.png`, `heatmap_missing_group_2_D.png`, `heatmap_missing_group_3_D.png`) provide a bird's-eye view of these patterns. They clearly delineate the two main eras and the large central gap. They also show how different groups of sensors light up or go dark in these distinct periods.

**Conclusion from Visuals:** The visualizations strongly support segmenting the data. Era 1 stands out as the period offering the most consistent availability of a *core set of environmental variables* necessary for foundational feature engineering. Attempting to use the entire dataset monolithically would require extreme measures to handle missing data and the drastically changing availability of key sensors, likely leading to less reliable features for the LSTM model.

**3.2. Visual Rationale: Insights from Outlier Detection Plots**

Further refining our understanding of the data quality within different periods, the outlier detection plots (IQR and Rolling Z-score methods for `air_temp_c`, `co2_measured_ppm`, `radiation_w_m2`, and `relative_humidity_percent`) provide critical insights for tailoring data cleaning strategies, especially for "Era 1".

* **Method-Specific Sensitivity:**
  * **IQR Method Plots** (e.g., `outliers_iqr_air_temp_c.png`, `outliers_iqr_co2_measured_ppm.png`): These plots show outliers based on the global distribution of the data across the entire plotted timeframe. The upper and lower bounds are fixed. This method is effective at identifying extreme values that are globally anomalous.
    * For `air_temp_c`, it flags sharp, short-duration spikes, particularly within Era 1 (April-July 2014).
    * For `co2_measured_ppm`, it identifies many points above ~889 ppm as outliers in both Era 1 and Era 2, and a few near-zero points post-Era 1.
    * For `radiation_w_m2`, it flags a significant number of higher readings (above ~381 W/m²) as outliers across both eras.
    * For `relative_humidity_percent`, it correctly identifies values outside the 0-100% physical range (e.g., points near 0% or above 103%).
  * **Rolling Z-score Plots** (e.g., `outliers_zscore_air_temp_c.png`, `outliers_zscore_co2_measured_ppm.png`): This method uses a rolling window (168 time periods in the plots, likely representing 168 hours or 7 days) to calculate local means and standard deviations. Outliers are points that deviate significantly from this *local* behavior.
    * The Z-score bounds are dynamic, adapting to trends and volatility in the data.
    * For `air_temp_c`, it flags similar spikes in Era 1 as the IQR method but is also sensitive to local variations that might not be global outliers.
    * For `co2_measured_ppm`, it's highly active, flagging points during periods of sustained high or low values if they differ from the rolling mean, even if those sustained periods are operationally valid but different from the window's average.
    * For `radiation_w_m2`, it also flags many of the higher peaks in Era 1 and a vast number of points in Era 2 during high radiation periods.
    * For `relative_humidity_percent`, it is more sensitive to local fluctuations within the 0-100% range than the IQR method.

* **Implications for Era 1 Data Cleaning Strategy:**
  * **`air_temp_c`:** The IQR and Z-score plots both highlight anomalous spikes within Era 1. The existing global `domain` rule in `data_processing_config.json` (-10°C to 50°C) is broad. For Era 1, the IQR suggests an upper bound closer to 42-43°C for "typical" data. Consideration should be given to whether these spikes are genuine extreme readings or sensor errors. A tighter domain rule or a segment-specific IQR/Z-score rule might be beneficial for Era 1.
  * **`co2_measured_ppm`:** This variable shows significant outlier activity in Era 1 by both methods. The IQR method's upper bound (~889 ppm) is much stricter than the configured domain rule (up to 5000 ppm). This discrepancy requires careful thought: Are CO2 readings above ~900 ppm in Era 1 truly errors, or are they valid but less frequent operational states? The definition of an "outlier" for CO2 needs to be robust and potentially context-aware for Era 1.
  * **`radiation_w_m2`:** The IQR method seems overly sensitive, flagging many potentially valid peak solar radiation values as outliers (upper bound ~381 W/m²). The configured `domain` rule (0-1500 W/m²) is more realistic for physical limits. The Rolling Z-score, with its current long window, also flags many high values. For Era 1, focusing on the domain rule for hard limits and perhaps a Z-score with a shorter, more adaptive window (e.g., daily) to catch unusual deviations from the typical diurnal solar cycle might be more appropriate than a global IQR.
  * **`relative_humidity_percent`:** The `domain` rule (0-100%) is the most critical. The IQR plot confirms it catches excursions beyond this. The Z-score method can identify local noise or instability within the 0-100% range, which might be useful for quality assessment within Era 1.

* **Configuration Adjustments (`data_processing_config.json` - Task 2 & 12):**
  * The current `outlier_detection.rules` are global. For Era 1, these rules need to be reviewed and potentially refined or overridden with segment-specific rules if `config.py` and `flow_main.py` are adapted to support this.
  * If global rules are maintained for simplicity, their parameters (e.g., IQR factor, Z-score window/threshold, domain bounds) must be chosen considering their impact on Era 1 data quality specifically, informed by the visual analysis in this section.
  * For instance, the `zscore_rolling` window of 168 periods (seen in plots) might be too long for capturing certain types of anomalies in Era 1 for some variables; shorter, more adaptive windows might be necessary. The `min_periods` for rolling calculations will also be important and should be configurable.
  * The `handling_strategy` (e.g., `to_nan`, `clip`) should be chosen carefully for each rule based on the nature of the variable and the likely type of outlier observed in Era 1.

**Conclusion from Outlier Visuals:** The outlier plots underscore the need for a nuanced approach to data cleaning, especially when focusing on Era 1. Relying solely on global rules or a single outlier detection method may not be optimal for all variables. A combination of domain knowledge (physical limits), robust statistical methods (like a carefully tuned Z-score for local anomalies), and potentially segment-specific configurations will be key to preparing high-quality data from Era 1 for the LSTM. These visualizations provide a strong empirical basis for the review and adaptation tasks (Task 2 and Task 12) outlined in this plan.

**4. Task Breakdown by File**

The following sections detail the required modifications and tasks for each relevant file within the `DataIngestion/simulation_data_prep/src/` directory.

**4.1. `config.py` and `data_processing_config.json`**

* **`data_processing_config.json`:**
  * **Task 1 (Verify):** Ensure the "Era1_CoreEnv" segment is correctly defined with the precise start and end dates.
    * **Status:** Appears **DONE** based on the provided file (Era1_CoreEnv is defined).
  * **Task 2 (Review & Adapt):** Review `outlier_detection`, `imputation`, and `segment_feature_configs.Era1_CoreEnv` sections, directly referencing insights from Section 3.1 and 3.2.
    * Ensure outlier rules (type, parameters like Z-score window, handling strategy) are appropriate for the sensors expected to be good and the outlier types observed in Era 1 plots.
    * Verify imputation strategies are sensible for Era 1 data characteristics (e.g., patchiness of `co2_measured_ppm` as noted in Sec 3.1).
    * Confirm that feature calculation parameters under `segment_feature_configs.Era1_CoreEnv` are aligned with the sensors available and reliable in Era 1 (Sec 3.1). For instance, if a sensor needed for a configured `delta_col` or `rate_of_change_col` is not good in Era 1, it should be removed from the Era 1 specific config.
    * **Status:** **TODO**
  * **Task 3 (Consider):** Clean up or clearly mark other segments in `data_segments` (like "Era2_Actuators", "Full_Range_Test") as "experimental" or "future work" to avoid confusion during this focused phase.
    * **Status:** **TODO**

* **`config.py` (Pydantic models):**
  * **Task 4 (Verify):** Ensure Pydantic models (`DataSegment`, `SegmentFeatureConfig`, `DataProcessingConfig`, etc.) correctly load and validate the "Era1_CoreEnv" segment definition and its specific feature configurations from `data_processing_config.json`.
  * **Task 5 (Refinement - Optional):** If `SegmentFeatureConfig` needs to specify *which exact raw sensor columns are considered primary inputs* for that segment (informed by Sec 3.1 availability), this model might need a new field (e.g., `era1_primary_sensor_columns: List[str]`). This could help downstream tasks to dynamically select columns.
    * **Status:** **TODO** (Verify existing models first).

**4.2. `flow_main.py` (Prefect Orchestration)**

* **Task 6 (Critical):** Ensure the `main_feature_flow` correctly uses the `segment_name` parameter (e.g., "Era1_CoreEnv") to:
  * Drive `determine_segment_task` to fetch the correct start/end timestamps for Era 1.
  * Pass these exact timestamps to `extract_data_task`.
  * Pass the `segment_name` to `clean_data_task` and `transform_features_task` so they can potentially use segment-specific configurations from `DataProcessingConfig`.
  * **Status:** Partially implemented (flow takes `segment_name`). Needs verification that all downstream tasks correctly utilize segment-specific configurations or data boundaries.
* **Task 7 (Enhancement):** Modify `transform_features_task` (and the underlying `transforms.core.transform_features` function) to accept `segment_name` and use it to fetch the appropriate `SegmentFeatureConfig` from `DataProcessingConfig`. This will allow dynamic feature calculation based on the segment being processed, informed by Era 1's specific sensor suite (Sec 3.1).
  * **Status:** **TODO**
* **Task 8 (Logging & Output):** Ensure log messages and any output artifacts (e.g., persisted Parquet files from `persist_features_task`) clearly indicate which segment (`Era1_CoreEnv`) was processed.
  * **Status:** Partially implemented. Review for consistency.

**4.3. `retriever.py` (or `dao/sensor_repository.py` if it's the active one)**

* **Task 9 (Verify):** Confirm that the data retrieval logic (`retrieve_data` function or `SensorRepository.get_sensor_data` method) strictly adheres to the `start_time` (inclusive) and `end_time` (exclusive) parameters passed from `extract_data_task` in `flow_main.py`. No data outside this window should be fetched for Era 1.
  * **Status:** Assumed to be working but needs explicit testing in the context of Era 1 isolation.
* **Task 10 (Column Selection - Optional but Recommended):** Instead of `SELECT *`, consider dynamically building the `SELECT` clause based on a list of *known good and relevant sensors for Era 1* (informed by Sec 3.1 availability plots). This list could come from `DataProcessingConfig` (see Task 5) or be determined in an earlier step of the flow after basic availability checks. This reduces data transfer and initial processing load.
  * **Status:** **TODO** (Currently uses `SELECT *` or user-provided list).

**4.4. `data_cleaning.py`**

* **Task 11 (Segment-Aware Rules):** The visual analysis (Sec 3.2) strongly suggests that global outlier rules might be insufficient. Modify `apply_outlier_treatment_pl` and `impute_missing_data_pl` to enhance their ability to use segment-specific rules from `DataProcessingConfig` if such a structure is adopted (e.g., allowing `OutlierDetectionConfig` to have a `segment_rules: Dict[str, List[OutlierRule]]` field).
  * **Status:** Current config structure leans global. Code adaptation for true segment-specific outlier/imputation rule lookup is **TODO**.
* **Task 12 (Review Rules for Era 1):** This is a critical task. Based on the verified list of good sensors in Era 1 and the outlier characteristics observed in Section 3.2, meticulously review and tune the `outlier_detection.rules` (and `imputation` strategies) in `data_processing_config.json` to be effective for Era 1. This includes adjusting parameters (e.g., Z-score windows, IQR factors, domain bounds) and handling strategies. If segment-specific rules are not implemented (Task 11), ensure global rules are primarily tuned for Era 1 during this phase.
  * **Status:** **TODO**

**4.5. `feature_calculator.py`**

* **Task 13 (Review & Adapt Functions for Era 1):** For each feature calculation function:
  * **Input Sensor Check:** Explicitly use the verified list of sensors that are reliable and of good quality *after cleaning* in Era 1 (informed by Sec 3.1 & 3.2).
  * **Graceful Handling & Dependencies:** If a required input for a feature is not reliable in Era 1 (even after cleaning attempts), the feature should ideally not be calculated for this segment, or the function should return a clear indicator. The calling function (`transforms.core.transform_features`) will need to handle this based on Era 1's defined feature set in `SegmentFeatureConfig`.
  * **Parameterization for Era 1:** Ensure that parameters passed to these functions (e.g., `t_base` for GDD, `window_str` and `min_periods` for rolling features) are appropriate for Era 1, sourced from `SegmentFeatureConfig.Era1_CoreEnv`.
  * **Status:** **TODO** (Requires systematic review of each function against Era 1 sensor list, cleaned data characteristics, and configs).

**4.6. `feature_engineering.py` (Higher-level feature construction, if used beyond `create_time_features`)**

* **Task 14 (`create_time_features`):**
  * **Status:** This function is generally applicable. The main check is that the `time_col` passed to it is indeed the primary, clean timestamp column for Era 1. Seems **Mostly DONE**.
* **Task 15 (Other Feature Logic):** If this file contains logic to orchestrate calls to `feature_calculator.py` or implement other complex features:
  * Apply the same principles as for `feature_calculator.py`: ensure calls are made only with sensors reliable in Era 1 and with Era 1-specific configurations.
  * **Status:** **TODO** (Review any such logic).

**4.7. `test_feature_engineering.py` (Unit Tests)**

* **Task 16 (Era 1 Test Data):** Create or adapt test fixtures to generate sample Polars DataFrames that specifically mimic Era 1's expected data characteristics, including sensor availability (Sec 3.1) and representative outlier patterns (Sec 3.2) *before and after cleaning*.
  * **Status:** **TODO**
* **Task 17 (Targeted Tests):** For each feature calculated for Era 1:
  * Write unit tests validating its correctness using the Era 1 sample data (both clean and with pre-cleaning outlier scenarios).
  * Test how feature calculations behave with data cleaned according to Era 1-specific outlier rules.
  * Test calculations at the boundaries of Era 1 if relevant (e.g., for rolling features).
  * **Status:** In progress, but needs to be refocused with Era 1-specific data, cleaning pipeline, and an Era 1-specific feature set.

**4.8. `feature_selection.py`**

* **Task 18 (Defer or Scope):** Feature selection should ideally be performed *after* a stable, validated set of Era 1 features has been generated.
  * If applied now, it should operate *only* on the features generated from Era 1.
  * The relevance of selected features will be tied to the LSTM's target (predicting all numeric features from Era 1).
  * **Status:** **DEFER** active work on this file until Era 1 feature generation is mature.

**4.9. `__init__.py` and `db_connector.py`**

* **`__init__.py`:**
  * **Status:** **DONE** (Marks `src` as a package).
* **`db_connector.py`:**
  * **Status:** Assumed **DONE** and functional. Its role is to provide a connection; the Era 1 date filtering happens at a higher level.

**5. Timeline & Next Steps**

1. **Immediate (1-2 days):**
    * **Task 1 & 6 (Partial):** Implement and verify Era 1 data isolation in `flow_main.py` and `data_processing_config.json`. Ensure `extract_data_task` receives and uses correct Era 1 timestamps.
    * This means running the `main_feature_flow` with `segment_name="Era1_CoreEnv"` and checking the output data.

2. **Following (1-2 days):**
    * **Task (New - "Verify Era 1 Sensor Quality & Characteristics"):** Once isolated Era 1 Parquet files are available, load them. Programmatically analyze and document the list of truly reliable sensor columns *within this isolated Era 1 dataset, considering both availability (Sec 3.1) and outlier characteristics (Sec 3.2)*. This list is critical input for all subsequent feature engineering and cleaning decisions.

3. **Core Data Cleaning and Feature Refinement for Era 1 (1-2 weeks):**
    * **Tasks related to `config.py`, `data_processing_config.json`, `data_cleaning.py`, `feature_calculator.py`, `feature_engineering.py`, `test_feature_engineering.py` (specifically Task 2, 4, 5, 7, 11, 12, 13, 14, 15, 16, 17):** Systematically review, adapt, and test configurations, data cleaning logic, and feature calculation logic using the verified list of Era 1 sensors and their observed characteristics. Iteratively tune outlier rules and ensure features are robustly calculated for Era 1.

**6. Risks and Mitigation**

* **Risk:** Era 1 data, even if best, might still have unforeseen quality issues for specific sensors that persist after initial cleaning attempts.
  * **Mitigation:** Thorough programmatic verification in Step 2. Iterative refinement of cleaning rules (Task 12) based on impact on data distributions and downstream feature sensibility.
* **Risk:** The list of "good" and *usable* sensors in Era 1 might be smaller than anticipated after rigorous quality checks, limiting the complexity or number of derivable features.
  * **Mitigation:** Be prepared to simplify some feature calculations or exclude features if their inputs are insufficient in Era 1. Prioritize features most critical for the LSTM based on domain knowledge and initial modeling experiments.
* **Risk:** Adapting all files and configurations, especially for segment-specific logic and detailed outlier tuning, might take longer than estimated.
  * **Mitigation:** Prioritize tasks. Get Era 1 data isolation and basic cleaning working first. Then iterate on refining outlier rules and features one by one or in small groups. Maintain clear version control for configurations.
* **Risk:** Defining and tuning effective outlier detection rules for Era 1 that balance removing genuine errors versus preserving valid data extremes might be complex and iterative.
  * **Mitigation:** Start with established domain knowledge for hard limits. Use insights from IQR and Z-score plots (Sections 3.2) to guide parameter tuning for statistical methods. Iteratively review the impact of chosen rules on data distributions and downstream feature calculations. Prioritize rules for sensors most critical to the LSTM.

This focused approach should provide a more stable and reliable foundation for your LSTM modeling efforts.
