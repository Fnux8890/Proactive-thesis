# Analysis of `sensor_data_merged` Table for Simulation Optimization Project

**Document Version:** 1.0
**Date of Analysis:** July 26, 2024
**Analyst:** AI Assistant
**Configuration Context:** `DataIngestion/simulation_data_prep/src/data_processing_config.json`

## 1. Introduction

This document presents a detailed analysis of the `sensor_data_merged` PostgreSQL table. This table is understood to be a critical data source for the greenhouse control strategy simulation and optimization project. The analysis focuses on data availability, integrity (with a focus on outliers as defined in the project's configuration), and overall suitability for downstream processes such as feature engineering, model training, and simulation. All assessments are guided by the specifications and rules defined within the `DataIngestion/simulation_data_prep/src/data_processing_config.json` file. The findings aim to inform data preprocessing strategies, identify potential data quality issues that could impact software components, and provide actionable recommendations for improving data robustness and the reliability of data-driven systems.

## 2. Executive Summary

The `sensor_data_merged` table, containing 127,206 records and spanning from December 2013 to September 2016, presents significant data quality challenges that have direct implications for software engineering efforts. Most notably, key environmental sensors crucial for the `Era1_CoreEnv` data segment (specifically `air_temp_c`, `relative_humidity_percent`, `light_intensity_umol`) exhibit severe data sparseness. Their data is primarily available only for an initial ~9-month period (approx. Dec 2013 - Sep 2014). This dramatically restricts the temporal scope of any software modules, analyses, or machine learning models relying on these sensors.

While `radiation_w_m2` and `co2_measured_ppm` demonstrate better temporal coverage, `co2_measured_ppm` still presents an ~11.7% missing data rate, which will necessitate careful imputation. Outlier detection, based on rules in `data_processing_config.json`, revealed a particularly high number of problematic values for `light_intensity_umol` (~22.8% of its already sparse data), suggesting potential sensor or data pipeline faults.

The system will heavily depend on imputation strategies defined in the configuration. The effectiveness and potential biases introduced by these methods, especially for columns with extensive missing periods or high outlier rates, require rigorous validation to prevent unreliable system behavior or simulation outcomes. These findings underscore the need for robust data validation and preprocessing modules within the data ingestion pipeline.

## 3. Methodology

The analysis was conducted by executing a series of SQL queries against the `sensor_data_merged` table within the project's PostgreSQL database. The queries were specifically designed to:

* **Establish Baseline:** Determine the total data volume (record count) and the overall temporal range (min/max timestamps).
* **Assess Data Completeness:** For key sensor columns identified in the `Era1_CoreEnv` segment of `data_processing_config.json` (`air_temp_c`, `relative_humidity_percent`, `radiation_w_m2`, `co2_measured_ppm`, `light_intensity_umol`), quantify NULL versus NON-NULL records and determine the actual time window for which data points exist.
* **Identify Outliers:** Query for records that violate the domain rules (e.g., min/max valid values) specified in the `outlier_detection` section of the configuration file.
* **Evaluate Imputation Needs:** Determine NULL counts for columns with specific imputation strategies outlined in the configuration (e.g., `rain_status`, `co2_measured_ppm` with its imputation limit).

## 4. Detailed Findings

### 4.1. General Table Characteristics

    *   **Total Records:** 127,206
    *   **Overall Time Range:** 2013-12-01 00:00:00Z to 2016-09-08 00:00:00Z.
    *   **Alignment with Configuration:** This observed time range precisely matches the `processing_start_date` and `processing_end_date` parameters within `data_processing_config.json`, indicating the dataset aligns with the intended global processing window.

### 4.2. Data Availability and Completeness

A critical finding is the significant disparity in data availability across key environmental sensors. This has direct consequences for the design and operational validity of any software components or models that consume this data.

| Column Name                 | Non-NULL Records | % Non-NULL | NULL Records | % NULL | First Data Point        | Last Data Point         | Software Engineering Implications                                                                                                                                                              |
|-----------------------------|------------------|------------|--------------|--------|-------------------------|-------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `radiation_w_m2`            | 121,372          | ~95.4%     | 5,834        | ~4.6%  | 2013-12-21T23:00:00.000Z | 2016-09-06T10:50:00.000Z | **High Reliability:** Good data coverage across most of the dataset's timespan. The low percentage of missing data suggests this sensor input is generally reliable for features/models.         |
| `co2_measured_ppm`          | 112,309          | ~88.3%     | 14,897       | ~11.7% | 2013-12-01T00:00:00.000Z | 2016-09-06T10:50:00.000Z | **Moderate Reliability:** Reasonable coverage. The ~11.7% missing data will require robust imputation. The configured `limit: 5` for linear fill needs validation against typical gap lengths to ensure efficacy. |
| `air_temp_c`                | 21,404           | ~16.8%     | 105,802      | ~83.2% | 2013-12-01T00:00:00.000Z | 2014-09-30T23:00:00.000Z | **Low Reliability (Temporal Constraint):** Severe sparseness. Usable data is primarily restricted to Dec 2013 - Sep 2014. This critically impacts any system component or model requiring this sensor data outside this specific window. |
| `relative_humidity_percent` | 10,047           | ~7.9%      | 117,159      | ~92.1% | 2013-12-01T00:00:00.000Z | 2014-09-30T23:00:00.000Z | **Very Low Reliability (Temporal Constraint):** Extreme sparseness. Similar critical impact as `air_temp_c`. High reliance on imputation, which may be statistically unsound or misleading over such long gaps. |
| `light_intensity_umol`      | 6,552            | ~5.1%      | 120,654      | ~94.9% | 2013-12-01T00:00:00.000Z | 2014-09-30T23:00:00.000Z | **Extremely Low Reliability (Temporal Constraint & Quality Issues):** Extreme sparseness. Data is practically unusable outside the initial ~9-month period without potentially misleading imputation. Also see significant outlier issues (Section 4.3). |

**Key Software Engineering Implications for Data Availability:**

* **Temporal Validity of Software Modules:** Any software modules, machine learning models, or feature engineering pipelines that depend on `air_temp_c`, `relative_humidity_percent`, or `light_intensity_umol` will have their operational validity largely confined to the December 2013 - September 2014 period. Attempting to use these components outside this timeframe with imputed data could lead to unpredictable or erroneous system behavior and simulation results. This necessitates careful versioning or conditional logic in software components.
* **`Era1_CoreEnv` vs. Full Range Processing:** The `Era1_CoreEnv` data segment, as defined in `data_processing_config.json` (ending 2014-08-31), aligns with the availability of these sparse sensors. Any data processing pipelines or feature engineering efforts intended for the "Full_Range_Test" or later periods (e.g., "Era2_Actuators" which starts Oct 2015) will encounter major data gaps for these core environmental variables, potentially leading to runtime errors or silent failures if not handled robustly.
* **Data Ingestion Pipeline & Monitoring:** The severe and temporally localized sparseness suggests potential past systemic issues with sensor data collection, data transmission protocols, or a significant change in sensor deployment or logging strategy that was not uniform. Future system design must incorporate robust data ingestion health checks and monitoring for these critical variables to prevent recurrence.

### 4.3. Outlier Detection and Data Integrity

Outliers were identified based on the domain rules specified in the `outlier_detection` section of `data_processing_config.json`. The configured handling strategy for these outliers is `clip`.

| Column Name                 | Domain Rule (from config) | Outlier Count | % of Non-NULL Data | Software Engineering Implications                                                                                                                                                                                               |
|-----------------------------|---------------------------|---------------|--------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `air_temp_c`                | -10 to 45 °C              | 28            | ~0.13%             | **Minor Issue:** Low impact on overall data quality. The `clip` strategy is a reasonable default for these few values. Preprocessing modules should log such clippings.                                                  |
| `relative_humidity_percent` | 0 to 100%                 | 0             | 0%                 | **Data Conforms:** Data is within the expected domain. No immediate action for outlier handling modules.                                                                                                                    |
| `radiation_w_m2`            | 0 to 1500 W/m²            | 0             | 0%                 | **Data Conforms:** Data is within the expected domain.                                                                                                                                                                      |
| `light_intensity_umol`      | 0 to 3000 µmol            | 1,494         | ~22.8%             | **Major Data Quality Issue:** A very high proportion of outliers relative to its already sparse available data. This strongly suggests sensor malfunction, persistent calibration issues, or an incorrect domain definition in the config. Clipping these values might mask serious underlying data integrity problems and lead to misleading model inputs. |
| `co2_measured_ppm`          | 100 to 1500 ppm           | 10            | ~0.009%            | **Minor Issue:** Very low impact. `clip` strategy is acceptable. Preprocessing modules should log clippings.                                                                                                              |

**Key Software Engineering Implications for Outliers:**

* **`light_intensity_umol` Integrity Risk:** The high outlier rate for `light_intensity_umol` is a critical data integrity concern. Relying on this data, even after the `clip` strategy is applied by a preprocessing module, could introduce significant noise or bias into any downstream system. This warrants urgent investigation into the data source or sensor itself. If the data is deemed fundamentally unreliable, the variable might need to be programmatically excluded from models or feature sets, or a specific data quality warning system implemented.
* **Effectiveness and Specificity of Domain Rules:** For most variables, the domain rules defined in the configuration appear effective in catching plausible outliers without flagging excessive amounts of data. However, the `light_intensity_umol` case highlights the need for potentially more adaptive or context-aware outlier detection mechanisms if such issues are recurrent. Preprocessing modules should be designed to be configurable and allow for updates to these rules.

### 4.4. Imputation Strategy Considerations and Impact

The `data_processing_config.json` file specifies imputation strategies for handling missing data, which will be invoked by data preparation modules.

* **`rain_status`**:
  * **NULL Count:** 120,654 records (~94.8% of total).
  * **Strategy:** `forward_fill` (as per config).
  * **SE Implication:** Extreme reliance on imputation. `forward_fill` assumes persistence of the last known state. While this might be a reasonable heuristic for `rain_status` over short durations, its application over potentially extended missing data gaps (given 94.8% missingness) could lead to significant inaccuracies. The root cause of this massive data loss should be investigated to improve future data collection. Imputation modules should log the extent of fill operations.
* **`co2_measured_ppm`**:
  * **NULL Count:** 14,897 records (~11.7% of total).
  * **Strategy:** `linear` interpolation with a `limit` of 5 consecutive NULLs (as per config).
  * **SE Implication:** The ~11.7% missing data is generally manageable if missing data gaps are typically short (i.e., <= 5 records). The `limit: 5` parameter is crucial; an analysis of consecutive NULL sequence lengths is strongly recommended to confirm this limit doesn't leave a substantial number of values unimputed, which would then require a secondary handling strategy or result in data loss. Imputation modules must correctly implement this limiting behavior.
* **Default Imputation Strategy (`linear` for other numeric columns):**
  * Columns like `air_temp_c` (83.2% NULLs), `relative_humidity_percent` (92.1% NULLs), and `light_intensity_umol` (94.9% NULLs) will be heavily processed by linear interpolation by data preparation modules due to their extremely high NULL counts.
  * **SE Implication:** Linear interpolation over very long gaps (which are highly likely given the temporal sparseness observed) can introduce artificial trends, smooth out genuine variability, and create data that does not reflect reality. The imputed values for these sensors outside their reliable data window (Dec 2013 - Sep 2014) should be treated with extreme caution by any consuming software module. These modules should ideally be designed to recognize or be configured to handle periods of low-confidence imputed data, possibly by down-weighting their influence or switching to alternative logic.

## 5. Overall System Impact and Actionable Recommendations

The observed state of the `sensor_data_merged` table has profound implications for the design, development, testing, and operational reliability of the entire simulation and optimization system.

**Key System-Level Impacts:**

1. **Limited Temporal Scope for Core Models & Simulations:** Software components (models, feature extractors, simulation engines) relying on `air_temp_c`, `relative_humidity_percent`, or `light_intensity_umol` will be most reliable *only* for the Dec 2013 - Sep 2014 period. Any system functionality attempting to use these sensors outside this window operates with data of significantly lower confidence, potentially leading to incorrect simulations or suboptimal control decisions.
2. **Data Quality Concerns for `light_intensity_umol`:** This sensor's data is highly problematic due to both extreme sparseness and a very high outlier rate. Its inclusion in any critical calculation or model input is risky without substantial cleanup, validation, or explicit acknowledgment of its low quality.
3. **High Reliance on Imputation Logic:** The system's data preprocessing modules will perform a large volume of imputation. The chosen methods and their parameters (e.g., `limit` for linear fill) must be robustly implemented and validated. Consider implementing mechanisms to log or flag imputed values to trace their influence on system outputs and facilitate debugging.
4. **Evidence of Potential Historical Data Collection Deficiencies:** The distinct patterns of missing data strongly suggest possible past failures or inconsistencies in data collection pipelines for several critical sensors. These are important lessons for the design of current or future data ingestion systems.

**Actionable Recommendations for Software Engineering and Data Management:**

1. **Prioritize Urgent Investigation of `light_intensity_umol` Data:**
    * **Action:** Conduct a root cause analysis for the `light_intensity_umol` data issues. This involves checking historical sensor calibration records, known operational issues during the Dec 2013 - Sep 2014 period, or verifying if the domain range in `data_processing_config.json` is appropriate for all operational conditions.
    * **Decision Point & Software Adaptation:** Based on findings, make an informed decision:
        * If data is salvageable: Implement specific data correction routines in the preprocessing pipeline.
        * If data is unreliable: Programmatically exclude this sensor from feature sets and models, or implement clear warnings in system outputs if its use is unavoidable. Update configuration to reflect its status.
2. **Validate and Refine Imputation Module Logic:**
    * **Action:** For `co2_measured_ppm`, perform a specific analysis of the distribution of consecutive NULL sequence lengths. This will empirically validate or suggest adjustments to the `limit: 5` parameter in the `data_processing_config.json`.
    * **Action:** For `air_temp_c`, `relative_humidity_percent`, and `light_intensity_umol`, critically evaluate the impact of applying linear interpolation over extensive gaps. Consider developing or integrating more sophisticated imputation methods if feasible (e.g., using correlations with other, more complete sensors, simple model-based imputation, or leveraging domain knowledge).
    * **Software Enhancement:** Enhance imputation modules to log detailed statistics about their operations (e.g., number of values imputed per column, length of gaps filled). Implement a mechanism to flag imputed data points (e.g., add a boolean companion column `[sensor_name]_is_imputed`) to allow downstream modules to assess their impact or handle them differently.
3. **Adapt Software to Data Availability Constraints (Temporal Segmentation):**
    * **Action:** Clearly define and document within the system's architecture which features, models, and simulation scenarios can operate reliably across the full 2013-2016 range versus those that are strictly confined to the ~2013-2014 period due to sensor data availability.
    * **Software Logic:** Implement conditional logic or distinct processing paths in software components based on the time period being processed, selecting appropriate models or feature sets that match data reliability for that period.
    * **Configuration Update:** Review and potentially update the `data_segments` in `data_processing_config.json` or ensure downstream logic correctly interprets these segments in light of the actual data availability, especially if a segment like "Full_Range_Test" is intended to use these sparse sensors.
4. **Strengthen Data Ingestion and Monitoring in Current/Future Systems:**
    * **Action:** For any ongoing or future data collection efforts related to this project, design and implement robust real-time or near-real-time monitoring and alerting for sensor data streams. This should detect gaps, anomalies, and communication failures promptly.
    * **System Design Principle:** Document these findings as critical lessons learned for future sensor deployment strategies and data pipeline architecture to prioritize data quality, completeness, and fault tolerance.
5. **Implement Iterative Data Validation in Preprocessing Pipeline:**
    * **Action:** Design the data preprocessing pipeline to include validation steps *after* cleaning and imputation. Re-run summary statistics and outlier checks post-imputation to quantify the impact of these operations and ensure they behave as expected, without introducing new artifacts.
    * **Automated Testing:** Develop automated tests for the preprocessing pipeline that use known input data with gaps/outliers to verify that cleaning and imputation logic works correctly according to configuration.

## 6. Conclusion

The `sensor_data_merged` table provides a valuable historical dataset that, despite its challenges, can yield insights for the simulation and optimization project. However, its direct use without addressing the highlighted data quality issues would pose significant risks to the reliability and accuracy of any software system built upon it. While `radiation_w_m2` and `co2_measured_ppm` offer broader utility, the severe sparseness and notable quality issues with `air_temp_c`, `relative_humidity_percent`, and especially `light_intensity_umol` necessitate careful, context-aware data preprocessing. This includes robust imputation logic, clear handling of outliers, and an architectural approach in the software that acknowledges and adapts to the variable temporal reliability of different data streams.

Addressing the recommendations outlined in this report will be crucial for developing a robust, reliable, and ultimately successful simulation and optimization system. This involves not just data science efforts but also careful software engineering to build resilient data pipelines and adaptable analytical components.
