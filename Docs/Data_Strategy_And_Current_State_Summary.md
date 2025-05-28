# Data Strategy & Current State Summary for Greenhouse Simulation Project

**Document Version:** 1.0
**Date:** October 27, 2023
**Author:** AI Assistant (Gemini)
**Context:** This summary is based on detailed data analysis (see `Docs/sql_data_analysis.md`, `Docs/plan_for_segemnt.md`, and visual data exploration artifacts) and informs the project's data processing and modeling strategy.

## 1. Executive Summary

Recent comprehensive analysis of the historical sensor data (`sensor_data_merged` table, spanning Dec 2013 - Sep 2016) has revealed significant data quality and availability challenges. These findings critically impact our approach to feature engineering and machine learning model development for the greenhouse simulation and optimization project.

Key environmental sensors crucial for robust modeling—specifically `air_temp_c`, `relative_humidity_percent`, and `light_intensity_umol`—exhibit severe sparseness, with reliable data largely confined to an initial ~9-month period (approx. Dec 2013 - Aug 2014). This period is now designated **"Era 1" (`Era1_CoreEnv`)**. Furthermore, `light_intensity_umol` shows a very high percentage of outliers even within its limited availability, raising concerns about its fundamental reliability.

Due to these constraints, a strategic decision has been made to **initially focus all data processing, feature engineering, and baseline modeling efforts exclusively on the "Era 1" data segment.** This approach aims to build a high-quality, coherent feature set from the most reliable portion of the dataset, enabling the development of a sound baseline model before tackling the complexities of other data periods or more challenging sensors.

This document summarizes the key data challenges identified and outlines the strategic, phased approach being taken to mitigate these issues and build a reliable data foundation for the project.

## 2. Key Data Challenges Identified

Our analyses (SQL queries, visual availability plots, and outlier detection plots) have highlighted several critical issues:

1. **Severe Temporal Sparseness of Core Environmental Sensors:**
    * `air_temp_c`: ~83.2% missing overall. Reliable data primarily Dec 2013 - Sep 2014.
    * `relative_humidity_percent`: ~92.1% missing overall. Reliable data primarily Dec 2013 - Sep 2014.
    * `light_intensity_umol`: ~94.9% missing overall. Reliable data primarily Dec 2013 - Sep 2014.
    * **Implication:** Meaningful feature engineering and modeling relying on the simultaneous availability of these core sensors is only feasible for the "Era 1" period.

2. **Significant Data Quality Issues for `light_intensity_umol`:**
    * Within its already sparse available data, ~22.8% of `light_intensity_umol` readings were flagged as outliers based on the configured domain rules (0-3000 µmol).
    * **Implication:** This sensor's data is highly suspect and requires urgent investigation. Its use, even within Era 1, must be approached with extreme caution or it might need to be excluded.

3. **Variable Availability and Intermittency for Other Sensors:**
    * `co2_measured_ppm`: While having better overall temporal coverage (~11.7% missing), it exhibits patchiness and requires robust imputation.
    * Many actuator statuses, pipe temperatures, and AFD humidity sensors only become available in a later period ("Era 2", from late 2015 onwards), where the core environmental sensors from Era 1 are largely missing. This creates a significant disconnect in available feature sets across different time periods.

4. **High Reliance on Imputation:**
    * The extensive missing data, especially outside Era 1 for core sensors, means any attempt to use the full dataset would heavily rely on imputation. Linear interpolation over such long gaps (months/years) is statistically unsound and likely to produce misleading data.
    * **Implication:** This underscores the necessity of segmenting the data and focusing on periods where imputation requirements are minimized for key variables.

5. **Outlier Presence:**
    * Visual outlier analysis (IQR, Rolling Z-score) confirms the presence of anomalous readings in various sensors within Era 1. This necessitates careful tuning of outlier detection rules (`data_processing_config.json`) to be effective for Era 1 data characteristics.
    * **Implication:** Data cleaning is a critical pre-requisite. The parameters for outlier detection (e.g., domain bounds, Z-score windows) need to be specifically reviewed and adapted for Era 1.

## 3. Strategic Approach: Focus on "Era 1" (`Era1_CoreEnv`)

Given the data challenges, the project will adopt the following focused strategy, detailed in `Docs/plan_for_segemnt.md`:

1. **Isolate Era 1 Data:** The immediate priority is to adapt the data processing pipeline (`DataIngestion/simulation_data_prep/src/`) to strictly extract and process data only from the "Era1_CoreEnv" segment (Jan 1, 2014 - Aug 31, 2014).
    * This involves configuring and running the Prefect flow (`flow_main.py`) for this specific segment.

2. **Verify Era 1 Sensor Quality & Characteristics:** Once Era 1 data is isolated, a detailed programmatic analysis will be performed to:
    * Confirm the precise list of reliably available sensors within this segment.
    * Assess their data quality (NULL percentages, outlier rates after initial cleaning) specifically for Era 1.
    * This step will definitively guide which sensors can be trusted for feature engineering in Era 1.

3. **Refine Data Cleaning for Era 1:**
    * Based on the Era 1 sensor verification, the outlier detection rules and imputation strategies in `data_processing_config.json` will be meticulously reviewed and tuned to be optimal for Era 1 data. This includes addressing the `light_intensity_umol` issues and ensuring CO2 imputation limits are appropriate for observed gap lengths in Era 1.

4. **Refine Feature Engineering for Era 1:**
    * The feature set defined in `data_processing_config.json` for `Era1_CoreEnv` will be revised based *only* on the verified list of reliable and cleaned sensors from Era 1.
    * Feature calculation logic in `feature_calculator.py` will be reviewed and tested to ensure robustness with Era 1 data.

5. **Iterative LSTM Model Development:**
    * The LSTM surrogate model (`DataIngestion/model_builder/`) will initially be trained and evaluated using only the high-quality features generated from the cleaned Era 1 data.
    * This provides a solid baseline and allows for focused model tuning on reliable data.

## 4. Implications for Stakeholders & Next Steps

* **Initial Model Scope:** The first iteration of reliable LSTM models and any simulations will be based on, and validated against, the conditions and sensor data present during Era 1. Generalizing to other periods will be a subsequent, more complex task.
* **`light_intensity_umol` Decision:** A critical decision point will be the handling of `light_intensity_umol`. If investigation proves its data fundamentally unreliable even in Era 1, it may need to be excluded, impacting any features or model components that rely on precise PAR data.
* **Future Eras (Era 2, etc.):** Addressing other data periods (like "Era2_Actuators") will require a separate, dedicated analysis and potentially different modeling strategies or feature sets due to the different sensor availability profiles. This will be considered future work after establishing a robust process for Era 1.

**Immediate Next Steps (as per `Docs/plan_for_segemnt.md`):**

1. **Complete Era 1 Data Isolation:** Execute the data pipeline for the `Era1_CoreEnv` segment and verify the output.
2. **Perform In-depth Sensor Quality Analysis for Isolated Era 1 Data:** Generate a definitive list of reliable sensors for this period.
3. **Tune Data Cleaning Configurations:** Update `data_processing_config.json` based on Era 1 sensor analysis.

This focused, sequential approach is deemed the most effective way to manage the identified data complexities and build a trustworthy foundation for the project's analytical and simulation goals.
