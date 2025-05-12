# Chapter 3: Structural Analysis, Era Definition, and Data Segmentation

This chapter focuses on understanding and handling the larger structural characteristics of the time-series data, particularly the definition of distinct operational "Eras" and the subsequent segmentation of data within these Eras into continuous blocks. This is crucial because data availability, sensor behavior, and even control strategies can change significantly over the long multi-year span of the dataset.

## 3.1. Objectives

* Formally define "Eras" based on significant shifts in data availability or known operational periods.
* Discuss methods for programmatic Era identification (e.g., changepoint detection), relating to concepts from toolkits like "Augurs" or libraries such as `ruptures`.
* Detail the implementation of within-Era data segmentation to create contiguous data blocks, suitable for imputation and sequence-based modeling.
* Ensure that subsequent processing steps (imputation, feature engineering) can operate on these well-defined, continuous segments.

## 3.2. Rationale for Era Definition and Segmentation

* **Handling Non-Stationarity and Concept Drift:** Long time-series datasets from operational environments often exhibit non-stationarity (statistical properties change over time) and concept drift (relationships between variables or optimal control strategies change). Treating the entire dataset as a single homogeneous block can lead to poor model performance.
* **Addressing Major Data Gaps:** As identified in `Jupyter/classes/data_availability_analyzer/report.md`, there's a significant year-long gap in the `sensor_data_merged` table. Processing data in Eras that do not span this major gap is essential.
* **Varying Sensor Availability:** Different sets of sensors are active and reliable during different periods (Era 1 vs. Era 2). Era-specific processing allows for tailored feature sets.
* **Improving Imputation Quality:** Imputing missing values is more reliable within continuous segments where underlying trends and patterns are more consistent, rather than across major structural breaks or long gaps.
* **Preparing for Sequence-Based Models (LSTMs, GANs):** These models learn from sequences of data. Providing them with continuous, consistently characterized segments from specific Eras helps them learn more effectively.

## 3.3. Defining Major Operational Eras

* **Current Approach (Configuration-Driven based on Analysis):
  * The `data_processing_config.json` file, under `era_definitions`, explicitly defines `start_date` and `end_date` for "Era1" and "Era2".
  * These dates were determined by:
        1. Initial high-level analysis from `Jupyter/classes/data_availability_analyzer/report.md` identifying broad periods of differing sensor activity.
        2. Refined by SQL queries on `public.sensor_data_merged` to pinpoint more precise start/end times based on the presence of key indicator sensors for each conceptual Era.
  * The `preprocess.py` script's `fetch_source_data` function loads data only for the specified Era's date range.

* **Potential for Programmatic Era/Changepoint Detection (Conceptual - "Augurs" / `ruptures`):
  * **Concept:** Instead of manual date setting, advanced time-series analysis toolkits can programmatically identify significant changepoints in the data where statistical properties (mean, variance, trend, seasonality) shift.
  * **Toolkits:**
    * "Augurs" (if it refers to a specialized time-series toolkit with these capabilities and Python bindings).
    * Python libraries like `ruptures` are specifically designed for changepoint detection and offer various algorithms (e.g., Pelt, BinSeg, DynP).
  * **Methodology (High-Level):**
        1. Select one or more key, relatively continuous sensor signals from `sensor_data_merged` that span the entire dataset (e.g., `radiation_w_m2` might be a candidate).
        2. Apply a changepoint detection algorithm to identify dates where significant structural breaks occur.
        3. These detected changepoints would then inform the `start_date` and `end_date` for the Era definitions in `data_processing_config.json`.
  * **Advantages:** More data-driven and potentially more precise Era delineation. Can adapt if new data with different structural breaks is added.
  * **Current Status:** This programmatic detection is **not currently implemented** in `preprocess.py`. Era definition is manual via config. This is an area for future enhancement if finer-grained or automated Era/regime identification is needed.

## 3.4. Within-Era Data Segmentation (`preprocess.py` - `DataSegmenter` from `processing_steps.py`)

Even within a broadly defined Era, there can be shorter periods of missing data (e.g., a sensor offline for a few days). The `DataSegmenter` class aims to break down the loaded Era data (which has already been resampled to a regular frequency in Chapter 2) into smaller, continuous blocks.

* **Rationale:** Many imputation techniques and time-series models work best on fully continuous data. Segmenting by significant gaps prevents inappropriate imputation across these gaps and provides cleaner sequences.

* **Configuration (`data_processing_config.json`):
  * `segmentation.min_gap_hours`: An integer defining the minimum duration of a data gap (e.g., 24 hours) that will trigger the end of one segment and the beginning of a new one.
  * `common_settings.time_col` (referenced via `segmentation.time_col_ref` or passed to `DataSegmenter`): The name of the time column (which should be the `DatetimeIndex` at this stage after resampling).

* **Implementation (`DataSegmenter.segment_by_availability` method):
    1. **Input:** Receives the DataFrame for a specific Era, which has already been resampled to a regular `target_frequency` (output of Chapter 2 processes).
    2. **Ensure Time Column and Sort:** Verifies the time column exists and sorts the DataFrame by it (though it should already be sorted and indexed by time after resampling).
    3. **Iterate and Detect Gaps:**
        * Calculates the time difference (`time_diff`) between consecutive timestamps in the `DatetimeIndex`.
        * If `time_diff` exceeds `min_gap_for_new_segment` (converted from `min_gap_hours` to `pd.Timedelta`), a segment boundary is identified.
    4. **Create Segments:** The original DataFrame is sliced into multiple smaller DataFrames based on these boundaries.
    5. **Output:** Returns a list of Pandas DataFrames, where each DataFrame in the list represents a continuous block of data from the input Era.
    6. **Logging:** Reports the number of segments found and the time range and row count for each segment.

* **Application Point in `preprocess.py`:**
  * Called in the main Era processing loop *after* data loading, initial cleaning, resampling, and outlier handling.
  * Subsequent steps like imputation (Chapter 4) and scaling (Chapter 5) are then applied *per segment*.

## 3.5. Outputs from this Chapter

* For each processed Era:
  * A list of Pandas DataFrames, each representing a continuous data segment from that Era, with a regular `DatetimeIndex`.
  * Updated entries in the summary report (`preprocessing_summary_report_{era_identifier}.txt`):
    * The configured `min_gap_hours` for segmentation.
    * The number of continuous data segments found within the Era.
    * For each segment: its start time, end time, and number of rows.

## 3.6. Considerations for Subsequent Steps & Toolkits

* **Targeted Imputation (Chapter 4):** Each data segment, being continuous, is now ideally suited for imputation techniques. Imputation will fill NaNs *within* these segments, respecting their continuity.
* **Sequence Preparation for LSTMs/GANs:** These continuous segments are the basis for creating fixed-length sequences for model training. The windowing process (sliding a window of `max_sequence_len`) will operate on these segments.
* **`tsfresh` and Feature Engineering (Chapter 5):** If `tsfresh` is used, it can be applied per segment, or segments can be concatenated (with appropriate group IDs if needed) before feature extraction. Domain-specific features will also be calculated on these cleaned, segmented, and resampled DataFrames.
* **Impact of `target_frequency` and `min_gap_hours`:** The interplay between the resampling frequency (Chapter 2) and the segmentation gap threshold is important. A very high resampling frequency might create many small gaps that could then be bridged by imputation if `min_gap_hours` is large, or could lead to many very short segments if `min_gap_hours` is small.
