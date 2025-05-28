# Chapter 1: Data Ingestion and Initial Cleaning

This chapter provides an in-depth discussion of the foundational steps for accessing, loading, and performing initial cleaning and validation of time-series sensor data. The primary source is the `public.sensor_data_merged` table within the TimescaleDB database. All processes are designed to be configuration-driven, leveraging `data_processing_config.json`, and tailored for preparing horticultural datasets for advanced modeling techniques such as LSTMs and Generative Adversarial Networks (GANs), including DoppelGANger.

## 1.1. Objectives - Detailed View

The primary objectives of this initial data ingestion and cleaning phase are:

1. **Reliable and Segmented Data Access:** To establish a robust mechanism for securely connecting to the database and fetching data specifically for predefined operational periods, termed "Eras." These Eras are critical as sensor availability and data characteristics can vary significantly across the dataset's timespan, as identified in preliminary analyses (e.g., `Jupyter/classes/data_availability_analyzer/report.md`).
2. **Time Column Standardization:** To ensure the primary time column (e.g., `time`) is consistently formatted as a Pandas `datetime` object, localized to UTC, and serves as the basis for chronological sorting and indexing. Rows with unparseable or missing primary timestamps must be removed to maintain data integrity for time-series operations.
3. **Strategic Column Exclusion:** To systematically remove columns from each Era-specific dataset that are irrelevant, uninformative (e.g., static, metadata not used for modeling), globally "dead" (consistently empty or near-empty across the entire dataset), or characteristic of other Eras and not pertinent to the current Era's focus. This step is crucial for reducing noise, computational load, and focusing models on relevant signals.
4. **Data Type Integrity and Conversion:** To verify and enforce appropriate data types for key columns. This includes converting boolean or binary status indicators (e.g., `rain_status`, `lamp_grpX_status`) into a consistent numerical format (0/1 integers) suitable for machine learning model input.
5. **Robust Outlier Management:** To identify and treat extreme or obviously erroneous sensor readings that could disproportionately influence subsequent statistical analyses, feature engineering, and model training. The primary method employed is value clipping (Winsorization) based on domain-specific or empirically determined thresholds.
6. **Baseline DataFrame Creation:** To produce a cleaned, validated, and consistently formatted Pandas DataFrame for each defined Era. This DataFrame serves as the foundational input for subsequent, more advanced preprocessing steps detailed in later chapters (e.g., resampling, imputation, feature scaling, and feature engineering).
7. **Comprehensive Logging and Reporting:** To maintain a clear audit trail of all operations, including data shapes at each step, configurations used, columns dropped, and outlier treatments applied. This is captured in console logs and a per-Era summary text file (`preprocessing_summary_report_{era_identifier}.txt`).

## 1.2. Configuration-Driven Data Loading for Defined Eras - Detailed Implementation

This section details the methodology for loading Era-specific data segments from the database, driven by `data_processing_config.json` and implemented in `preprocess.py` using `db_utils.py`.

* **Rationale for Era-Based Loading:**
  * The `Jupyter/classes/data_availability_analyzer/report.md` clearly indicated that the `public.sensor_data_merged` table contains distinct periods ("Eras") with significantly different sensor availability patterns and data quality. A monolithic approach to processing this data would obscure these differences or lead to feature sets dominated by the lowest common denominator of available sensors.
  * Processing data on an Era-by-Era basis allows for tailored data cleaning, feature selection, and potentially Era-specific model training, leading to more robust and accurate outcomes.

* **`data_processing_config.json` Structure for Data Loading:**
  * **`database_connection` Object:**
    * Contains keys like `user`, `password`, `host`, `port`, `dbname`.
    * Values are ideally hardcoded for the Docker environment (e.g. `host: "db"`) or reference environment variable names (e.g. `env_var_user: "PG_USER"`) for flexibility and security, which `preprocess.py` then resolves using `os.getenv()`.
  * **`era_definitions` Object:**
    * A parent object containing nested objects, one for each defined Era (e.g., `"Era1"`, `"Era2"`).
    * Each Era object **must** contain:
      * `"db_table"`: String, the fully qualified name of the source table (e.g., `"public.sensor_data_merged"`).
      * `"start_date"`: String, ISO 8601 timestamp (e.g., `"2013-12-01T00:00:00Z"`) marking the beginning of the Era.
      * `"end_date"`: String, ISO 8601 timestamp (e.g., `"2014-08-27T23:59:59Z"`) marking the end of the Era.
    * May also contain a `"description"` field for clarity.
  * **`common_settings` Object:**
    * Must contain `"time_col"`: String, the name of the primary timestamp column in the database table (e.g., `"time"`).

* **Python Implementation (`preprocess.py` - `fetch_source_data` function):**
  * **Signature:** `fetch_source_data(era_identifier: str, era_config: dict, global_config: dict) -> pd.DataFrame`
  * **Connection:**
    * Retrieves database connection parameters from `global_config["database_connection"]`.
    * Instantiates `SQLAlchemyPostgresConnector` from `db_utils.py` using these credentials. The connector handles engine creation and connection pooling.
  * **Query Construction:**
    * Extracts `db_table`, `start_date`, `end_date` from the passed `era_config`.
    * Extracts the `time_col` name from `global_config["common_settings"]`.
    * Dynamically constructs a SQL query string like:

            ```sql
            SELECT *
            FROM {db_table}
            WHERE "{time_col}" >= :start_date AND "{time_col}" <= :end_date
            ORDER BY "{time_col}" ASC;
            ```

    * Uses SQLAlchemy-style named parameters (e.g., `:start_date`) for clarity and security with `text().bindparams()`.
  * **Data Fetching:**
    * Calls `db_connector.fetch_data_to_pandas(sql_text(query).bindparams(**params))`, where `params` is a dictionary like `{'start_date': start_date_val, 'end_date': end_date_val}`.
    * `pd.read_sql_query` (via `db_utils.py`) executes the query and loads results into a Pandas DataFrame.
  * **Error Handling:** Includes `try-except` blocks for database connection errors and query execution errors, returning an empty DataFrame on failure and logging the issue.
  * **Logging:** Prints information about the Era being fetched, the constructed query (or its key parameters like date range and table), and the shape of the fetched DataFrame.

## 1.3. Initial Data Integrity and Formatting - Detailed Implementation

This step, primarily handled by the `sort_and_prepare_df` function in `preprocess.py`, ensures fundamental data consistency after loading.

* **Time Column Processing (Primary Key):**
  * **Requirement:** A valid, chronological time index is paramount for all subsequent time-series operations.
  * **Implementation:**
        1. The function receives the DataFrame loaded by `fetch_source_data`.
        2. Identifies the time column name from `config.get('common_settings',{}).get('time_col', 'time')`.
        3. **Conversion & Validation:** `df[time_col_name] = pd.to_datetime(df[time_col_name], errors='coerce', utc=True)`.
            *`errors='coerce'`: Any values that cannot be parsed into a valid datetime will be converted to `NaT` (Not a Time).
            * `utc=True`: Standardizes all timestamps to UTC, crucial for avoiding timezone ambiguities, especially if data originates from multiple sources or across daylight saving transitions.
        4. **NaN Timestamp Removal:** `df.dropna(subset=[time_col_name], inplace=True)`. Rows where the primary timestamp could not be parsed (and are thus `NaT`) are removed as they cannot be placed chronologically.
        5. **Sorting:** `df.sort_values(by=time_col_name, inplace=True)`. Ensures data is in strict chronological order.
        6. **Index Reset:** `df.reset_index(drop=True, inplace=True)`. Provides a clean default integer index after sorting.
  * **Logging:** Reports the processed time range (`min` and `max` of the time column).

* **Default ID Column Creation:**
  * **Requirement:** Some time-series libraries (e.g., `tsfresh` if used later for comprehensive feature extraction) work more effectively if there's an identifier column, especially for datasets with multiple independent time series. For a single greenhouse, this might be a single, constant ID across all records of an Era.
  * **Implementation:**
        1. Identifies the `id_col` name from `config.get('common_settings',{}).get('id_col', 'entity_id')`.
        2. Checks if this column already exists in the DataFrame.
        3. If not, it creates the column and populates it with a default value. This default value is derived from the Era's description or its identifier (e.g., `config.get('era_definitions',{}).get(era_identifier,{}).get('description', era_identifier)`), making it specific to the Era being processed.
  * **Relevance for GANs:** For DoppelGANger, if each sequence is treated as an independent sample from a single process (the greenhouse in a specific Era), this explicit ID column might be less critical for the GAN training itself but remains good practice for data management.

* **Boolean/Status Column Type Conversion to Integer (0/1):**
  * **Requirement:** Ensure consistent numerical representation for boolean flags and status indicators for direct use in mathematical models and scaling.
  * **Configuration (`data_processing_config.json`):** A list of column names intended to be boolean/status is provided under `common_settings.boolean_columns_to_int` or an Era-specific override `era_definitions.[era_id].boolean_columns_to_int`.
  * **Implementation:**
        1. The script retrieves the appropriate list of `boolean_columns_to_int`.
        2. For each column in this list present in the DataFrame:
            *If `df[col].dtype == 'bool'`: Directly convert using `df[col] = df[col].astype(int)`.
            * If `pd.api.types.is_numeric_dtype(df[col])` and `df[col].dropna().isin([0, 1]).all()`: This handles cases where a status might have been loaded as `0.0`/`1.0` floats. It's converted to `float` first (to handle potential mixed types like `int` and `float` if `dropna().isin()` was on a subset) and then to `pd.Int64Dtype()` (nullable integer) to preserve `NaN`s if any exist after other processing but ensure an integer base.
            * Logs warnings if a column marked for conversion does not meet the expected type criteria.
  * **Rationale:** This step ensures these features are directly usable by scalers and neural networks, which expect numeric inputs. `pd.Int64Dtype()` is preferred over `int` when NaNs might still be present to avoid errors or unintended conversion of NaNs to a number like 0.

## 1.4. Strategic Column Exclusion - Detailed Criteria & Implementation

This step, performed in `sort_and_prepare_df` by dropping columns listed in `era_definitions.[era_id].dead_columns_to_drop` from the config, is vital for creating focused and efficient datasets for each Era.

* **Rationale:** Not all columns in `sensor_data_merged` are useful for every analytical task or for training generative models for every defined Era. Excluding irrelevant or problematic columns reduces dimensionality, noise, training time, and can lead to better model performance and interpretability.
* **Process:** The `df.drop(columns=cols_to_drop, inplace=True)` operation is used.
* **Detailed Criteria for Populating `dead_columns_to_drop` (per Era):**
    1. **Pure Metadata/Non-Sensor Data:**
        * Examples: `format_type`, `uuid`, `source_file`, `source_system`, `lamp_group` (unless `lamp_group` or `source_system` are intentionally retained to be used as categorical attributes for conditional GAN generation).
        * Reason: These typically don't carry direct information about the greenhouse state for a numerical model, or their high cardinality (like `uuid`) makes them unsuitable as features.
    2. **Semantically Uninformative or Globally "Dead" Columns:**
        * Examples: `behov`, `value`, `status_str`, `oenske_ekstra_lys`, `timer_on`, `timer_off`, `lampe_timer_on`, `lampe_timer_off`, `measured_status_bool`.
        * Reason: These were identified in `report.md` as being largely empty or uninformative across the entire dataset. Even if SQL queries on `sensor_data_merged` show them as 100% present for the *rows included* in a specific Era's date range, their *actual values* must be verified. If they are constant, default, or carry no dynamic information relevant to the greenhouse environment or control, they should be dropped.
    3. **Clear Redundancy (within the context of an Era):**
        * Example for Era 1: If `air_temp_c` is selected as the primary air temperature, `air_temp_middle_c` might be dropped if it measures a very similar point or is less reliable for Era 1. If multiple light sensors like `light_intensity_umol` (PAR), `radiation_w_m2` (global solar), `light_intensity_lux`, and `outside_light_w_m2` are all present, select the most horticulturally relevant and reliable ones for Era 1 (typically PAR and global solar) and drop others to avoid multicollinearity and simplify the feature set.
        * Example for Era 2: If `air_temp_middle_c` becomes the primary temperature sensor, `air_temp_c` might be dropped. If `radiation_w_m2` is the chosen global light sensor, other light metrics might be dropped if they don't add unique information for Era 2's focus.
        * Reason: Reduces model complexity and avoids issues related to multicollinearity.
    4. **Forecasts/Alternative "Actuals":**
        * Examples: `temperature_forecast_c`, `sun_radiation_forecast_w_m2`, `temperature_actual_c`, `sun_radiation_actual_w_m2`.
        * Reason: If direct, reliable sensor measurements (e.g., `air_temp_c`, `radiation_w_m2`) are available and chosen as primary for an Era, their corresponding forecast or alternative "actual" columns are often less useful for training a GAN to learn the fundamental sensor dynamics and can be dropped.
    5. **Aggregated Values or Setpoints (if raw data is preferred for GAN modeling):**
        * Examples: `dli_sum` (if PAR or global radiation is available to calculate it from), `co2_required_ppm`, `heating_setpoint_c`.
        * Reason: For a GAN, it's often better to learn from the underlying raw sensor data that leads to these aggregates or to model the system's response to setpoints rather than treating setpoints themselves as a primary feature to be generated (unless the goal is to generate setpoint strategies).
    6. **Columns Predominantly Characterizing *Other* Eras:**
        * Reason: This is key for Era-specific modeling. If a sensor set is characteristic of Era 1's operational mode and largely absent or irrelevant in Era 2 (e.g., `light_intensity_umol` being good in Era 1 but mostly missing in Era 2 as per `report.md`), it should be dropped from Era 2's dataset to avoid introducing noise or requiring excessive imputation for a feature that isn't central to that Era. Conversely, Era 2-specific actuators (like detailed lamp group statuses or AFD vent data) should be dropped from Era 1 if they are largely inactive or missing there.
        * The definition of these is guided by the comprehensive data availability analysis (`report.md`) and confirmed by SQL queries on `sensor_data_merged` for each Era's full date range.

## 1.5. Outlier Detection and Treatment - Detailed Implementation

This step, performed by `preprocess.py` calling `OutlierHandler` from `processing_steps.py`, aims to mitigate the impact of erroneous or non-physical sensor readings.

* **Rationale for Outlier Treatment:**
  * Sensor malfunctions, transmission errors, or other anomalies can introduce extreme values that do not reflect the true greenhouse environment.
  * These outliers can severely skew statistical calculations (means, standard deviations), negatively impact the fitting of scalers (MinMaxScaler is very sensitive to outliers), and disproportionately influence the training of neural networks (LSTMs, GANs), potentially leading them to learn incorrect patterns or struggle to converge.
* **Configuration (`data_processing_config.json`):
  * The `preprocessing_rules` section contains named rule sets (e.g., `"default_outlier_rules"`, `"era1_specific_outlier_rules"`).
  * Each Era definition in `era_definitions` references one of these rule sets via `"outlier_rules_ref"`.
  * Each rule within a set is a dictionary: `{"column": "column_name", "min_value": X, "max_value": Y, "clip": true}`.
    * `"column"`: The target column name.
    * `"min_value"`, `"max_value"`: Plausible lower and upper bounds for the sensor reading.
    * `"clip": true`: Indicates that values outside these bounds should be capped (Winsorized).
* **Method - Clipping (Winsorization via `OutlierHandler.clip_outliers`):**
  * The `OutlierHandler` iterates through the rules defined for the current Era.
  * For each column with a rule, it uses Pandas' `.clip(lower=min_val, upper=max_val)` method. This replaces values less than `min_value` with `min_value`, and values greater than `max_value` with `max_value`.
  * **Reason for Choosing Clipping:** It's a straightforward and robust method that retains the data point (unlike removal) but limits its extreme influence. It's generally preferred over replacing with `NaN` as an initial step if the goal is to preserve as much temporal continuity as possible before imputation of *originally* missing values.
* **Alternative Treatments Considered (and why clipping is chosen for this stage):**
  * **Treating as Missing (`NaN`):** This is a valid alternative. Outliers would be replaced with `NaN` and then handled by the subsequent imputation step (Chapter 4). This might be preferred if capping feels too artificial for certain sensors. However, it converts an existing (though erroneous) data point into a missing one, which then requires imputation that might be less accurate than capping if the outlier was isolated.
  * **Transformation (e.g., Log, Box-Cox):** These can reduce the skewness caused by outliers in some distributions but fundamentally change the scale and interpretation of the feature. This is more of a feature engineering choice than a simple cleaning step and might be considered later if specific features require it.
  * **Statistical Methods (Z-score, IQR):** While useful for EDA, applying them automatically to diverse time series with different seasonalities, trends, and operational setpoint changes can be risky, as they might incorrectly flag valid extreme operational values as outliers. Domain-based min/max thresholds are generally safer for initial cleaning.
* **Determining Thresholds (`min_value`, `max_value`):
  * **Primary Source:** Sensor specifications and horticultural domain knowledge (e.g., air temperature in a greenhouse is unlikely to be -50°C or +70°C).
  * **Secondary Source:** Exploratory Data Analysis (EDA) and visualizations (like those produced by `Jupyter/classes/data_availability_analyzer/data_availability.py` or ad-hoc plotting) can help identify current data ranges and potentially erroneous spikes, informing reasonable bounds.
* **Application Point in Pipeline:**
  * Outlier treatment is applied *after* initial data loading, time column standardization, and irrelevant column dropping for an Era.
  * It is performed *before* time-series resampling (Chapter 2) and detailed imputation of missing values (Chapter 4). This ensures that resampling and imputation operate on data that is already cleaned of extreme non-physical values.

## 1.6. Outputs from this Chapter (per Era)

Upon completion of the steps outlined in this chapter, for each processed Era, the following will be generated:

* **Cleaned and Filtered Pandas DataFrame:** This DataFrame will have:
  * A standardized, UTC-localized, and sorted `Datetime` column (typically as the index after `sort_and_prepare_df` but before resampling).
  * Only the columns selected for retention for that specific Era (irrelevant/dead columns dropped).
  * Boolean/status columns converted to 0/1 integer representation.
  * Specified numerical columns clipped to their defined plausible min/max ranges.
* **Updated Summary Report (`preprocessing_summary_report_{era_identifier}.txt`):** This text file will include entries detailing:
  * The specific Era identifier and its configuration from `data_processing_config.json`.
  * The shape (rows, columns) of the data loaded directly from the database for the Era's date range.
  * The list of columns that were dropped based on the `dead_columns_to_drop` configuration for that Era.
  * The shape of the DataFrame after these columns were dropped.
  * The shape of the DataFrame after outlier clipping has been applied.
  * (Optionally, a count of values changed by clipping per column if verbose logging is enabled in `OutlierHandler`).

## 1.7. Considerations for Subsequent Steps & Toolkits (e.g., `tsfresh`, "Augurs")

This initial data ingestion and cleaning phase is foundational for all subsequent preprocessing and feature engineering efforts.

* **Input for Time Regularization (Chapter 2):** The cleaned DataFrame per Era, with its validated time column, is the direct input for the resampling process. Resampling will establish a perfectly regular time grid, which may introduce NaNs if upsampling occurs.
* **Improved Basis for Imputation (Chapter 4):** By treating outliers *before* imputation, we prevent extreme values from unduly influencing imputation methods (e.g., linear interpolation between a valid point and an extreme outlier would be skewed). This leads to more realistic imputation of genuinely missing data points.
* **Reliable Input for Advanced Analysis Toolkits:**
  * **`tsfresh`:** If `tsfresh` is to be used later for automatic feature extraction, it requires cleaned numerical data and benefits from a consistent time index and an entity identifier (which our `id_col` provides).
  * **"Augurs" (or similar for changepoint/seasonality):** Specialized toolkits for detecting structural breaks, trends, or seasonality also perform best on data that has been cleaned of obvious errors and outliers. The Era-specific DataFrames produced here are a better starting point for such analyses than the raw, unfiltered table.
  * Ensuring data types are appropriate (e.g., numerical for sensor readings, 0/1 for statuses) is critical for the correct functioning of these downstream tools.
