# Implementing DoppelGANger for Synthetic Time-Series Sensor Data Generation

This report details the implementation strategy for using the DoppelGANger GAN architecture to generate synthetic time-series sensor data, tailored for the greenhouse simulation and optimization project. It builds upon the general considerations previously discussed for GANs and focuses specifically on DoppelGANger due to its strengths in handling multivariate and potentially attribute-conditioned time series.

## 1. Why DoppelGANger?

DoppelGANger (Lin et al., 2020) is a strong candidate for this project for several reasons:

* **Multivariate Time Series Fidelity:** It's designed to generate multiple correlated time-series features simultaneously, which is crucial for sensor data where variables (temperature, humidity, CO2, light) are often interdependent.
* **Capturing Temporal Dynamics:** It utilizes LSTMs within its generator, enabling it to learn and reproduce temporal patterns, trends, and seasonality present in the original data.
* **Handling of Attributes/Metadata (Optional but Powerful):** While not a primary focus for the initial sensor data, DoppelGANger can incorporate static attributes that describe each time series (e.g., sensor type, location, greenhouse section). This could be leveraged if such metadata becomes relevant.
* **Improved Training Stability:** It often employs Wasserstein GAN with Gradient Penalty (WGAN-GP), which can lead to more stable training and help mitigate issues like mode collapse compared to vanilla GANs.
* **Per-Sequence Normalization:** The architecture can handle features with different scales more effectively due to its design and preprocessing suggestions (e.g., per-example scaling).
* **Available Implementations:** Mature and optimized PyTorch implementations are available (e.g., Gretel Synthetics' DGAN), which support GPU acceleration and integrate well with Python-based data pipelines.

## 2. Key Architectural Features Relevant to Sensor Data

* **Generator:** Produces both static attributes (if any) and the time-series features. The LSTM component is key for sequential data generation. The "multiple time-steps per LSTM cell emission" trick can help in learning longer-range dependencies more efficiently.
* **Discriminator:** Evaluates the realism of both the generated attributes and the time series, encouraging the generator to produce coherent outputs.
* **Attribute and Feature Separation:** Explicitly models attributes separately from time-dependent features, allowing for conditional generation if attributes are used.
* **WGAN-GP Loss:** Contributes to training stability and quality of generated samples.

## 3. Recommended Python Implementation: Gretel Synthetics DGAN

Gretel.ai provides an optimized PyTorch implementation of DoppelGANger, often referred to as `DGAN` within their `gretel-synthetics` library.

* **Installation:** `uv pip install gretel-synthetics`
* **Advantages:**
  * GPU support.
  * Relatively user-friendly API.
  * Can train from Pandas DataFrames or NumPy arrays.
  * Provides reasonable default configurations.

## 4. Step-by-Step Implementation Guidance

### Step 4.1: Data Preprocessing

This is a critical step for DoppelGANger, especially with sparse and irregular sensor data.

1. **Data Loading & Initial Cleaning:**
    * Fetch raw data from TimescaleDB (e.g., into a Pandas DataFrame using SQLAlchemy), focusing on the selected columns for the target Era.
    * Ensure data is sorted by `time`.
    * Handle timezones consistently (e.g., convert all to UTC using `df['time'] = pd.to_datetime(df['time'], utc=True)`).
    * Drop columns identified as "dead" or almost entirely empty for the specific Era being processed (as per `db_data_description.md` and `report.md`).

2. **Resampling to a Consistent Frequency (Crucial for Time Series Uniformity):**
    * **Context:** Time-series models, including DoppelGANger when creating fixed-length sequences, typically expect data sampled at regular intervals.
    * **Action:** Determine a target frequency for your time series (e.g., 1-minute, 5-minute, 10-minute, 1-hour intervals). This choice depends on the original data's granularity and the dynamics you want to capture.
        * If data is already at a consistent high frequency (e.g., 1-minute from the sample in `db_data_description.md`), this step might primarily involve ensuring no implicit gaps exist that break the regularity.
        * If data is sparse or irregular, or if you wish to model at a coarser granularity:
            * **Upsampling (if increasing frequency or filling small, regular gaps):** If upsampling to a finer grid (e.g., from 5-min to 1-min to fill expected slots), this will introduce NaNs that must be handled by subsequent imputation. Use with caution if original data is very sparse.
            * **Downsampling (if decreasing frequency):** Aggregate data to a coarser interval (e.g., from 1-min to hourly). Use appropriate aggregation functions for each column (e.g., `mean` for continuous sensors like temperature, `sum` for countable events like light ON duration if pre-aggregated, `last` or `first` for status columns).
                * Example: `df.set_index('time').resample('1H').mean()` (adjust `.mean()` as needed per column type).
    * **Set DatetimeIndex:** Ensure `time` is the DataFrame index before resampling: `df.set_index('time', inplace=True)`.
    * **Output:** A DataFrame with a regular DatetimeIndex at your chosen frequency.

3. **Defining Training Examples (Sequences) from Regularized Data:**
    * DoppelGANger trains on multiple "examples," where each example consists of (optional) static attributes and a multivariate time series.
    * **Strategy:** From the regular-frequency DataFrame, extract fixed-length windows of time as individual training examples.
        * **Choose `max_sequence_len`:** This is a crucial hyperparameter. It should be long enough to capture meaningful daily or weekly patterns (e.g., if your data is now at 1-minute frequency, `max_sequence_len = 24 * 60 = 1440` for daily patterns; if resampled to hourly, `max_sequence_len = 24 * 7 = 168` for weekly patterns).
        * **Sliding Window:** Extract sequences of `max_sequence_len` by sliding a window over your resampled dataset for the current Era. Overlapping windows can increase the number of training samples.
    * **Multivariate Input:** Each sequence example will contain all relevant sensor columns (features) for that time window.

4. **Handling Missing Data (`NaN`s) within Sequences (Post-Resampling):**
    * DoppelGANger (especially the Gretel implementation) expects numeric inputs without `NaN`s for the feature sequences. NaNs might have been introduced by upsampling or were present in the original data at the chosen frequency.
    * **Short Gaps within a sequence:** For small internal gaps within an extracted sequence:
        * **Forward-fill (LOCF):** `df_sequence.fillna(method='ffill', inplace=True)` - Often a good first choice for sensor data, assuming a value persists until a new one is recorded.
        * **Backward-fill (NOCB):** `df_sequence.fillna(method='bfill', inplace=True)`.
        * **Linear Interpolation:** `df_sequence.interpolate(method='linear', inplace=True)` - Use cautiously; can create artificial smoothness or unrealistic intermediate values for some sensor types (e.g., status flags).
        * **Consider `_is_valid` flags (mentioned in Section 7):** Even after imputation, these flags can inform the GAN about original data presence.
    * **Leading/Trailing NaNs in a sequence:** If a sequence starts or ends with NaNs after windowing (and after within-sequence imputation attempts), these might need to be handled by either ensuring imputation covers them (e.g., if global `ffill` and `bfill` are applied before windowing after resampling) or by carefully considering if such sequences are valid for training. The Gretel DGAN might require fully populated sequences.
    * **Long Gaps / Highly Incomplete Sequences:** If a sequence window (after resampling and initial imputation) still contains too many `NaN`s for critical features (e.g., >50% for a key sensor), it's generally better to discard that window from the training set.

5. **Normalization (per feature, per Era model):**
    * **Feature Scaling:** Scale each sensor feature. Min-max scaling to `[-1, 1]` or `[0, 1]` is common for GANs.
        * Apply scaling *per feature (column)* across the entire training dataset of sequences for the current Era.
        * Save the scalers (e.g., `sklearn.preprocessing.MinMaxScaler`) for each feature to inverse-transform the generated synthetic data later.
    * **Attributes (if any, per Era model):** If using static categorical attributes (e.g., `Era_ID` if a unified model was chosen, though segmented training is recommended), they would need to be one-hot encoded. Numerical attributes should also be scaled.

6. **Data Formatting for `gretel-synthetics DGAN`:**
    * The `model.train_numpy()` method typically expects:
        * `features`: A NumPy array of shape `(num_examples, max_sequence_len, num_features)`.
        * `attributes` (optional): A NumPy array of shape `(num_examples, num_attributes)`.
    * You'll need to transform your list of processed DataFrame windows into this 3D NumPy array for features.

### Step 4.2: Model Configuration and Training

Using `gretel-synthetics DGAN`:

```python
from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.config import DGANConfig

# Assuming 'features_train_np' is your (num_examples, max_seq_len, num_features) NumPy array
# Assuming 'attributes_train_np' is your (num_examples, num_attributes) NumPy array (or None)

config = DGANConfig(
    max_sequence_len=max_sequence_len, # Your chosen sequence length
    sample_len=max_sequence_len, # Or control generation length sampling
    batch_size=64, # Adjust based on GPU memory (e.g., 32, 64, 128)
    epochs=500,    # Start with a moderate number, e.g., 300-1000, may need more
    generator_learning_rate=1e-4,
    discriminator_learning_rate=1e-4,
    attribute_noise_dim=50, # If using attributes
    feature_noise_dim=50,
    attribute_num_layers=3, # Deeper networks for attributes
    attribute_num_units=100,
    feature_num_layers=3,   # Deeper LSTMs for features
    feature_num_units=100,
    use_attribute_discriminator=True, # If using attributes
    cuda=True # Set to True to use GPU if available
)

model = DGAN(config)
# If you have attributes:
# model.train_numpy(attributes=attributes_train_np, features=features_train_np)
# If no attributes:
model.train_numpy(features=features_train_np)

# Save the trained model
model.save_model("doppelganger_sensor_model.pkl")
```

* **Hyperparameters:** `max_sequence_len`, `batch_size`, `epochs`, learning rates, network layer sizes (`num_units`, `num_layers`) are key. Start with library defaults or common values and tune based on results and computational resources.
* **GPU Usage:** `cuda=True` in the config will enable GPU training if PyTorch and CUDA are correctly set up in your environment.

### Step 4.3: Synthetic Data Generation

```python
# Load the trained model
# model = DGAN.load_model("doppelganger_sensor_model.pkl")

num_synthetic_sequences = 1000 # Or however many you need

# If trained with attributes:
# synthetic_attributes, synthetic_features = model.generate_numpy(num_sequences=num_synthetic_sequences)
# If trained without attributes:
_, synthetic_features = model.generate_numpy(num_sequences=num_synthetic_sequences)
# synthetic_features will be (num_synthetic_sequences, sample_len, num_features)
```

* `sample_len` in `DGANConfig` can control the length of generated sequences if you want them different from `max_sequence_len`.

### Step 4.4: Post-processing

1. **Inverse Scaling:**
    * Apply the inverse transformation of your scalers to `synthetic_features` to get them back into their original physical units. This must be done per feature using the scaler fitted for that feature.
2. **Reshape and Timestamping:**
    * The `synthetic_features` array needs to be reshaped into a 2D format suitable for a DataFrame (e.g., concatenate sequences).
    * Assign timestamps. If you aim to create a continuous synthetic timeline:
        * Determine a start date and frequency (e.g., 1-minute intervals matching your `sensor_data_merged`).
        * Generate a DatetimeIndex in Pandas.
        * Align the generated feature sequences to this index.
3. **Create DataFrame:** Convert the post-processed NumPy array into a Pandas DataFrame with appropriate column names (your `TARGET_COLUMNS` excluding `time` if time is the index, or including `time` if it's a column).

### Step 4.5: Output

1. **CSV File:**

    ```python
    # Assuming 'final_synthetic_df' is your post-processed DataFrame
    # with a 'time' column or DatetimeIndex
    csv_output_path = Path("output") / "doppelganger_synthetic_data.csv"
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    final_synthetic_df.to_csv(csv_output_path, index=(final_synthetic_df.index.name == 'time'))
    print(f"Saved DoppelGANger synthetic data to {csv_output_path}")
    ```

2. **TimescaleDB Insertion (Optional):**
    * Use SQLAlchemy and Pandas `to_sql` to write `final_synthetic_df` to a new table (e.g., `sensor_data_synthetic_doppelganger`).
    * Execute `SELECT create_hypertable(...)` on the new table.

## 5. Integration into Dockerized Pipeline (Conceptual)

Referencing the Prefect flow from the general GAN report:

* **`extract_and_preprocess_task`:** Fetches data from `sensor_data` (or `sensor_data_merged`), performs windowing, imputation for small gaps, normalization, and formats data into NumPy arrays (`features_train_np`, `attributes_train_np` if any). Saves these arrays and the scalers.
* **`train_doppelganger_task`:** Loads preprocessed arrays and scalers. Initializes and trains `DGAN` model. Saves the trained `model.pkl`. This task should be configured to run on a GPU-enabled Docker container.
* **`generate_synthetic_data_task`:** Loads the trained `model.pkl` and scalers. Generates a specified number of synthetic sequences. Performs inverse scaling. Reshapes and assigns timestamps to create a final DataFrame.
* **`load_synthetic_data_task`:** Takes the final synthetic DataFrame. Writes it to a CSV file and/or inserts it into a new TimescaleDB table (e.g., `sensor_data_synthetic_doppelganger`).

## 6. Specific Considerations for Your Sensor Data

* **Number of Training Examples:** The primary challenge for DoppelGANger is often having enough distinct sequence examples. If you have 3 years of 1-minute data, using a sliding window of, say, 1 day (1440 points) with a reasonable stride will generate a large number of training examples.
* **`max_sequence_len` Choice:**
  * Too short: May not capture daily/weekly patterns.
  * Too long: Increases training complexity and memory requirements; may make it harder for the LSTM to learn dependencies.
  * Experimentation might be needed. Start with daily (1440 points if 1-min data) or weekly patterns.
* **Attribute Use:** Initially, you might not use explicit attributes. The model will learn the joint distribution of all sensor features. If you later identify distinct operational modes or greenhouse sections that significantly alter sensor behavior, these could be encoded as attributes.
* **Evaluation:** Rigorous evaluation (statistical properties, domain expert review, downstream task performance) as outlined in the "GAN_Synthetic_Data_Considerations.md" is critical.

This focused report on DoppelGANger should provide a clearer path for its implementation within your project. Remember that GAN development is iterative, and hyperparameter tuning and preprocessing choices will likely require experimentation.

## 7. Next Steps and Action Items

* **TODO: Finalize Detailed Feature Engineering Plan for `sensor_data_merged` (as input to DoppelGANger)**
  * **Context:** The `sensor_data_merged` table (described in `DataIngestion/feature_extraction/db_data_description.md`) will be the primary source for training the DoppelGANger model. The `Jupyter/classes/data_availability_analyzer/report.md` highlights significant variations in data availability across different time periods ("Eras"). This necessitates a feature engineering plan that is acutely aware of these data characteristics to ensure the GAN learns meaningful patterns and generates realistic synthetic data.
  * **Overarching Strategy:** Given the distinct data availability profiles, consider the following primary approaches:
      1. **Segmented Training (Recommended):** Train separate DoppelGANger models for "Era 1" (approx. Early 2014 - Aug 2014) and "Era 2" (approx. Oct 2015 - Late 2016). Each model would be trained on a feature set derived from columns reliably available *within its respective Era*.
      2. **Unified Training with Era as Attribute:** Train a single model on data from both Eras, but include a mandatory static attribute per sequence indicating `Era_ID` (e.g., 0 for Era 1, 1 for Era 2). The feature set would need to be the *intersection* of reliably available columns across both Eras, or use extensive `_is_valid` flags for columns missing in one Era.
      *Initial recommendation is **Segmented Training** for better fidelity within each distinct data regime.*

  * **Action Required: Define Era-Specific Feature Sets & Transformations:**
    1. **Column Selection per Era (Crucial First Step):**
        * **Era 1 (Core Environmental Focus):**
            * **Primary Features:** `time` (for windowing), `air_temp_c`, `relative_humidity_percent`, `light_intensity_umol`, `radiation_w_m2`, `outside_temp_c`, `co2_measured_ppm` (mind patchiness), `rain_status`, `curtain_1_percent` (and other available curtains), `flow_temp_1_c`, `flow_temp_2_c`.
            * **Exclude:** Columns largely missing in Era 1 (most actuators like `lamp_grpX_status`, `vent_..._percent`, `pipe_temp_X_c`). Exclude "dead" columns (e.g., `behov`, `value`, `window_1_percent`).
        * **Era 2 (Actuator & AFD Focus):**
            * **Primary Features:** `time` (for windowing), `radiation_w_m2`, `co2_measured_ppm` (mind intermittency), `pipe_temp_1_c`, `pipe_temp_2_c`, all `lamp_grpX_status` columns, all `vent_..._afd..._percent` columns, `humidity_deficit_afd3_g_m3`, `relative_humidity_afd3_percent`, `humidity_deficit_afd4_g_m3`, `relative_humidity_afd4_percent`.
            * **Exclude:** Columns largely missing in Era 2 (e.g., `air_temp_c`, `relative_humidity_percent`, `light_intensity_umol`). Exclude "dead" columns.
        * **Note on CSV-derived features:** The features listed in `lightgroup_feature.md` (derived from the separate CSV analysis) are insightful but must be mapped to available columns within `sensor_data_merged` for the GAN training based on this table. For example, `målt status_X` from CSVs would correspond to the various `lamp_grpX_status` columns in `sensor_data_merged` if the mapping is direct.

    2. **Encoding & Transformation Strategies (per Era-specific feature set):**
        * **Numerical Continuous Features (e.g., `air_temp_c`, `radiation_w_m2`):**
            * Apply Min-Max scaling to a range like `[-1, 1]` or `[0, 1]`. Save scalers *per feature, per Era model* for inverse transformation.
        * **Numerical Discrete/Status Features (e.g., `co2_status`, potentially vent/curtain positions if they are step-wise):**
            * If truly categorical (few unique values representing states), treat as categorical. Otherwise, scale as numerical.
        * **Boolean Features (e.g., `rain_status`, `lamp_grpX_status`):**
            * Convert to `0/1` numerical representation. These are already suitable for GAN input after this conversion.
        * **Categorical Columns (Low Cardinality for Attributes - if using Unified Training):**
            * If `source_system` or a simplified `format_type` were to be used as a static attribute in a unified model, apply one-hot encoding. For features within the time series, one-hot encoding can drastically increase dimensionality; consider if these should be features or if the information is implicitly captured by other sensors for a given Era.
        * **Handling Intermittency (especially in Era 2):**
            * For columns like `co2_measured_ppm` (Era 2), `pipe_temp_X_c` (Era 2), and vent/lamp statuses (Era 2) that are noted as intermittent in `report.md`:
                * **Imputation (within `max_sequence_len` windows):** Use forward-fill (LOCF) for short internal gaps. Avoid complex interpolation that might create unrealistic values.
                * **`_is_valid` flags:** For each key intermittent feature, generate a corresponding binary feature (e.g., `co2_measured_ppm_is_valid`) which is 1 if the original value was present, 0 if imputed/missing. This allows the GAN to learn the pattern of missingness itself. This is highly recommended for Era 2 features.
                * **Discarding sequences:** If a sequence window has excessive missingness for critical features even after LOCF, discard it from training.

    3. **Derived Time-Series Specific Features (per Era-specific feature set):**
        * **Lagged Features:** Create lagged versions of key continuous and status features (e.g., `air_temp_c_lag1h`, `lamp_grp1_no3_status_lag1step`). The number of lags should be sensible for `max_sequence_len`.
        * **Rolling Aggregates:** For continuous data, calculate rolling means, stddev over short windows (e.g., 10-min, 30-min, 1-hour). Use `min_periods` appropriately given potential intermittency.
        * **Time Since Last Change (for status/actuator columns):** For `lamp_grpX_status` or `vent_..._percent`, a feature representing time since the value last changed could be valuable.
        * **Cyclical Time Features:** Encode `hour_of_day` and `day_of_year` using sin/cos transformations. These should be part of the feature set for both Eras.

    4. **Timestamp Handling:**
        * `time` column is used for windowing/sequencing and for re-attaching to synthetic data. It is generally *not* directly used as an input feature to the LSTM core of DoppelGANger, unless transformed (e.g., seconds since start of Era, which might be less useful than cyclical time features).

    5. **Attribute Definition (if using Unified Training with Era as Attribute):**
        * The primary attribute would be `Era_ID` (categorical, one-hot encoded).
        * Other *static* attributes for each sequence (e.g., `source_system` if it remains constant for long periods that align with `max_sequence_len`) could be considered, but start simple.

  * **Rationale:** A clearly defined, Era-aware feature set is paramount. This ensures the GAN is trained on consistent data segments and learns patterns relevant to the specific operational and sensor realities of each period. For GANs, explicitly providing information about data validity (`_is_valid` flags) for known intermittent sensors can significantly improve the quality and realism of synthetic data, especially for complex datasets like `sensor_data_merged`.

  * **@Research & Decision:**
    * Decide on the primary strategy: **Segmented Training (recommended)** vs. Unified Training with Era attribute.
    * For the chosen strategy, meticulously list every column from `sensor_data_merged` that will be included for each Era/model. Document its transformation (scaling, encoding, `_is_valid` flag generation, derived features like lags/rolling means).
    * Confirm the choice of `max_sequence_len` for each Era, considering the typical duration of horticultural patterns and data sampling frequency (likely 1-minute from sample).
    * Prototype the preprocessing pipeline for one Era first to validate the steps.
