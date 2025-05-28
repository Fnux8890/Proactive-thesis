# Chapter 5: Normalization, Feature Engineering, and Final Model Preparation

This chapter covers the final stages of data preprocessing before the data is used for feature engineering or directly fed into sequence-based models like LSTMs or GANs (e.g., DoppelGANger). These steps include numerical feature scaling/normalization and a discussion on the transition to feature engineering, including considerations for `tsfresh` and domain-specific features.

## 5.1. Objectives

* Apply numerical scaling (normalization) to appropriate features to ensure they are on a comparable range, which is beneficial for training neural networks.
* Manage and persist scalers for consistent application to new data and for inverse transformation of synthetic data.
* Outline the principles of subsequent feature engineering, including time-based features, lagged variables, rolling statistics, and domain-specific horticultural metrics.
* Discuss strategies for splitting time-series data into training, validation, and test sets.
* Ensure the output data segments are in a final state ready for sequence windowing (for GANs/LSTMs) or direct feature calculation.

## 5.2. Numerical Feature Scaling/Normalization (`preprocess.py` - `scale_data_for_era` function)

Neural networks, including those in GANs and LSTMs, generally perform better and converge faster when input numerical features are on a similar scale.

* **Rationale:**
  * Prevents features with larger magnitudes from dominating the learning process.
  * Helps optimization algorithms (like gradient descent) navigate the loss landscape more effectively.
  * Required by some activation functions and distance-based calculations within models.

* **Configuration (`data_processing_config.json`):
  * While specific scaling parameters aren't usually in the main config, the choice of which columns to scale is critical. The `scale_data_for_era` function identifies candidate columns.
  * `common_settings.boolean_columns_to_int`: This list helps `scale_data_for_era` identify 0/1 integer columns that might be excluded from standard Min-Max scaling if they are intended to remain as 0/1 flags.

* **Implementation (`scale_data_for_era` function in `preprocess.py`):
    1. **Input:** Receives an imputed data segment (Pandas DataFrame) for a specific Era.
    2. **Scaler Storage:** Defines a path for saving/loading scaler objects for the Era (e.g., `plots/scalers/{era_identifier}_scalers.joblib`).
    3. **Column Selection for Scaling:**
        * Identifies numerical columns (typically `float64`, `float32`).
        * Includes numeric columns mentioned in outlier/imputation rules (in case they became integer but represent continuous data).
        * **Excludes** columns listed in `boolean_columns_to_int` from the general Min-Max scaling to `[-1,1]`, assuming these 0/1 features are suitable as is or might receive a different mapping if needed for a specific GAN input range (e.g., mapping 0 to -1, 1 to 1 if `[-1,1]` is strictly required for all features).
    4. **Scaling Method - MinMaxScaler:**
        * Uses `sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))` by default. This scales features to the range `[-1, 1]`, which is common for GANs with `tanh` activation functions in the output layer.
        * Alternative common range: `[0, 1]` (achieved with `MinMaxScaler(feature_range=(0, 1))`).
    5. **Fitting Scalers (`fit_scalers=True` - typically for training data process):
        * For each selected column in the input segment, a new `MinMaxScaler` is instantiated.
        * The scaler is `fit_transform`ed on the column's data. This learns the min/max for that column *within the current data segment* and applies the scaling.
        * The fitted scaler object is stored in a dictionary (`fitted_scalers`) keyed by column name.
        * This dictionary of all fitted scalers for the Era (or segment) is saved to a file using `joblib.dump()`.
        * **CRITICAL NOTE for GAN/LSTM Training:** For robust model training, scalers should be fitted **once** on a representative *training dataset* for an entire Era (or the training portion of all segments from an Era combined). Then, these same fitted scalers must be used to `transform` the training, validation, and any test sets for that Era, as well as for `inverse_transforming` synthetic data. The current `preprocess.py` scales each segment with `fit_scalers=True`, which is suitable for EDA or if each segment is treated as an independent training set. For a cohesive GAN model per Era, this scaling strategy needs adjustment (e.g., scale entire Era data post-imputation *before* final segmentation for windowing, or fit on first/training segments and apply to others).
    6. **Applying Scalers (`fit_scalers=False` - for validation/test/new data):
        * Loads the previously saved `fitted_scalers` dictionary using `joblib.load()`.
        * For each selected column, applies the corresponding loaded scaler using `.transform()`.
    7. **Output:** Returns the scaled DataFrame and the dictionary of scalers used.

* **Alternative Scaling Method - Standardization (`StandardScaler`):
  * Rescales features to have zero mean and unit variance (`(x - mean) / std_dev`).
  * Less sensitive to extreme outliers than `MinMaxScaler` but doesn't guarantee a fixed range.
  * Could be an alternative if `MinMaxScaler` proves problematic.

## 5.3. Transition to Feature Engineering

Once data is cleaned, resampled to a regular frequency, imputed, and scaled, it forms the basis for feature engineering. The goal of feature engineering is to create new, informative variables from the existing ones that can improve model performance or capture specific domain knowledge.

* **Input:** The scaled (and imputed, resampled) data segments from each Era.
* **Types of Features (referencing `DataIngestion/feature_extraction/lightgroup_feature.md` and `db_data_description.md`):
    1. **Time-Based Features:**
        * Cyclical encoding of time (hour of day, day of week, day of year, month) using sine/cosine transformations. `preprocess.py` should ideally add these before scaling other numerics or handle their scaling separately if needed.
        * Categorical time features (e.g., `is_weekday`).
    2. **Lagged Features:** Values of sensors from previous time steps (e.g., `air_temp_c_lag_1h`, `co2_measured_ppm_lag_30m`). Crucial for LSTMs and GANs to learn temporal dependencies.
    3. **Rolling Statistics:** Rolling mean, median, min, max, standard deviation over defined windows (e.g., `air_temp_c_roll_mean_6h`). Smooths noise and captures trends.
    4. **Interaction Features:** Combining two or more features (e.g., `temperature * humidity_deficit`, difference between internal and external temperature).
    5. **Domain-Specific Horticultural Features:**
        * **Daily Light Integral (DLI):** Crucial. Calculate from `light_intensity_umol` (Era 1) or estimated from `radiation_w_m2` (Era 1 & 2).
        * **Vapor Pressure Deficit (VPD):** If not directly available and reliable (e.g., `vpd_hpa`), calculate from air temperature and relative humidity (needs co-availability of these sensors within an Era).
        * **Growing Degree Days (GDD):** Accumulate thermal units.
        * Heating/Cooling Degree Proxies.
        * Time since last significant event (e.g., rain, screen closure, lamp ON).
* **Tooling for Feature Engineering:**
  * **Pandas:** Primary tool for creating lagged features (`.shift()`), rolling windows (`.rolling()`), and custom calculations.
  * **NumPy:** For numerical operations.
  * **`tsfresh` (Optional - for automated broad feature generation):**
    * Can automatically extract hundreds of generic time-series characteristics.
    * **Pros:** Can uncover unexpected informative features.
    * **Cons:** High dimensionality (requires careful feature selection), features may lack horticultural interpretability, computationally intensive.
    * **Recommendation:** Consider as a supplementary tool after domain-driven feature engineering. Apply per segment or on concatenated, ID-labelled segments.
  * **"Augurs" / Specialized Libraries (Conceptual):** If specific algorithms for seasonality decomposition or complex trend feature extraction are available and relevant, they would be applied here.

## 5.4. Data Splitting for Time Series Models

Before training any model (LSTM, GAN), the preprocessed and feature-engineered data must be split correctly into training, validation, and (optionally) test sets.

* **Chronological Split (Essential):** Time series data must be split chronologically to prevent data leakage and obtain realistic performance estimates. Future data cannot be used to train models predicting the past or present.
  * Example for a single Era: Train on first 70% of data, validate on next 15%, test on final 15%.
* **Avoid Random Shuffling:** Standard k-fold cross-validation with random shuffling is inappropriate as it breaks temporal dependencies.
* **Walk-Forward Validation / Time Series Cross-Validation:** For more robust model evaluation, especially for forecasting models, these techniques involve iteratively training on past data and testing on subsequent data blocks.
* **Consideration for Segmented Eras:** If training separate models per Era, splitting would occur within each Era's dataset.

## 5.5. Final Preparation for Model Ingestion (LSTMs, DoppelGANger)

* **Sequence Windowing:** LSTMs and GANs like DoppelGANger operate on sequences of data. The preprocessed (and feature-engineered) data segments need to be transformed into overlapping or non-overlapping windows of a fixed `max_sequence_len`.
  * Each window becomes one training example for the GAN.
  * Input shape for DoppelGANger features: `(num_examples, max_sequence_len, num_features)`.
* **Attribute Preparation (for DoppelGANger):** If using static attributes (e.g., `Era_ID` if training a unified model, `source_system` if relevant and constant per sequence), these need to be prepared in the shape `(num_examples, num_attributes)` and typically one-hot encoded if categorical.
* **Final NaN Check:** Ensure no NaNs remain in the data passed to the model, especially after windowing and feature engineering (some lag/rolling operations can re-introduce NaNs at the beginning of series/segments).

## 5.6. Outputs from this Chapter

* For each processed Era/segment:
  * A Pandas DataFrame (or NumPy array) containing scaled numerical features, including any newly engineered features.
  * Saved scaler objects for each Era (if not already saved per segment).
* A clear strategy for splitting data into train/validation/test sets.
* Data structured into sequences ready for model input.
* Updated entries in the summary report (`preprocessing_summary_report_{era_identifier}.txt`):
  * Details of the scaling process (columns scaled, range).
  * List of key engineered features created.
  * Notes on data splitting strategy applied.

This chapter concludes the main preprocessing pipeline, yielding data that is as clean, complete, and well-structured as possible for the demanding task of training advanced time-series models.
