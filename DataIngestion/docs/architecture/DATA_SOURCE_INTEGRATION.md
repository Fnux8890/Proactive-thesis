# Data Source Integration for LightGBM & MOEA NSGA-III Optimization

## Overview

This document describes how various data sources from the greenhouse monitoring system are integrated to train LightGBM surrogate models and used in Multi-Objective Evolutionary Algorithm (MOEA) optimization, specifically NSGA-III.

## Data Sources

(This document outlines primary tables. Other auxiliary or intermediate tables might exist.)

### 1. Sensor Data (Merged)
**Table:** `sensor_data_merged` (TimescaleDB hypertable)
- **Purpose:** Comprehensive time-series measurements from greenhouse sensors, including metadata about data origin.
- **Key Columns (selected examples):**
  - `time`: Timestamp (timestamp with time zone)
  - `source_system`, `source_file`, `format_type`, `uuid`: Data origin and identifiers (text)
  - Climate sensors: `air_temp_c`, `air_temp_middle_c`, `outside_temp_c`, `relative_humidity_percent`, `co2_measured_ppm` (double precision)
  - Light sensors: `light_intensity_umol`, `dli_sum`, `radiation_w_m2`, `outside_light_w_m2` (double precision)
  - Control actuators: `heating_setpoint_c`, `vent_pos_1_percent`, `pipe_temp_1_c`, `curtain_1_percent`, `lamp_grp1_no3_status` (double precision, boolean)
  - Plant environment: `vpd_hpa`, `humidity_deficit_g_m3` (double precision)
  - Forecasts: `temperature_forecast_c`, `sun_radiation_forecast_w_m2` (double precision)
  - `soil_moisture_percent`, `soil_temperature_c`, `leaf_wetness_percent` (double precision)
- **Note:** The original `sensor_data` table might exist for even rawer, less processed data stages.

### 2. Era Labels (A, B, C)
**Tables:** `era_labels_level_a`, `era_labels_level_b`, `era_labels_level_c` (and potentially a unified `era_labels` table)
- **Purpose:** Segmentation of time-series into operational periods (eras) based on different algorithms, allowing for context-aware feature extraction and analysis.
- **Algorithms:**
  - Level A: PELT (Pruned Exact Linear Time) - Major operational changes
  - Level B: BOCPD (Bayesian Online Changepoint Detection) - Medium-term patterns
  - Level C: HMM (Hidden Markov Model) - Short-term state transitions
- **Current Status:** Era detection has created excessive segments (1.88M total) due to stable greenhouse data. See ERA_DETECTION_IMPROVEMENT_PLAN.md for proposed fixes.
- **Usage:** Defines boundaries for feature extraction windows for subsequent feature engineering steps.

### 3. Preprocessed Features
**Table:** `preprocessed_features`
- **Purpose:** Data that has undergone initial cleaning, transformation, and is segmented by era, ready for feature engineering or direct model input.
- **Current Status:** Contains 291,457 records (2013-12-01 to 2016-09-08) with only 1 era_identifier, suggesting era segmentation may not be properly applied yet.
- **Key Columns (selected examples from table schema):**
  - `time`: Timestamp (timestamp with time zone)
  - `era_identifier`: Identifier for the era segment (text)
  - `air_temp_c`: Air temperature (real)
  - `relative_humidity_percent`: Relative humidity (real)
  - `co2_measured_ppm`: CO2 concentration (real)
  - `light_intensity_umol`: Light intensity (real)
  - `radiation_w_m2`: Solar radiation (real)
  - `total_lamps_on`: Calculated number of lamps active (real)
  - `dli_sum`: Daily Light Integral sum (real)
  - `vpd_hpa`: Vapor Pressure Deficit (real)
  - `heating_setpoint_c`: Heating setpoint (real)
  - `co2_status`: CO2 dosing status (integer)
  - `source_file`, `format_type`: Provenance information (text)
  - `extended_features`: Additional calculated features (jsonb)
- **Processing Steps (Conceptual - actual implementation may vary):**
  1. Outlier detection
  2. Missing value imputation
  3. Time regularization (e.g., 5-minute intervals)
  4. Derivation of features like `total_lamps_on` or content for `extended_features`.
- **Nature of Features (Conceptual):**
  - Temporal features: hour, day of week, season
  - Lag features: previous values at t-1, t-2, etc.
  - Rolling statistics: mean, std, min, max over windows

### 4. Extracted Features (tsfresh)
**Tables:** `tsfresh_features_level_a`, `tsfresh_features_level_b`, `tsfresh_features_level_c`, `gpu_features_level_a`, `gpu_features_level_b`, `gpu_features_level_c`
- **Note:** Multiple feature tables exist for different era levels and extraction methods (CPU vs GPU). The generic `tsfresh_features` table also exists.
- **Purpose:** Time-series feature engineering for each era, designed to capture various characteristics of the signals.
- **GPU Features:** Enhanced feature extraction using CUDA, computing extended statistics (percentiles, entropy, energy) with ~1.3KB shared memory usage.
- **Conceptual Features per Signal (examples if using tsfresh defaults like `EfficientFCParameters` or `MinimalFCParameters`):
  - Statistical: mean, median, variance, standard_deviation, skewness, kurtosis, minimum, maximum
  - Temporal: autocorrelation, partial_autocorrelation, measures of trend or seasonality
  - Frequency domain: FFT coefficients, spectral centroid/entropy
  - Complexity: approximate_entropy, sample_entropy, c3_statistic
- **Conceptual Structure (if stored in a separate long-format table):**
  - `era_id` (or `era_identifier`), `signal_name` (e.g., `air_temp_c`), `feature_name` (e.g., `tsfresh.feature_extraction.feature_calculators.mean`), `feature_value`

### 5. Phenotype Data (Literature-Based)
**Table:** `literature_kalanchoe_phenotypes`
- **Source:** Curated data from scientific literature regarding Kalanchoe plant phenotypes under various experimental conditions.
- **Purpose:** Provides crucial biological context for defining realistic optimization objectives, constraints, and understanding plant responses in MOEA.
- **Key Columns (selected examples):**
  - `publication_source`: Bibliographic source of the data (character varying)
  - `species`, `cultivar_or_line`: Specific plant variety (character varying)
  - `phenotype_name`: Name of the observed biological trait (character varying)
  - `phenotype_value`, `phenotype_unit`: Measured value and its unit for the trait (real, character varying)
  - `experiment_description`, `measurement_condition_notes`: Contextual details (text)
  - `environment_temp_day_c`, `environment_photoperiod_h`, `environment_dli_mol_m2_d`, `environment_light_intensity_umol_m2_s`: Key environmental conditions during the experiment (real)
- **Usage:** Incorporated as domain knowledge in plant growth models, for setting biologically relevant targets, and for validating simulation outputs.

### 6. External Weather Data (Aarhus)
**Table:** `external_weather_aarhus`
- **Source:** External weather provider (e.g., Open-Meteo API, for Aarhus region).
- **Purpose:** Offers historical and potentially forecast weather conditions. This data can influence greenhouse internal conditions, predict external impacts (e.g., solar gain, heat loss), and inform control strategies.
- **Key Columns (selected examples):**
  - `time`: Timestamp (timestamp with time zone)
  - `temperature_2m`: External air temperature at 2 meters (real)
  - `relative_humidity_2m`: External relative humidity at 2 meters (real)
  - `precipitation`: External precipitation (real)
  - `rain`, `snowfall`: Specific precipitation types (real)
  - `shortwave_radiation`: External solar radiation (real)
  - `direct_normal_irradiance`, `diffuse_radiation`: Components of solar radiation (real)
  - `cloud_cover`, `cloud_cover_low`, `cloud_cover_mid`, `cloud_cover_high`: Cloud cover details (real)
  - `wind_speed_10m`, `wind_direction_10m`: Wind conditions (real)
  - `weathercode`: Code representing weather conditions (real)
- **Usage:** Context for greenhouse climate control decisions, energy demand forecasting.

### 7. Energy Prices (Denmark)
**Table:** `external_energy_prices_dk`
- **Source:** Danish energy market (e.g., specific price areas like DK1, DK2).
- **Purpose:** Provides spot prices for energy, crucial for cost-related objectives in MOEA and for optimizing energy consumption strategies.
- **Key Columns:**
  - `HourUTC`: Timestamp for the price point (timestamp with time zone)
  - `PriceArea`: Geographic area for the energy price (character varying)
  - `SpotPriceDKK`: Energy price in Danish Kroner per unit (real)
- **Usage:** Economic optimization of energy consumption, scheduling of high-demand operations.

## Data Integration Pipeline

### Phase 1: Feature Engineering
```
sensor_data + era_labels → preprocessed_greenhouse_data → tsfresh_features
```

1. **Era-based Segmentation:** Use era labels to define feature extraction windows
2. **Signal Selection:** Optimal signals based on coverage and importance
3. **Feature Extraction:** Apply tsfresh to each era segment
4. **Feature Aggregation:** Combine features with phenotype data

### Phase 2: LightGBM Training

#### Data Flow:
```python
# PostgreSQL → Pandas DataFrame
features = load_from_db("tsfresh_features")
phenotypes = load_from_db("phenotypes")
weather = load_from_db("external_weather_data")
energy = load_from_db("energy_prices")

# Feature matrix construction
X = merge_features(features, phenotypes, weather, energy)
y = calculate_target(objective_name)  # e.g., energy_consumption
```

#### Target Variables:
1. **Energy Consumption:** `heating_energy + cooling_energy + lighting_energy`
2. **Plant Growth:** `biomass_accumulation_rate` (derived from growth measurements)
3. **Water Usage:** `irrigation_volume + evapotranspiration`
4. **Crop Quality:** `compactness * uniformity * flower_quality`
5. **Climate Stability:** `temp_variance + humidity_variance + co2_variance`

#### Model Configuration:
- **GPU Acceleration:** CUDA-enabled LightGBM for faster training
- **Hyperparameters:** Objective-specific tuning (e.g., more trees for plant growth)
- **Cross-validation:** Time-series split to respect temporal dependencies

### Phase 3: MOEA NSGA-III Optimization

#### Surrogate Model Usage:
```python
# Load trained models
models = {
    'energy': lgb.Booster(model_file='models/energy_consumption/model.txt'),
    'growth': lgb.Booster(model_file='models/plant_growth/model.txt'),
    'water': lgb.Booster(model_file='models/water_usage/model.txt')
}

# Evaluate objectives
def evaluate_solution(decision_variables):
    # Prepare features from decision variables
    features = prepare_features(decision_variables, current_state)
    
    # Predict objectives using surrogate models
    objectives = []
    for name, model in models.items():
        prediction = model.predict(features)
        objectives.append(prediction)
    
    return objectives
```

#### Decision Variables:
- Temperature setpoint (18-28°C)
- Humidity setpoint (60-85%)
- CO2 setpoint (400-1000 ppm)
- Light intensity (0-600 μmol/m²/s)
- Light hours (0-18 hours)
- Ventilation rate (0-100%)

#### Constraints:

- **Environmental:** Min/max temperature, humidity, CO2, VPD limits
- **Operational:** Max change rates, ventilation temperature thresholds
- **Economic:** Daily energy/water cost limits

## Feature Importance by Objective


### Energy Consumption
1. **Primary signals:** Heating pipes, ventilation positions, lamp status
2. **External factors:** Outside temperature, energy prices
3. **Derived features:** Heating degree days, cooling requirements
4. **Data Mapping & Engineering Notes:**

    - **Primary signals source:** Columns like `heating_pipe_temperature_c`, `ventilation_position_percent`, `total_lamps_on` from `sensor_data_merged` or `preprocessed_features`.
    - **External factors source:** `temperature_2m` from `external_weather_aarhus`; `SpotPriceDKK` from `external_energy_prices_dk`.
    - **Derived features engineering:**
        - `Heating Degree Days (HDD)`: Calculated as `max(0, T_base - T_outside_avg_daily)`. `T_base` (e.g., 18°C) from domain knowledge, `T_outside_avg_daily` from `external_weather_aarhus`.
        - `Cooling Requirements`: Could be inferred from periods where `air_temp_c` (internal) exceeds a setpoint despite cooling actions (e.g., ventilation).
        - **Storage:** These derived features could be added as new columns to `preprocessed_features`, stored in its `extended_features` JSONB field, or calculated on-the-fly by modeling scripts.

### Plant Growth
1. **Primary signals:** DLI, CO2, temperature, humidity, VPD
2. **Phenotype integration:** Growth rate parameters, light response curves
3. **Temporal features:** Photoperiod, accumulated light integral
4. **Data Mapping & Engineering Notes:**

    - **Primary signals source:** Columns like `dli_sum`, `co2_measured_ppm`, `air_temp_c`, `relative_humidity_percent`, `vpd_hpa` from `preprocessed_features`.
    - **Phenotype integration:** Parameters (e.g., base growth rate, light/temp response coefficients) sourced from `literature_kalanchoe_phenotypes` or domain expertise, used to parameterize growth models or create phenotype-specific features.
    - **Temporal features engineering:**
        - `Photoperiod`: Calculated based on lamp activity (`total_lamps_on`) and natural daylight hours (derived from `time` and potentially `external_weather_aarhus` radiation data).
        - `Accumulated Light Integral (ALI)`: Sum of `dli_sum` over relevant periods (e.g., per era, per week).
        - **Storage:** As per Energy Consumption, these can be new columns in `preprocessed_features`, part of `extended_features`, or calculated on-the-fly.

### Water Usage
1. **Primary signals:** Humidity, VPD, radiation, temperature
2. **Plant factors:** Leaf area, transpiration coefficients
3. **Environmental:** Wind speed, external humidity
4. **Data Mapping & Engineering Notes:**

    - **Primary signals source:** `relative_humidity_percent`, `vpd_hpa`, `radiation_w_m2`, `air_temp_c` from `preprocessed_features`.
    - **Plant factors:** `Leaf Area Index (LAI)` could be a modeled output or a simplified proxy. `Transpiration coefficients` might be derived from `literature_kalanchoe_phenotypes` or estimated.
    - **Environmental factors source:** `wind_speed_10m`, `relative_humidity_2m` from `external_weather_aarhus`.
    - **Derived features engineering:** Features related to evapotranspiration (e.g., Penman-Monteith components if data allows) could be calculated.
    - **Storage:** As above.

### Crop Quality
1. **Primary signals:** Light uniformity, temperature stability
2. **Phenotype factors:** Compactness genes, flower development
3. **Stress indicators:** VPD extremes, light stress markers
4. **Data Mapping & Engineering Notes:**

    - **Primary signals source:** `light_intensity_umol` (spatial variance if multiple sensors available, otherwise temporal stability), `air_temp_c` (temporal stability/fluctuations) from `preprocessed_features`.
    - **Phenotype factors:** Information like `compactness` or `time_to_flower` from `literature_kalanchoe_phenotypes` can serve as targets or inform feature engineering.
    - **Stress indicators engineering:**
        - `VPD extremes`: Count of instances or duration where `vpd_hpa` exceeds critical thresholds (from literature or domain expertise).
        - `Light stress markers`: Could involve periods of excessively high `light_intensity_umol` or insufficient `dli_sum`.
        - `Temperature stress`: Periods where `air_temp_c` is outside optimal ranges from `literature_kalanchoe_phenotypes`.
    - **Storage:** As above.

## Benchmark and Test Data

This section describes datasets primarily used for testing, benchmarking, and validating various components of the data processing and modeling pipeline.

**Tables:** `benchmark_test_1000`, `benchmark_test_10000`, `benchmark_test_50000` (schemas are likely similar, differing by data volume)
- **Purpose:** Provide standardized datasets for evaluating feature engineering processes, model training performance, prediction accuracy, and the impact of data scale.
- **Key Columns (example from `benchmark_test_1000`):
  - `time`: Timestamp (timestamp without time zone)
  - `era_identifier`: Identifier for era segment (text)
  - `temperature`, `humidity`, `co2_level`, `light_intensity`: Base sensor readings (double precision)
  - `energy_consumption`: A target variable, potentially for benchmarking energy prediction models (double precision)
  - `feature_1`, `feature_2`, `feature_3`: Generic columns representing example engineered features (double precision)
- **Potential Usage Scenarios:**

  - Validating the correctness and performance of feature extraction logic.
  - Testing the efficiency and accuracy of model training and prediction pipelines.
  - Comparing the impact of different data volumes or feature sets on model outcomes.
  - Serving as datasets for continuous integration (CI) checks to ensure pipeline integrity and reproducibility.

## LightGBM Specific Considerations

LightGBM is a gradient boosting framework that uses tree-based learning algorithms. It's often used for tasks like regression and classification due to its efficiency and performance.

### Input Data and Feature Mapping
- **Primary Data Source:** The `preprocessed_features` table is the main source of input data. Features are selected based on relevance to the prediction target (e.g., predicting future temperature, energy consumption, or a plant growth metric).
- **Feature Types:**
  - Direct sensor readings: `air_temp_c`, `relative_humidity_percent`, `co2_measured_ppm`, `dli_sum`, etc.
  - Derived features: As detailed in the "Feature Importance by Objective" section (e.g., HDD, ALI, VPD extremes). These might be pre-calculated and stored in `preprocessed_features` or its `extended_features` JSONB column, or engineered on-the-fly.
  - Tsfresh features: If generated and stored (conceptually in `tsfresh_features` or embedded in `extended_features`), these provide a rich set of time-series characteristics.
  - External data: Relevant columns from `external_weather_aarhus` (e.g., lagged outside temperature) or `external_energy_prices_dk` (e.g., future energy prices if forecasting cost) can be joined and included.
- **Data Preprocessing for LightGBM:**
  - **Numerical Encoding:** All features must be numerical. Categorical features (e.g., `era_identifier` if used directly, or weather codes from `external_weather_aarhus`) need to be appropriately encoded (e.g., one-hot encoding, label encoding, or target encoding).
  - **Handling Missing Values:** LightGBM can handle missing values internally, but imputation strategies (as mentioned for `preprocessed_features`) are often applied beforehand.
  - **Time-Series Aspects:** For time-dependent predictions, features often include:
    - Lagged values of the target variable and relevant exogenous variables.
    - Rolling window statistics (mean, min, max, std over past N periods).
    - Time-based features (hour of day, day of week, month).
- **Target Variable:** Depends on the specific modeling task (e.g., `air_temp_c` at t+1, `energy_consumption` over the next era, a calculated growth metric).

## NSGA-III Specific Considerations

### Reference Points

- Structured reference points for many-objective optimization (>3 objectives)
- Adaptive reference point generation based on Pareto front shape

### Data Linkage for Objectives and Constraints

The NSGA-III algorithm, as part of the MOEA framework, requires objectives and constraints to be defined as functions that can be evaluated for each candidate solution. These functions will draw upon the documented data sources.

- **Objectives Quantification:**
  - **Minimize Energy Consumption:** Calculated by summing energy usage (e.g., from `preprocessed_features.total_lamps_on` converted to kWh, heating energy estimated from `heating_pipe_temperature_c` and flow rates if available) and multiplying by `external_energy_prices_dk.SpotPriceDKK` over the evaluation period.
  - **Maximize Plant Growth:** This is typically a modeled output. The model would take inputs like `dli_sum`, `co_measured_ppm`, `air_temp_c` from `preprocessed_features`, and potentially growth parameters from `literature_kalanchoe_phenotypes`, to predict a growth metric (e.g., biomass accumulation, increase in leaf area index).
  - **Minimize Water Usage:** Estimated based on irrigation actions (if logged) or modeled based on `vpd_hpa`, `radiation_w_m2`, `air_temp_c` from `preprocessed_features` and plant LAI (Leaf Area Index, potentially from a plant model).
  - **Maximize Crop Quality:** This can be complex. It might involve minimizing deviations from ideal `air_temp_c` or `vpd_hpa` ranges (from `preprocessed_features`), ensuring `light_intensity_umol` uniformity, or achieving target `dli_sum`. Quality metrics could also be informed by `literature_kalanchoe_phenotypes` (e.g., achieving conditions known to promote compactness).

- **Constraints Quantification:**
  - **Environmental Limits:** `Min/max temperature` constraints are checked against `preprocessed_features.air_temp_c`. Similarly for `humidity` (`relative_humidity_percent`), `CO2` (`co_measured_ppm`), and `VPD` (`vpd_hpa`).
  - **Operational Limits:** `Max change rates` for setpoints would be compared against the proposed changes in control actions. `Ventilation temperature thresholds` would use `preprocessed_features.air_temp_c` to decide if ventilation is allowed.
  - **Economic Limits:** `Daily energy/water cost limits` would be checked against the calculated costs as described in the objectives.

### Population Initialization

- Seeded with historical good solutions from database
- Domain knowledge from phenotype data constrains initial population

### Fitness Evaluation

- Parallel evaluation using GPU-accelerated LightGBM
- Batch processing of population members
- Caching of repeated evaluations

### Convergence Monitoring

- Hypervolume indicator tracking
- Diversity metrics (spacing, spread)
- Constraint violation statistics

## Implementation Details

### Database Queries

```sql
-- Example: Load features for specific era
SELECT f.*, p.growth_rate, p.light_saturation
FROM tsfresh_features f
LEFT JOIN phenotypes p ON p.plant_type = 'Kalanchoe'
WHERE f.era_id = ? 
  AND f.signal_name IN (?, ?, ...)
```

### Model Pipeline

```python
class SurrogateModelPipeline:
    def __init__(self):
        self.data_loader = PostgreSQLDataLoader()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = LightGBMTrainer()

    def train(self, objective):
        # Load multi-source data
        features = self.data_loader.load_features()
        phenotypes = self.data_loader.load_phenotypes()
        weather = self.data_loader.load_weather()

        # Engineer features
        X = self.feature_engineer.combine_sources(features, phenotypes, weather)

        X = self.feature_engineer.combine_sources(
            features, phenotypes, weather
        )
        
        # Train model
        model = self.model_trainer.train(X, y)
        return model
```

## Performance Considerations

1. **Data Volume:** ~10M sensor readings → ~100K era segments → ~10K feature vectors
2. **GPU Utilization:** LightGBM GPU reduces training time by 5-10x
3. **Parallel MOEA:** 4-8 parallel evaluations per generation
4. **Memory Management:** Chunked data loading for large datasets
5. **Caching:** Feature matrices cached between MOEA generations

## Future Enhancements

1. **Online Learning:** Update surrogate models with new greenhouse data
2. **Transfer Learning:** Adapt models between different greenhouse types
3. **Uncertainty Quantification:** Bayesian surrogate models for robust optimization
4. **Multi-fidelity Models:** Combine high-fidelity simulations with data-driven models
5. **Adaptive Sampling:** Active learning for targeted data collection