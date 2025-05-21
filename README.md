# Thesis project

### 🌱 Data-Driven Greenhouse Climate Control & Optimization 🌿

**Optimizing plant growth & energy efficiency through proactive simulation, multi-objective optimization, and investigation of advanced computational acceleration.**

This project develops an advanced **data-driven climate control and optimization system** for greenhouses. It aims to enhance plant health and resource efficiency by integrating **proactive strategies**, **species-specific plant growth simulation** (initially focusing on *Kalanchoe blossfeldiana*), and **multi-objective evolutionary algorithms (MOEAs)**. Building upon concepts from frameworks like DynaGrow, this work seeks to provide more tailored and computationally feasible solutions for sustainable greenhouse management.

### 🔬 **How It Works**

Leveraging historical data, predictive modeling, and (future) IoT-driven environmental adjustments, this system dynamically balances energy efficiency and plant growth. Key aspects include:

- **Data Pipeline & Feature Engineering**: Ingesting historical greenhouse data, performing preprocessing (including era detection based on techniques like PELT/BOCPD/HMM), and extracting features crucial for the subsequent simulation and optimization phases (FR-1, FR-2).
- **Species-Specific Plant Simulation**: Implementing and calibrating plant growth models (e.g., for Kalanchoe) to predict responses to various control strategies (FR-3, NFR-6.1). This is a core component for evaluating potential outcomes.
- **Multi-Objective Optimization (MOEA)**: Employing algorithms like NSGA-II to find Pareto-optimal control strategies that balance conflicting objectives, such as minimizing energy costs and maximizing simulated plant growth (FR-4).
- **Addressing Computational Bottlenecks**: A primary research focus is tackling the significant computational demands of detailed plant-specific simulations and extensive MOEA evaluations. This involves:
  - Identifying performance limitations on traditional CPU architectures.
  - **Investigating and implementing GPU acceleration using CUDA** for critical components (plant model simulation, MOEA fitness evaluations) across various GPU tiers (consumer, professional, data-center class).
- **Scalable, Modular Software Architecture**: Designing the system (primarily Python for simulation/optimization, potentially orchestrated with Elixir/Docker for broader pipeline elements) for extensibility and real-world integration (NFR-2, NFR-3, NFR-4).

### 📊 Available Data

This project utilizes a diverse set of data sources for feature extraction, plant model building, and multi-objective optimization:

- **Greenhouse Sensor Data**:
  - Collected from two primary locations: **KnudJepsen** and **Aarslev**.
  - Data spans varying, overlapping time periods, offering a rich historical view of greenhouse operations.
  - Key sensor measurements include (but are not limited to):
    - `time` (Timestamp)
    - `dli_sum` (Daily Light Integral)
    - `air_temp_middle_c` (Air Temperature - middle sensor)
    - `co2_measured_ppm` (CO2 concentration)
    - `radiation_w_m2` (Solar Radiation)
    - `relative_humidity_percent`
    - `pipe_temp_1_c` & `pipe_temp_2_c` (Heating pipe temperatures)
    - `curtain_1_percent` & `curtain_2_percent` (Climate screen positions)
    - `vent_pos_1_percent` & `vent_pos_2_percent` (Ventilation positions)
    - `outside_temp_c` (Outside Temperature)
- **External Weather Data**:
  - Fetched from external meteorological services (e.g., Open-Meteo), covering the same overall time span as the sensor data.
  - Provides broader meteorological context.
  - Includes variables such as: `temperature_2m`, `relative_humidity_2m`, `precipitation`, `rain`, `snowfall`, `weathercode`, `pressure_msl`, `surface_pressure`, `cloud_cover` (total, low, mid, high), `shortwave_radiation`, `direct_normal_irradiance`, `diffuse_radiation`, `wind_speed_10m`, and `wind_direction_10m`.
- **Energy Prices**:
  - Historical spot price data for electricity, corresponding to the operational period of the greenhouses.
  - Sourced for relevant price areas (e.g., `DK1`, `DK2`).
  - Includes columns like `HourUTC`, `PriceArea`, and `SpotPriceDKK`.
  - Crucial for the energy cost optimization objective.
- **Plant-Specific Phenotype Data**:
  - Structured phenotype data extracted from horticultural literature, primarily focusing on *Kalanchoe blossfeldiana* cultivars (e.g., 'Molly' and its Ri-lines).
  - Stored in `DataIngestion/feature_extraction/pre_process/phenotype.json`, referencing `phenotype.schema.json`.
  - Each entry details specific measurements like `plant_height`, `number_of_nodes`, `internode_length`, along with their units and the environmental conditions under which they were recorded (e.g., `environment_temp_day_C`, `environment_photoperiod_h`).
  - This data is crucial for informing plant model parameterization, calibration, and validation (as per memory `fcc8193c-7024-45f4-97b0-989a1d8fbd1c` and project requirements FR-3.1, NFR-6.1).
- **Era Detection Labels**:
  - Generated from the sensor data using various changepoint detection algorithms.
  - Includes era labels derived from:
    - PELT (Pruned Exact Linear Time)
    - BOCPD (Bayesian Online Changepoint Detection)
    - HMM (Hidden Markov Models) with Viterbi decoding
  - These labels help segment time-series data into distinct operational or environmental periods, aiding feature engineering.

This comprehensive dataset forms the foundation for developing and evaluating advanced, data-driven optimization strategies for greenhouse environments.

### 🚀 **Key Features**

✅ **Species-Specific Plant Simulation**: Utilizing detailed plant models (e.g., for Kalanchoe) to evaluate the outcomes of different control strategies as part of the MOEA fitness evaluation.
✅ **Multi-Objective Optimization**: Finding optimal trade-offs between energy costs and plant productivity using MOEAs.
✅ **Computational Performance Research**: Actively investigating and developing methods to accelerate complex simulations and optimizations using GPU (CUDA) technology.
✅ **Data-Driven Insights**: Utilizing historical data to inform simulations and guide optimization processes.
✅ **Energy-Efficient Growth Strategies**: Aiming to identify control schedules that maximize plant productivity while minimizing resource consumption.
✅ **Modular & Extensible Design**: Facilitating the integration of new plant models, optimization algorithms, or data sources.
✅ **Sustainability-Focused**: Contributing to reducing the carbon footprint and operational costs in horticulture.

### ⚙️ Pipeline Structure

The data processing pipeline is orchestrated using Docker Compose and consists of several distinct stages, each potentially implemented with different technologies to leverage their strengths:

1. **Stage 1: Initial Data Ingestion & Processing (`DataIngestion/rust_pipeline`)**
   - This stage, primarily implemented in Rust, is responsible for the initial ingestion of raw data from various sources.
   - It likely handles tasks such as reading different file formats, basic cleaning, validation, and preparing data for subsequent processing steps.

2. **Stage 2: Pre-processing (`DataIngestion/feature_extraction/pre_process`)**
   - Focuses on pre-processing tasks before feature extraction, likely implemented in Python.
   - This may involve data type conversions, handling missing values (e.g., converting sentinel values like -1 in `dli_sum` to `np.nan`), alignment of time-series data, and initial data transformations.

3. **Stage 3: Era Detection (`DataIngestion/feature_extraction/era_detection_rust`)**
   - Utilizes a dedicated Rust-based service for performing era detection on time-series data.
   - This service applies algorithms such as PELT, BOCPD, and HMM to segment data into meaningful operational or environmental periods, based on changes in signal characteristics (e.g., `dli_sum`).

4. **Stage 4: Feature Extraction (`DataIngestion/feature_extraction/feature`)**
   - Responsible for extracting relevant features from the pre-processed and era-segmented data, likely using Python.
   - This stage will involve generating a comprehensive set of time-series features (e.g., using `tsfresh` with `MinimalFCParameters` or `EfficientFCParameters`) from key sensor signals like `air_temp_c`, `relative_humidity_percent`, `co2_measured_ppm`, `radiation_w_m2`, `light_intensity_umol`, and `dli_sum`.
   - *(Note: This stage is currently under development).*

5. **Stage 5: Model Building & Optimization (`DataIngestion/model_builder`)**
   - The final stage, intended for building predictive plant models and running Multi-Objective Evolutionary Algorithms (MOEA).
   - This stage will take the engineered features to train and calibrate species-specific simulation models (e.g., for Kalanchoe) and then use MOEAs (e.g., NSGA-III) to find Pareto-optimal control strategies.
   - *(Note: This stage is not yet fully built).*

The overall orchestration of these stages, their dependencies, and data flows are managed via the `docker-compose.yml` file.

### 📚 **Research-Backed & State-of-the-Art**

This project builds upon foundational work in greenhouse control (e.g., DynaGrow) and advances it by:

- Focusing on **species-specific modeling** to provide more accurate and tailored optimization.
- Addressing the **computational feasibility** of advanced modeling and optimization techniques through **GPU acceleration research**.
- Implementing **multi-objective evolutionary algorithms** for sophisticated, data-driven control strategy generation.
- Systematically analyzing **performance trade-offs across different hardware architectures (CPU vs. GPU)** for horticultural optimization tasks.

**Contributing to sustainable agriculture**, this work aligns with Denmark’s **green transition goals**, ensuring **future-proof, resource-conscious food production.**
