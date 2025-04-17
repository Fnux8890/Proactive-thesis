# Data Handling and Output Management Document

## Document Information

- **Document Title:** Data Handling and Output Management Document
- **Project:** Simulation-Based Greenhouse Control Optimization
- **Version:** 1.0 (Based on Facilitation Report)
- **Last Updated:** [Date]

---

## 1. Introduction

### 1.1 Purpose

This document outlines the strategy for managing input data (historical environmental data, configuration) and handling the outputs (simulation results, optimization strategies, logs) for the simulation-based greenhouse control optimization system.

### 1.2 Scope

This document covers:

- Sources and formats of input data (TimescaleDB, Configuration Files).
- Storage mechanisms for input data and generated outputs.
- The flow of data through the Rust application modules (Data Access, Input Prep, Simulation, MOEA, Logging).
- Formats for intermediate data structures and final output files.
- Basic data quality considerations for historical input.

*Note: This document focuses on the data specific to the offline simulation and optimization process. It does not cover real-time data streams or complex operational analytics pipelines.*

---

## 2. Data Architecture Overview

### 2.1 Data Flow

```mermaid
graph TD
    A[TimescaleDB: Historical Environmental Data] --> B(Data Access Module - Rust);
    C[Configuration File (e.g., TOML)] --> D(Input Preparation & MOEA Modules - Rust);
    B --> E(Input Preparation Module - Rust);
    E -- Prepared Simulation Inputs --> F(Simulation Module - Rust);
    F -- Simulated Growth Metrics --> G(MOEA Module - Rust);
    E -- Energy Cost Data --> G;
    G -- Candidate Strategies --> F;
    G -- Final Strategies & Logs --> H(Logging/Output Module - Rust);
    H --> I[Output Files (Logs, Results - CSV/JSON)];

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style C fill:#ccf,stroke:#333,stroke-width:2px
    style I fill:#ccf,stroke:#333,stroke-width:2px
```

*Diagram Interpretation:* Historical data is read from TimescaleDB. Configuration is read from files. These are processed by the Rust application. The simulation and MOEA modules exchange data internally. Final results and logs are written to the file system.

### 2.2 Data Categories

1. **Historical Input Data:** Time-series environmental readings (temperature, humidity, CO2, light, etc.) stored in TimescaleDB.
2. **Configuration Data:** Parameters for simulation, MOEA, database connection, etc., stored in configuration files (e.g., TOML).
3. **Intermediate Data:** Internal Rust structs representing prepared simulation inputs, control strategy candidates, and simulation outputs.
4. **Output Data:**
    - Non-dominated control strategies found by the MOEA (e.g., CSV, JSON files).
    - Process logs (e.g., text files, structured logs).

---

## 3. Data Input and Sources

### 3.1 Historical Data (TimescaleDB)

- **Source:** TimescaleDB instance containing historical greenhouse environmental data.
- **Access Mechanism:** `Data Access` module in Rust using `sqlx` or similar to execute SQL `SELECT` queries.
- **Schema:** Defined relational schema in TimescaleDB (See Appendix D in ICD). Key fields typically include timestamp, sensor type/location, and value.
- **Quality Considerations:** Input data is assumed to be reasonably clean. The `Input Preparation` module may handle basic filtering, gap-filling (interpolation if necessary), or sanity checks based on configuration.

### 3.2 Configuration Data (Files)

- **Source:** Configuration files (e.g., `config.toml`).
- **Access Mechanism:** Rust file I/O operations within relevant modules (Input Prep, MOEA, Data Access). Parsing using libraries like `toml`.
- **Content:** Database connection details, simulation model parameters, MOEA settings (population size, generations, termination criteria), energy cost calculation parameters, logging settings, file paths.

---

## 4. Data Storage

### 4.1 Input Data Storage

- **Historical Environmental Data:** Persisted in TimescaleDB.
- **Configuration Data:** Stored as files (e.g., TOML, YAML) in the project repository or a designated configuration directory.

### 4.2 Output Data Storage

- **Optimization Results (Strategies):** Written to the file system in a structured format (CSV or JSON) in a designated output directory.
- **Logs:** Written to the file system (plain text or structured logs) in a designated output or logs directory.

### 4.3 Intermediate Data

- Stored transiently in memory within the Rust application's data structures (structs, enums) during execution. Not persisted unless logged.

---

## 5. Data Preparation and Usage

### 5.1 Input Preparation Module

- **Responsibility:** Transforms raw historical data (from Data Access) and configuration into the specific formats required by the `Simulation` and `MOEA` modules.
- **Tasks:**
  - Filtering data based on time range or other criteria.
  - Aggregating data if needed (e.g., averaging over intervals).
  - Handling potential gaps (e.g., via interpolation based on configuration).
  - Structuring data into time steps suitable for the simulation model.
  - Calculating or retrieving energy cost information.
  - Loading and providing simulation/MOEA parameters from configuration.
- **Technology:** Python using libraries like `pandas` or `polars` for data manipulation, transformation, and feature calculation.

### 5.1.1 Defined Derived Features

The following derived features are calculated by the `Input Preparation Module` based on the raw historical data and configuration, to serve as inputs for the `Simulation Module` or for calculating objective functions in the `MOEA Module`:

- **For Simulation Model:**
    1. `temp_delta_in_out`: `sensor_temperature_ins - sensor_temperature_outs` - Represents the temperature gradient driving heat transfer.
    2. `VPD_ins`: `SVP(sensor_temperature_ins) * (1 - (sensor_humidity_ins / 100))` - Vapor Pressure Deficit inside, driving transpiration. Requires a standard function for Saturation Vapor Pressure (`SVP`).
    3. `humidity_delta_in_out`: `sensor_humidity_ins - sensor_humidity_outs` - Represents the humidity gradient relevant for moisture exchange.
    4. `temp_rate_of_change`: Rate of change of `sensor_temperature_ins` (e.g., difference from previous time step / time_step_duration) - Indicates thermal dynamics.
    5. `humidity_rate_of_change`: Rate of change of `sensor_humidity_ins` - Indicates moisture dynamics.
    6. `temp_rolling_avg_Xmin`: Rolling average of `sensor_temperature_ins` over a configurable `X` minutes period - Smoothed temperature trend.
    7. `humidity_rolling_avg_Xmin`: Rolling average of `sensor_humidity_ins` over a configurable `X` minutes period - Smoothed humidity trend.

- **For Objective Functions:**
    8.  `total_heating_energy`: Sum/Integral of (`actuator_heating_power * time_step_duration`) - Total energy consumed by heating.
    9.  `total_ventilation_energy`: *Calculation TBD* (Depends on how `actuator_ventilation_pos` relates to energy use, if ventilation is powered).
    10. `total_lighting_energy`: Sum/Integral of (`actuator_lighting_power` or equivalent * `time_step_duration`) - Total energy consumed by lighting.
    11. `total_light_par`: Sum/Integral of (`sensor_light_par * time_step_duration`) - Total photosynthetic active radiation received, proxy for growth potential.
    12. `mean_abs_deviation_temp_from_setpoint`: `mean(abs(sensor_temperature_ins - temp_setpoint))` - Average deviation from the temperature target. Requires `temp_setpoint`.
    13. `mean_abs_deviation_humidity_from_setpoint`: `mean(abs(sensor_humidity_ins - humidity_setpoint))` - Average deviation from the humidity target. Requires `humidity_setpoint`.
    14. `mean_abs_deviation_light_from_setpoint`: `mean(abs(sensor_light_par - light_setpoint))` - Average deviation from light target (if applicable, e.g., DLI target). Requires `light_setpoint`.
    15. `time_in_optimal_temp_range`: Percentage of time `sensor_temperature_ins` is within a defined optimal band. Requires defined range.
    16. `time_in_optimal_humidity_range`: Percentage of time `sensor_humidity_ins` is within a defined optimal band. Requires defined range.

- **Implementation Notes:**
  - The specific formula for `SVP(temperature)` (e.g., Magnus formula) should be documented in the code or a technical appendix.
  - The time window `X` for rolling averages needs to be configurable.
  - The calculation for `total_ventilation_energy` needs clarification based on the specific actuators.
  - Setpoint values and optimal ranges are external inputs or configurations.

---

## 6. Optimization and Analysis

### 6.1 Core Analytical Engine (MOEA Module)

- **Purpose:** Uses the prepared input data and simulation results to perform multi-objective optimization.
- **Analysis Type:** Finds a set of non-dominated solutions representing the trade-off between minimizing energy cost and maximizing simulated plant growth.
- **Inputs:**
  - Simulated growth metrics (from `Simulation` module).
  - Energy cost data/parameters (from `Input Preparation` module).
  - MOEA configuration settings (population size, generations, etc.).
- **Output:** A collection of non-dominated `ControlStrategy` representations and their corresponding objective function values.

### 6.2 Post-Processing Analysis (Manual/External)

- **Purpose:** Interpretation of the optimization results.
- **Method:** Analysis of the output files (e.g., CSV/JSON containing non-dominated solutions) using external tools (e.g., Python scripts with Pandas/Matplotlib, R, spreadsheet software) to visualize the Pareto front and understand the trade-offs.
- **Scope:** This analysis is performed *after* the Rust application completes its run and is part of the research interpretation, not an automated component of the system itself.

---

## 7. Output and Logging

### 7.1 Output Files

- **Optimization Results:**
  - **Format:** CSV or JSON (specified in configuration).
  - **Content:** Each record represents a non-dominated solution, including the parameters defining the control strategy and the achieved objective values (simulated growth, energy cost). (See Appendix C in ICD for schema).
  - **Location:** Designated output directory.
- **Logs:**
  - **Format:** Plain text or structured JSON lines (specified in configuration), managed by logging crates (`tracing`, `log`).
  - **Content:** Timestamps, log levels, messages indicating process stages (start/end of data loading, simulation, optimization generations), configuration used, errors, warnings, potentially performance timings.
  - **Location:** Designated output/logs directory.

### 7.2 Log Levels

- Standard levels (ERROR, WARN, INFO, DEBUG, TRACE) used to control verbosity. Configurable via the configuration file.

---

## 8. Data Security

### 8.1 Considerations

- **Database Credentials:** Secure management of TimescaleDB connection details (environment variables or restricted config files preferred over hardcoding).
- **Configuration Files:** Ensure appropriate file permissions if sensitive data (like credentials) is stored.
- **Output Data:** No specific security measures applied beyond standard file system permissions, as the output represents simulation results, not sensitive personal or operational data.

---

## 9. Performance Considerations

### 9.1 Data Handling Performance

- **Database Queries:** Efficient SQL queries in the `Data Access` module are important, especially if dealing with large historical datasets. Indexing in TimescaleDB is crucial.
- **Input Preparation:** Processing steps should be efficient to avoid becoming a bottleneck before simulation/optimization.
- **Output Writing:** Writing results and logs should be efficient, especially if logging verbosely during long optimization runs. Asynchronous logging might be considered if performance is critical.

---

## Appendices

*(Appendices defining specific formats will be detailed during implementation and referenced here and in the ICD)*

- **Appendix A:** TimescaleDB Input Schema Reference (Link to ICD Appendix D)
- **Appendix B:** Configuration File Format Details (Link to ICD Appendix B)
- **Appendix C:** Output Result File Format (CSV/JSON Schema) (Link to ICD Appendix C)
