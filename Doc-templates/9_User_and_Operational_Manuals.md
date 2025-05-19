# Usage Guide: Simulation-Based Greenhouse Control Optimization Tool

## Document Information

- **Document Title:** Usage Guide
- **Project:** Simulation-Based Greenhouse Control Optimization
- **Version:** 1.0 (Based on Facilitation Report)
- **Last Updated:** [Date]

---

## 1. Introduction

### 1.1 Purpose

This guide explains how to set up, configure, execute, and understand the outputs of the Simulation-Based Greenhouse Control Optimization tool. It is intended for researchers or developers running the simulation and optimization experiments.

### 1.2 System Overview

The tool is a command-line Rust application designed to:

- Read historical environmental data from a TimescaleDB database.
- Read configuration parameters from a file (e.g., TOML).
- Run a plant growth simulation based on the historical data and simulation parameters.
- Execute a Multi-Objective Evolutionary Algorithm (MOEA) to find trade-offs between simulated plant growth and estimated energy cost based on defined control strategies (e.g., lighting schedules).
- Output the results (non-dominated strategies) and process logs to the file system.
- Run within a Docker container for environment consistency.

---

## 2. Getting Started

### 2.1 System Requirements

- **Runtime Environment:**
  - Docker (Recommended for consistent environment)
  - Alternatively, a Python environment (>= [Specify Version, e.g., 3.10+]) with `uv` installed.
- **Database:** Access to a running TimescaleDB instance (local via Docker or remote).
- **Operating System:** Linux, macOS, or Windows (with appropriate Python/`uv`/Docker setup).
- **Source Code:** Cloned project repository.

### 2.2 Setup

1. **Clone Repository:**

    ```bash
    git clone [Repository URL]
    cd [Repository Directory]
    ```

2. **Database Setup:**
    - Ensure your TimescaleDB instance is running.
    - Apply the necessary database schema (refer to `[Path to Schema Files or Instructions]`).
    - Load historical environmental data into the appropriate TimescaleDB tables. (Format details in Section 4.1).
3. **Configuration:**
    - Create or modify the configuration file (e.g., `config.toml`). See Section 3 for details. Ensure database connection details are correct.
4. **Python Environment Setup (if not using Docker):**
    - Navigate to the project root directory.
    - Create a virtual environment: `uv venv`
    - Activate the environment (e.g., `. .venv/bin/activate` on Linux/macOS, `.venv\Scripts\activate` on Windows PowerShell).
    - Install dependencies: `uv pip install -r requirements.txt` (assuming a `requirements.txt` file exists).
5. **(Optional) Docker Setup:**
    - Build the Docker image:

      ```bash
      docker build -t gh-sim-opt .
      ```

    - Ensure the Docker container can access the TimescaleDB instance (network configuration might be needed).

---

## 3. Configuration

### 3.1 Configuration File

The application requires a configuration file (default: `config.toml` or specified via command-line argument) to control its behavior.

```toml
# Example config.toml structure

[database]
connection_string = "postgres://user:password@host:port/database"

[simulation]
# Parameters specific to the chosen plant growth model
# e.g., light_use_efficiency = 0.5
# e.g., temp_optimum = 25.0
# ... other model parameters ...
evaluation_start_time = "YYYY-MM-DDTHH:MM:SSZ"
evaluation_end_time = "YYYY-MM-DDTHH:MM:SSZ"

[optimization]
# MOEA Settings (e.g., for NSGA-II)
population_size = 100
max_generations = 50
# Crossover/Mutation probabilities, etc.
# ... other MOEA parameters ...

# Control Strategy Definition
# e.g., define optimization variables for lighting schedule
# variable_bounds = { light_intensity = [0, 100], duration = [0, 12] }

[objectives]
# Parameters for energy cost calculation
# e.g., energy_price_kwh = 0.15
# e.g., lighting_watts_per_unit = 50

[output]
log_file = "output/run.log"
results_file = "output/results.csv" # or .json
log_level = "INFO" # e.g., ERROR, WARN, INFO, DEBUG, TRACE
results_format = "CSV" # CSV or JSON

```

### 3.2 Key Parameters

- **`[database]`**: Connection string for TimescaleDB.
- **`[simulation]`**: Parameters required by the specific plant growth model being used, and the time period for evaluation.
- **`[optimization]`**: Settings for the MOEA (population size, generations, algorithm-specific parameters), definition of the control strategy variables and their bounds.
- **`[objectives]`**: Parameters needed for calculating the objective functions (e.g., energy cost coefficients).
- **`[output]`**: Paths for log and result files, desired log verbosity, and format for the results file (CSV or JSON).

---

## 4. Running the Application

### 4.1 Native Execution (using Python)

1. **Build:**

    *No explicit build step usually required for Python.* Ensure dependencies are installed (See Setup Section 2.2, Step 4).

2. **Run:**

    ```bash
    # Ensure your virtual environment is activated
    # Assuming config.toml is in the root directory and the main script is main.py
    python main.py --config config.toml
    # Or specify a different config path
    # python main.py --config path/to/your/config.toml
    ```

### 4.2 Docker Execution

1. **Run Container:**

    ```bash
    docker run --rm \
      -v $(pwd)/config.toml:/app/config.toml \
      -v $(pwd)/output:/app/output \
      --network=[Your_Docker_Network_With_DB] \
      gh-sim-opt # Add command if not specified in Dockerfile ENTRYPOINT/CMD
      # Example if CMD/ENTRYPOINT is just 'python':
      # gh-sim-opt main.py --config /app/config.toml
    ```

    *Explanation:*
    - `--rm`: Removes the container after execution.
    - `-v $(pwd)/config.toml:/app/config.toml`: Mounts your local config file into the container. Adjust path if needed.
    - `-v $(pwd)/output:/app/output`: Mounts a local `output` directory into the container to retrieve results/logs. Create this directory locally first (`mkdir output`).
    - `--network=[Your_Docker_Network_With_DB]`: Connects the container to the network where your TimescaleDB is accessible. Replace `[Your_Docker_Network_With_DB]` with the actual network name.
    - `gh-sim-opt`: The name of the built Docker image.
    *(Command to run the Python script, e.g., `main.py --config /app/config.toml`, might be part of the command above or defined in the Dockerfile's `ENTRYPOINT` or `CMD`)*

### 4.3 Input Data Format (TimescaleDB)

- The application expects historical data in specific TimescaleDB tables.
- Refer to the Database Schema documentation (e.g., ICD Appendix D or schema definition files) for table names, column names (e.g., `timestamp`, `sensor_type`, `value`), and expected data types.

---

## 5. Output Interpretation

### 5.1 Log File

- **Location:** Specified by `output.log_file` in the configuration.
- **Content:** Provides information about the execution process, including:
  - Configuration parameters used.
  - Start and end times of major stages (data loading, simulation, optimization).
  - MOEA progress (e.g., generation number).
  - Any warnings or errors encountered.
  - Log level controlled by `output.log_level`.

### 5.2 Results File

- **Location:** Specified by `output.results_file` in the configuration.
- **Format:** CSV or JSON, as specified by `output.results_format`.
- **Content:** Contains the set of non-dominated solutions found by the MOEA.
  - **CSV:** Each row is a solution. Columns represent the optimized control strategy parameters and the corresponding objective values (e.g., `simulated_growth`, `energy_cost`, `lighting_intensity_param1`, `lighting_duration_param2`, ...).
  - **JSON:** Typically an array of objects, each object representing a solution with key-value pairs for parameters and objectives.
- **Analysis:** This file is the primary output for research analysis. Use external tools (Python scripts with Pandas/Matplotlib, R, etc.) to plot the Pareto front and analyze the trade-offs.

---

## 6. Troubleshooting

### 6.1 Common Issues

- **Database Connection Errors:**
  - Verify the `database.connection_string` in `config.toml`.
  - Check if the TimescaleDB instance is running and accessible from where the application is running (including Docker network settings).
  - Ensure correct username/password/database name.
- **Configuration Errors:**
  - Application fails to start, often with a parsing error message.
  - Carefully check the `config.toml` syntax.
  - Ensure all required parameters are present for the selected simulation model and MOEA.
- **File Path Errors:**
  - Errors reading config or writing output files.
  - Verify the paths specified in `config.toml` (`output.log_file`, `output.results_file`).
  - Ensure the application (or Docker container) has write permissions to the output directory.
- **Simulation/Optimization Errors:**
  - Check the log file (`output.log_file`) for specific error messages (e.g., numerical issues, invalid inputs detected).
  - Increase `log_level` to `DEBUG` or `TRACE` for more detailed information.
  - Review simulation model parameters and MOEA settings.

### 6.2 Getting Help

- Consult the project's README file.
- Check existing GitHub Issues for similar problems.
- If necessary, file a new GitHub Issue with details about the error, your configuration, and steps to reproduce.

---

## Appendices

*(Appendices might be added during the project)*

### Appendix A: Example Configuration Files

Complete examples for different scenarios.

### Appendix B: Output File Schema Details

Precise column names/JSON structure for the results file.

### Appendix C: Glossary

Definitions of terms specific to the project.
