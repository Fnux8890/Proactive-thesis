# Interface Control Document (ICD)

## Document Information

- **Title:** Interface Control Document
- **Project:** Simulation-Based Greenhouse Control Optimization
- **Version:** 1.0 (Based on Facilitation Report)
- **Last Updated:** [Date]

---

## 1. Introduction

### 1.1 Purpose

This document specifies the interfaces among the internal software components of the Simulation-Based Greenhouse Control Optimization system, as well as interfaces with external data storage (TimescaleDB) and the file system. It ensures clarity on how these components interact and exchange data.

### 1.2 Scope

The ICD covers:

- Interfaces between the core Rust modules (Data Access, Input Preparation, Simulation, MOEA, Logging).
- The interface between the Data Access module and the TimescaleDB database.
- Interfaces for reading configuration files and writing output files (logs, results).
- Data formats used for internal communication and external file storage.
- Error handling related to these interfaces.

*Note: This project is simulation-based and operates offline on historical data. Interfaces related to live sensors, actuators, real-time external APIs (weather, energy), or user interfaces are out of scope.*

---

## 2. Interface Overview

### 2.1 System Context Diagram

```mermaid
graph TD
    subgraph Rust Application
        DataAccess[Data Access]
        InputPrep[Input Preparation]
        Simulation[Plant Simulation]
        MOEA[Optimization MOEA]
        Logging[Logging/Output]
    end

    TimescaleDB[(TimescaleDB: Historical Data)]
    ConfigFile[Configuration File (e.g., TOML)]
    ResultFiles[/Output Files (Logs, Results)/]

    ConfigFile -- Reads --> InputPrep;
    ConfigFile -- Reads --> MOEA; # e.g., for MOEA parameters
    DataAccess -- Reads --> TimescaleDB;
    InputPrep -- Uses --> DataAccess;
    Simulation -- Uses --> InputPrep;
    MOEA -- Uses --> Simulation;
    MOEA -- Uses --> InputPrep; # For Energy Cost Data/Params
    Logging -- Collects from --> MOEA;
    Logging -- Collects from --> Simulation; # Potentially intermediate logs
    Logging -- Writes to --> ResultFiles;

    style TimescaleDB fill:#f9f,stroke:#333,stroke-width:2px
    style ConfigFile fill:#ccf,stroke:#333,stroke-width:2px
    style ResultFiles fill:#ccf,stroke:#333,stroke-width:2px
```

### 2.2 Interface Categories

1. Internal Component Interfaces (Rust Modules)
2. Database Interface (TimescaleDB)
3. File System Interfaces (Configuration, Output)

---

## 3. Internal Component Interfaces (Rust Modules)

## 3. Internal Component Interfaces (Python Modules)

These interfaces represent function calls and data passing between the Python modules within the application. Data is primarily exchanged via standard Python objects, classes, and potentially Pandas/Polars DataFrames.

### 3.1 Data Access -> Input Preparation

- **Description:** Provides raw historical time-series data retrieved from TimescaleDB, likely as a list of tuples/dicts or directly as a DataFrame.
- **Data Format:** Python list of tuples/dicts, or preferably a Polars/Pandas DataFrame.
- **Mechanism:** Function/method call returning the data structure.

### 3.2 Input Preparation -> Simulation

- **Description:** Provides processed and structured environmental data (e.g., a DataFrame or NumPy arrays) and simulation configuration parameters.
- **Data Format:** Polars/Pandas DataFrame or NumPy arrays for time-series data, Python dict or custom class instance for configuration.
- **Mechanism:** Function/method call passing the prepared input objects.

### 3.3 Input Preparation -> MOEA

- **Description:** Provides necessary data for the MOEA's objective function calculation related to energy cost.
- **Data Format:** Python dict, class instance, or relevant values (e.g., floats) derived from prepared data.
- **Mechanism:** Function/method call passing the required energy data.

### 3.4 Simulation -> MOEA

- **Description:** Returns the output metrics from a simulation run (e.g., simulated plant growth).
- **Data Format:** Python float or a simple object/dict containing the result(s).
- **Mechanism:** Function/method call returning the simulation output.

### 3.5 MOEA -> Simulation

- **Description:** Provides a candidate control strategy (e.g., a list/array of parameters) to the simulation module.
- **Data Format:** Python list, NumPy array, or custom class instance representing the control strategy parameters.
- **Mechanism:** Function/method call passing the candidate strategy representation.

### 3.6 MOEA / Simulation -> Logging

- **Description:** Sends log messages or final results to the logging system.
- **Data Format:** Standard Python strings for log messages; DataFrames or lists/dicts for results.
- **Mechanism:** Calls to the standard Python `logging` module methods.

*(Note: If the simulation core is later implemented in Rust for performance, a Python-Rust Foreign Function Interface (FFI) using tools like PyO3 will need to be defined to handle the MOEA -> Simulation and Simulation -> MOEA interactions across the language boundary.)*

---

## 4. Database Interface (TimescaleDB)

### 4.1 Data Access Module <-> TimescaleDB

- **Description:** The Data Access module reads historical environmental data from TimescaleDB.
- **Protocol:** Standard SQL over a database connection (e.g., TCP/IP).
- **Queries:** `SELECT` statements targeting specific tables and time ranges containing historical greenhouse data.
- **Data Format:** Relational database rows, fetched into Python data structures (e.g., list of tuples, potentially directly into a Polars/Pandas DataFrame) by the Data Access module (using libraries like `psycopg2`).
- **Authentication:** Database credentials (username/password) managed via configuration or environment variables.

---

## 5. File System Interfaces

### 5.1 Configuration File -> Input Prep / MOEA

- **Description:** Modules (primarily Input Preparation and MOEA) read configuration parameters from a file.
- **Location:** Defined path, potentially relative to the executable or specified via command-line argument.
- **Format:** Structured text file (e.g., TOML, YAML). Content includes simulation model parameters, MOEA settings, objective function details, database connection details.
- **Mechanism:** Standard file read operations using Python's file I/O. Parsing handled by relevant libraries (e.g., `tomli`, `PyYAML`).

### 5.2 Logging Module -> Output Files

- **Description:** The Logging module writes process logs and final optimization results to files.
- **Location:** Defined output directory.
- **Format:**
  - **Logs:** Plain text or structured logs (e.g., JSON lines) via Python's `logging` module and its formatters.
  - **Results:** Structured data format (CSV or JSON) containing the non-dominated solutions.
- **Mechanism:** Standard file write operations using Python's file I/O and potentially `pandas` or the `csv`/`json` modules.

---

## 6. Data Exchange Formats

### 6.1 Internal Data Structures (Rust)

### 6.1 Internal Data Structures (Python)

- **Description:** Data exchanged between Rust modules.
- **Format:** Defined `struct` and `enum` types within the Rust codebase. Examples include structures for historical data points, simulation inputs/outputs, control strategies, and optimization results. Type safety is enforced by the Rust compiler.

- **Description:** Data exchanged between Python modules/functions.

- **Format:** Standard Python types (lists, dicts, floats), custom classes, Pandas/Polars DataFrames, NumPy arrays. Type hinting used where appropriate.

### 6.2 Configuration File Format

- **Description:** Format for external configuration files.
- **Format:** TOML or YAML recommended for readability. Defines sections for database connection, simulation parameters, MOEA settings, etc. (Specific structure to be defined during implementation).

### 6.3 Output File Format (Results)

- **Description:** Format for storing the final non-dominated solutions.
- **Format:** CSV or JSON.
  - **CSV:** Each row represents a solution. Columns include parameters of the control strategy and the corresponding objective values (simulated growth, energy cost).
  - **JSON:** An array of objects, where each object represents a solution with its strategy parameters and objective values.
- **Schema:** To be defined, detailing specific fields and types.

---

## 7. Communication Protocols

- **Internal:** Standard Python function/method calls between modules within the same process. Potential for FFI calls if Rust simulation core is used.
- **Database:** SQL protocol over a TCP/IP connection managed by the Python database driver (e.g., `psycopg2`).
- **File System:** Standard OS-level file I/O protocols.

*(Note: Network protocols like MQTT, REST, WebSocket are not used).*

---

## 8. Error Handling

### 8.1 Error Types

- **Database Errors:** Connection failures, query errors, authentication issues.
- **File I/O Errors:** File not found, permission denied, parsing errors (configuration), write errors (output).
- **Configuration Errors:** Missing parameters, invalid values, incorrect format.
- **Simulation/Optimization Errors:** Numerical instability, invalid inputs, convergence issues.
- **Internal Logic Errors:** Panics, assertion failures (should be minimized through robust design and testing).

### 8.2 Handling Strategy

- Utilize Python's standard exception handling (`try`/`except`) for recoverable errors.
- Propagate exceptions appropriately between modules/functions.
- Provide informative error messages in logs using the `logging` module.
- Handle critical errors (e.g., inability to read config, connect to DB) by catching exceptions at a high level and terminating the application gracefully with an error message.
- Use the standard `logging` module to record errors and warnings.

---

## 9. Security

### 9.1 Considerations

- **Database Credentials:** Securely manage database connection strings/credentials (e.g., use environment variables, configuration files with appropriate permissions, or a secrets management system if deployed in a shared environment - though less likely for a thesis project). Avoid hardcoding credentials.
- **File Permissions:** Ensure configuration files containing sensitive information (like DB credentials) have restricted read permissions. Ensure the application has write permissions to the designated output directory.
- **Input Validation:** Validate data read from configuration files to prevent parsing errors or unexpected behavior.

*(Note: Complex security mechanisms like OAuth, network encryption beyond standard DB protocols, or detailed access control are likely out of scope for this offline analysis tool).*

---

## 10. Performance Requirements

### 10.1 Focus Areas

- **Data Loading Time:** Efficient retrieval and preparation of historical data from TimescaleDB.
- **Simulation Execution Speed:** Individual simulation runs should be fast enough to allow for many evaluations within the MOEA.
- **Optimization Time:** The overall MOEA process should complete within a reasonable timeframe for analysis (e.g., hours rather than days, depending on complexity and data size).
- **Resource Usage:** Memory and CPU consumption should remain within the limits of a typical development/research machine.

*(Note: Real-time response constraints are not applicable).*

---

## 11. Monitoring and Logging

### 11.1 Logged Information

- **Process Flow:** Start/end of major stages (data loading, simulation run, MOEA generation).
- **Configuration:** Parameters used for the run.
- **Errors & Warnings:** Any issues encountered during execution.
- **Optimization Progress:** MOEA generation number, population statistics (optional).
- **Results:** Final non-dominated solutions found.
- **Performance Metrics:** Execution time for key stages (optional).

### 11.2 Log Format

- **Recommended:** Structured logging (e.g., JSON) using crates like `tracing` with `tracing-subscriber` for flexible output formatting and filtering. Plain text logs are also acceptable.
- **Content:** Timestamp, log level (INFO, WARN, ERROR, DEBUG), message, potentially module/function origin.

---

## Appendices

*(Appendices will be developed during the project)*

- **Appendix A:** Internal API Specifications (Detailed Rust function signatures and struct definitions)
- **Appendix B:** Configuration File Format Specification
- **Appendix C:** Output File Format Specification (Results - CSV/JSON schema)
- **Appendix D:** TimescaleDB Schema Details (Relevant tables/columns)
