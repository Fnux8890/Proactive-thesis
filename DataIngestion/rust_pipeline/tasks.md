# Task List: Rust Greenhouse Data Pipeline

**Last Updated:** Wednesday, April 9, 2025

**Overall Objective:** Develop a robust, end-to-end pipeline using Rust for data ingestion/validation, TimescaleDB for storage, and Python for analysis, modeling, and proactive multi-objective optimization of greenhouse environments.

**Note:** Refer to `Docs/*.md` (especially `Rust_Coding_Guidelines.md`, `Architecture.md`, etc.) for detailed implementation rules and standards when working on these tasks.

---

## Report 1: Data Ingestion (Rust)

- [ ] **1.1. Setup Rust Ingestion Environment** *(Partially Done)*
  - [x] Initialize Rust project, basic deps (csv, serde), basic structure (`main.rs`).
  - [ ] Configure core dependencies in `Cargo.toml` (`sqlx`, `tokio` features beyond basic runtime, `tracing`, `config`/`dotenv`).
  - [ ] Establish full module structure (`src/parser.rs`, `src/validation.rs`, `src/db_inserter.rs`, `src/config.rs`, `src/errors.rs`).
  - [ ] Add initial `README.md` and basic module/function doc comments per `Docs/Documentation_Guidelines.md`.

- [ ] **1.2. Implement Robust CSV Parsing** *(Mostly Done)*
  - [x] Implement parsing for multiple specific CSV formats (Aarslev/Knudjepsen).
  - [x] Handle required delimiters (`;` and `,`).
  - [x] Handle comma decimals (Knudjepsen).
  - [x] Implement `SensorValue` enum and custom `Deserialize` for robust Aarslev Celle parsing (handles quoted numbers, unquoted numbers, empty fields).
  - [x] Basic error handling per file is functional.
  - [ ] Refine error handling strategy (implement custom errors in `src/errors.rs` using `thiserror`, integrate `tracing` for logging, define bad record strategy - skip/log/quarantine).
  - [ ] Implement timestamp parsing for various formats using `chrono`.
  - [ ] Investigate and fix premature container exit (exit code 0) when processing full dataset. (Deferred until core features like timestamp parsing are complete).

- [ ] **1.3. Implement Initial Data Validation in Rust** *(Not Started)*
  - [ ] Create `src/validation.rs` module.
  - [ ] Implement validation logic *after* successful parsing (e.g., range checks for temp/humidity, timestamp validity, required field checks).
  - [ ] Integrate validation failures into the error handling strategy (e.g., specific error variants, logging).

- [x] **1.4. Develop Proactive Schema/Format Handling** *(Partially Done)*
  - [x] Parse `aarslev/celle*/*.csv.json` config files into `AarslevCelleJsonConfig` struct.
  - [x] **(NEXT PRIORITY)** Utilize parsed `AarslevCelleJsonConfig` during CSV parsing for `aarslev/celle*` files:
    - [x] Store parsed configs effectively (e.g., `HashMap<PathBuf, AarslevCelleJsonConfig>`).
    - [x] Lookup the relevant config when processing a `celle` CSV file path.
    - [x] Use the `delimiter` field from the loaded config in `csv::ReaderBuilder`.
    - [x] Integrate `date_format` and `time_format` using `chrono` for timestamp parsing.
    - [ ] *Future Use:* Plan integration for `Variables` field from config.
  - [x] Develop a broader configuration-driven strategy (e.g., using `config` crate with `config.toml` or extending data_files.json) to define formats for *all* data sources, replacing hardcoded path checks.
  - [ ] Implement logging of the format configuration being used for each file.
  - [ ] Define alerting strategy for format mismatches or high error rates.

- [ ] **(Implicit) Handling JSON Data Parsing** *(Partially Done - Config JSON Only)*
  - [x] Implemented parsing for `AarslevCelleJsonConfig` JSON files.
  - [x] Implement parsing logic for other potential JSON data files if identified (e.g., `AarslevStreamJSON`).

---

## Report 2: Data Storage (TimescaleDB) *(Not Started)*

- [ ] **2.1. Setup TimescaleDB Instance and Schema**
  - [ ] Set up TimescaleDB instance (Docker preferred for dev).
  - [ ] Define schema: tables, columns, data types (`TIMESTAMPTZ`, `FLOAT8`, etc.).
  - [ ] Convert data tables to Hypertables (`create_hypertable`) with appropriate `chunk_time_interval`.
- [ ] **2.2. Implement Rust-TimescaleDB Connection Pooling**
  - [ ] Choose pooling library (`sqlx` built-in recommended).
  - [ ] Configure pool options (`max_connections`, timeouts).
  - [ ] Implement secure credential loading (`dotenv`/`config`).
  - [ ] Initialize pool asynchronously at startup (`Arc<PgPool>`).
- [ ] **2.3. Implement Efficient Batch Inserts from Rust**
  - [ ] Implement `UNNEST` batch insert strategy.
  - [ ] Restructure Rust structs to column vectors.
  - [ ] Construct `INSERT ... SELECT * FROM UNNEST(...)` query with explicit type casts.
  - [ ] Wrap inserts in transactions.
  - [ ] Make batch size configurable.
  - [ ] Consider parallelism (`tokio::spawn`) if needed.
- [ ] **2.4. Configure TimescaleDB Indexing and Compression**
  - [ ] Add indexes on frequently filtered/grouped columns (e.g., sensor ID).
  - [ ] Ensure unique indexes include partitioning columns.
  - [ ] Enable and configure TimescaleDB compression (`ALTER TABLE ... SET (timescaledb.compress = true)`).
  - [ ] Define `compress_orderby` and `compress_segmentby`.
  - [ ] Set up automatic compression policy (`add_compression_policy`).

---

## Report 3: Data Analysis & Modeling (Python) *(Not Started)*

- [ ] **3.1. Setup Python Analysis Environment**
- [ ] **3.2. Implement Python-TimescaleDB Connection**
- [ ] **3.3. Implement Efficient Data Querying from Python** (incl. `time_bucket`, chunking)
- [ ] **3.4. Implement Data Cleansing and Validation in Python** (Missing values, outliers, etc.)
- [ ] **3.5. Implement Feature Extraction Strategy** (Hybrid: `tsfresh` + domain-specific)
- [ ] **3.6. Develop and Train Predictive Models (ANNs)**

---

## Report 4: Proactive Multi-Objective Optimization *(Not Started)*

- [ ] **4.1. Define Optimization Objectives and Constraints**
- [ ] **4.2. Integrate Predictive Models into Optimization**
- [ ] **4.3. Implement Multi-Objective Genetic Algorithm (MOGA)**
- [ ] **4.4. Develop Control Strategy Selection Logic**
- [ ] **4.5. Integrate Optimization with Control System**

---

## Report 5: Pipeline Orchestration & Integration *(Not Started)*

- [ ] **5.1. Select and Setup Orchestration Tool** (e.g., Dagster, Prefect)
- [ ] **5.2. Define Pipeline Workflow in Orchestrator**
- [ ] **5.3. Implement Monitoring and Alerting**
- [ ] **5.4. Investigate PyO3 for Performance Bottlenecks (Optional)**

---
