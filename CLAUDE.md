# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive data-driven greenhouse climate control optimization system that balances plant growth and energy efficiency through simulation and multi-objective optimization. The project focuses on species-specific plant simulation (initially Kalanchoe blossfeldiana) and uses Multi-Objective Evolutionary Algorithms (MOEAs) to find Pareto-optimal control strategies.

The primary research focus is addressing computational bottlenecks in plant simulations and MOEA evaluations through GPU acceleration using CUDA.

## Architecture & Key Components

### Data Pipeline Structure

The project uses a multi-stage Docker-orchestrated pipeline:

1. **Stage 1: Rust Data Ingestion** (`DataIngestion/rust_pipeline/`)
   - Handles raw data ingestion from CSV/JSON sources
   - Uses `sqlx` + `tokio` for async TimescaleDB operations
   - Validates and batch inserts sensor data

2. **Stage 2: Python Pre-processing** (`DataIngestion/feature_extraction/pre_process/`)
   - Data cleaning, time regularization, missing value handling
   - External data fetching (weather, energy prices)
   - Phenotype data ingestion from literature

3. **Stage 3: Era Detection** (`DataIngestion/feature_extraction/era_detection_rust/`)
   - Rust-based changepoint detection (PELT, BOCPD, HMM)
   - Segments time-series into operational periods
   - GPU-accelerated implementations under development

4. **Stage 4: Feature Extraction** (`DataIngestion/feature_extraction/feature/`)
   - Python-based using `tsfresh`
   - GPU-accelerated feature extraction (`feature-gpu/`)
   - Supervised feature selection capabilities

5. **Stage 5: Model Building & Optimization** (`DataIngestion/model_builder/`)
   - Plant simulation models (LSTM-based)
   - MOEA implementation using `pymoo`
   - MLflow experiment tracking

### Database Architecture

- **TimescaleDB** (PostgreSQL 16) for time-series data
- Hypertables: `sensor_data`, `preprocessed_greenhouse_data`, `feature_data`
- External data tables: `external_weather_data`, `energy_prices`

## Common Development Commands

### Building & Running the Pipeline

```bash
# From DataIngestion/ directory

# Build all services
docker compose build

# Run the full orchestrated pipeline (PowerShell)
./run_orchestration.ps1

# Run for a specific date
./run_orchestration.ps1 -SingleDate "2014-01-15"

# Run the full orchestrated pipeline (Bash)
./run_orchestrated.sh

# Run tests
./run_tests.sh
```

### Individual Component Commands

```bash
# Rust pipeline
cd DataIngestion/rust_pipeline
docker compose up -d --build

# Feature extraction
cd DataIngestion/feature_extraction/feature
docker build -f feature.dockerfile -t feature-extraction .
docker run -v $(pwd):/app feature-extraction

# GPU feature extraction
cd DataIngestion/feature_extraction/feature-gpu
docker build -f feature_gpu.dockerfile -t feature-gpu .
docker run --gpus all -v $(pwd):/app feature-gpu
```

### Python Development

```bash
# Using uv for dependency management (preferred)
uv pip install -r requirements.txt
uv run python script.py

# Run specific tests with pytest
uv run pytest tests/test_feature_engineering.py -v

# Linting (when available)
uv run ruff check .
uv run black --check .
```

### Rust Development

```bash
# Format code
cargo fmt

# Run clippy lints
cargo clippy -- -D warnings

# Build in release mode
cargo build --release

# Run tests
cargo test

# Run with environment variables
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres cargo run
```

## Key Technical Patterns

### Database Access
- All Python database access goes through `db_utils.py` modules
- Use connection pooling with `psycopg2` or `sqlalchemy`
- Rust uses `sqlx` with compile-time checked queries

### Error Handling
- Python: Use custom exceptions, proper logging with `logging` module
- Rust: Use `Result<T, E>` with `thiserror` for custom errors, avoid `.unwrap()`

### Configuration
- Python: Environment variables loaded via `python-dotenv`
- Rust: Environment variables via `dotenv` crate or `config` crate
- Docker: `.env` files for docker-compose configuration

### GPU Acceleration
- CUDA kernels for computationally intensive operations
- CuPy/Numba for Python GPU code
- Focus on batch processing for efficiency

## Important Project Context

### Data Sources
- **Greenhouse sensors**: KnudJepsen and Aarslev locations
- **External weather**: Open-Meteo API
- **Energy prices**: Danish spot prices (DK1, DK2)
- **Plant phenotype**: Literature-based JSON data

### Key Libraries & Frameworks
- **Python**: pandas, numpy, scipy, tsfresh, pymoo, pytorch, sqlalchemy
- **Rust**: tokio, sqlx, serde, chrono, polars
- **GPU**: CUDA, CuPy, Numba
- **Orchestration**: Docker Compose, Prefect
- **ML Tracking**: MLflow

### Performance Considerations
- Batch operations for database inserts (1000+ rows)
- GPU acceleration for feature extraction and simulation
- Efficient time-series operations using specialized libraries
- Profile before optimizing (Python: cProfile, Rust: cargo-flamegraph)

## Testing Strategy

- Unit tests for individual components
- Integration tests with test database
- GPU tests require CUDA-capable hardware
- Use pytest for Python, cargo test for Rust

## Monitoring & Debugging

- Prefect UI for flow monitoring: http://localhost:4200
- MLflow UI for experiment tracking: http://localhost:5000
- pgAdmin for database inspection: http://localhost:5050
- Check logs: `docker compose logs -f <service-name>`

## Critical Files to Understand

1. `Doc-templates/*.md` - Comprehensive project documentation
2. `DataIngestion/docker-compose.yml` - Service orchestration
3. `DataIngestion/feature_extraction/data_processing_config.json` - Feature configuration
4. `.cursor/rules/*.mdc` - Language-specific development guidelines
5. `DataIngestion/simulation_data_prep/prefect.yaml` - Flow deployment config