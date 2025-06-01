# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive data-driven greenhouse climate control optimization system that balances plant growth and energy efficiency through simulation and multi-objective optimization. The project focuses on species-specific plant simulation (initially Kalanchoe blossfeldiana) and uses Multi-Objective Evolutionary Algorithms (MOEAs) to find Pareto-optimal control strategies.

The primary research focus is addressing computational bottlenecks in plant simulations and MOEA evaluations through GPU acceleration using CUDA, while handling extreme data sparsity (91.3% missing values) in greenhouse sensor data.

## Current Goals & Objectives

1. **Enhanced Sparse Pipeline**: Implement and optimize a hybrid Rust+Python pipeline that efficiently processes sparse greenhouse data with GPU acceleration
2. **Multi-Level Feature Extraction**: Extract meaningful features at three temporal scales (hours/days, days/weeks, weeks/months) to capture plant growth dynamics
3. **MOEA GPU Acceleration**: Optimize multi-objective evolutionary algorithms for finding Pareto-optimal greenhouse control strategies
4. **Production-Ready Deployment**: Ensure all components run via Docker Compose without additional scripts or manual intervention
5. **Unified Architecture**: Maintain a single, cohesive pipeline that avoids temporary workarounds or duplicate implementations

## Architecture & Key Components

### Data Pipeline Structure

The project uses an **Enhanced Sparse Pipeline** architecture that integrates multiple stages for efficiency:

1. **Stage 1: Rust Data Ingestion** (`DataIngestion/rust_pipeline/`)
   - Handles raw data ingestion from CSV/JSON sources
   - Uses `sqlx` + `tokio` for async TimescaleDB operations
   - Validates and batch inserts sensor data (~10K rows/second)
   - Output: `sensor_data` hypertable

2. **Stages 2-4: Integrated Sparse Pipeline** (`DataIngestion/gpu_feature_extraction/`)
   - **Hybrid Rust+Python architecture** for optimal performance
   - **Sub-stage 2**: Sparse-aware data aggregation and resampling
   - **Sub-stage 3**: Hybrid feature extraction:
     - Rust: CPU-bound sparse operations (coverage, gap analysis)
     - Python: GPU-accelerated complex features (statistics, patterns)
   - **Sub-stage 4**: Simplified era creation based on data availability
   - Handles 91.3% sparse data efficiently
   - Output: `sparse_features` table (50-100 features per sensor)

3. **Stage 5: Model Building** (`DataIngestion/model_builder/`)
   - Trains surrogate models using PyTorch (LSTM) and LightGBM
   - GPU-accelerated training with MLflow experiment tracking
   - Multi-objective models: plant growth, energy, resource efficiency
   - Output: Trained models stored in `/models` directory

4. **Stage 6: MOEA Optimization** (`DataIngestion/moea_optimizer/`)
   - CPU version: pymoo library (NSGA-III)
   - GPU version: Custom PyTorch-based NSGA-III
   - Finds Pareto-optimal greenhouse control strategies
   - Output: Optimization results in `/results` directory

### Database Architecture

- **TimescaleDB** (PostgreSQL 16) for time-series data
- **Hypertables**: 
  - `sensor_data`: Raw sensor measurements
  - `preprocessed_greenhouse_data`: Cleaned and regularized data
  - `sparse_features`: Extracted features optimized for sparse data
  - `enhanced_sparse_features`: Multi-level features with external data
- **External data tables**: 
  - `external_weather_data`: Open-Meteo API weather data
  - `energy_prices`: Danish spot prices (DK1, DK2)
  - `phenotype_data`: Literature-based plant parameters
- **Era tables**: `era_labels_a`, `era_labels_b`, `era_labels_c` (multi-level temporal segmentation)

## Common Development Commands

### IMPORTANT: Use Docker Compose ONLY

**DO NOT CREATE**: Shell scripts (.sh), PowerShell scripts (.ps1), or Python runner scripts  
**ALWAYS USE**: Docker Compose commands directly

### Building & Running the Pipeline

```bash
# From DataIngestion/ directory

# Build all services
docker compose build

# Run the standard pipeline
docker compose up

# Run the enhanced sparse pipeline (RECOMMENDED)
docker compose -f docker-compose.yml -f docker-compose.enhanced.yml up

# Run production pipeline with monitoring
docker compose -f docker-compose.yml -f docker-compose.prod.yml up

# Run specific service
docker compose up enhanced_sparse_pipeline

# Check service logs
docker compose logs -f enhanced_sparse_pipeline

# Run with custom dates (via environment variables)
START_DATE=2014-01-15 END_DATE=2014-02-15 docker compose up enhanced_sparse_pipeline
```

### Individual Component Testing (Development Only)

```bash
# Test Rust compilation
cd DataIngestion/gpu_feature_extraction
cargo check
cargo test

# Test Python GPU features locally
cd DataIngestion/gpu_feature_extraction
python minimal_gpu_features.py < test_input.json

# But for actual pipeline runs, ALWAYS use Docker Compose:
docker compose up enhanced_sparse_pipeline
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
- **Hybrid Processing Model**: Rust handles CPU-bound operations, Python handles GPU operations
- **Python GPU Libraries**: PyTorch (primary), CuPy, RAPIDS cuDF
- **GPU Feature Categories**:
  - Statistical features (mean, std, percentiles)
  - Temporal features (trends, seasonality)
  - Cross-sensor correlations
  - Sparse-specific features (coverage, gap statistics)
- **Performance**: ~1M samples/second in hybrid mode
- **Memory Management**: Batch processing to fit GPU memory constraints

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
- **Database**: Batch operations (1000+ rows), connection pooling
- **Sparse Data Handling**: 
  - Skip computations for windows with <5% data coverage
  - Use sparse-aware algorithms (e.g., nanmean instead of mean)
  - Vectorized operations for gap analysis
- **GPU Optimization**:
  - Batch size tuning (typically 1000-10000 samples)
  - Minimize CPU-GPU memory transfers
  - Use mixed precision where appropriate
- **Multi-Level Processing**: Parallel computation of era levels
- **Profiling Tools**: 
  - Python: cProfile, line_profiler, nvidia-smi
  - Rust: cargo-flamegraph, tokio-console

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

## Docker Services Reference

### Core Services
- **db**: TimescaleDB (PostgreSQL 16) - All pipeline data storage
- **rust_pipeline**: Stage 1 - Raw data ingestion from CSV/JSON files
- **enhanced_sparse_pipeline**: Stages 2-4 - Integrated sparse-aware processing with GPU features
- **model_builder**: Stage 5 - GPU-accelerated model training (LSTM, LightGBM)
- **moea_optimizer**: Stage 6 - Multi-objective optimization (CPU/GPU variants)

### Supporting Services (Production)
- **prometheus**: Metrics collection
- **grafana**: Visualization dashboards
- **dcgm-exporter**: NVIDIA GPU metrics
- **node-exporter**: System metrics

### Service Configuration Files
- `docker-compose.yml`: Base configuration with core services
- `docker-compose.enhanced.yml`: Enhanced sparse pipeline override
- `docker-compose.prod.yml`: Production monitoring stack
- `docker-compose.cloud.yml`: Cloud deployment (4x A100 GPUs)

## Critical Files to Understand

1. **Pipeline Entry Points**:
   - `DataIngestion/docker-compose.enhanced.yml` - Enhanced pipeline configuration
   - `DataIngestion/gpu_feature_extraction/src/main.rs` - Rust orchestrator with hybrid mode
   - `DataIngestion/gpu_feature_extraction/src/python_bridge.rs` - Rust-Python interface

2. **Configuration Files**:
   - `DataIngestion/feature_extraction/data_processing_config.json` - Feature definitions
   - `DataIngestion/moea_optimizer/config/moea_config_gpu.toml` - MOEA settings
   - `DataIngestion/.env` - Environment variables for all services

3. **Documentation**:
   - `DataIngestion/docs/architecture/PIPELINE_FLOW.md` - Current architecture
   - `DataIngestion/docs/MULTI_LEVEL_FEATURE_EXTRACTION.md` - Feature strategy
   - `Doc-templates/*.md` - Comprehensive project documentation

4. **Key Implementation Files**:
   - `DataIngestion/gpu_feature_extraction/minimal_gpu_features.py` - GPU feature extraction
   - `DataIngestion/gpu_feature_extraction/src/hybrid_pipeline.rs` - Hybrid mode implementation
   - `DataIngestion/model_builder/src/utils/multi_level_data_loader.py` - Multi-level data handling

## Critical Development Guidelines

### File Management
1. **NEVER CREATE**: New Python scripts to run the pipeline (e.g., `run_pipeline.py`)
2. **NEVER CREATE**: Shell/PowerShell scripts for orchestration
3. **ALWAYS EDIT**: Existing files instead of creating new ones
4. **ALWAYS USE**: Docker Compose for running any pipeline components
5. **Organization**: Keep similar files together in appropriate directories

### Documentation Standards
1. **Location**: All documentation goes in `DataIngestion/docs/` with appropriate subdirectories:
   - `docs/architecture/`: System design and technical decisions
   - `docs/operations/`: Operational guides and troubleshooting
   - `docs/deployment/`: Deployment configurations and guides
   - `docs/testing/`: Test strategies and results
2. **Format**: Use Markdown with clear headings and code examples
3. **Diagrams**: Include ASCII diagrams or Mermaid diagrams in documentation
4. **Updates**: Keep documentation in sync with code changes

### Docker Compose Best Practices
1. **Service Dependencies**: Use `depends_on` with health checks
2. **Environment Variables**: Define in `.env` files, not hardcoded
3. **Volume Mounts**: Use named volumes for data persistence
4. **GPU Access**: Use `deploy.resources.reservations.devices` for GPU
5. **Networking**: Use custom networks for service isolation

### Pipeline Execution Flow
1. **Standard Pipeline**: `docker compose up`
2. **Enhanced Pipeline**: `docker compose -f docker-compose.yml -f docker-compose.enhanced.yml up`
3. **Production**: `docker compose -f docker-compose.yml -f docker-compose.prod.yml up`
4. **Testing**: `docker compose -f docker-compose.yml -f docker-compose.test.yml up`

### Key Architecture Decisions
1. **Hybrid Mode**: Rust binary calls Python scripts via subprocess (not Docker-in-Docker)
2. **Sparse Handling**: Integrated pipeline stages 2-4 for efficiency
3. **GPU Usage**: Python handles all GPU operations, Rust handles CPU operations
4. **Data Flow**: All data passes through TimescaleDB between stages

## Common Issues & Troubleshooting

### Pipeline Not Running?
1. **Check service health**: `docker compose ps`
2. **View logs**: `docker compose logs -f enhanced_sparse_pipeline`
3. **Database connection**: Ensure `db` service is healthy before starting pipeline
4. **GPU access**: Verify with `nvidia-smi` and check Docker GPU runtime

### Data Processing Issues
1. **Sparse data**: Pipeline handles 91.3% missing data - this is expected
2. **Memory errors**: Reduce batch size in environment variables
3. **Slow processing**: Check if GPU is being utilized (`nvidia-smi`)
4. **Missing features**: Verify all pipeline stages completed successfully

### Docker Compose Issues
1. **Build failures**: Clean rebuild with `docker compose build --no-cache`
2. **Permission errors**: Check volume mount permissions
3. **Network issues**: Ensure all services are on the same network
4. **GPU not found**: Install NVIDIA Container Toolkit

## Environment Variables Reference

```bash
# Database
DATABASE_URL=postgresql://postgres:postgres@db:5432/postgres

# Pipeline Configuration
START_DATE=2013-12-01
END_DATE=2016-09-08
BATCH_SIZE=1000
MIN_ERA_ROWS=100
FEATURES_TABLE=enhanced_sparse_features

# Feature Flags
ENABLE_WEATHER_FEATURES=true
ENABLE_ENERGY_FEATURES=true
ENABLE_GROWTH_FEATURES=true

# GPU Settings
USE_GPU=true
CUDA_VISIBLE_DEVICES=0
```

## Important Reminders

- **NEVER** create Python scripts to run the pipeline - ALWAYS use Docker Compose
- **NEVER** create shell/PowerShell scripts for orchestration - ALWAYS use Docker Compose
- **ALWAYS** put documentation in `DataIngestion/docs/` with appropriate subdirectories
- **ALWAYS** prefer editing existing files over creating new ones
- **ALWAYS** use the hybrid pipeline mode (`--hybrid-mode`) for sparse data processing