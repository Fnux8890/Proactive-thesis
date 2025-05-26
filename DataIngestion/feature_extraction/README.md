# Feature Extraction Pipeline

This directory contains the complete feature extraction pipeline for greenhouse sensor data processing.

## Overview

The feature extraction pipeline consists of four main stages:
1. **Pre-processing** - Data cleaning, normalization, and enrichment
2. **Era Detection** - Identifying operational periods using changepoint algorithms
3. **Feature Extraction** - Computing time-series features using tsfresh
4. **Model Building** - Training predictive models on extracted features

## Directory Structure

```
feature_extraction/
├── benchmarks/            # Performance benchmarking tools
│   ├── src/              # Benchmark source code
│   ├── docker/           # Docker configurations
│   └── results/          # Benchmark results
├── config/               # Configuration files
├── db/                   # Database utilities
│   ├── __init__.py
│   ├── connection.py     # Connection pooling
│   ├── chunked_query.py  # Chunked data retrieval
│   └── metrics.py        # Performance metrics
├── docs/                 # Comprehensive documentation
│   ├── operations/       # Operational guides
│   ├── database/         # Database documentation
│   ├── migrations/       # Migration guides
│   └── testing/          # Test documentation
├── examples/             # Usage examples
├── features/             # Feature adapters and utilities
├── pre_process/          # Pre-processing stage
├── era_detection/        # Python era detection
├── era_detection_rust/   # Rust era detection
├── feature/              # Feature extraction stage
├── feature-gpu/          # GPU-accelerated extraction
├── tests/                # Test suite
├── docker-compose.test.yaml
├── Makefile
└── README.md
```

## Quick Start

### Using Make Commands

```bash
# Run all tests
make test

# Run specific test suites
make test-db          # Database tests
make test-processing  # Processing tests
make test-features    # Feature tests

# Run benchmarks
make benchmark

# Clean up
make clean
```

### Using Docker Compose

```bash
# Run the complete pipeline
docker-compose up

# Run tests
docker-compose -f docker-compose.test.yaml up

# Run specific stage
docker-compose up pre_process
docker-compose up era_detector
docker-compose up feature_extraction
```

## Pipeline Stages

### 1. Pre-processing (`pre_process/`)
- Loads raw sensor data from multiple sources
- Handles missing values and outliers
- Normalizes timestamps and resamples data
- Enriches with external weather and energy price data
- Outputs to `preprocessed_features` table

### 2. Era Detection (`era_detection_rust/`)
- Detects changepoints using PELT, BOCPD, and HMM algorithms
- Identifies operational periods (eras) in time series
- Supports both JSONB and hybrid table formats
- Outputs to `era_labels` tables

### 3. Feature Extraction (`feature/` and `feature-gpu/`)
- Computes comprehensive time-series features
- Uses tsfresh for feature calculation
- GPU acceleration available for large datasets
- Outputs to `feature_data` table

### 4. Model Building (`../model_builder/`)
- Trains LSTM models for prediction
- Implements surrogate models for optimization
- Uses MLflow for experiment tracking

## Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@host:5432/db
DB_HOST=localhost
DB_PORT=5432
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=postgres

# Logging
RUST_LOG=info
PYTHONUNBUFFERED=1

# GPU (optional)
CUDA_VISIBLE_DEVICES=0
```

### Data Processing Configuration

See `data_processing_config.json` for feature extraction settings.

## Documentation

- **[Operations Guide](docs/operations/)** - How to operate the pipeline
- **[Database Guide](docs/database/)** - Database optimization and utilities
- **[Migration Guide](docs/migrations/)** - System migration procedures
- **[Testing Guide](docs/operations/TESTING_GUIDE.md)** - Testing best practices
- **[Benchmarks](benchmarks/)** - Performance benchmarking

## Development

### Running Tests

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_connection.py -v
```

### Code Quality

```bash
# Format code
black .
isort .

# Lint code
flake8 .
mypy .
```

## Performance

### Benchmarking

See [benchmarks/](benchmarks/) for performance testing tools.

### Optimization

- Connection pooling for database efficiency
- Chunked queries for large datasets
- GPU acceleration for feature extraction
- Parallel processing for era detection

## Troubleshooting

### Common Issues

1. **Database Connection Errors**
   - Check DATABASE_URL environment variable
   - Verify TimescaleDB is running
   - Check network connectivity

2. **Memory Issues**
   - Reduce batch size in configuration
   - Enable chunked processing
   - Use GPU acceleration if available

3. **Performance Issues**
   - Run benchmarks to identify bottlenecks
   - Check database indexes
   - Enable parallel processing

### Logs

- Python logs: Standard output
- Rust logs: Set RUST_LOG=debug for verbose output
- Database logs: Check TimescaleDB container logs

## Contributing

1. Follow the existing code structure
2. Add tests for new features
3. Update documentation
4. Run tests before committing
5. Use meaningful commit messages

## License

See main repository LICENSE file.