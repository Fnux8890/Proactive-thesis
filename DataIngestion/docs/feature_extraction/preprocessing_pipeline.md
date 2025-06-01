# Preprocessing Pipeline

A robust, type-safe preprocessing pipeline for greenhouse sensor data with comprehensive testing and monitoring.

## 🏗️ Architecture

```
pre_process/
├── core/               # Core processing logic
│   ├── config_handler.py    # Pydantic config validation
│   ├── island_detector.py   # Gap detection & segmentation
│   ├── models.py           # Data models & types
│   └── processing_steps.py # Outlier & imputation handlers
├── external/           # External data fetchers
│   ├── fetch_energy.py     # Energy price data
│   ├── fetch_external_weather.py  # Weather data
│   └── phenotype_ingest.py # Literature phenotypes
├── utils/              # Utilities & helpers
│   ├── data_enrichment_utils.py   # Feature engineering
│   ├── data_preparation_utils.py  # Data prep functions
│   ├── database_operations.py     # DB operations
│   ├── database_operations_hybrid.py # Hybrid storage
│   └── db_utils.py         # Base DB connector
├── tests/              # Comprehensive test suite
├── scripts/            # Standalone scripts
├── output/             # Processing outputs
└── preprocess.py       # Main entry point
```

## 🚀 Quick Start

### Development Setup

```bash
# Install dependencies
make install-dev

# Run tests
make test

# Run with watch mode
make test-watch

# Format & lint
make format lint

# Type check
make typecheck
```

### Running the Pipeline

```bash
# Using uv (recommended)
uv run python preprocess.py

# Or with Docker
docker build -f preprocess.dockerfile -t preprocessing .
docker run -v $(pwd)/data:/app/data preprocessing
```

## 🔧 Configuration

### Environment Variables

```bash
PREPROCESSING_DB_HOST=localhost
PREPROCESSING_DB_PORT=5432
PREPROCESSING_DB_NAME=greenhouse_db
PREPROCESSING_DB_USER=postgres
PREPROCESSING_DB_PASSWORD=postgres
PREPROCESSING_BATCH_SIZE=10000
PREPROCESSING_STORAGE_TYPE=timescale_columnar
```

### Configuration File

See `preprocess_config.json` for detailed configuration options.

## 🧪 Testing

### Unit Tests

```bash
# Run all tests
make test

# Run specific test file
uv run pytest tests/test_island_detector.py -v

# Run with coverage
uv run pytest --cov=. --cov-report=html
```

### Property-Based Testing

We use Hypothesis for property-based testing:

```python
@given(data_frames(...))
def test_island_detection_properties(df):
    # Test invariants hold for random data
```

## 📊 Key Features

### Island Detection

Automatically segments time series based on temporal gaps:

```python
from core import label_islands

# Detect segments with >15min gaps
island_ids = label_islands(time_series, gap="15min")
```

### Type Safety

Comprehensive type hints with runtime validation:

```python
from core import PreprocessingConfig

# Pydantic validates at runtime
config = PreprocessingConfig(**json_config)
```

### Metrics Collection

Detailed processing metrics:

```python
handler = OutlierHandler(rules)
result = handler.clip_outliers(df)
print(f"Clipped {handler.metrics.outliers_clipped} outliers")
```

## 🔄 Processing Steps

1. **Gap Detection**: Identify continuous segments
2. **Outlier Handling**: Clip or flag outliers
3. **Imputation**: Fill missing values
4. **Resampling**: Regularize time intervals
5. **Feature Engineering**: Generate derived features
6. **Storage**: Save to TimescaleDB/Parquet

## 🛠️ Development

### Pre-commit Hooks

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Adding New Features

1. Add data model to `core/models.py`
2. Implement logic in appropriate module
3. Add comprehensive tests
4. Update type hints
5. Run `make ci` to verify

## 📈 Performance

- Batch processing: 10k rows default
- Parallel external data fetching
- Efficient TimescaleDB operations
- Memory-conscious pandas operations

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're in the correct directory
2. **DB Connection**: Check environment variables
3. **Memory Issues**: Reduce batch size
4. **Type Errors**: Run `make typecheck`

### Debug Mode

```bash
export DEBUG=1
uv run python preprocess.py
```

## 📚 References

- [TimescaleDB Docs](https://docs.timescale.com/)
- [Pydantic Docs](https://docs.pydantic.dev/)
- [tsfresh Docs](https://tsfresh.readthedocs.io/)