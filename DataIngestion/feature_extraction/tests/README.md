# Feature Extraction Tests

This directory contains comprehensive tests for all feature extraction components.

## Quick Start

```bash
# Run all tests
../run_tests.sh all

# Run specific test suite
../run_tests.sh backend
../run_tests.sh thread-safety
../run_tests.sh observability

# Check test results
../test_summary.py
```

## Test Organization

### Unit Tests
- `test_dtypes.py` - Backend dtype helper functions
- `test_adapters_type_safety.py` - Type safety in adapters
- `test_sentinel_performance.py` - Performance optimization tests
- `test_parquet_parity.py` - Parquet file consistency

### Integration Tests  
- `test_connection_thread_safety.py` - Thread-safe connection pooling
- `test_connection_metrics.py` - Observability and metrics
- `test_connection_soak.py` - Long-running stability tests
- `test_sql_date_filters.py` - SQL safety and parameterization

### Component Tests
- `test_processing_steps.py` - Data processing pipeline steps

## Docker-Based Testing

All tests can be run in isolated Docker containers using `docker-compose.test.yaml`:

```yaml
# Run backend tests only
docker compose -f docker-compose.test.yaml up test-backend-adapter

# Run all tests with database
docker compose -f docker-compose.test.yaml up test-all

# Clean up test resources
docker compose -f docker-compose.test.yaml down -v
```

## Test Services

| Service | Description | Dependencies |
|---------|-------------|--------------|
| `test-backend-adapter` | Tests pandas/cuDF compatibility | None |
| `test-thread-safety` | Tests connection pool thread safety | PostgreSQL |
| `test-observability` | Tests metrics collection | PostgreSQL |
| `test-performance` | Tests performance optimizations | None |
| `test-sql-safety` | Tests SQL injection prevention | PostgreSQL |
| `test-integration` | Full integration test suite | PostgreSQL |
| `test-soak` | Long-running stability tests | PostgreSQL |
| `test-components-isolated` | Tests without any dependencies | None |
| `test-gpu-backend` | GPU backend tests (optional) | NVIDIA runtime |

## Writing New Tests

### 1. Backend Compatibility Test
```python
def test_backend_function():
    from backend import pd, DataFrame
    # Test both pandas and cuDF code paths
```

### 2. Thread Safety Test
```python
def test_concurrent_operations():
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Test concurrent access
```

### 3. Metrics Test
```python
def test_metrics_collection():
    from db.metrics import get_metrics_collector
    collector = get_metrics_collector()
    # Verify metrics are collected correctly
```

## Test Patterns

### Mocking Database Connections
```python
@patch('psycopg2.connect')
def test_without_real_db(mock_connect):
    mock_conn = Mock()
    mock_connect.return_value = mock_conn
    # Test database operations
```

### Testing with Timeouts
```python
def test_with_timeout():
    with pytest.raises(TimeoutError):
        get_connection(timeout=0.1)
```

### Verifying Thread Safety
```python
def test_no_race_conditions():
    results = []
    barrier = threading.Barrier(num_threads)
    # Synchronize thread starts
    # Verify no data corruption
```

## CI/CD Integration

Add to your CI pipeline:

```yaml
# GitHub Actions
- name: Run Feature Extraction Tests
  run: |
    cd DataIngestion/feature_extraction
    ./run_tests.sh all
    ./test_summary.py

# GitLab CI
test:feature-extraction:
  script:
    - cd DataIngestion/feature_extraction
    - ./run_tests.sh all
    - ./test_summary.py
```

## Debugging Failed Tests

```bash
# View logs for failed test
./run_tests.sh logs test-thread-safety

# Run test in foreground with verbose output
./run_tests.sh thread-safety -v

# Run with debugger
docker compose -f docker-compose.test.yaml run --rm test-backend-adapter python -m pdb -m pytest tests/test_dtypes.py
```

## Performance Benchmarks

Some tests include performance benchmarks:

- **Sentinel replacement**: Should show 4x+ speedup
- **Connection acquisition**: Should complete in <100ms  
- **Metrics collection**: Should handle 1000+ ops/sec

## Known Issues

1. **GPU tests**: Require NVIDIA Docker runtime
2. **Database tests**: Need PostgreSQL running
3. **Import errors**: Use Docker environment for dependencies

## Maintenance

- Update test dependencies in `test.dockerfile`
- Add new test services to `docker-compose.test.yaml`
- Include new tests in `test_summary.py` analysis