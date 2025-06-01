# Hybrid GPU Feature Extraction Pipeline

This hybrid pipeline combines the strengths of Rust for efficient data handling and Python for GPU-accelerated feature extraction.

## Architecture

The hybrid pipeline consists of:

1. **Rust Component** (`src/hybrid_pipeline.rs`):
   - Handles sparse data loading and preprocessing
   - Manages sliding windows and data batching
   - Performs era detection
   - Orchestrates the overall pipeline flow

2. **Python GPU Component** (`minimal_gpu_features.py`):
   - Receives data via JSON over stdin/stdout
   - Uses CuDF/CuPy for GPU acceleration
   - Extracts statistical features
   - Returns results as JSON

## Quick Start

### 1. Build the Docker Images

```bash
# Build Python GPU service
docker build -f Dockerfile.python-gpu -t gpu-feature-python:latest .

# Build Rust pipeline
docker build -f Dockerfile -t gpu-feature-rust:latest .
```

### 2. Run the Hybrid Pipeline

Using Docker Compose:
```bash
docker-compose -f docker-compose.hybrid.yml up
```

Using cargo directly:
```bash
cargo run -- --hybrid-mode --start-date 2014-01-01 --end-date 2014-12-31
```

### 3. Test the Python GPU Service

```bash
# Test directly
python test_python_gpu.py

# Or with Docker
docker run --rm -it --gpus all gpu-feature-python:latest python /app/test_python_gpu.py
```

## Configuration

### Environment Variables

- `DATABASE_URL`: PostgreSQL connection string
- `DISABLE_GPU`: Set to "true" to use CPU fallback
- `RUST_LOG`: Log level (info, debug, trace)

### Command Line Options

```bash
cargo run -- --hybrid-mode \
  --start-date 2014-01-01 \
  --end-date 2014-12-31 \
  --batch-size 24 \
  --database-url postgresql://user:pass@host:5432/db
```

## Features Extracted

The Python GPU service extracts:

1. **Basic Statistics**:
   - Mean, std, min, max
   - Percentiles (25th, 50th, 75th)

2. **Rolling Window Features**:
   - 30-minute rolling statistics
   - 2-hour rolling statistics

3. **Change Features**:
   - Mean change (diff)
   - Standard deviation of changes

## Performance

The hybrid approach provides:

- **Efficient sparse data handling** via Rust
- **GPU-accelerated feature computation** via Python/CuDF
- **Scalable sliding window processing**
- **Fault tolerance** with graceful fallback to CPU

## Troubleshooting

### GPU Not Detected

If the Python service doesn't detect GPU:

1. Check NVIDIA drivers: `nvidia-smi`
2. Verify Docker GPU support: `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`
3. Check CUDA installation in container

### Memory Issues

For large datasets:

1. Reduce `batch_size` parameter
2. Increase `slide_hours` for less overlap
3. Monitor GPU memory with `nvidia-smi`

### Python Service Errors

Enable debug logging:
```bash
export RUST_LOG=debug
python minimal_gpu_features.py < test_data.json
```

## Development

### Adding New Features

1. Update `minimal_gpu_features.py` to add new feature calculations
2. Modify `src/python_bridge.rs` if the interface changes
3. Update tests in `test_python_gpu.py`

### Testing

Run unit tests:
```bash
# Rust tests
cargo test

# Python tests
python -m pytest tests/
```

## Advantages of Hybrid Approach

1. **Language Strengths**: Uses Rust for systems programming and Python for data science libraries
2. **Easier GPU Integration**: Leverages mature Python GPU ecosystem (RAPIDS)
3. **Maintainability**: Simpler than Rust-CUDA integration
4. **Flexibility**: Easy to swap feature extraction implementations
5. **Compatibility**: Works with existing Python data science tools

## Future Improvements

- [ ] Add streaming support for real-time processing
- [ ] Implement feature caching
- [ ] Add more sophisticated feature engineering
- [ ] Support for distributed processing
- [ ] Integration with MLflow for feature versioning