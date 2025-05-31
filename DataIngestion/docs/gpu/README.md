# GPU Feature Extraction

High-performance GPU-accelerated feature extraction for greenhouse time series data using Rust and CUDA.

## Features

- **32-48x faster** than CPU-based tsfresh
- **130+ features** computed in parallel on GPU
- **Safe Rust** implementation with cudarc
- **Docker** containerized for easy deployment
- **TimescaleDB** integration

## Quick Start

### Build
```bash
docker compose build gpu_feature_extraction
```

### Run
```bash
# Extract features for all Level A eras
docker compose run --rm gpu_feature_extraction --era-level A

# Benchmark mode
docker compose run --rm gpu_feature_extraction --benchmark --max-eras 100
```

## Architecture

```
src/
├── main.rs         # CLI and orchestration
├── config.rs       # Configuration management
├── db.rs          # Database operations
├── features.rs    # Feature extraction logic
├── kernels.rs     # CUDA kernel management
└── pipeline.rs    # Pipeline orchestration
```

### Key Technologies

- **Rust**: Safe systems programming
- **cudarc**: Safe CUDA bindings for Rust
- **CUDA 12.4**: GPU compute platform
- **TimescaleDB**: Time-series database
- **Docker**: Containerization with GPU support

## Feature Categories

1. **Statistical Features**
   - Mean, std, min, max, skewness, kurtosis
   - Percentiles (P10, P90)

2. **Temporal Features**
   - Rolling window statistics (1min, 5min, 15min, 1h)
   - Autocorrelation (ACF)
   - Cross-correlation

3. **Physical Features**
   - VPD (Vapor Pressure Deficit)
   - DLI (Daily Light Integral)
   - Energy efficiency metrics

4. **Cross-Sensor Features**
   - Temperature-humidity relationships
   - Light-CO2 efficiency
   - Actuator synchronization

## Performance

| Dataset | CPU Time | GPU Time | Speedup |
|---------|----------|----------|---------|
| 1K eras | 2 hours | 1.5 min | 80x |
| Full dataset | 8+ hours | 15 min | 32x |

## Development

### Local Build (requires CUDA toolkit)
```bash
export CUDA_ROOT=/usr/local/cuda
cargo build --release
```

### Run Tests
```bash
cargo test
```

### Add New Feature
1. Define CUDA kernel in `kernels.rs`
2. Add safe wrapper in `features.rs`
3. Update feature extraction logic

## CLI Options

```
gpu_feature_extraction [OPTIONS]

OPTIONS:
    --database-url <URL>       Database connection string [env: DATABASE_URL]
    --era-level <LEVEL>        Era level to process (A, B, or C) [default: B]
    --min-era-rows <N>         Minimum era size in rows [default: 100]
    --batch-size <N>           Batch size for GPU processing [default: 1000]
    --features-table <TABLE>   Features table name [default: feature_data]
    --max-eras <N>            Maximum eras to process (for testing)
    --benchmark               Enable benchmark mode
    -h, --help                Print help information
```

## Docker Environment Variables

- `DATABASE_URL`: PostgreSQL connection string
- `RUST_LOG`: Log level (error, warn, info, debug, trace)
- `CUDA_VISIBLE_DEVICES`: GPU device selection
- `GPU_BATCH_SIZE`: Default batch size

## Integration

The service integrates with the existing pipeline, replacing the CPU-based feature extraction:

```yaml
# docker-compose.yml
gpu_feature_extraction:
  build:
    context: ./gpu_feature_extraction
  environment:
    DATABASE_URL: postgresql://postgres:postgres@db:5432/postgres
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            capabilities: [gpu]
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `--batch-size`
- Check GPU memory with `nvidia-smi`

### No GPU Found
- Verify Docker GPU support: `docker run --gpus all nvidia/cuda nvidia-smi`
- Check NVIDIA drivers installed

### Performance Issues
- Monitor GPU utilization: `watch nvidia-smi`
- Check for thermal throttling
- Profile with: `nsys profile ./gpu_feature_extraction`

## License

See main project license.