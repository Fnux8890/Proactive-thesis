# Parallel Feature Extraction Guide

## Overview

The feature extraction pipeline supports parallel processing to handle large-scale time-series data efficiently. While not a fully distributed cloud-native architecture, it provides significant performance improvements through batch processing and CPU/GPU parallelization.

## Architecture

### Parallelization Levels

1. **Era Batching**
   - Processes multiple eras simultaneously in batches
   - Configurable via `BATCH_SIZE` environment variable
   - Default: 100 eras per batch

2. **Feature Calculation Parallelism**
   - tsfresh internally parallelizes feature calculation
   - Configurable via `N_JOBS` environment variable
   - Default: -1 (use all available CPU cores)

3. **GPU Acceleration** (when available)
   - Uses RAPIDS AI for GPU-accelerated operations
   - Enabled via `USE_GPU=true`
   - Requires NVIDIA GPU with CUDA support

## Configuration

### Environment Variables

```bash
# Database Configuration
DB_USER=postgres
DB_PASSWORD=postgres
DB_HOST=db
DB_PORT=5432
DB_NAME=postgres

# Feature Extraction Configuration
FEATURES_TABLE=tsfresh_features
USE_GPU=true                    # Enable GPU acceleration
FEATURE_SET=efficient           # Options: minimal, efficient, comprehensive
BATCH_SIZE=200                  # Number of eras per batch
ERA_LEVEL=B                     # Era detection level (A, B, or C)
MIN_ERA_ROWS=100               # Minimum rows per era
N_JOBS=-1                      # CPU cores for parallel processing (-1 = all)
```

## Running Feature Extraction

### 1. Standalone Mode (Testing)

```bash
cd DataIngestion/feature_extraction

# Build and run with default settings
docker compose -f docker-compose.feature.yml up --build

# Run with custom parallel settings
export BATCH_SIZE=500
export N_JOBS=-1
docker compose -f docker-compose.feature.yml up --build
```

### 2. As Part of Main Pipeline

```bash
cd DataIngestion

# Run complete pipeline with feature extraction
docker compose up --build feature_extraction
```

### 3. GPU-Accelerated Mode

```bash
# Ensure GPU support is enabled
export USE_GPU=true

# Run with GPU acceleration
docker compose -f docker-compose.feature.yml up --build
```

## Performance Optimization

### Batch Size Selection

- **Small datasets (<10,000 rows)**: BATCH_SIZE=50-100
- **Medium datasets (10,000-100,000 rows)**: BATCH_SIZE=100-200
- **Large datasets (>100,000 rows)**: BATCH_SIZE=200-500

### CPU Core Allocation

- **Development/Testing**: N_JOBS=2-4
- **Production (dedicated server)**: N_JOBS=-1 (all cores)
- **Shared environment**: N_JOBS=<number_of_cores/2>

### Memory Considerations

- Each batch requires ~2-5GB RAM depending on feature set
- Monitor memory usage: `docker stats feature_extraction`
- Reduce BATCH_SIZE if experiencing OOM errors

## Scaling Strategies

### 1. Vertical Scaling (Single Machine)
- Increase CPU cores for better N_JOBS parallelism
- Add GPU for accelerated processing
- Increase RAM for larger batch sizes

### 2. Horizontal Scaling (Future Enhancement)
While not currently implemented, the architecture supports:
- Multiple worker containers processing different era batches
- Redis/RabbitMQ for task distribution
- Kubernetes deployment for cloud scaling

### 3. Hybrid Approach
- Use GPU for preprocessing and data transformation
- Use CPU cores for tsfresh feature calculation
- Optimal for mixed workloads

## Monitoring & Debugging

### Check Processing Status
```bash
# View logs
docker compose -f docker-compose.feature.yml logs -f feature_extraction

# Monitor resource usage
docker stats feature_extraction
```

### Common Issues

1. **Out of Memory**
   - Reduce BATCH_SIZE
   - Reduce N_JOBS
   - Use FEATURE_SET=minimal

2. **Slow Processing**
   - Increase BATCH_SIZE
   - Ensure N_JOBS=-1
   - Enable GPU with USE_GPU=true

3. **GPU Not Detected**
   - Verify NVIDIA drivers installed
   - Check Docker GPU support: `docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi`
   - Ensure nvidia-container-runtime installed

## Example: Production Configuration

```yaml
# docker-compose.override.yml
services:
  feature_extraction:
    environment:
      # Optimized for 32-core server with GPU
      BATCH_SIZE: 300
      N_JOBS: 30  # Leave 2 cores for system
      USE_GPU: true
      FEATURE_SET: efficient
      MIN_ERA_ROWS: 200  # Skip very small eras
    deploy:
      resources:
        limits:
          cpus: '30'
          memory: 64G
        reservations:
          cpus: '20'
          memory: 32G
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Performance Benchmarks

Based on testing with greenhouse sensor data:

| Configuration | Dataset Size | Processing Time | Throughput |
|--------------|--------------|-----------------|------------|
| Single-threaded | 100K rows | 45 min | 37 rows/sec |
| CPU Parallel (16 cores) | 100K rows | 8 min | 208 rows/sec |
| GPU + CPU Parallel | 100K rows | 3 min | 556 rows/sec |

## Future Enhancements

1. **Distributed Processing**
   - Implement task queue (Celery/RQ)
   - Multiple worker containers
   - Cloud-native deployment

2. **Streaming Processing**
   - Real-time feature extraction
   - Apache Kafka integration
   - Incremental feature updates

3. **Advanced GPU Utilization**
   - Custom CUDA kernels for specific features
   - Multi-GPU support
   - GPU memory optimization

## Conclusion

The current parallel feature extraction implementation provides significant performance improvements through:
- Batch processing of eras
- CPU parallelization via tsfresh
- Optional GPU acceleration
- Efficient memory management

While not a fully distributed system, it can efficiently process millions of sensor readings on a single high-performance machine, making it suitable for most greenhouse monitoring applications.