# Parallel Feature Extraction Architecture

## Overview

This module implements parallel feature extraction optimized for both local development and cloud deployment. It addresses the CPU-bound nature of tsfresh by using a hybrid approach.

## How GPU Acceleration Works with tsfresh

Since tsfresh is CPU-only, we use GPUs for:

### 1. **Data Preprocessing (GPU)**
- **Loading**: cuDF for fast CSV/Parquet reading
- **Transformation**: GPU-accelerated melting and reshaping
- **Imputation**: Fast forward/backward fill on GPU
- **Rolling Statistics**: GPU-computed features that supplement tsfresh

### 2. **Feature Extraction (Hybrid)**
- **GPU Features**: Rolling mean/std/min/max at multiple windows
- **CPU Features**: tsfresh statistical features (600+ features)
- **Parallel Processing**: Multiple workers process different eras

### 3. **Feature Selection (GPU)**
- **Correlation Matrix**: GPU-accelerated correlation computation
- **Variance Filtering**: Fast variance calculation on GPU
- **PCA/Dimensionality Reduction**: cuML GPU algorithms

## Architecture

```
┌─────────────────┐
│   Coordinator   │
│ (Task Distribution)
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼───┐
│  GPU   │ │ CPU  │
│Workers │ │Workers│
└───┬───┘ └──┬───┘
    │        │
    └────┬───┘
         │
    ┌────▼────┐
    │ Database│
    └─────────┘
```

## Local Development Setup

```bash
# Uses docker-compose.override.yml automatically
docker compose up

# This gives you:
# - Minimal feature set (faster)
# - CPU-only processing
# - Smaller batch sizes
# - Debug logging
# - Code hot-reloading
```

## Cloud Deployment Setup

```bash
# Use cloud configuration
docker compose -f docker-compose.yml -f docker-compose.cloud.yml up

# This gives you:
# - Comprehensive feature set
# - 4 GPU + 4 CPU workers
# - Optimized batch sizes
# - Production logging
# - Full monitoring stack
```

## Feature Sets

### Minimal (Local Development)
- Basic statistics: mean, median, std, min, max
- Simple aggregations: sum, count
- Fast to compute for quick iteration

### Efficient (Default)
- Minimal features plus:
- Autocorrelation, partial autocorrelation
- Basic spectral features
- Change detection

### Comprehensive (Cloud Production)
- All tsfresh features (600+)
- Complex entropy calculations
- Full spectral analysis
- Advanced time series features

## GPU vs CPU Distribution

**GPU Workers Handle:**
- Large eras (>500k rows)
- Data preprocessing
- Rolling window features
- Feature selection
- High-frequency sensors (temperature, humidity, light)

**CPU Workers Handle:**
- Small/medium eras
- tsfresh feature extraction
- Binary/categorical sensors
- Complex statistical features

## Performance Expectations

| Environment | Feature Set | Time per Era | Total Time |
|-------------|------------|--------------|------------|
| Local (CPU) | Minimal | 5-10 min | 8-12 hours |
| Local (CPU) | Efficient | 10-20 min | 15-20 hours |
| Cloud (GPU+CPU) | Comprehensive | 2-5 min | 2-4 hours |

## Configuration

### Environment Variables

```bash
# Feature extraction
FEATURE_SET=minimal|efficient|comprehensive
N_JOBS=4  # CPU parallelism
BATCH_SIZE=1000  # Rows per batch

# GPU settings
CUDA_VISIBLE_DEVICES=0,1,2,3
GPU_MEMORY_LIMIT=30GB

# Worker distribution
GPU_WORKERS=4
CPU_WORKERS=4
GPU_THRESHOLD=500000  # Rows
USE_SMART_DISTRIBUTION=true
```

### Smart Distribution

The coordinator analyzes each era to determine optimal processing:

1. **Data Volume**: Large eras → GPU
2. **Sensor Types**: Continuous sensors → GPU, Binary → CPU  
3. **Feature Complexity**: Simple stats → GPU, Entropy → CPU

## Monitoring

Access dashboards at:
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090

Key metrics:
- Worker utilization
- Era processing time
- Memory usage
- GPU utilization (cloud only)