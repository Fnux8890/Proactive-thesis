# Sparse Pipeline Architecture

## Overview

The sparse pipeline is a GPU-accelerated, integrated approach designed specifically for greenhouse data with >90% missing values and many temporal islands. Unlike traditional pipelines that fail with sparse data, this pipeline handles everything in a single container with intelligent windowing and conservative gap filling.

## Why Sparse Pipeline?

### Traditional Pipeline Problems:
- **Preprocessing fails**: Can't regularize time series with 91.3% missing data
- **Era detection meaningless**: PELT/BOCPD/HMM can't find changepoints in disconnected islands
- **Feature extraction breaks**: TSFresh expects continuous data
- **Multiple containers inefficient**: Data transfer overhead between stages

### Sparse Pipeline Solution:
- **Single GPU container**: All processing in one place
- **Adaptive windowing**: Adjusts to data quality dynamically
- **Conservative gap filling**: Only fills small gaps (max 2 hours)
- **Island-aware**: Works with disconnected data periods

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Sparse GPU Pipeline                       │
│                                                              │
│  Stage 1: Hourly Aggregation                                │
│  └─> Groups sparse data into hourly buckets                │
│  └─> Calculates coverage metrics per hour                  │
│  └─> Only keeps hours with >10% coverage                   │
│                                                              │
│  Stage 2: Conservative Gap Filling                          │
│  └─> Forward fills gaps up to 2 hours only                 │
│  └─> Preserves data sparsity characteristics               │
│  └─> Tracks fill statistics                                 │
│                                                              │
│  Stage 3: GPU Feature Extraction                            │
│  └─> Adaptive window sizing (12-24 hours)                  │
│  └─> Quality-aware processing                              │
│  └─> GPU acceleration for feature computation              │
│                                                              │
│  Stage 4: Era Creation                                      │
│  └─> Monthly aggregation of features                       │
│  └─> Creates eras from feature windows                     │
│  └─> No changepoint detection needed                       │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Data Quality Analyzer
```rust
pub struct DataQualityAnalyzer {
    min_coverage_threshold: f64,  // 0.1 for sparse data
    min_quality_score: f64,       // 0.5 minimum
}
```

Calculates:
- **Coverage**: % of non-null values
- **Continuity**: Largest continuous segment
- **Consistency**: Value stability

### 2. Adaptive Window Configuration
```rust
pub struct AdaptiveWindowConfig {
    window_size: usize,      // 12-24 hours based on quality
    overlap_ratio: f64,      // 25-50% overlap
    min_quality_score: f64,  // 0.5 minimum
}
```

### 3. Sparse Pipeline Config
```rust
pub struct SparsePipelineConfig {
    min_hourly_coverage: f32,     // 0.1 (10%)
    max_interpolation_gap: i64,   // 2 hours
    enable_parquet_checkpoints: bool,
    checkpoint_dir: PathBuf,
    window_hours: usize,          // 24
    slide_hours: usize,           // 6
}
```

## Data Flow

1. **Input**: Raw sensor data with 91.3% missing values
2. **Stage 1**: Aggregate to hourly → ~309 viable hours/month
3. **Stage 2**: Fill small gaps → ~12 CO2 gaps, ~8 humidity gaps filled
4. **Stage 3**: Extract features → ~16 window features/month
5. **Stage 4**: Create eras → 1-2 monthly eras
6. **Output**: Features and eras saved to database

## Running the Pipeline

### Docker Compose
```bash
# Use the sparse compose file
docker compose -f docker-compose.sparse.yml up

# Or run stages individually
docker compose -f docker-compose.sparse.yml up rust_pipeline
docker compose -f docker-compose.sparse.yml up sparse_pipeline
```

### Direct Execution
```bash
docker run --rm \
  --network container:dataingestion-db-1 \
  -e DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres \
  -e RUST_LOG=gpu_feature_extraction=info \
  -e SPARSE_MODE=true \
  -e DISABLE_GPU=false \
  -v $(pwd)/gpu_feature_extraction/checkpoints:/tmp/gpu_sparse_pipeline:rw \
  dataingestion-gpu-sparse-pipeline:latest \
  --sparse-mode \
  --start-date 2014-01-01 \
  --end-date 2014-12-31 \
  --batch-size 24
```

## Performance

- Processes 1 month of sparse data in ~11 seconds
- Handles data with >90% missing values
- GPU acceleration provides 10-100x speedup for feature extraction
- Checkpoint system allows recovery from failures

## Advantages

1. **Handles Real Data**: Works with actual greenhouse data sparsity
2. **Single Container**: Reduces complexity and data transfer overhead
3. **GPU Accelerated**: Fast feature extraction even with large datasets
4. **Adaptive**: Adjusts processing based on data quality
5. **Robust**: Checkpointing and error recovery built-in