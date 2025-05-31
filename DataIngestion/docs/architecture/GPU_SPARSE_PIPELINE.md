# GPU Sparse Pipeline Architecture

## Overview

The GPU Sparse Pipeline is a complete reimplementation of the data processing pipeline designed specifically to handle the reality of 91.3% sparse greenhouse sensor data. Instead of fighting against data sparsity, this approach embraces it and uses GPU acceleration to extract meaningful features from the available data.

## Key Innovation: GPU-First Processing

Traditional pipeline flow:
```
Ingestion → Preprocessing → Era Detection → Feature Extraction → Model Building
```

Sparse pipeline flow:
```
Ingestion → GPU Sparse Pipeline (All-in-One) → Model Building
```

## Implementation Details

### Stage 1: Hourly Aggregation

The pipeline starts by aggregating minute-level data to hourly intervals:

```rust
// In sparse_pipeline.rs
pub async fn stage1_aggregate_hourly() -> Result<DataFrame> {
    // SQL aggregation with coverage tracking
    SELECT 
        DATE_TRUNC('hour', time) as hour,
        AVG(air_temp_c) as air_temp_c_mean,
        COUNT(air_temp_c) as air_temp_c_count,  // Track coverage
        ...
}
```

Key features:
- Calculates coverage metrics for each sensor
- Filters hours with >30% data coverage
- Preserves statistical information (mean, std, min, max)

### Stage 2: Conservative Gap Filling

Physics-constrained interpolation for small gaps:

```rust
fn fill_column_with_stats() -> Result<(DataFrame, usize)> {
    // Only fill gaps ≤2 hours
    // Apply physical constraints:
    //   - Temperature: 10-40°C, max 2°C/hour change
    //   - CO2: 300-1500 ppm, max 100 ppm/hour change
    //   - Humidity: 30-95%, max 10%/hour change
}
```

### Stage 3: GPU Feature Extraction

Sliding window feature extraction using CUDA kernels:

```rust
pub async fn stage3_gpu_features(&self, filled_df: DataFrame) -> Result<Vec<FeatureSet>> {
    // Create 24-hour sliding windows with 6-hour overlap
    let windows = self.create_sliding_windows(filled_df)?;
    
    // GPU acceleration for each window
    for (window_start, window_df) in windows {
        let features = self.gpu_extractor.extract_batch(&[era_data])?;
    }
}
```

GPU kernels compute:
- Statistical features (mean, std, min, max, percentiles)
- Temporal features (trends, autocorrelation)
- Domain-specific features (photoperiod, day/night differences)

### Stage 4: Feature-Based Era Creation

Instead of detecting changepoints in sparse data, we create monthly eras from feature similarity:

```rust
pub async fn stage4_create_eras(&self, features: Vec<FeatureSet>) -> Result<Vec<Era>> {
    // Group features by month
    // Compute era statistics from feature aggregates
    // Store as sparse_monthly_eras
}
```

## Configuration

The pipeline is highly configurable via `SparsePipelineConfig`:

```rust
pub struct SparsePipelineConfig {
    pub min_hourly_coverage: f32,      // Default: 0.3 (30%)
    pub max_interpolation_gap: i64,    // Default: 2 hours
    pub enable_parquet_checkpoints: bool,  // Default: true
    pub checkpoint_dir: PathBuf,       // Default: /tmp/gpu_sparse_pipeline
    pub window_hours: usize,           // Default: 24
    pub slide_hours: usize,            // Default: 6
}
```

## Intermediate Outputs

The pipeline saves checkpoints at each stage for debugging and recovery:

```
/tmp/gpu_sparse_pipeline/
├── stage1_hourly_20240531_143022.parquet      # Hourly aggregated data
├── stage2_filled_20240531_143145.parquet      # Gap-filled data
├── stage3_features_20240531_144512.json       # Extracted features
└── stage4_eras_20240531_144623.json           # Monthly eras
```

## Database Schema

### Input
- `sensor_data_merged` - Raw minute-level sensor data (91.3% sparse)

### Outputs
- `sparse_window_features` - Features extracted from sliding windows
- `sparse_monthly_eras` - Monthly operational periods with statistics

## Performance Characteristics

Expected performance on RTX 4070:
- Hourly aggregation: ~1M rows/second
- Gap filling: ~100K rows/second  
- GPU feature extraction: ~50 windows/second
- Total pipeline time for 1 year: ~5 minutes

## Usage

### Command Line
```bash
./gpu_feature_extraction \
    --sparse-mode \
    --start-date 2014-06-01 \
    --end-date 2014-07-31 \
    --batch-size 48
```

### Docker Compose
```bash
docker compose -f docker-compose.sparse.yml up gpu-sparse-pipeline
```

### Programmatic
```rust
let config = SparsePipelineConfig::default();
let pipeline = SparsePipeline::new(pool, config)
    .with_gpu_extractor(gpu_extractor);
    
let results = pipeline.run_pipeline(start_time, end_time).await?;
```

## Advantages Over Traditional Pipeline

1. **Handles Sparse Data**: Works with 8.7% data coverage instead of requiring dense time series
2. **GPU Acceleration**: 10-100x faster feature extraction
3. **Simpler Architecture**: Single container replaces 4 processing stages  
4. **Better Features**: Sliding windows capture more temporal patterns
5. **Recoverable**: Checkpoints allow resuming from any stage

## Future Enhancements

1. **Adaptive Windows**: Adjust window size based on data density
2. **Multi-GPU Support**: Process multiple windows in parallel
3. **Streaming Mode**: Process data as it arrives
4. **Advanced Imputation**: Use neural networks for gap filling
5. **Online Learning**: Update features incrementally

## Validation

The pipeline includes built-in validation:
- Coverage metrics at each stage
- Physics constraint checking
- Feature quality scores
- Era consistency validation

See `validate_sparse_pipeline.py` for comprehensive tests.