# Sparse Pipeline Enhancements

## Overview

The sparse pipeline has been enhanced with intelligent data quality assessment and adaptive window sizing to automatically optimize processing based on data characteristics.

## Key Enhancements

### 1. Data Quality Scoring

The `DataQualityAnalyzer` module provides comprehensive metrics for each data window:

```rust
pub struct DataQualityMetrics {
    pub overall_score: f64,      // 0.0 to 1.0
    pub coverage: f64,           // Percentage of non-null values
    pub continuity: f64,         // How continuous the data is (few gaps)
    pub consistency: f64,        // How consistent values are (low noise)
    pub sensor_availability: f64, // How many sensors have data
}
```

**Quality Score Components:**
- **Coverage (40%)**: How much data is available vs missing
- **Continuity (30%)**: How few gaps exist in the time series
- **Consistency (20%)**: Inverse of noise/variability (coefficient of variation)
- **Sensor Availability (10%)**: How many key sensors have data

### 2. Adaptive Window Sizing

The pipeline now automatically adjusts window parameters based on data quality:

```rust
pub struct AdaptiveWindowConfig {
    pub window_size: usize,      // 12-48 hours based on coverage
    pub overlap_ratio: f64,      // 0.25-0.75 based on continuity
    pub quality_threshold: f64,  // Minimum quality to process window
    pub min_sensors: usize,      // Minimum sensors required
}
```

**Adaptation Rules:**
- High coverage (>80%) → 12-hour windows
- Medium coverage (50-80%) → 24-hour windows  
- Low coverage (<50%) → 48-hour windows
- Poor continuity → Higher overlap (up to 75%)

### 3. Quality-Based Window Filtering

Windows are now evaluated before processing:
- Quality score calculated for each candidate window
- Windows below quality threshold are skipped
- Prevents wasting GPU resources on poor data

## Implementation Details

### Stage 3 Enhancement

The GPU feature extraction stage now:

1. Analyzes overall data quality
2. Determines optimal window configuration
3. Creates windows with quality filtering
4. Logs quality metrics for monitoring

```rust
// In sparse_pipeline.rs
pub async fn stage3_gpu_features(&self, filled_df: DataFrame) -> Result<Vec<FeatureSet>> {
    // Analyze data quality
    let quality_metrics = self.quality_analyzer.analyze_window(&filled_df)?;
    
    // Get adaptive configuration
    let adaptive_config = AdaptiveWindowConfig::from_quality_metrics(&quality_metrics);
    
    // Create quality-filtered windows
    let windows = self.create_adaptive_windows(filled_df, &adaptive_config)?;
}
```

### Quality Analysis Algorithm

1. **Coverage Calculation**: Non-null values / total possible values
2. **Continuity Score**: 1.0 - (gap_count / theoretical_max_gaps)
3. **Consistency Score**: exp(-coefficient_of_variation)
4. **Overall Score**: Weighted average of all metrics

## Benefits

1. **Automatic Optimization**: No manual tuning required
2. **Resource Efficiency**: Skip processing poor quality windows
3. **Better Features**: Larger windows when data is sparse
4. **Quality Tracking**: Built-in monitoring of data quality

## Usage

The enhancements are automatically applied when using sparse mode:

```bash
# Docker
docker compose -f docker-compose.sparse.yml up gpu-sparse-pipeline

# Local testing
cargo run -- \
    --sparse-mode \
    --start-date "2014-06-01" \
    --end-date "2014-06-30"
```

## Example Output

```
Stage 3: GPU feature extraction with adaptive windows...
Overall data quality score: 0.67
  Coverage: 72.3%, Continuity: 65.4%, Consistency: 58.9%
Adaptive window config: size=24 hours, overlap=50%
Created 89 quality windows from 156 candidates
```

## Future Enhancements

1. **Machine Learning Quality Prediction**: Use ML to predict optimal windows
2. **Multi-Scale Windows**: Process at multiple time scales simultaneously
3. **Sensor Importance Weighting**: Weight sensors by their predictive importance
4. **Dynamic Thresholds**: Adjust quality thresholds based on downstream task performance

## Configuration

Quality parameters can be adjusted in `SparsePipelineConfig`:

```rust
pub struct SparsePipelineConfig {
    pub min_hourly_coverage: f32,      // Minimum coverage (default: 0.3)
    pub max_interpolation_gap: i64,    // Max gap to fill (default: 2 hours)
    pub window_hours: usize,           // Base window size (default: 24)
    pub slide_hours: usize,            // Base slide step (default: 6)
}
```

The system will adapt these based on actual data quality.