# Enhanced Sparse Pipeline with External Data Integration

This enhanced version of the sparse pipeline integrates external weather data, energy prices, and plant phenotype information to create comprehensive features for MOEA optimization.

## New Features

### 1. External Data Integration
- **Weather Data**: Fetches from Open-Meteo API or local database
- **Energy Prices**: Danish spot prices (DK1/DK2) for cost optimization
- **Plant Phenotype**: Species-specific parameters from literature

### 2. Enhanced GPU Kernels
- **Extended Statistics**: Percentiles (25th, 50th, 75th, 90th), skewness, kurtosis
- **Growth Features**: Growing Degree Days (GDD), Daily Light Integral (DLI)
- **Energy Features**: Cost-weighted consumption, peak/off-peak ratios
- **Weather Coupling**: Temperature differentials, solar efficiency

### 3. Multi-Resolution Feature Extraction
Computes features at multiple time scales:
- 5 minutes
- 15 minutes
- 1 hour
- 4 hours
- 1 day

## Usage

### Basic Example

```bash
# Run with default settings (GPU + external data)
cargo run --bin test_enhanced_sparse -- \
    --start-date 2014-01-01 \
    --end-date 2014-01-31 \
    --database-url postgresql://user:pass@localhost/greenhouse

# Run without external data
cargo run --bin test_enhanced_sparse -- \
    --start-date 2014-01-01 \
    --end-date 2014-01-31 \
    --database-url postgresql://user:pass@localhost/greenhouse \
    --enable-external-data false

# Custom location and species
cargo run --bin test_enhanced_sparse -- \
    --start-date 2014-01-01 \
    --end-date 2014-01-31 \
    --database-url postgresql://user:pass@localhost/greenhouse \
    --lat 55.676 \
    --lon 12.568 \
    --price-area DK2 \
    --species "Rosa hybrida"
```

### Programmatic Usage

```rust
use gpu_feature_extraction::sparse_pipeline::{SparsePipeline, SparsePipelineConfig};
use gpu_feature_extraction::features::GpuFeatureExtractor;
use chrono::Utc;

// Configure pipeline
let config = SparsePipelineConfig {
    min_hourly_coverage: 0.1,
    max_interpolation_gap: 2,
    enable_parquet_checkpoints: true,
    checkpoint_dir: PathBuf::from("/tmp/sparse_pipeline"),
    window_hours: 24,
    slide_hours: 6,
    enable_external_data: true,
    greenhouse_lat: 56.16,  // Queens, Denmark
    greenhouse_lon: 10.20,
    price_area: "DK1".to_string(),
    phenotype_species: "Kalanchoe blossfeldiana".to_string(),
};

// Create pipeline with GPU support
let mut pipeline = SparsePipeline::new(pool, config);
if let Ok(gpu_extractor) = create_gpu_extractor() {
    pipeline = pipeline.with_gpu_extractor(gpu_extractor);
}

// Run pipeline
let results = pipeline.run_pipeline(start_time, end_time).await?;
```

## Feature Categories

### Growth Features
- `gdd_phenotype_specific`: Growing degree days with species-specific base temperature
- `dli_mol_m2_d`: Daily light integral in mol/m²/day
- `photoperiod_hours`: Hours of supplemental lighting

### Weather Coupling
- `temp_differential_mean`: Average temperature difference (inside - outside)
- `temp_differential_std`: Variability in temperature differential
- `solar_efficiency_ratio`: Light transmission efficiency

### Energy Features
- `total_energy_cost`: Total energy cost for the period
- `peak_hours_ratio`: Proportion of energy used during peak hours
- `peak_off_peak_ratio`: Ratio of peak to off-peak energy prices

### Multi-Resolution Statistics
For each sensor and time resolution:
- `{sensor}_{resolution}_p25`: 25th percentile
- `{sensor}_{resolution}_median`: Median value
- `{sensor}_{resolution}_p75`: 75th percentile
- `{sensor}_{resolution}_p90`: 90th percentile

## GPU Kernel Details

### Extended Statistics Kernel
Computes percentiles using a sorted approach with Thrust library:
```cuda
__global__ void compute_percentiles_kernel(
    const float* sorted_data,
    float* percentiles,
    const float* percentile_values,
    int n,
    int num_percentiles
)
```

### Growth Energy Features Kernel
Combines multiple calculations in a single kernel:
```cuda
__global__ void compute_growth_energy_features(
    const float* temperature,
    const float* light_intensity,
    const int* lamp_status,
    const float* outside_temp,
    const float* solar_radiation,
    const float* energy_price,
    const float* power_consumption,
    // ... parameters and outputs
)
```

## Performance Optimization

1. **Batch Processing**: Processes multiple windows in parallel on GPU
2. **Shared Memory**: Uses shared memory for reductions
3. **Coalesced Access**: Ensures memory access patterns are optimized
4. **Multi-Stream**: Can use multiple CUDA streams for concurrent operations

## Dependencies

External data fetching requires:
- Network access for Open-Meteo API
- Pre-populated tables for energy prices:
  - `external_weather_aarhus`
  - `external_energy_prices_dk`

## Output

The pipeline produces:
1. **Hourly aggregated data** with gap filling
2. **Window-level features** (700+ features per window)
3. **Monthly eras** with summary statistics

Results are saved to:
- Database tables: `sparse_window_features`, `sparse_monthly_eras`
- Parquet checkpoints: `/tmp/gpu_sparse_pipeline_enhanced/`
- JSON summary: `/tmp/gpu_sparse_pipeline_enhanced/summary.json`

## Troubleshooting

### Missing External Data
If external data is not available, the pipeline will:
- Log warnings but continue processing
- Skip features that require external data
- Use only internal sensor data

### GPU Memory Issues
For large datasets:
- Reduce `window_hours` to decrease memory usage
- Process in smaller date ranges
- Disable some feature categories

### Performance Tuning
- Adjust `slide_hours` for different overlap ratios
- Modify `min_hourly_coverage` for sparser data
- Enable/disable checkpointing based on needs