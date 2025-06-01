# Enhanced Sparse Pipeline for Greenhouse Climate Optimization

## Overview

The Enhanced Sparse Pipeline is a comprehensive GPU-accelerated feature extraction system designed for greenhouse climate control optimization. It processes extremely sparse sensor data (91.3% missing values) and integrates external data sources to create rich feature sets for Multi-Objective Evolutionary Algorithm (MOEA) optimization.

## Key Features

### 🚀 **GPU Acceleration**
- Extended statistical features (percentiles, skewness, kurtosis) via CUDA kernels
- Weather coupling analysis on GPU
- Energy optimization calculations
- Plant growth metrics computation
- **Expected GPU utilization**: 85-95% (vs 65-75% in basic pipeline)

### 🌍 **External Data Integration**
- **Weather Data**: Open-Meteo API integration (17 variables)
- **Energy Prices**: Danish spot market data (DK1, DK2)
- **Plant Phenotypes**: Species-specific growth parameters

### 📊 **Multi-Resolution Analysis**
- **5 Resolution Levels**: 15min, 1h, 4h, 12h, 24h
- **Adaptive Windows**: Based on data quality
- **~1,200+ Features**: vs 350 in basic pipeline

### 🎯 **MOEA Optimization Ready**
Three primary objectives:
1. **Maximize Plant Growth** (GDD, DLI, temperature optimality)
2. **Minimize Energy Costs** (cost-weighted consumption, peak/off-peak optimization)
3. **Minimize Plant Stress** (stress degree days, environmental stability)

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Sensor Data   │    │   Weather API    │    │  Energy Prices  │
│  (91.3% sparse) │    │  (Open-Meteo)    │    │   (DK1, DK2)   │
└─────────┬───────┘    └─────────┬────────┘    └─────────┬───────┘
          │                      │                       │
          └──────────────────────┼───────────────────────┘
                                 │
┌─────────────────────────────────▼─────────────────────────────────┐
│                    Enhanced Sparse Pipeline                       │
│                                                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │
│  │ Stage 1:        │  │ Stage 2:        │  │ Stage 3:        │   │
│  │ Multi-Source    │→ │ Conservative    │→ │ GPU Multi-      │   │
│  │ Aggregation     │  │ Gap Filling     │  │ Resolution      │   │
│  │                 │  │                 │  │ Features        │   │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘   │
│                                                     │             │
│  ┌─────────────────┐  ┌─────────────────┐         │             │
│  │ Stage 4:        │← │ Phenotype       │←────────┘             │
│  │ Enhanced Era    │  │ Integration     │                       │
│  │ Creation        │  │                 │                       │
│  └─────────────────┘  └─────────────────┘                       │
└───────────────────────────┬───────────────────────────────────────┘
                            │
┌───────────────────────────▼───────────────────────────────────────┐
│                    MOEA Optimizer                                 │
│                                                                   │
│  Objectives:                                                      │
│  • Maximize Growth (GDD, DLI, optimality)                        │
│  • Minimize Cost (energy efficiency, peak usage)                 │
│  • Minimize Stress (temperature stability, stress days)          │
│                                                                   │
│  Output: Pareto-optimal control strategies                        │
└───────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Prerequisites

```bash
# GPU Requirements
nvidia-smi  # Verify NVIDIA driver
docker run --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

# Database Setup
# Ensure TimescaleDB with sensor_data, external_weather_aarhus, external_energy_prices_dk tables
```

### 2. Environment Configuration

```bash
# Copy enhanced configuration
cp .env.sparse .env

# Key settings for enhanced mode
DISABLE_GPU=false
SPARSE_MODE=true
ENHANCED_MODE=true
SPARSE_START_DATE=2014-01-01
SPARSE_END_DATE=2014-12-31
```

### 3. Basic Usage

```bash
# Build enhanced pipeline
docker compose -f docker-compose.sparse.yml build sparse_pipeline

# Run enhanced mode
docker compose -f docker-compose.sparse.yml run --rm \
  -e ENHANCED_MODE=true \
  sparse_pipeline \
  --enhanced-mode \
  --start-date 2014-01-01 \
  --end-date 2014-07-01 \
  --batch-size 24
```

### 4. Direct Binary Usage

```bash
# Compile and run directly
cd gpu_feature_extraction
cargo build --release

./target/release/gpu_feature_extraction \
  --enhanced-mode \
  --start-date 2014-01-01 \
  --end-date 2014-07-01 \
  --batch-size 24 \
  --database-url postgresql://postgres:postgres@localhost:5432/postgres
```

## Feature Categories

### 1. Extended Statistical Features (GPU)
```
Per sensor (temperature, CO2, humidity, light):
• Basic: mean, std, min, max
• Percentiles: p5, p25, p50, p75, p95
• Moments: skewness, kurtosis
• Spread: IQR, MAD
• Information: entropy
Total: ~80 features
```

### 2. Weather Coupling Features (GPU)
```
• Temperature differential (internal - external)
• Solar radiation efficiency 
• Weather response lag correlation
• Thermal mass indicators
• Ventilation effectiveness
Total: ~15 features
```

### 3. Energy Optimization Features (GPU)
```
• Cost-weighted consumption
• Peak vs off-peak usage ratio
• Energy efficiency scores
• Optimal load shifting opportunities
• Hours until cheap energy
Total: ~12 features
```

### 4. Plant Growth Features (GPU)
```
• Growing Degree Days (GDD)
• Daily Light Integral (DLI)
• Photoperiod calculation
• Temperature optimality scores
• Light sufficiency ratios
• Stress degree days
• Flowering signals (species-specific)
• Expected growth rate
Total: ~20 features
```

### 5. Multi-Resolution Features
```
Each feature category computed at:
• 15-minute resolution
• 1-hour resolution  
• 4-hour resolution
• 12-hour resolution
• 24-hour resolution
Multiplier: 5x all features
```

### 6. MOEA Optimization Metrics
```
• Growth performance score
• Energy cost efficiency
• Environmental coupling score
• Sustainability score (combined)
• Objective 1: Minimize energy cost
• Objective 2: Maximize growth rate
• Objective 3: Minimize stress
Total: ~10 optimization-ready metrics
```

## Expected Performance

### Basic vs Enhanced Pipeline
```
                    Basic      Enhanced    Improvement
GPU Utilization:    65-75%     85-95%      +20-30%
Feature Count:      ~350       ~1,200      +3.4x
Memory Usage:       2 GB       6-8 GB      3-4x
Processing Speed:   77 feat/s  150+ feat/s +2x
Data Coverage:      6 months   1+ year     +2x
```

### Processing Time Estimates
```
Dataset Size    Enhanced Pipeline    Features Generated
1 Month         3-5 seconds         ~240 feature sets
6 Months        15-20 seconds       ~1,440 feature sets  
1 Year          30-40 seconds       ~2,880 feature sets
```

## Data Requirements

### Internal Sensor Data (Required)
- Temperature sensors
- CO2 measurements  
- Humidity sensors
- Light intensity
- Lamp status indicators
- Heating setpoints
- Ventilation positions

### External Weather Data (Optional)
Database table: `external_weather_aarhus`
```sql
CREATE TABLE external_weather_aarhus (
    time TIMESTAMPTZ PRIMARY KEY,
    temperature_2m REAL,
    relative_humidity_2m REAL,
    precipitation REAL,
    shortwave_radiation REAL,
    wind_speed_10m REAL,
    pressure_msl REAL,
    cloud_cover REAL
);
```

### Energy Price Data (Optional)
Database table: `external_energy_prices_dk`
```sql
CREATE TABLE external_energy_prices_dk (
    "HourUTC" TIMESTAMPTZ NOT NULL,
    "PriceArea" VARCHAR(10) NOT NULL,
    "SpotPriceDKK" REAL,
    PRIMARY KEY ("HourUTC", "PriceArea")
);
```

### Phenotype Data (Optional)
File: `feature_extraction/pre_process/phenotype.json`
```json
{
  "phenotype": [
    {
      "species": "Kalanchoe blossfeldiana",
      "environment_temp_day_C": 22.0,
      "environment_temp_night_C": 18.0,
      "environment_photoperiod_h": 8.0
    }
  ]
}
```

## Configuration Options

### Enhanced Pipeline Config
```rust
pub struct EnhancedPipelineConfig {
    // Basic settings
    pub min_hourly_coverage: f32,      // 0.1 = 10% minimum
    pub max_interpolation_gap: i64,    // 2 hours max gap fill
    pub window_hours: usize,           // 24 hour windows
    pub slide_hours: usize,            // 6 hour slide
    
    // Feature toggles
    pub enable_weather_features: bool,  // External weather
    pub enable_energy_features: bool,   // Energy prices
    pub enable_growth_features: bool,   // Plant biology
    
    // Advanced features
    pub enable_multiresolution: bool,   // Multiple time scales
    pub enable_extended_statistics: bool, // GPU percentiles
    pub enable_coupling_features: bool,   // Cross-domain
    pub enable_temporal_features: bool,   // Time patterns
    
    // Resolution windows
    pub resolution_windows: Vec<Duration>, // [15min, 1h, 4h, 12h, 24h]
}
```

### Environment Variables
```bash
# GPU Control
DISABLE_GPU=false                    # Enable GPU acceleration
CUDA_VISIBLE_DEVICES=0              # GPU device selection

# Pipeline Mode
SPARSE_MODE=true                    # Handle sparse data
ENHANCED_MODE=true                  # Full feature set

# Data Range  
SPARSE_START_DATE=2014-01-01       # Start date
SPARSE_END_DATE=2014-12-31         # End date
SPARSE_BATCH_SIZE=24               # Window size (hours)

# External Data
ENABLE_WEATHER_FEATURES=true       # Weather integration
ENABLE_ENERGY_FEATURES=true        # Energy price integration
ENABLE_GROWTH_FEATURES=true        # Plant phenotype features

# Performance
GPU_BATCH_SIZE=500                 # GPU processing batch
MAX_MEMORY_GB=8                    # Memory limit
```

## Output Structure

### Enhanced Feature Sets
```rust
pub struct EnhancedFeatureSet {
    pub era_id: i64,
    pub computed_at: DateTime<Utc>,
    pub resolution: String,           // "15min", "60min", etc.
    
    // Feature categories
    pub sensor_features: HashMap<String, f64>,      // ~40 features
    pub extended_stats: HashMap<String, f64>,       // ~80 features  
    pub weather_features: HashMap<String, f64>,     // ~15 features
    pub energy_features: HashMap<String, f64>,      // ~12 features
    pub growth_features: HashMap<String, f64>,      // ~20 features
    pub temporal_features: HashMap<String, f64>,    // ~30 features
    pub optimization_metrics: HashMap<String, f64>, // ~10 features
}
```

### MOEA-Ready Outputs
```rust
// Direct optimization objectives
feature_set.optimization_metrics = {
    "moea_obj1_growth_score": 0.85,        // Maximize (0-1)
    "moea_obj2_cost_efficiency": 0.72,     // Maximize (0-1)  
    "moea_obj3_stress_min": 0.91,          // Maximize (0-1)
    
    "growth_performance_score": 0.85,       // Combined growth
    "energy_cost_efficiency": 0.72,         // Energy efficiency
    "sustainability_score": 0.83,           // Overall sustainability
}
```

## MOEA Integration

### Python Integration Example
```python
from enhanced_sparse_pipeline import EnhancedSparsePipeline

# Load features for MOEA
pipeline = EnhancedSparsePipeline()
results = await pipeline.run_enhanced_pipeline(start_date, end_date)

# Extract MOEA objectives
for resolution, feature_sets in results.multiresolution_features.items():
    for feature_set in feature_sets:
        objectives = [
            feature_set.optimization_metrics["moea_obj1_growth_score"],
            -feature_set.optimization_metrics["moea_obj2_cost_efficiency"],  # Minimize cost
            -feature_set.optimization_metrics["moea_obj3_stress_min"]        # Minimize stress
        ]
        
        # Feed to MOEA optimizer
        moea_problem.evaluate(control_variables, objectives)
```

## Monitoring & Debugging

### GPU Utilization
```bash
# Monitor during execution
watch -n 1 nvidia-smi

# Expected during feature extraction:
# GPU-Util: 85-95%
# Memory-Usage: 6-8 GB / 12 GB
# Temperature: 70-80°C
```

### Performance Profiling
```bash
# Enable detailed logging
RUST_LOG=debug ./gpu_feature_extraction --enhanced-mode

# Check feature extraction times
grep "Stage 3" logs/*.log
# Expected: ~15-20s for 6 months of data
```

### Feature Validation
```bash
# Check feature counts per resolution
grep "feature sets" logs/*.log

# Expected output:
# 15min: 2,880 feature sets (6 months * 24 hours * 4 per hour * 5 resolutions)
# 60min: 720 feature sets  
# 240min: 180 feature sets
# Total: ~1,200+ feature sets
```

## Troubleshooting

### Common Issues

**1. GPU Not Activated**
```bash
# Check DISABLE_GPU setting
echo $DISABLE_GPU  # Should be "false"

# Verify GPU availability
nvidia-smi

# Check Docker GPU runtime
docker run --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

**2. External Data Missing**
```bash
# Weather data check
psql -c "SELECT COUNT(*) FROM external_weather_aarhus WHERE time BETWEEN '2014-01-01' AND '2014-12-31';"

# Energy data check  
psql -c "SELECT COUNT(*) FROM external_energy_prices_dk WHERE \"HourUTC\" BETWEEN '2014-01-01' AND '2014-12-31';"
```

**3. Memory Issues**
```bash
# Reduce batch size
export SPARSE_BATCH_SIZE=12  # From default 24

# Monitor memory usage
nvidia-smi -l 1

# Check system memory
free -h
```

**4. Performance Issues**
```bash
# Enable GPU profiling
nsys profile ./gpu_feature_extraction --enhanced-mode

# Check CPU usage
htop

# Verify SSD performance
iostat -x 1
```

## Roadmap

### Phase 1: Core Implementation ✅
- [x] GPU kernels for extended statistics
- [x] External data integration
- [x] Multi-resolution processing
- [x] MOEA objective calculation

### Phase 2: Optimization (Next)
- [ ] RAPIDS cuDF integration for 10x preprocessing speedup
- [ ] Multi-GPU support for parallel processing
- [ ] Real-time streaming mode
- [ ] Adaptive batch sizing

### Phase 3: Advanced Features (Future)
- [ ] Anomaly detection features
- [ ] Predictive features (forecasting)
- [ ] Cross-greenhouse comparison
- [ ] Automated hyperparameter tuning

## Contributing

### Development Setup
```bash
# Clone and build
git clone <repo>
cd DataIngestion/gpu_feature_extraction
cargo build

# Run tests
cargo test

# Format code
cargo fmt
cargo clippy
```

### Adding New Features
1. Add GPU kernel to `src/kernels/extended_statistics.cu`
2. Create Rust wrapper in `src/enhanced_features.rs`
3. Integrate in `src/enhanced_sparse_pipeline.rs`
4. Add to MOEA objectives if applicable
5. Update documentation

### Performance Testing
```bash
# Benchmark script
./scripts/benchmark_enhanced_pipeline.sh

# Compare with baseline
./scripts/compare_basic_vs_enhanced.sh
```

## Support

For issues, questions, or contributions:
- 📖 Documentation: `docs/` directory
- 🐛 Bug reports: Include GPU info, logs, and data characteristics
- 💡 Feature requests: Describe MOEA optimization use case
- 🚀 Performance issues: Include `nvidia-smi` output and timing logs

---

**Ready to optimize your greenhouse with enhanced AI? Start with the quick start guide above! 🌱🚀**