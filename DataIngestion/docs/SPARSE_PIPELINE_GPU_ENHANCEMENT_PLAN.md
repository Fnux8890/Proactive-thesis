# Sparse Pipeline GPU Enhancement Plan

Generated: May 31, 2025

## Executive Summary

This plan outlines comprehensive enhancements to the sparse feature extraction pipeline to leverage more GPU acceleration, incorporate external data sources (weather, energy prices, phenotype), and generate richer features for the MOEA optimizer. The goal is to maximize GPU utilization (currently at 65-75%) while extracting more meaningful features from our sparse greenhouse data.

## Current State Analysis

### What We Have
1. **Core Sensor Data** (91.3% sparse):
   - Temperature, CO2, humidity, radiation, VPD
   - Lamp status indicators
   - Ventilation positions
   - Heating setpoints

2. **External Data Sources** (underutilized):
   - **Weather Data**: 17 variables from Open-Meteo API
   - **Energy Prices**: Danish spot prices (DK1, DK2)
   - **Phenotype Data**: Kalanchoe blossfeldiana growth parameters

3. **Current GPU Features** (partially implemented):
   - Basic statistics (mean, std, min, max)
   - Rolling window features (60s, 300s, 3600s)
   - Limited cross-sensor features

### Performance Metrics
- **GPU Utilization**: 65-75% (room for improvement)
- **Memory Usage**: 1.8-2.1 GB of 12 GB available
- **Processing Speed**: 76.6 features/second

## Enhancement Strategy

### 1. Extended Time Period Processing

#### Rationale
- Current: 6 months (Jan-Jul 2014)
- Proposed: Full year (2014) + multi-year capability
- Benefits: More training data for MOEA, seasonal pattern capture

#### Implementation
```rust
// Extend date range handling
pub struct ExtendedPipelineConfig {
    pub enable_multi_year: bool,
    pub year_batch_size: usize,  // Process N years at once
    pub seasonal_segmentation: bool,  // Spring, Summer, Fall, Winter eras
}
```

### 2. GPU-Accelerated Feature Categories

#### A. Advanced Statistical Features (GPU)
```cuda
// New GPU kernels for extended statistics
__global__ void compute_extended_stats_kernel(
    const float* data, 
    int n,
    ExtendedStats* output
) {
    // Existing: mean, std, min, max
    // New additions:
    // - Skewness and kurtosis
    // - Percentiles (5, 25, 50, 75, 95)
    // - Interquartile range
    // - Median absolute deviation
    // - Entropy measures
    // - Hurst exponent
}
```

#### B. Temporal Pattern Features (GPU)
```cuda
// GPU kernel for autocorrelation and spectral features
__global__ void compute_temporal_patterns_kernel(
    const float* time_series,
    int length,
    int max_lag,
    TemporalFeatures* output
) {
    // - Autocorrelation function (ACF) up to lag 24
    // - Partial autocorrelation (PACF)
    // - Fourier transform magnitudes
    // - Dominant frequencies
    // - Circadian rhythm strength
    // - Weekly patterns
}
```

#### C. Weather-Climate Coupling Features (GPU)
```cuda
// Cross-correlation between internal and external conditions
__global__ void compute_weather_coupling_kernel(
    const float* internal_temp,
    const float* external_temp,
    const float* solar_radiation,
    int length,
    WeatherCouplingFeatures* output
) {
    // - Temperature differential dynamics
    // - Solar gain efficiency
    // - Thermal mass indicators
    // - Weather responsiveness lag
    // - Predictive power of external on internal
}
```

#### D. Energy Efficiency Features (GPU)
```cuda
// Energy consumption patterns and efficiency metrics
__global__ void compute_energy_features_kernel(
    const float* lamp_status,
    const float* heating_status,
    const float* energy_prices,
    const float* internal_conditions,
    int length,
    EnergyFeatures* output
) {
    // - Energy use intensity (kWh/m²)
    // - Cost-weighted consumption
    // - Peak vs off-peak usage ratio
    // - Efficiency scores (output/input)
    // - Optimal control windows
}
```

#### E. Plant Growth Features (GPU)
```cuda
// Phenotype-aware growth conditions
__global__ void compute_growth_features_kernel(
    const float* temperature,
    const float* light_intensity,
    const float* photoperiod,
    const PhenotypeParams* phenotype,
    int length,
    GrowthFeatures* output
) {
    // - Growing degree days (GDD)
    // - Daily light integral (DLI)
    // - Photoperiod accumulation
    // - Stress degree days
    // - Optimal growth hours
    // - Development stage indicators
}
```

### 3. Multi-Resolution Feature Extraction

```rust
pub struct MultiResolutionConfig {
    pub resolutions: Vec<Duration>,  // 5min, 15min, 1h, 4h, 1d
    pub adaptive_resolution: bool,   // Based on data density
}

impl SparsePipeline {
    pub async fn extract_multiresolution_features(
        &self,
        data: DataFrame,
    ) -> Result<HashMap<String, FeatureSet>> {
        let mut resolution_features = HashMap::new();
        
        for resolution in &self.config.resolutions {
            // Resample data to resolution
            let resampled = self.resample_to_resolution(data.clone(), resolution)?;
            
            // Extract features at this resolution
            let features = self.gpu_extractor.extract_features_batch(resampled)?;
            
            resolution_features.insert(
                format!("res_{}", resolution.num_minutes()),
                features
            );
        }
        
        Ok(resolution_features)
    }
}
```

### 4. External Data Integration

#### A. Weather Features
```rust
pub struct WeatherFeatures {
    // Direct measurements
    pub external_temp: f32,
    pub external_humidity: f32,
    pub solar_radiation: f32,
    pub wind_speed: f32,
    pub precipitation: f32,
    
    // Derived features
    pub temp_differential: f32,  // internal - external
    pub solar_efficiency: f32,    // internal_gain / solar_radiation
    pub weather_volatility: f32,  // std of changes
    pub forecast_accuracy: f32,   // if forecasts available
}
```

#### B. Energy Market Features
```rust
pub struct EnergyMarketFeatures {
    // Price signals
    pub current_price: f32,
    pub price_percentile: f32,  // where in daily range
    pub price_trend: f32,       // derivative
    pub price_volatility: f32,
    
    // Optimization signals
    pub hours_until_cheap: i32,
    pub hours_until_expensive: i32,
    pub optimal_load_shift: f32,
}
```

#### C. Phenotype-Informed Features
```rust
pub struct PhenotypeFeatures {
    // Growth conditions
    pub temp_optimality: f32,    // how close to optimal
    pub light_sufficiency: f32,  // DLI vs requirement
    pub stress_index: f32,       // cumulative stress
    
    // Development tracking
    pub thermal_time: f32,       // accumulated GDD
    pub photoperiod_sum: f32,    // for flowering
    pub expected_height: f32,    // based on conditions
}
```

### 5. GPU Memory Optimization

```rust
// Batch processing with memory management
pub struct GpuBatchProcessor {
    pub max_batch_size: usize,
    pub feature_cache: HashMap<String, CudaSlice<f32>>,
    pub stream_pool: Vec<CudaStream>,
}

impl GpuBatchProcessor {
    pub fn process_adaptive_batch(&self, data: &[f32]) -> Result<Features> {
        // Dynamically adjust batch size based on available memory
        let available_memory = self.get_available_gpu_memory()?;
        let optimal_batch = self.calculate_optimal_batch_size(
            data.len(),
            available_memory
        );
        
        // Process in parallel streams
        let chunks = data.chunks(optimal_batch);
        let results = chunks.par_iter()
            .enumerate()
            .map(|(i, chunk)| {
                let stream = &self.stream_pool[i % self.stream_pool.len()];
                self.process_chunk_on_stream(chunk, stream)
            })
            .collect::<Result<Vec<_>>>()?;
        
        // Merge results
        self.merge_features(results)
    }
}
```

### 6. Advanced Cross-Domain Features

#### A. Control-Response Features
```rust
pub struct ControlResponseFeatures {
    // System dynamics
    pub heating_response_time: f32,
    pub ventilation_effectiveness: f32,
    pub lamp_heat_contribution: f32,
    
    // Control efficiency
    pub control_stability: f32,
    pub actuator_usage_rate: f32,
    pub energy_per_degree: f32,
}
```

#### B. Anomaly Detection Features
```rust
pub struct AnomalyFeatures {
    // Statistical anomalies
    pub isolation_forest_score: f32,
    pub mahalanobis_distance: f32,
    pub local_outlier_factor: f32,
    
    // Pattern anomalies
    pub sequence_likelihood: f32,
    pub changepoint_probability: f32,
}
```

#### C. Predictive Features
```rust
pub struct PredictiveFeatures {
    // Future state indicators
    pub temp_trend_strength: f32,
    pub co2_depletion_rate: f32,
    pub humidity_stability: f32,
    
    // Forecast horizons
    pub hours_until_threshold: f32,
    pub predicted_peak_temp: f32,
    pub control_action_urgency: f32,
}
```

## Implementation Roadmap

### Phase 1: GPU Kernel Development (1-2 weeks)
1. Implement extended statistical kernels
2. Add temporal pattern analysis
3. Create weather coupling kernels
4. Optimize memory management

### Phase 2: External Data Integration (1 week)
1. Connect weather API pipeline
2. Integrate energy price data
3. Load phenotype parameters
4. Create cross-domain features

### Phase 3: Multi-Resolution Processing (1 week)
1. Implement adaptive resampling
2. Create resolution-specific features
3. Optimize GPU batch processing
4. Test with full year data

### Phase 4: Advanced Features (2 weeks)
1. Implement anomaly detection
2. Add predictive features
3. Create control-response metrics
4. Validate feature importance

## Expected Improvements

### Performance Gains
- **GPU Utilization**: 65-75% → 85-95%
- **Feature Count**: ~350 → 1,200+
- **Processing Speed**: 76 feat/s → 150+ feat/s
- **Memory Efficiency**: 2GB → 6-8GB utilized

### MOEA Benefits
1. **Richer Solution Space**: 3x more features
2. **Better Predictions**: Weather/energy awareness
3. **Improved Control**: Plant-specific optimization
4. **Cost Optimization**: Energy price integration

### Data Quality Improvements
1. **Temporal Coverage**: 6 months → 1+ years
2. **Feature Diversity**: 12 categories vs 5
3. **External Context**: Weather + energy + phenotype
4. **Multi-Scale Analysis**: 5 time resolutions

## Code Examples

### 1. Enhanced GPU Feature Extraction
```rust
// New comprehensive feature extractor
impl GpuFeatureExtractor {
    pub fn extract_enhanced_features(
        &self,
        sensor_data: &HashMap<String, Vec<f32>>,
        weather_data: &WeatherData,
        energy_data: &EnergyData,
        phenotype: &PhenotypeParams,
    ) -> Result<EnhancedFeatureSet> {
        // Allocate GPU memory for all data
        let d_sensors = self.upload_sensor_data(sensor_data)?;
        let d_weather = self.upload_weather_data(weather_data)?;
        let d_energy = self.upload_energy_data(energy_data)?;
        
        // Launch parallel feature extraction
        let features = vec![
            self.launch_statistical_kernels(&d_sensors),
            self.launch_temporal_kernels(&d_sensors),
            self.launch_coupling_kernels(&d_sensors, &d_weather),
            self.launch_energy_kernels(&d_sensors, &d_energy),
            self.launch_growth_kernels(&d_sensors, phenotype),
            self.launch_anomaly_kernels(&d_sensors),
        ];
        
        // Synchronize and merge
        self.stream.synchronize()?;
        self.merge_feature_results(features)
    }
}
```

### 2. Adaptive Window Configuration
```rust
// Dynamic window sizing based on data density
impl AdaptiveWindowConfig {
    pub fn calculate_optimal_window(
        &self,
        data_density: f32,
        feature_importance: &HashMap<String, f32>,
    ) -> Duration {
        // High density → smaller windows
        // Low density → larger windows
        let base_window = match data_density {
            d if d > 0.8 => Duration::hours(6),
            d if d > 0.5 => Duration::hours(12),
            d if d > 0.2 => Duration::hours(24),
            _ => Duration::hours(48),
        };
        
        // Adjust based on feature importance
        self.adjust_for_importance(base_window, feature_importance)
    }
}
```

### 3. Cross-Domain Feature Pipeline
```rust
// Integrated feature extraction pipeline
pub async fn run_enhanced_pipeline(
    &self,
    start_date: DateTime<Utc>,
    end_date: DateTime<Utc>,
) -> Result<EnhancedPipelineResults> {
    // Stage 1: Load all data sources
    let (sensor_data, weather_data, energy_data) = tokio::join!(
        self.load_sensor_data(start_date, end_date),
        self.load_weather_data(start_date, end_date),
        self.load_energy_data(start_date, end_date),
    );
    
    // Stage 2: Multi-resolution processing
    let resolutions = vec![
        Duration::minutes(5),
        Duration::minutes(15),
        Duration::hours(1),
        Duration::hours(4),
        Duration::days(1),
    ];
    
    let mut all_features = HashMap::new();
    for resolution in resolutions {
        let features = self.extract_resolution_features(
            &sensor_data?,
            &weather_data?,
            &energy_data?,
            resolution,
        ).await?;
        all_features.insert(resolution, features);
    }
    
    // Stage 3: Create enhanced eras
    let enhanced_eras = self.create_enhanced_eras(all_features)?;
    
    Ok(EnhancedPipelineResults {
        feature_count: self.count_total_features(&enhanced_eras),
        eras: enhanced_eras,
        quality_metrics: self.calculate_quality_metrics(),
    })
}
```

## Validation Strategy

### 1. Feature Importance Analysis
- Use SHAP values to validate feature relevance
- Compare MOEA performance with/without new features
- Measure prediction accuracy improvements

### 2. Performance Benchmarking
- GPU utilization monitoring
- Memory usage profiling
- Processing speed comparison
- Scaling tests with multi-year data

### 3. Scientific Validation
- Compare growth features with phenotype data
- Validate energy calculations with bills
- Cross-reference weather coupling with physics

## Conclusion

This enhancement plan transforms the sparse pipeline from a basic feature extractor to a comprehensive, GPU-accelerated system that leverages all available data sources. By increasing GPU utilization to 85-95% and extracting 1,200+ features across multiple domains and resolutions, we provide the MOEA optimizer with rich, contextual information for finding optimal greenhouse control strategies.

The integration of weather, energy, and phenotype data, combined with advanced GPU kernels and multi-resolution processing, positions this system at the forefront of smart greenhouse optimization technology.