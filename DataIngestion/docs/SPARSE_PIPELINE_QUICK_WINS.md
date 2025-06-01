# Sparse Pipeline Enhancement: Quick Wins Implementation Guide

Generated: May 31, 2025

## Overview

This guide focuses on practical, high-impact enhancements that can be implemented quickly to improve the sparse pipeline's GPU utilization and feature richness. These "quick wins" are prioritized by impact/effort ratio.

## Quick Win #1: Extended Statistical Features (2-3 days)

### Current State
- Basic stats: mean, std, min, max (4 features per sensor)
- Total: ~20 statistical features

### Enhancement
Add GPU kernels for extended statistics that are valuable for greenhouse control:

```cuda
// Add to kernels/statistics.cu
__global__ void compute_percentiles_kernel(
    const float* data,
    int n,
    float* percentiles,  // Output: p5, p25, p50, p75, p95
    int* histogram       // Shared memory histogram
) {
    // Fast GPU percentile calculation using histogram method
    extern __shared__ int shared_hist[];
    
    // Build histogram collaboratively
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // ... histogram building logic ...
    
    // Calculate percentiles from histogram
    if (threadIdx.x < 5) {
        float percentile_points[] = {0.05f, 0.25f, 0.5f, 0.75f, 0.95f};
        percentiles[threadIdx.x] = calculate_percentile_from_hist(
            shared_hist, n, percentile_points[threadIdx.x]
        );
    }
}

// Add skewness and kurtosis
__global__ void compute_moments_kernel(
    const float* data,
    int n,
    float mean,
    float std,
    float* skewness,
    float* kurtosis
) {
    // Parallel computation of 3rd and 4th moments
    __shared__ float partial_m3[256];
    __shared__ float partial_m4[256];
    
    // Each thread computes partial moments
    // ... implementation ...
}
```

### Expected Impact
- **Features added**: 7 per sensor (5 percentiles + skewness + kurtosis)
- **Total new features**: ~35
- **GPU utilization increase**: +5-10%
- **MOEA benefit**: Better understanding of data distribution

## Quick Win #2: Weather Coupling Features (1-2 days)

### Current State
- No external weather integration
- Missing critical environmental context

### Enhancement
Simple weather-sensor coupling features:

```rust
// Add to sparse_pipeline.rs
impl SparsePipeline {
    async fn compute_weather_coupling_features(
        &self,
        internal_data: &DataFrame,
        weather_data: &DataFrame,
    ) -> Result<HashMap<String, f64>> {
        let mut features = HashMap::new();
        
        // Temperature coupling
        let internal_temp = internal_data.column("air_temp_c_mean")?;
        let external_temp = weather_data.column("temperature_2m")?;
        
        // Simple features first
        features.insert(
            "temp_differential_mean".to_string(),
            (internal_temp.mean().unwrap() - external_temp.mean().unwrap())
        );
        
        // Solar efficiency (internal temp gain per unit radiation)
        let solar_rad = weather_data.column("shortwave_radiation")?;
        let temp_gain = self.calculate_temp_gain(internal_temp, external_temp)?;
        let solar_efficiency = temp_gain / solar_rad.mean().unwrap();
        features.insert("solar_efficiency".to_string(), solar_efficiency);
        
        // Time lag correlation (how long external changes take to affect internal)
        let lag_correlation = self.compute_lag_correlation_gpu(
            external_temp.f64()?,
            internal_temp.f64()?,
            24  // max lag hours
        )?;
        features.insert("weather_response_lag_hours".to_string(), lag_correlation.0);
        features.insert("weather_correlation_strength".to_string(), lag_correlation.1);
        
        Ok(features)
    }
}
```

### Expected Impact
- **Features added**: 10-15 weather coupling metrics
- **GPU utilization increase**: +10-15% (correlation calculations)
- **MOEA benefit**: Environmental awareness for predictive control

## Quick Win #3: Energy-Aware Features (1 day)

### Current State
- Lamp status tracked but not correlated with energy prices
- No cost optimization signals

### Enhancement
Add energy price awareness:

```rust
// Simple energy features
pub struct EnergyFeatures {
    pub fn calculate_cost_weighted_usage(
        lamp_status: &[bool],
        heating_power: &[f32],
        energy_prices: &[f32],
    ) -> HashMap<String, f64> {
        let mut features = HashMap::new();
        
        // Weighted energy consumption
        let weighted_consumption: f64 = lamp_status.iter()
            .zip(energy_prices.iter())
            .map(|(on, price)| if *on { *price as f64 } else { 0.0 })
            .sum();
            
        features.insert("cost_weighted_lamp_usage".to_string(), weighted_consumption);
        
        // Peak vs off-peak ratio
        let price_threshold = percentile(energy_prices, 0.75);
        let peak_usage = count_usage_above_threshold(lamp_status, energy_prices, price_threshold);
        let offpeak_usage = count_usage_below_threshold(lamp_status, energy_prices, price_threshold);
        
        features.insert("peak_offpeak_ratio".to_string(), peak_usage / offpeak_usage.max(1.0));
        
        // Hours until cheap energy
        let next_cheap_hour = find_next_cheap_period(energy_prices, current_hour);
        features.insert("hours_until_cheap_energy".to_string(), next_cheap_hour);
        
        features
    }
}
```

### Expected Impact
- **Features added**: 5-8 energy features
- **GPU utilization increase**: +5% (minimal computation)
- **MOEA benefit**: Direct cost optimization capability

## Quick Win #4: Multi-Window Feature Extraction (2 days)

### Current State
- Fixed window sizes: 60s, 300s, 3600s
- Missing medium-term patterns

### Enhancement
Add adaptive multi-scale windows:

```rust
// Enhanced window configuration
pub fn get_adaptive_windows(data_density: f32) -> Vec<Duration> {
    match data_density {
        d if d > 0.7 => vec![
            Duration::minutes(15),
            Duration::hours(1),
            Duration::hours(4),
            Duration::hours(12),
        ],
        d if d > 0.3 => vec![
            Duration::hours(1),
            Duration::hours(6),
            Duration::hours(12),
            Duration::days(1),
        ],
        _ => vec![
            Duration::hours(6),
            Duration::hours(12),
            Duration::days(1),
            Duration::days(2),
        ],
    }
}

// GPU kernel for multi-window processing
__global__ void extract_multiwindow_features(
    const float* data,
    int data_length,
    const int* window_sizes,
    int num_windows,
    float* features_out  // [num_windows x num_features]
) {
    int window_idx = blockIdx.x;
    if (window_idx >= num_windows) return;
    
    int window_size = window_sizes[window_idx];
    
    // Extract features for this window size
    // Shared memory for efficiency
    extern __shared__ float shared_data[];
    
    // Collaborative loading and processing
    // ...
}
```

### Expected Impact
- **Features added**: 3x current features (multiple scales)
- **GPU utilization increase**: +20-25%
- **MOEA benefit**: Multi-scale temporal patterns

## Quick Win #5: Phenotype-Aware Growth Features (1 day)

### Current State
- No plant-specific features
- Generic control without biological context

### Enhancement
Add simple growth-related features:

```rust
// Growth features based on Kalanchoe phenotype
pub fn calculate_growth_features(
    temperature: &[f32],
    light_intensity: &[f32],
    phenotype: &KalanchoePhenotype,
) -> HashMap<String, f64> {
    let mut features = HashMap::new();
    
    // Growing Degree Days (GDD)
    let base_temp = 10.0;  // Kalanchoe base temperature
    let gdd: f64 = temperature.iter()
        .map(|t| (t - base_temp).max(0.0) as f64)
        .sum() / 24.0;  // Daily accumulation
    features.insert("growing_degree_days".to_string(), gdd);
    
    // Daily Light Integral (DLI)
    let dli = calculate_dli(light_intensity);
    features.insert("daily_light_integral".to_string(), dli);
    
    // Temperature optimality (22°C day, 18°C night optimal)
    let temp_optimality = calculate_temp_optimality(
        temperature,
        phenotype.optimal_day_temp,
        phenotype.optimal_night_temp
    );
    features.insert("temperature_optimality".to_string(), temp_optimality);
    
    // Photoperiod for flowering (8h critical for Kalanchoe)
    let photoperiod_hours = calculate_photoperiod(light_intensity);
    let flowering_signal = if photoperiod_hours <= 8.0 { 1.0 } else { 0.0 };
    features.insert("flowering_photoperiod_met".to_string(), flowering_signal);
    
    features
}
```

### Expected Impact
- **Features added**: 6-10 plant-specific features
- **GPU utilization increase**: +5%
- **MOEA benefit**: Biologically meaningful optimization

## Implementation Priority & Timeline

### Week 1
1. **Day 1-2**: Extended Statistical Features
   - Implement percentile kernel
   - Add skewness/kurtosis
   - Test and validate

2. **Day 3-4**: Weather Coupling Features
   - Load weather data
   - Implement correlation kernels
   - Calculate efficiency metrics

3. **Day 5**: Energy Features
   - Integrate price data
   - Simple cost calculations
   - Peak/off-peak analysis

### Week 2
1. **Day 1-2**: Multi-Window Processing
   - Adaptive window selection
   - Parallel extraction kernel
   - Memory optimization

2. **Day 3**: Phenotype Features
   - Load phenotype data
   - GDD/DLI calculations
   - Growth optimality metrics

3. **Day 4-5**: Integration & Testing
   - Combine all features
   - Performance profiling
   - MOEA integration test

## Expected Overall Impact

### Before Enhancement
- **Features**: ~350
- **GPU Utilization**: 65-75%
- **Processing Speed**: 76.6 feat/s

### After Quick Wins
- **Features**: ~750-850 (+115-140%)
- **GPU Utilization**: 85-90% (+20-25%)
- **Processing Speed**: ~120 feat/s (+57%)

### MOEA Benefits
1. **2x richer feature space** for optimization
2. **Environmental awareness** through weather coupling
3. **Cost optimization** via energy price integration
4. **Biological relevance** with growth features
5. **Multi-scale patterns** from adaptive windows

## Validation Checklist

- [ ] GPU utilization reaches 85%+
- [ ] All new features have unit tests
- [ ] Performance regression tests pass
- [ ] MOEA shows improved convergence
- [ ] Memory usage stays under 8GB
- [ ] Processing time < 10s for 1 year

## Code Structure

```
gpu_feature_extraction/
├── src/
│   ├── kernels/
│   │   ├── statistics_extended.cu    # New
│   │   ├── weather_coupling.cu       # New
│   │   └── multiwindow.cu           # New
│   ├── features/
│   │   ├── energy.rs                # New
│   │   ├── growth.rs                # New
│   │   └── weather.rs               # New
│   └── sparse_pipeline_enhanced.rs  # Updated
```

## Next Steps

1. **Start with Extended Statistics** - Highest impact/effort ratio
2. **Run A/B tests** - Compare MOEA performance with/without features
3. **Monitor GPU metrics** - Ensure utilization improvements
4. **Document feature meanings** - For MOEA interpretation

These quick wins provide maximum impact with minimal implementation effort, setting the foundation for more advanced enhancements later.