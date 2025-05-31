# Missing Features Implementation Roadmap

## Overview
This document tracks the missing features from the GPU-aware feature engineering catalogue and provides implementation priorities.

## Implementation Status

### ✅ Implemented Features (~120 features)

#### Basic Set (Initial Implementation)
- [x] Basic statistics (mean, std, min, max, skewness, kurtosis)
- [x] Rolling window mean and std
- [x] VPD calculation
- [x] Simple energy efficiency metric

#### Thermal Time Features (COMPLETE)
- [x] Growing Degree Days (GDD)
- [x] Photo-Thermal Time (PTT)
- [x] Cumulative VPD Index (CVPI)
- [x] Accumulated Light Integral (ALI)
- [x] Heat stress hours
- [x] Thermal efficiency index

#### Psychrometric Features (COMPLETE)
- [x] Dew point temperature
- [x] Wet bulb temperature
- [x] Humidity ratio
- [x] Enthalpy of moist air
- [x] Specific volume
- [x] Bowen ratio (latent:sensible)
- [x] Condensation risk index

#### Actuator Dynamics (COMPLETE)
- [x] Edge counting (state transitions)
- [x] Duty cycle calculation
- [x] Ramp rates (rate of change)
- [x] Overshoot/undershoot detection
- [x] Settling time
- [x] Control effort metric
- [x] Actuator synchronization coefficient
- [x] Oscillation detection

#### Frequency Domain Features (COMPLETE)
- [x] Spectral power in frequency bands
- [x] Spectral centroid
- [x] Peak frequency detection
- [x] Spectral entropy
- [x] Spectral rolloff
- [x] Spectral flux

#### Temporal Dependencies (COMPLETE)
- [x] Autocorrelation function (ACF) for multiple lags
- [x] Partial autocorrelation (PACF) via Durbin-Levinson
- [x] Cross-correlation between signals
- [x] Lag of maximum cross-correlation
- [x] Lagged feature values
- [x] Time-delayed mutual information

#### Entropy & Complexity Features (COMPLETE)
- [x] Shannon entropy
- [x] Sample entropy (SampEn)
- [x] Permutation entropy
- [x] Approximate entropy (ApEn)
- [x] Higuchi fractal dimension
- [x] Lempel-Ziv complexity

#### Wavelet Features (COMPLETE)
- [x] Discrete Wavelet Transform (DWT) with db4
- [x] Multi-level DWT energy
- [x] Wavelet packet energy
- [x] Continuous Wavelet Transform (CWT) with Morlet
- [x] Wavelet coherence
- [x] Wavelet entropy

#### Economic Features (COMPLETE)
- [x] Price gradient
- [x] Price volatility
- [x] Price-weighted energy consumption
- [x] Cost efficiency ratio
- [x] Peak vs off-peak energy ratio
- [x] Demand response opportunity
- [x] Time-of-use optimization score
- [x] Carbon-weighted energy
- [x] Energy ROI
- [x] Forecast error cost impact

### 🟢 Remaining Integration Tasks

#### 1. Thermal Time Features (CRITICAL for plant growth)
```rust
// src/kernels/thermal_time.cu
__global__ void compute_gdd(float* temp, float* gdd, float base_temp, int n);
__global__ void compute_ptt(float* temp, float* light, float* ptt, int n);
__global__ void compute_cvpi(float* vpd, float* cvpi, int n);
```

#### 2. Advanced Rolling Statistics
```rust
// src/kernels/rolling_advanced.cu
__global__ void rolling_percentiles(float* data, float* p10, float* p90, int window, int n);
__global__ void cumulative_sum(float* data, float* cumsum, int n);
__global__ void heating_cooling_hours(float* temp, float* hdd, float* cdh, float base, int n);
```

#### 3. Psychrometric Features
```rust
// src/kernels/psychrometric.cu
__global__ void dew_point(float* temp, float* rh, float* tdew, int n);
__global__ void enthalpy(float* temp, float* rh, float* h, int n);
__global__ void wet_bulb(float* temp, float* rh, float* twb, int n);
```

### 🟠 Medium Priority Missing Features (~50 features)

#### 4. Temporal Dependencies (ACF/PACF)
```rust
// src/kernels/temporal.cu
__global__ void autocorrelation(float* data, float* acf, int max_lag, int n);
__global__ void cross_correlation(float* x, float* y, float* xcorr, int n);
```

#### 5. Actuator Dynamics
```rust
// src/kernels/actuator.cu
__global__ void edge_count(float* signal, int* edges, float threshold, int n);
__global__ void duty_cycle(bool* binary_signal, float* duty, int window, int n);
__global__ void ramp_rate(float* signal, float* rate, int n);
```

#### 6. Frequency Domain (FFT-based)
```rust
// Using cuFFT
- Spectral power in bands (0-0.1Hz, 0.1-1Hz, 1-10Hz)
- Spectral centroid
- Peak frequency
```

### 🔴 Lower Priority Missing Features (~20 features)

#### 7. Entropy & Complexity
```rust
// src/kernels/entropy.cu
__global__ void shannon_entropy(float* data, float* entropy, int bins, int n);
__global__ void sample_entropy(float* data, float* sampen, int m, float r, int n);
```

#### 8. Economic Features
```rust
// src/kernels/economic.cu
__global__ void price_weighted_energy(float* energy, float* price, float* pwe, int n);
```

## Implementation Plan

### Phase 1: Critical Features (Week 1)
1. **Thermal Time Kernels**
   - GDD, PTT, CVPI
   - These directly impact plant growth modeling

2. **Advanced Rolling Stats**
   - Percentiles for outlier detection
   - Cumulative sums for resource tracking

### Phase 2: Climate Features (Week 2)
3. **Psychrometric Kernels**
   - Full set of moist air properties
   - Critical for energy balance

4. **Temporal Dependencies**
   - ACF/PACF for time series analysis
   - Cross-correlation for lag detection

### Phase 3: Control Features (Week 3)
5. **Actuator Dynamics**
   - Essential for control optimization
   - Helps identify inefficiencies

6. **Frequency Domain**
   - Spectral features via cuFFT
   - Identify periodic patterns

### Phase 4: Advanced Analytics (Week 4)
7. **Entropy Measures**
   - System complexity metrics
   - Predictability assessment

8. **Economic Integration**
   - Cost-aware features
   - ROI optimization support

## Code Structure for New Features

```rust
// src/features.rs - Add new feature categories
impl GpuFeatureExtractor {
    fn compute_thermal_features(&self, era_data: &EraData) -> Result<HashMap<String, f64>> {
        // GDD, PTT, CVPI calculations
    }
    
    fn compute_psychrometric_features(&self, era_data: &EraData) -> Result<HashMap<String, f64>> {
        // Dew point, enthalpy, wet bulb
    }
    
    fn compute_actuator_features(&self, era_data: &EraData) -> Result<HashMap<String, f64>> {
        // Edge counts, duty cycles, ramp rates
    }
}
```

## Testing Strategy

1. **Unit Tests**: Each kernel tested individually
2. **Reference Values**: Compare with Python/tsfresh
3. **Performance Benchmarks**: Ensure >30GB/s throughput
4. **Integration Tests**: Full pipeline validation

## Expected Impact

| Feature Category | Expected Model Improvement | Computation Cost |
|-----------------|---------------------------|------------------|
| Thermal Time | High (15-20%) | Low |
| Psychrometric | High (10-15%) | Low |
| Actuator Dynamics | Medium (5-10%) | Medium |
| Frequency Domain | Medium (5-10%) | High (FFT) |
| Entropy | Low (2-5%) | High |

## Dependencies

- **cuFFT**: For frequency domain features
- **cuBLAS**: For PACF computation
- **Thrust**: For sorting/percentiles

## Next Steps

1. Implement Phase 1 features (thermal time + advanced rolling)
2. Benchmark and validate against Python reference
3. Update model builder to use new features
4. Measure impact on MOEA optimization