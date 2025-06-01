// Extended Statistics GPU Kernels for Sparse Pipeline
// Implements percentiles, skewness, kurtosis, and other advanced statistical features

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>
#include <math.h>

// Structure to hold extended statistical features
struct ExtendedStatistics {
    float mean;
    float std;
    float min;
    float max;
    float p5;      // 5th percentile
    float p25;     // 25th percentile  
    float p50;     // Median
    float p75;     // 75th percentile
    float p95;     // 95th percentile
    float skewness;
    float kurtosis;
    float mad;     // Median Absolute Deviation
    float iqr;     // Interquartile Range
    float entropy; // Shannon entropy (binned)
    int count;
};

// Structure for weather coupling features
struct WeatherCouplingFeatures {
    float temp_differential_mean;
    float temp_differential_std;
    float solar_efficiency;
    float weather_response_lag;
    float correlation_strength;
    float thermal_mass_indicator;
    float ventilation_effectiveness;
};

// Structure for energy features
struct EnergyFeatures {
    float cost_weighted_consumption;
    float peak_offpeak_ratio;
    float hours_until_cheap;
    float energy_efficiency_score;
    float cost_per_degree_hour;
    float optimal_load_shift_hours;
};

// Structure for growth features
struct GrowthFeatures {
    float growing_degree_days;
    float daily_light_integral;
    float photoperiod_hours;
    float temperature_optimality;
    float light_sufficiency;
    float stress_degree_days;
    float flowering_signal;
    float expected_growth_rate;
};

// Fast percentile calculation using histogram method
__global__ void compute_percentiles_kernel(
    const float* data,
    int n,
    float data_min,
    float data_max,
    ExtendedStatistics* result
) {
    __shared__ int histogram[256];
    __shared__ float bin_width;
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // Initialize shared memory
    if (tid < 256) {
        histogram[tid] = 0;
    }
    if (tid == 0) {
        bin_width = (data_max - data_min) / 256.0f;
    }
    __syncthreads();
    
    // Build histogram collaboratively
    for (int i = tid; i < n; i += blockDim.x) {
        if (data[i] >= data_min && data[i] <= data_max) {
            int bin = min(255, (int)((data[i] - data_min) / bin_width));
            atomicAdd(&histogram[bin], 1);
        }
    }
    __syncthreads();
    
    // Calculate percentiles from histogram
    if (tid == 0) {
        int target_5 = (int)(0.05f * n);
        int target_25 = (int)(0.25f * n);
        int target_50 = (int)(0.50f * n);
        int target_75 = (int)(0.75f * n);
        int target_95 = (int)(0.95f * n);
        
        int cumulative = 0;
        for (int i = 0; i < 256; i++) {
            cumulative += histogram[i];
            
            if (cumulative >= target_5 && result->p5 == 0.0f) {
                result->p5 = data_min + i * bin_width;
            }
            if (cumulative >= target_25 && result->p25 == 0.0f) {
                result->p25 = data_min + i * bin_width;
            }
            if (cumulative >= target_50 && result->p50 == 0.0f) {
                result->p50 = data_min + i * bin_width;
            }
            if (cumulative >= target_75 && result->p75 == 0.0f) {
                result->p75 = data_min + i * bin_width;
            }
            if (cumulative >= target_95 && result->p95 == 0.0f) {
                result->p95 = data_min + i * bin_width;
                break;
            }
        }
        
        // Calculate IQR
        result->iqr = result->p75 - result->p25;
        
        // Calculate entropy from histogram
        float entropy = 0.0f;
        for (int i = 0; i < 256; i++) {
            if (histogram[i] > 0) {
                float p = (float)histogram[i] / n;
                entropy -= p * log2f(p);
            }
        }
        result->entropy = entropy;
    }
}

// Compute higher order moments (skewness and kurtosis)
__global__ void compute_moments_kernel(
    const float* data,
    int n,
    float mean,
    float std,
    ExtendedStatistics* result
) {
    __shared__ float partial_m3[256];
    __shared__ float partial_m4[256];
    
    int tid = threadIdx.x;
    int grid_size = blockDim.x * gridDim.x;
    
    partial_m3[tid] = 0.0f;
    partial_m4[tid] = 0.0f;
    
    // Each thread computes partial moments
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += grid_size) {
        float z = (data[i] - mean) / std;
        float z2 = z * z;
        partial_m3[tid] += z * z2;
        partial_m4[tid] += z2 * z2;
    }
    __syncthreads();
    
    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_m3[tid] += partial_m3[tid + stride];
            partial_m4[tid] += partial_m4[tid + stride];
        }
        __syncthreads();
    }
    
    // Write block results
    if (tid == 0) {
        atomicAdd(&result->skewness, partial_m3[0]);
        atomicAdd(&result->kurtosis, partial_m4[0]);
    }
}

// Finalize moments calculation
__global__ void finalize_moments_kernel(
    ExtendedStatistics* result,
    int n
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        result->skewness = result->skewness / n;
        result->kurtosis = (result->kurtosis / n) - 3.0f; // Excess kurtosis
    }
}

// Compute Median Absolute Deviation
__global__ void compute_mad_kernel(
    const float* data,
    int n,
    float median,
    float* temp_deviations,
    ExtendedStatistics* result
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Compute absolute deviations from median
    if (tid < n) {
        temp_deviations[tid] = fabsf(data[tid] - median);
    }
    
    __syncthreads();
    
    // Use CUB to find median of deviations (simplified version)
    // In practice, would use percentile calculation on temp_deviations
    // For now, approximate with mean of deviations
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += temp_deviations[i];
        }
        result->mad = sum / n * 1.4826f; // Scale factor for normal distribution
    }
}

// Weather coupling analysis
__global__ void compute_weather_coupling_kernel(
    const float* internal_temp,
    const float* external_temp,
    const float* solar_radiation,
    int n,
    WeatherCouplingFeatures* result
) {
    __shared__ float temp_diff_sum;
    __shared__ float temp_diff_sq_sum;
    __shared__ float solar_gain_sum;
    __shared__ float solar_rad_sum;
    
    int tid = threadIdx.x;
    int grid_size = blockDim.x * gridDim.x;
    
    if (tid == 0) {
        temp_diff_sum = 0.0f;
        temp_diff_sq_sum = 0.0f;
        solar_gain_sum = 0.0f;
        solar_rad_sum = 0.0f;
    }
    __syncthreads();
    
    // Calculate temperature differentials and solar effects
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += grid_size) {
        float temp_diff = internal_temp[i] - external_temp[i];
        atomicAdd(&temp_diff_sum, temp_diff);
        atomicAdd(&temp_diff_sq_sum, temp_diff * temp_diff);
        
        if (solar_radiation[i] > 0.0f) {
            float temp_gain = fmaxf(0.0f, temp_diff - 5.0f); // Baseline diff
            atomicAdd(&solar_gain_sum, temp_gain);
            atomicAdd(&solar_rad_sum, solar_radiation[i]);
        }
    }
    __syncthreads();
    
    if (tid == 0) {
        result->temp_differential_mean = temp_diff_sum / n;
        float variance = (temp_diff_sq_sum / n) - 
                        (result->temp_differential_mean * result->temp_differential_mean);
        result->temp_differential_std = sqrtf(variance);
        
        result->solar_efficiency = (solar_rad_sum > 0.0f) ? 
                                  (solar_gain_sum / solar_rad_sum) : 0.0f;
    }
}

// Lag correlation analysis for weather response
__global__ void compute_lag_correlation_kernel(
    const float* external_signal,
    const float* internal_signal,
    int n,
    int max_lag,
    float* correlation_results
) {
    int lag = blockIdx.x; // Each block handles one lag
    if (lag > max_lag) return;
    
    __shared__ float sum_ext;
    __shared__ float sum_int;
    __shared__ float sum_ext_sq;
    __shared__ float sum_int_sq;
    __shared__ float sum_cross;
    __shared__ int valid_count;
    
    int tid = threadIdx.x;
    
    if (tid == 0) {
        sum_ext = 0.0f;
        sum_int = 0.0f;
        sum_ext_sq = 0.0f;
        sum_int_sq = 0.0f;
        sum_cross = 0.0f;
        valid_count = 0;
    }
    __syncthreads();
    
    // Calculate correlation for this lag
    int valid_n = n - lag;
    for (int i = tid; i < valid_n; i += blockDim.x) {
        float ext_val = external_signal[i];
        float int_val = internal_signal[i + lag];
        
        atomicAdd(&sum_ext, ext_val);
        atomicAdd(&sum_int, int_val);
        atomicAdd(&sum_ext_sq, ext_val * ext_val);
        atomicAdd(&sum_int_sq, int_val * int_val);
        atomicAdd(&sum_cross, ext_val * int_val);
        atomicAdd(&valid_count, 1);
    }
    __syncthreads();
    
    if (tid == 0 && valid_count > 1) {
        float mean_ext = sum_ext / valid_count;
        float mean_int = sum_int / valid_count;
        
        float numerator = sum_cross - valid_count * mean_ext * mean_int;
        float denom_ext = sum_ext_sq - valid_count * mean_ext * mean_ext;
        float denom_int = sum_int_sq - valid_count * mean_int * mean_int;
        
        float denominator = sqrtf(denom_ext * denom_int);
        correlation_results[lag] = (denominator > 0.0f) ? 
                                  (numerator / denominator) : 0.0f;
    }
}

// Energy cost analysis
__global__ void compute_energy_features_kernel(
    const float* lamp_power,
    const float* heating_power,
    const float* energy_prices,
    int n,
    EnergyFeatures* result
) {
    __shared__ float total_cost;
    __shared__ float total_energy;
    __shared__ float peak_usage;
    __shared__ float offpeak_usage;
    
    int tid = threadIdx.x;
    
    if (tid == 0) {
        total_cost = 0.0f;
        total_energy = 0.0f;
        peak_usage = 0.0f;
        offpeak_usage = 0.0f;
    }
    __syncthreads();
    
    // Calculate price threshold (75th percentile approximation)
    float price_threshold = 0.0f;
    if (tid == 0) {
        // Simple approximation - would use percentile calculation in practice
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += energy_prices[i];
        }
        price_threshold = sum / n * 1.2f; // Rough 75th percentile
    }
    __syncthreads();
    
    // Calculate energy costs and usage patterns
    for (int i = tid; i < n; i += blockDim.x) {
        float total_power = lamp_power[i] + heating_power[i];
        float hour_cost = total_power * energy_prices[i];
        
        atomicAdd(&total_cost, hour_cost);
        atomicAdd(&total_energy, total_power);
        
        if (energy_prices[i] > price_threshold) {
            atomicAdd(&peak_usage, total_power);
        } else {
            atomicAdd(&offpeak_usage, total_power);
        }
    }
    __syncthreads();
    
    if (tid == 0) {
        result->cost_weighted_consumption = total_cost;
        result->peak_offpeak_ratio = (offpeak_usage > 0.0f) ? 
                                    (peak_usage / offpeak_usage) : 999.0f;
        result->energy_efficiency_score = (total_cost > 0.0f) ? 
                                         (total_energy / total_cost) : 0.0f;
    }
}

// Growing degree days and plant growth features
__global__ void compute_growth_features_kernel(
    const float* temperature,
    const float* light_intensity,
    const float* photoperiod,
    int n,
    float base_temp,
    float optimal_temp_day,
    float optimal_temp_night,
    float light_requirement,
    GrowthFeatures* result
) {
    __shared__ float gdd_sum;
    __shared__ float dli_sum;
    __shared__ float photoperiod_sum;
    __shared__ float temp_optimality_sum;
    __shared__ float stress_sum;
    
    int tid = threadIdx.x;
    
    if (tid == 0) {
        gdd_sum = 0.0f;
        dli_sum = 0.0f;
        photoperiod_sum = 0.0f;
        temp_optimality_sum = 0.0f;
        stress_sum = 0.0f;
    }
    __syncthreads();
    
    // Calculate growth metrics
    for (int i = tid; i < n; i += blockDim.x) {
        // Growing Degree Days
        float gdd = fmaxf(0.0f, temperature[i] - base_temp);
        atomicAdd(&gdd_sum, gdd);
        
        // Daily Light Integral (assuming hourly data)
        float dli = light_intensity[i] * 3600.0f / 1000000.0f; // Convert to mol/m²/hour
        atomicAdd(&dli_sum, dli);
        
        // Photoperiod accumulation
        atomicAdd(&photoperiod_sum, photoperiod[i]);
        
        // Temperature optimality (0-1 score)
        float temp_opt = 1.0f - fabsf(temperature[i] - optimal_temp_day) / 10.0f;
        temp_opt = fmaxf(0.0f, fminf(1.0f, temp_opt));
        atomicAdd(&temp_optimality_sum, temp_opt);
        
        // Stress degree days (temperatures outside optimal range)
        float stress = 0.0f;
        if (temperature[i] < optimal_temp_night - 2.0f || 
            temperature[i] > optimal_temp_day + 2.0f) {
            stress = fabsf(temperature[i] - optimal_temp_day);
        }
        atomicAdd(&stress_sum, stress);
    }
    __syncthreads();
    
    if (tid == 0) {
        result->growing_degree_days = gdd_sum / 24.0f; // Daily accumulation
        result->daily_light_integral = dli_sum / 24.0f; // Daily total
        result->photoperiod_hours = photoperiod_sum / n;
        result->temperature_optimality = temp_optimality_sum / n;
        result->light_sufficiency = result->daily_light_integral / light_requirement;
        result->stress_degree_days = stress_sum / 24.0f;
        
        // Flowering signal for Kalanchoe (short day plant)
        result->flowering_signal = (result->photoperiod_hours <= 8.0f) ? 1.0f : 0.0f;
        
        // Expected growth rate based on conditions
        result->expected_growth_rate = result->temperature_optimality * 
                                      fminf(1.0f, result->light_sufficiency);
    }
}

// Kernel launcher functions (to be called from Rust)
extern "C" {
    void launch_extended_statistics_kernel(
        const float* data,
        int n,
        float data_min,
        float data_max,
        ExtendedStatistics* result,
        cudaStream_t stream
    );
    
    void launch_weather_coupling_kernel(
        const float* internal_temp,
        const float* external_temp,
        const float* solar_radiation,
        int n,
        WeatherCouplingFeatures* result,
        cudaStream_t stream
    );
    
    void launch_energy_features_kernel(
        const float* lamp_power,
        const float* heating_power,
        const float* energy_prices,
        int n,
        EnergyFeatures* result,
        cudaStream_t stream
    );
    
    void launch_growth_features_kernel(
        const float* temperature,
        const float* light_intensity,
        const float* photoperiod,
        int n,
        float base_temp,
        float optimal_temp_day,
        float optimal_temp_night,
        float light_requirement,
        GrowthFeatures* result,
        cudaStream_t stream
    );
}

// Implementation of launcher functions
void launch_extended_statistics_kernel(
    const float* data,
    int n,
    float data_min,
    float data_max,
    ExtendedStatistics* result,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid(1);
    
    // Initialize result structure
    cudaMemsetAsync(result, 0, sizeof(ExtendedStatistics), stream);
    
    // Launch percentiles kernel
    compute_percentiles_kernel<<<grid, block, 256 * sizeof(int), stream>>>(
        data, n, data_min, data_max, result
    );
    
    // Calculate basic statistics first (mean, std) - assumed to be done elsewhere
    // Then launch moments kernel
    // compute_moments_kernel<<<grid, block, 256 * sizeof(float) * 2, stream>>>(
    //     data, n, mean, std, result
    // );
}

void launch_weather_coupling_kernel(
    const float* internal_temp,
    const float* external_temp,
    const float* solar_radiation,
    int n,
    WeatherCouplingFeatures* result,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    
    compute_weather_coupling_kernel<<<grid, block, 0, stream>>>(
        internal_temp, external_temp, solar_radiation, n, result
    );
}

void launch_energy_features_kernel(
    const float* lamp_power,
    const float* heating_power,
    const float* energy_prices,
    int n,
    EnergyFeatures* result,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid(1);
    
    compute_energy_features_kernel<<<grid, block, 0, stream>>>(
        lamp_power, heating_power, energy_prices, n, result
    );
}

void launch_growth_features_kernel(
    const float* temperature,
    const float* light_intensity,
    const float* photoperiod,
    int n,
    float base_temp,
    float optimal_temp_day,
    float optimal_temp_night,
    float light_requirement,
    GrowthFeatures* result,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid(1);
    
    compute_growth_features_kernel<<<grid, block, 0, stream>>>(
        temperature, light_intensity, photoperiod, n,
        base_temp, optimal_temp_day, optimal_temp_night, light_requirement,
        result
    );
}