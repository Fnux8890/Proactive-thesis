
/// Extended rolling and cumulative statistics (full 34 features)
pub fn rolling_statistics_extended_kernels() -> &'static str {
    r#"
extern "C" {

// Percentile calculation using partial sorting
__global__ void rolling_percentiles(
    const float* __restrict__ signal,
    float* __restrict__ p10,
    float* __restrict__ p25,
    float* __restrict__ p50,
    float* __restrict__ p75,
    float* __restrict__ p90,
    const int window_size,
    const int n
) {
    extern __shared__ float shared_window[];
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n - window_size + 1) {
        // Copy window to shared memory
        for (int i = threadIdx.x; i < window_size; i += blockDim.x) {
            if (i < window_size) {
                shared_window[i] = signal[tid + i];
            }
        }
        __syncthreads();
        
        // Simple sorting for small windows (bubble sort is fine for small windows)
        if (threadIdx.x == 0) {
            for (int i = 0; i < window_size - 1; i++) {
                for (int j = 0; j < window_size - i - 1; j++) {
                    if (shared_window[j] > shared_window[j + 1]) {
                        float temp = shared_window[j];
                        shared_window[j] = shared_window[j + 1];
                        shared_window[j + 1] = temp;
                    }
                }
            }
            
            // Extract percentiles
            p10[tid] = shared_window[(int)(0.10f * window_size)];
            p25[tid] = shared_window[(int)(0.25f * window_size)];
            p50[tid] = shared_window[(int)(0.50f * window_size)];
            p75[tid] = shared_window[(int)(0.75f * window_size)];
            p90[tid] = shared_window[(int)(0.90f * window_size)];
        }
    }
}

// Heating/Cooling Degree Hours
__global__ void compute_degree_hours(
    const float* __restrict__ temperature,
    float* __restrict__ hdd,  // Heating degree hours
    float* __restrict__ cdd,  // Cooling degree hours
    const float heating_base,  // e.g., 18°C
    const float cooling_base,  // e.g., 24°C
    const int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        float temp = temperature[tid];
        
        // HDD: how much below heating base
        hdd[tid] = fmaxf(heating_base - temp, 0.0f);
        
        // CDD: how much above cooling base
        cdd[tid] = fmaxf(temp - cooling_base, 0.0f);
    }
}

// Rolling range (max - min)
__global__ void rolling_range(
    const float* __restrict__ signal,
    float* __restrict__ range,
    const int window_size,
    const int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n - window_size + 1) {
        float min_val = CUDART_INF_F;
        float max_val = -CUDART_INF_F;
        
        for (int i = 0; i < window_size; i++) {
            float val = signal[tid + i];
            min_val = fminf(min_val, val);
            max_val = fmaxf(max_val, val);
        }
        
        range[tid] = max_val - min_val;
    }
}

// Rolling skewness
__global__ void rolling_skewness(
    const float* __restrict__ signal,
    float* __restrict__ skewness,
    const int window_size,
    const int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n - window_size + 1) {
        // First pass: compute mean
        float sum = 0.0f;
        for (int i = 0; i < window_size; i++) {
            sum += signal[tid + i];
        }
        float mean = sum / window_size;
        
        // Second pass: compute moments
        float m2 = 0.0f, m3 = 0.0f;
        for (int i = 0; i < window_size; i++) {
            float diff = signal[tid + i] - mean;
            float diff2 = diff * diff;
            m2 += diff2;
            m3 += diff2 * diff;
        }
        
        m2 /= window_size;
        m3 /= window_size;
        
        if (m2 > 1e-10f) {
            skewness[tid] = m3 / powf(m2, 1.5f);
        } else {
            skewness[tid] = 0.0f;
        }
    }
}

// Rolling kurtosis
__global__ void rolling_kurtosis(
    const float* __restrict__ signal,
    float* __restrict__ kurtosis,
    const int window_size,
    const int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n - window_size + 1) {
        // First pass: compute mean
        float sum = 0.0f;
        for (int i = 0; i < window_size; i++) {
            sum += signal[tid + i];
        }
        float mean = sum / window_size;
        
        // Second pass: compute moments
        float m2 = 0.0f, m4 = 0.0f;
        for (int i = 0; i < window_size; i++) {
            float diff = signal[tid + i] - mean;
            float diff2 = diff * diff;
            m2 += diff2;
            m4 += diff2 * diff2;
        }
        
        m2 /= window_size;
        m4 /= window_size;
        
        if (m2 > 1e-10f) {
            kurtosis[tid] = m4 / (m2 * m2) - 3.0f;  // Excess kurtosis
        } else {
            kurtosis[tid] = 0.0f;
        }
    }
}

// Rolling coefficient of variation
__global__ void rolling_cv(
    const float* __restrict__ signal,
    float* __restrict__ cv,
    const int window_size,
    const int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n - window_size + 1) {
        float sum = 0.0f, sum_sq = 0.0f;
        
        for (int i = 0; i < window_size; i++) {
            float val = signal[tid + i];
            sum += val;
            sum_sq += val * val;
        }
        
        float mean = sum / window_size;
        float variance = (sum_sq / window_size) - (mean * mean);
        
        if (fabsf(mean) > 1e-10f && variance > 0.0f) {
            cv[tid] = sqrtf(variance) / fabsf(mean);
        } else {
            cv[tid] = 0.0f;
        }
    }
}

// Rolling median absolute deviation (MAD)
__global__ void rolling_mad(
    const float* __restrict__ signal,
    const float* __restrict__ median,  // Pre-computed rolling median
    float* __restrict__ mad,
    const int window_size,
    const int n
) {
    extern __shared__ float shared_deviations[];
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n - window_size + 1) {
        float med = median[tid];
        
        // Compute absolute deviations
        for (int i = threadIdx.x; i < window_size; i += blockDim.x) {
            if (i < window_size) {
                shared_deviations[i] = fabsf(signal[tid + i] - med);
            }
        }
        __syncthreads();
        
        // Find median of deviations (simplified for small windows)
        if (threadIdx.x == 0) {
            // Sort deviations
            for (int i = 0; i < window_size - 1; i++) {
                for (int j = 0; j < window_size - i - 1; j++) {
                    if (shared_deviations[j] > shared_deviations[j + 1]) {
                        float temp = shared_deviations[j];
                        shared_deviations[j] = shared_deviations[j + 1];
                        shared_deviations[j + 1] = temp;
                    }
                }
            }
            
            mad[tid] = shared_deviations[window_size / 2];
        }
    }
}

// Rolling interquartile range (IQR)
__global__ void rolling_iqr(
    const float* __restrict__ p25,
    const float* __restrict__ p75,
    float* __restrict__ iqr,
    const int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        iqr[tid] = p75[tid] - p25[tid];
    }
}

// Exponentially weighted moving average (EWMA)
__global__ void compute_ewma(
    const float* __restrict__ signal,
    float* __restrict__ ewma,
    const float alpha,  // Smoothing factor (0-1)
    const int n
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        ewma[0] = signal[0];
        
        for (int i = 1; i < n; i++) {
            ewma[i] = alpha * signal[i] + (1.0f - alpha) * ewma[i - 1];
        }
    }
}

// Double exponential smoothing (Holt's method)
__global__ void compute_double_exponential(
    const float* __restrict__ signal,
    float* __restrict__ level,
    float* __restrict__ trend,
    const float alpha,  // Level smoothing
    const float beta,   // Trend smoothing
    const int n
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // Initialize
        level[0] = signal[0];
        trend[0] = signal[1] - signal[0];
        
        for (int i = 1; i < n; i++) {
            float prev_level = level[i - 1];
            float prev_trend = trend[i - 1];
            
            level[i] = alpha * signal[i] + (1.0f - alpha) * (prev_level + prev_trend);
            trend[i] = beta * (level[i] - prev_level) + (1.0f - beta) * prev_trend;
        }
    }
}

// Rolling z-score normalization
__global__ void rolling_zscore(
    const float* __restrict__ signal,
    const float* __restrict__ rolling_mean,
    const float* __restrict__ rolling_std,
    float* __restrict__ zscore,
    const int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        float std = rolling_std[tid];
        if (std > 1e-10f) {
            zscore[tid] = (signal[tid] - rolling_mean[tid]) / std;
        } else {
            zscore[tid] = 0.0f;
        }
    }
}

// Cumulative product
__global__ void cumulative_product(
    const float* __restrict__ signal,
    float* __restrict__ cumprod,
    const int n
) {
    // Simple sequential implementation for now
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        cumprod[0] = signal[0];
        for (int i = 1; i < n; i++) {
            cumprod[i] = cumprod[i - 1] * signal[i];
        }
    }
}

// Rolling geometric mean
__global__ void rolling_geometric_mean(
    const float* __restrict__ signal,
    float* __restrict__ geom_mean,
    const int window_size,
    const int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n - window_size + 1) {
        float log_sum = 0.0f;
        int valid_count = 0;
        
        for (int i = 0; i < window_size; i++) {
            float val = signal[tid + i];
            if (val > 0.0f) {
                log_sum += logf(val);
                valid_count++;
            }
        }
        
        if (valid_count > 0) {
            geom_mean[tid] = expf(log_sum / valid_count);
        } else {
            geom_mean[tid] = 0.0f;
        }
    }
}

// Rolling harmonic mean
__global__ void rolling_harmonic_mean(
    const float* __restrict__ signal,
    float* __restrict__ harm_mean,
    const int window_size,
    const int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n - window_size + 1) {
        float reciprocal_sum = 0.0f;
        int valid_count = 0;
        
        for (int i = 0; i < window_size; i++) {
            float val = signal[tid + i];
            if (fabsf(val) > 1e-10f) {
                reciprocal_sum += 1.0f / val;
                valid_count++;
            }
        }
        
        if (valid_count > 0 && reciprocal_sum > 0.0f) {
            harm_mean[tid] = valid_count / reciprocal_sum;
        } else {
            harm_mean[tid] = 0.0f;
        }
    }
}

} // extern "C"
"#
}