
/// Temporal dependency features (ACF, PACF, cross-correlation)
pub fn temporal_dependency_kernels() -> &'static str {
    r#"
extern "C" {

// Compute autocorrelation function (ACF) for multiple lags
__global__ void compute_acf(
    const float* __restrict__ signal,
    float* __restrict__ acf,
    const float* __restrict__ mean,    // Pre-computed mean
    const float* __restrict__ variance, // Pre-computed variance
    const int max_lag,
    const int n
) {
    const int lag = blockIdx.x;
    if (lag > max_lag) return;
    
    extern __shared__ float shared_sum[];
    const int tid = threadIdx.x;
    
    float local_sum = 0.0f;
    float signal_mean = mean[0];
    float signal_var = variance[0];
    
    // Each thread computes part of the sum for this lag
    for (int i = tid; i < n - lag; i += blockDim.x) {
        local_sum += (signal[i] - signal_mean) * (signal[i + lag] - signal_mean);
    }
    
    shared_sum[tid] = local_sum;
    __syncthreads();
    
    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    // Normalize and store result
    if (tid == 0) {
        float covariance = shared_sum[0] / (n - lag);
        acf[lag] = covariance / signal_var;  // Normalize by variance
    }
}

// Compute partial autocorrelation function (PACF) using Durbin-Levinson
__global__ void compute_pacf_durbin(
    const float* __restrict__ acf,
    float* __restrict__ pacf,
    const int max_lag
) {
    // This is a sequential algorithm, so we use a single thread
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    
    // Durbin-Levinson algorithm
    float phi[64][64];  // Max lag = 64
    
    // PACF at lag 0 is always 1
    pacf[0] = 1.0f;
    
    // PACF at lag 1 equals ACF at lag 1
    if (max_lag >= 1) {
        pacf[1] = acf[1];
        phi[1][1] = acf[1];
    }
    
    // Compute PACF for lags 2 to max_lag
    for (int k = 2; k <= max_lag && k < 64; k++) {
        float numerator = acf[k];
        float denominator = 1.0f;
        
        // Compute numerator and denominator
        for (int j = 1; j < k; j++) {
            numerator -= phi[k-1][j] * acf[k-j];
            denominator -= phi[k-1][j] * acf[j];
        }
        
        // PACF at lag k
        if (fabsf(denominator) > 1e-10f) {
            phi[k][k] = numerator / denominator;
            pacf[k] = phi[k][k];
            
            // Update phi values
            for (int j = 1; j < k; j++) {
                phi[k][j] = phi[k-1][j] - phi[k][k] * phi[k-1][k-j];
            }
        } else {
            pacf[k] = 0.0f;
        }
    }
}

// Cross-correlation between two signals
__global__ void compute_cross_correlation(
    const float* __restrict__ signal_x,
    const float* __restrict__ signal_y,
    float* __restrict__ xcorr,
    const float* __restrict__ mean_x,
    const float* __restrict__ mean_y,
    const float* __restrict__ std_x,
    const float* __restrict__ std_y,
    const int max_lag,
    const int n
) {
    const int lag_idx = blockIdx.x;
    const int lag = lag_idx - max_lag;  // Negative to positive lags
    
    if (abs(lag) > max_lag) return;
    
    extern __shared__ float shared_sum[];
    const int tid = threadIdx.x;
    
    float local_sum = 0.0f;
    float mx = mean_x[0];
    float my = mean_y[0];
    
    // Compute cross-correlation for this lag
    if (lag >= 0) {
        // Positive lag: y leads x
        for (int i = tid; i < n - lag; i += blockDim.x) {
            local_sum += (signal_x[i] - mx) * (signal_y[i + lag] - my);
        }
    } else {
        // Negative lag: x leads y
        int abs_lag = -lag;
        for (int i = tid; i < n - abs_lag; i += blockDim.x) {
            local_sum += (signal_x[i + abs_lag] - mx) * (signal_y[i] - my);
        }
    }
    
    shared_sum[tid] = local_sum;
    __syncthreads();
    
    // Reduce
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    // Normalize and store
    if (tid == 0) {
        int effective_n = n - abs(lag);
        float correlation = shared_sum[0] / (effective_n * std_x[0] * std_y[0]);
        xcorr[lag_idx] = correlation;
    }
}

// Find lag of maximum cross-correlation
__global__ void find_max_xcorr_lag(
    const float* __restrict__ xcorr,
    int* __restrict__ max_lag,
    float* __restrict__ max_value,
    const int num_lags
) {
    extern __shared__ float shared_max[];
    extern __shared__ int shared_idx[];
    
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    
    float local_max = -2.0f;  // Correlation ranges from -1 to 1
    int local_idx = -1;
    
    // Find local maximum
    for (int i = gid; i < num_lags; i += gridDim.x * blockDim.x) {
        float val = fabsf(xcorr[i]);  // Consider absolute value
        if (val > local_max) {
            local_max = val;
            local_idx = i;
        }
    }
    
    shared_max[tid] = local_max;
    shared_idx[tid] = local_idx;
    __syncthreads();
    
    // Reduce
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_max[tid + stride] > shared_max[tid]) {
                shared_max[tid] = shared_max[tid + stride];
                shared_idx[tid] = shared_idx[tid + stride];
            }
        }
        __syncthreads();
    }
    
    if (tid == 0 && blockIdx.x == 0) {
        int idx = shared_idx[0];
        max_lag[0] = idx - num_lags / 2;  // Convert back to actual lag
        max_value[0] = (idx >= 0) ? xcorr[idx] : 0.0f;
    }
}

// Compute lagged features (values at specific lags)
__global__ void compute_lagged_features(
    const float* __restrict__ signal,
    float* __restrict__ lagged_values,
    const int* __restrict__ lags,
    const int num_lags,
    const int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lag_idx = tid / n;
    const int time_idx = tid % n;
    
    if (lag_idx < num_lags && time_idx < n) {
        int lag = lags[lag_idx];
        int source_idx = time_idx - lag;
        
        if (source_idx >= 0 && source_idx < n) {
            lagged_values[lag_idx * n + time_idx] = signal[source_idx];
        } else {
            lagged_values[lag_idx * n + time_idx] = 0.0f;  // Padding
        }
    }
}

// Compute time-delayed mutual information (nonlinear dependency)
__global__ void compute_delayed_mutual_info(
    const float* __restrict__ signal_x,
    const float* __restrict__ signal_y,
    float* __restrict__ mi,
    const int delay,
    const int n_bins,
    const int n
) {
    extern __shared__ int shared_hist[];
    
    const int tid = threadIdx.x;
    const int effective_n = n - delay;
    
    // Initialize 2D histogram in shared memory
    for (int i = tid; i < n_bins * n_bins; i += blockDim.x) {
        shared_hist[i] = 0;
    }
    __syncthreads();
    
    // Find min/max for binning
    float min_x = CUDART_INF_F, max_x = -CUDART_INF_F;
    float min_y = CUDART_INF_F, max_y = -CUDART_INF_F;
    
    for (int i = tid; i < effective_n; i += blockDim.x) {
        float x = signal_x[i];
        float y = signal_y[i + delay];
        
        min_x = fminf(min_x, x);
        max_x = fmaxf(max_x, x);
        min_y = fminf(min_y, y);
        max_y = fmaxf(max_y, y);
    }
    
    // Reduce min/max (simplified - in practice use proper reduction)
    __syncthreads();
    
    // Build joint histogram
    float range_x = max_x - min_x + 1e-10f;
    float range_y = max_y - min_y + 1e-10f;
    
    for (int i = tid; i < effective_n; i += blockDim.x) {
        float x = signal_x[i];
        float y = signal_y[i + delay];
        
        int bin_x = (int)((x - min_x) / range_x * (n_bins - 1));
        int bin_y = (int)((y - min_y) / range_y * (n_bins - 1));
        
        bin_x = min(max(bin_x, 0), n_bins - 1);
        bin_y = min(max(bin_y, 0), n_bins - 1);
        
        atomicAdd(&shared_hist[bin_y * n_bins + bin_x], 1);
    }
    __syncthreads();
    
    // Compute mutual information
    if (tid == 0) {
        float mi_value = 0.0f;
        
        // Marginal distributions
        int hist_x[32], hist_y[32];  // Assuming n_bins <= 32
        
        for (int i = 0; i < n_bins; i++) {
            hist_x[i] = 0;
            hist_y[i] = 0;
            for (int j = 0; j < n_bins; j++) {
                hist_x[i] += shared_hist[j * n_bins + i];
                hist_y[i] += shared_hist[i * n_bins + j];
            }
        }
        
        // Compute MI
        for (int i = 0; i < n_bins; i++) {
            for (int j = 0; j < n_bins; j++) {
                int joint = shared_hist[j * n_bins + i];
                if (joint > 0) {
                    float p_xy = (float)joint / effective_n;
                    float p_x = (float)hist_x[i] / effective_n;
                    float p_y = (float)hist_y[j] / effective_n;
                    
                    if (p_x > 0 && p_y > 0) {
                        mi_value += p_xy * log2f(p_xy / (p_x * p_y));
                    }
                }
            }
        }
        
        mi[0] = mi_value;
    }
}

} // extern "C"
"#
}