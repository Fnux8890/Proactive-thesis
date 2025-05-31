// Enhanced CUDA kernels for statistical feature extraction
// Uses more shared memory for better performance and more features

#ifndef CUDART_INF_F
#define CUDART_INF_F __int_as_float(0x7f800000)
#endif

extern "C" {

// Enhanced warp stats structure for shared memory
struct WarpStats {
    float sum;          // sum of values
    float sum_sq;       // sum of squared values (for variance)
    float sum_cube;     // sum of cubed values (for skewness)
    float sum_quad;     // sum of quad values (for kurtosis)
    float min_val;      // minimum value
    float max_val;      // maximum value
    float count;        // number of values processed
    int min_idx;        // index of minimum value
    int max_idx;        // index of maximum value
};

// Extended statistical features structure
struct StatisticalFeaturesExtended {
    float mean;
    float std;
    float min;
    float max;
    float skewness;
    float kurtosis;
    float q25;          // 25th percentile
    float q50;          // median
    float q75;          // 75th percentile
    float range;        // max - min
    float cv;           // coefficient of variation
    float iqr;          // interquartile range
    float energy;       // sum of squares
    float entropy;      // approximate entropy
    int min_idx;        // position of minimum
    int max_idx;        // position of maximum
};

// Warp reduction utilities with full stats
__device__ WarpStats warp_reduce_stats(WarpStats val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val.sum += __shfl_down_sync(0xffffffff, val.sum, offset);
        val.sum_sq += __shfl_down_sync(0xffffffff, val.sum_sq, offset);
        val.sum_cube += __shfl_down_sync(0xffffffff, val.sum_cube, offset);
        val.sum_quad += __shfl_down_sync(0xffffffff, val.sum_quad, offset);
        val.count += __shfl_down_sync(0xffffffff, val.count, offset);
        
        float other_min = __shfl_down_sync(0xffffffff, val.min_val, offset);
        int other_min_idx = __shfl_down_sync(0xffffffff, val.min_idx, offset);
        if (other_min < val.min_val) {
            val.min_val = other_min;
            val.min_idx = other_min_idx;
        }
        
        float other_max = __shfl_down_sync(0xffffffff, val.max_val, offset);
        int other_max_idx = __shfl_down_sync(0xffffffff, val.max_idx, offset);
        if (other_max > val.max_val) {
            val.max_val = other_max;
            val.max_idx = other_max_idx;
        }
    }
    return val;
}

// Enhanced statistical computation kernel
__global__ void compute_statistics_extended(
    const float* __restrict__ input,
    StatisticalFeaturesExtended* __restrict__ output,
    const unsigned int n
) {
    extern __shared__ char shared_mem[];
    
    // Partition shared memory
    WarpStats* warp_stats = (WarpStats*)shared_mem;
    unsigned int* histogram = (unsigned int*)&shared_mem[(blockDim.x / 32) * sizeof(WarpStats)];
    
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    const int lane = tid % 32;
    const int warp_id = tid / 32;
    const int num_warps = (blockDim.x + 31) / 32;
    
    // Initialize histogram bins (256 bins for 8-bit resolution)
    const int num_bins = 256;
    if (tid < num_bins) {
        histogram[tid] = 0;
    }
    __syncthreads();
    
    // Initialize warp accumulator
    WarpStats stats = {0.0f, 0.0f, 0.0f, 0.0f, CUDART_INF_F, -CUDART_INF_F, 0.0f, -1, -1};
    
    // First pass: compute all statistics and build histogram
    float global_min = CUDART_INF_F;
    float global_max = -CUDART_INF_F;
    
    // Pre-scan for range (needed for histogram binning)
    for (int i = gid; i < n; i += gridDim.x * blockDim.x) {
        float val = input[i];
        global_min = fminf(global_min, val);
        global_max = fmaxf(global_max, val);
    }
    
    // Reduce min/max across block
    __shared__ float s_min, s_max;
    if (tid == 0) {
        s_min = CUDART_INF_F;
        s_max = -CUDART_INF_F;
    }
    __syncthreads();
    atomicMin((int*)&s_min, __float_as_int(global_min));
    atomicMax((int*)&s_max, __float_as_int(global_max));
    __syncthreads();
    
    float range = s_max - s_min;
    float bin_width = range / num_bins;
    
    // Main statistics computation pass
    for (int i = gid; i < n; i += gridDim.x * blockDim.x) {
        float val = input[i];
        float normalized = val - s_min;  // Shift to start at 0
        
        // Update statistics
        stats.sum += val;
        stats.sum_sq += val * val;
        stats.sum_cube += val * val * val;
        stats.sum_quad += val * val * val * val;
        stats.count += 1.0f;
        
        if (val < stats.min_val) {
            stats.min_val = val;
            stats.min_idx = i;
        }
        if (val > stats.max_val) {
            stats.max_val = val;
            stats.max_idx = i;
        }
        
        // Update histogram
        if (bin_width > 0) {
            int bin = min((int)(normalized / bin_width), num_bins - 1);
            atomicAdd(&histogram[bin], 1);
        }
    }
    
    // Warp-level reduction
    stats = warp_reduce_stats(stats);
    
    // Store warp results in shared memory
    if (lane == 0 && warp_id < num_warps) {
        warp_stats[warp_id] = stats;
    }
    __syncthreads();
    
    // Final reduction and percentile calculation by first warp
    if (warp_id == 0) {
        WarpStats final_stats = {0.0f, 0.0f, 0.0f, 0.0f, CUDART_INF_F, -CUDART_INF_F, 0.0f, -1, -1};
        
        if (lane < num_warps) {
            final_stats = warp_stats[lane];
        }
        
        final_stats = warp_reduce_stats(final_stats);
        
        if (lane == 0) {
            // Calculate moments
            float mean = final_stats.sum / n;
            float variance = (final_stats.sum_sq / n) - (mean * mean);
            float std = sqrtf(fmaxf(variance, 0.0f));
            
            // Calculate standardized moments for skewness and kurtosis
            float skewness = 0.0f;
            float kurtosis = 0.0f;
            if (std > 0.0f) {
                float n_float = (float)n;
                float m3 = (final_stats.sum_cube / n) - 3.0f * mean * variance - mean * mean * mean;
                float m4 = (final_stats.sum_quad / n) - 4.0f * mean * (final_stats.sum_cube / n) + 
                          6.0f * mean * mean * variance + 3.0f * mean * mean * mean * mean;
                
                skewness = m3 / (std * std * std);
                kurtosis = m4 / (std * std * std * std) - 3.0f;  // Excess kurtosis
            }
            
            // Calculate percentiles from histogram
            unsigned int q25_count = n / 4;
            unsigned int q50_count = n / 2;
            unsigned int q75_count = 3 * n / 4;
            
            unsigned int cumsum = 0;
            float q25_val = s_min, q50_val = s_min, q75_val = s_min;
            
            for (int i = 0; i < num_bins; i++) {
                cumsum += histogram[i];
                if (cumsum >= q25_count && q25_val == s_min) {
                    q25_val = s_min + (i + 0.5f) * bin_width;
                }
                if (cumsum >= q50_count && q50_val == s_min) {
                    q50_val = s_min + (i + 0.5f) * bin_width;
                }
                if (cumsum >= q75_count && q75_val == s_min) {
                    q75_val = s_min + (i + 0.5f) * bin_width;
                    break;
                }
            }
            
            // Calculate entropy from histogram
            float entropy = 0.0f;
            for (int i = 0; i < num_bins; i++) {
                if (histogram[i] > 0) {
                    float p = (float)histogram[i] / n;
                    entropy -= p * log2f(p);
                }
            }
            
            // Fill output structure
            output->mean = mean;
            output->std = std;
            output->min = final_stats.min_val;
            output->max = final_stats.max_val;
            output->skewness = skewness;
            output->kurtosis = kurtosis;
            output->q25 = q25_val;
            output->q50 = q50_val;
            output->q75 = q75_val;
            output->range = final_stats.max_val - final_stats.min_val;
            output->cv = (mean != 0.0f) ? (std / fabsf(mean)) : 0.0f;
            output->iqr = q75_val - q25_val;
            output->energy = final_stats.sum_sq;
            output->entropy = entropy;
            output->min_idx = final_stats.min_idx;
            output->max_idx = final_stats.max_idx;
        }
    }
}

// Compute cross-correlation between two signals
__global__ void compute_cross_correlation(
    const float* __restrict__ signal1,
    const float* __restrict__ signal2,
    float* __restrict__ correlation,
    const unsigned int n,
    const int max_lag
) {
    extern __shared__ float s_cache[];
    
    int tid = threadIdx.x;
    int lag = blockIdx.x - max_lag;  // Lag from -max_lag to +max_lag
    
    if (abs(lag) > max_lag) return;
    
    float sum_xy = 0.0f;
    float sum_x = 0.0f, sum_y = 0.0f;
    float sum_x2 = 0.0f, sum_y2 = 0.0f;
    int count = 0;
    
    // Compute correlation for this lag
    int start = max(0, -lag);
    int end = min(n, n - lag);
    
    for (int i = start + tid; i < end; i += blockDim.x) {
        float x = signal1[i];
        float y = signal2[i + lag];
        
        sum_xy += x * y;
        sum_x += x;
        sum_y += y;
        sum_x2 += x * x;
        sum_y2 += y * y;
        count++;
    }
    
    // Reduce within block
    s_cache[tid] = sum_xy;
    s_cache[tid + blockDim.x] = sum_x;
    s_cache[tid + 2 * blockDim.x] = sum_y;
    s_cache[tid + 3 * blockDim.x] = sum_x2;
    s_cache[tid + 4 * blockDim.x] = sum_y2;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_cache[tid] += s_cache[tid + s];
            s_cache[tid + blockDim.x] += s_cache[tid + blockDim.x + s];
            s_cache[tid + 2 * blockDim.x] += s_cache[tid + 2 * blockDim.x + s];
            s_cache[tid + 3 * blockDim.x] += s_cache[tid + 3 * blockDim.x + s];
            s_cache[tid + 4 * blockDim.x] += s_cache[tid + 4 * blockDim.x + s];
        }
        __syncthreads();
    }
    
    if (tid == 0 && count > 0) {
        float n_float = (float)count;
        float mean_x = s_cache[blockDim.x] / n_float;
        float mean_y = s_cache[2 * blockDim.x] / n_float;
        
        float cov = (s_cache[0] / n_float) - (mean_x * mean_y);
        float std_x = sqrtf((s_cache[3 * blockDim.x] / n_float) - (mean_x * mean_x));
        float std_y = sqrtf((s_cache[4 * blockDim.x] / n_float) - (mean_y * mean_y));
        
        correlation[blockIdx.x] = (std_x > 0 && std_y > 0) ? (cov / (std_x * std_y)) : 0.0f;
    }
}

} // extern "C"