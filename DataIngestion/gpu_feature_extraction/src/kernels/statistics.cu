// CUDA kernels for statistical feature extraction

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <math.h>

// Define CUDA math constants
#ifndef CUDART_INF_F
#define CUDART_INF_F __int_as_float(0x7f800000)
#endif

// Compute mean, std, min, max in a single pass
__global__ void compute_basic_stats(
    const float* __restrict__ data,
    const int* __restrict__ era_offsets,
    float* __restrict__ means,
    float* __restrict__ stds,
    float* __restrict__ mins,
    float* __restrict__ maxs,
    int n_eras
) {
    int era_idx = blockIdx.x;
    if (era_idx >= n_eras) return;
    
    int start = era_offsets[era_idx];
    int end = era_offsets[era_idx + 1];
    int n = end - start;
    
    // Shared memory for reduction
    extern __shared__ float sdata[];
    float* s_sum = sdata;
    float* s_sum2 = &sdata[blockDim.x];
    float* s_min = &sdata[2 * blockDim.x];
    float* s_max = &sdata[3 * blockDim.x];
    
    int tid = threadIdx.x;
    int stride = blockDim.x * gridDim.y;
    
    // Initialize
    float local_sum = 0.0f;
    float local_sum2 = 0.0f;
    float local_min = CUDART_INF_F;
    float local_max = -CUDART_INF_F;
    
    // Grid-stride loop
    for (int i = start + tid; i < end; i += stride) {
        float val = data[i];
        local_sum += val;
        local_sum2 += val * val;
        local_min = fminf(local_min, val);
        local_max = fmaxf(local_max, val);
    }
    
    // Store in shared memory
    s_sum[tid] = local_sum;
    s_sum2[tid] = local_sum2;
    s_min[tid] = local_min;
    s_max[tid] = local_max;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sum2[tid] += s_sum2[tid + s];
            s_min[tid] = fminf(s_min[tid], s_min[tid + s]);
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }
    
    // Write results
    if (tid == 0) {
        float mean = s_sum[0] / n;
        means[era_idx] = mean;
        stds[era_idx] = sqrtf(s_sum2[0] / n - mean * mean);
        mins[era_idx] = s_min[0];
        maxs[era_idx] = s_max[0];
    }
}

// Compute quantiles using sorting network
__global__ void compute_quantiles(
    const float* __restrict__ data,
    const int* __restrict__ era_offsets,
    float* __restrict__ q10,
    float* __restrict__ q25,
    float* __restrict__ q50,
    float* __restrict__ q75,
    float* __restrict__ q90,
    int n_eras,
    float* __restrict__ workspace
) {
    int era_idx = blockIdx.x;
    if (era_idx >= n_eras) return;
    
    int start = era_offsets[era_idx];
    int end = era_offsets[era_idx + 1];
    int n = end - start;
    
    // Copy data to workspace for sorting
    float* era_workspace = workspace + start;
    if (threadIdx.x == 0) {
        // Use CUB for efficient sorting
        size_t temp_storage_bytes = 0;
        cub::DeviceRadixSort::SortKeys(
            nullptr, temp_storage_bytes,
            &data[start], era_workspace, n
        );
        
        // Compute quantile indices
        int idx_10 = (int)(0.1f * n);
        int idx_25 = (int)(0.25f * n);
        int idx_50 = (int)(0.5f * n);
        int idx_75 = (int)(0.75f * n);
        int idx_90 = (int)(0.9f * n);
        
        // Store quantiles
        q10[era_idx] = era_workspace[idx_10];
        q25[era_idx] = era_workspace[idx_25];
        q50[era_idx] = era_workspace[idx_50];
        q75[era_idx] = era_workspace[idx_75];
        q90[era_idx] = era_workspace[idx_90];
    }
}

// Compute autocorrelation at different lags
__global__ void compute_autocorrelation(
    const float* __restrict__ data,
    const int* __restrict__ era_offsets,
    const float* __restrict__ means,
    float* __restrict__ ac_lag1,
    float* __restrict__ ac_lag5,
    float* __restrict__ ac_lag10,
    int n_eras
) {
    int era_idx = blockIdx.x;
    if (era_idx >= n_eras) return;
    
    int start = era_offsets[era_idx];
    int end = era_offsets[era_idx + 1];
    int n = end - start;
    float mean = means[era_idx];
    
    extern __shared__ float s_corr[];
    int tid = threadIdx.x;
    
    // Compute autocorrelation for lag 1
    float corr1 = 0.0f, var = 0.0f;
    for (int i = start + tid; i < end - 1; i += blockDim.x) {
        float x0 = data[i] - mean;
        float x1 = data[i + 1] - mean;
        corr1 += x0 * x1;
        var += x0 * x0;
    }
    
    // Reduce and store
    s_corr[tid] = corr1;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_corr[tid] += s_corr[tid + s];
        __syncthreads();
    }
    
    if (tid == 0) {
        ac_lag1[era_idx] = s_corr[0] / (var + 1e-8f);
    }
    
    // Similar for lag 5 and 10...
}