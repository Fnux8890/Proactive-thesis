// CUDA kernels for growth and energy-aware features

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

extern "C" {

// Kernel for computing Growing Degree Days (GDD)
__global__ void compute_gdd_kernel(
    const float* temperature,
    float* gdd,
    float base_temp,
    int n
) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float local_sum = 0.0f;
    
    if (i < n && !isnan(temperature[i])) {
        float effective_temp = temperature[i] - base_temp;
        local_sum = fmaxf(effective_temp, 0.0f);
    }
    
    sdata[tid] = local_sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(gdd, sdata[0]);
    }
}

// Kernel for computing Daily Light Integral (DLI)
__global__ void compute_dli_kernel(
    const float* light_intensity,  // in µmol/m²/s
    const int* lamp_status,
    float* dli,                     // in mol/m²/day
    int n
) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float local_sum = 0.0f;
    
    if (i < n && !isnan(light_intensity[i]) && lamp_status[i] > 0) {
        // Convert µmol/m²/s to mol/m²/hour (multiply by 3600 / 1e6)
        local_sum = light_intensity[i] * 3600.0f / 1e6f;
    }
    
    sdata[tid] = local_sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(dli, sdata[0]);
    }
}

// Kernel for computing temperature differentials
__global__ void compute_temp_differential_kernel(
    const float* inside_temp,
    const float* outside_temp,
    float* temp_diff_mean,
    float* temp_diff_std,
    int n
) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float diff = 0.0f;
    float valid = 0.0f;
    
    if (i < n && !isnan(inside_temp[i]) && !isnan(outside_temp[i])) {
        diff = inside_temp[i] - outside_temp[i];
        valid = 1.0f;
    }
    
    sdata[tid] = diff;
    sdata[tid + blockDim.x] = valid;
    __syncthreads();
    
    // Reduction for sum and count
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata[tid + blockDim.x] += sdata[tid + s + blockDim.x];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(temp_diff_mean, sdata[0]);
        atomicAdd(temp_diff_std, sdata[blockDim.x]);  // Use as count for now
    }
}

// Kernel for computing energy cost features
__global__ void compute_energy_features_kernel(
    const float* energy_price,
    const float* power_consumption,  // Estimated from lamp status, ventilation, etc.
    float* total_cost,
    float* peak_hours,
    float* off_peak_hours,
    float price_threshold,
    int n
) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float cost = 0.0f;
    float is_peak = 0.0f;
    float is_off_peak = 0.0f;
    
    if (i < n && !isnan(energy_price[i]) && !isnan(power_consumption[i])) {
        cost = energy_price[i] * power_consumption[i];
        
        if (energy_price[i] > price_threshold) {
            is_peak = 1.0f;
        } else {
            is_off_peak = 1.0f;
        }
    }
    
    sdata[tid] = cost;
    sdata[tid + blockDim.x] = is_peak;
    sdata[tid + 2 * blockDim.x] = is_off_peak;
    __syncthreads();
    
    // Triple reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata[tid + blockDim.x] += sdata[tid + s + blockDim.x];
            sdata[tid + 2 * blockDim.x] += sdata[tid + s + 2 * blockDim.x];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(total_cost, sdata[0]);
        atomicAdd(peak_hours, sdata[blockDim.x]);
        atomicAdd(off_peak_hours, sdata[2 * blockDim.x]);
    }
}

// Kernel for computing solar efficiency
__global__ void compute_solar_efficiency_kernel(
    const float* solar_radiation,    // W/m²
    const float* light_intensity,    // µmol/m²/s inside greenhouse
    float* efficiency_ratio,
    int n
) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float ratio = 0.0f;
    float valid = 0.0f;
    
    if (i < n && !isnan(solar_radiation[i]) && !isnan(light_intensity[i]) 
        && solar_radiation[i] > 10.0f) {  // Only when there's meaningful solar radiation
        // Convert solar radiation to PAR (roughly 45% of total radiation)
        // Then convert W/m² to µmol/m²/s (factor ~4.6 for PAR)
        float solar_par = solar_radiation[i] * 0.45f * 4.6f;
        ratio = light_intensity[i] / solar_par;
        valid = 1.0f;
    }
    
    sdata[tid] = ratio;
    sdata[tid + blockDim.x] = valid;
    __syncthreads();
    
    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata[tid + blockDim.x] += sdata[tid + s + blockDim.x];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(efficiency_ratio, sdata[0]);
        atomicAdd(efficiency_ratio + 1, sdata[blockDim.x]);  // Store count in next position
    }
}

// Host function to compute growth and energy features
void compute_growth_energy_features(
    const float* d_temperature,
    const float* d_light_intensity,
    const int* d_lamp_status,
    const float* d_outside_temp,
    const float* d_solar_radiation,
    const float* d_energy_price,
    const float* d_power_consumption,
    int n,
    float base_temp,
    float price_threshold,
    float* gdd,
    float* dli,
    float* temp_diff_mean,
    float* temp_diff_std,
    float* total_energy_cost,
    float* peak_hours,
    float* off_peak_hours,
    float* solar_efficiency
) {
    // Allocate device memory for results
    float *d_gdd, *d_dli, *d_temp_diff_mean, *d_temp_diff_count;
    float *d_total_cost, *d_peak_hours, *d_off_peak_hours;
    float *d_efficiency;
    
    cudaMalloc(&d_gdd, sizeof(float));
    cudaMalloc(&d_dli, sizeof(float));
    cudaMalloc(&d_temp_diff_mean, sizeof(float));
    cudaMalloc(&d_temp_diff_count, sizeof(float));
    cudaMalloc(&d_total_cost, sizeof(float));
    cudaMalloc(&d_peak_hours, sizeof(float));
    cudaMalloc(&d_off_peak_hours, sizeof(float));
    cudaMalloc(&d_efficiency, 2 * sizeof(float));  // ratio sum and count
    
    // Initialize to zero
    cudaMemset(d_gdd, 0, sizeof(float));
    cudaMemset(d_dli, 0, sizeof(float));
    cudaMemset(d_temp_diff_mean, 0, sizeof(float));
    cudaMemset(d_temp_diff_count, 0, sizeof(float));
    cudaMemset(d_total_cost, 0, sizeof(float));
    cudaMemset(d_peak_hours, 0, sizeof(float));
    cudaMemset(d_off_peak_hours, 0, sizeof(float));
    cudaMemset(d_efficiency, 0, 2 * sizeof(float));
    
    // Configure kernel launch parameters
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    // Compute GDD
    compute_gdd_kernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(
        d_temperature, d_gdd, base_temp, n
    );
    
    // Compute DLI
    if (d_light_intensity && d_lamp_status) {
        compute_dli_kernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(
            d_light_intensity, d_lamp_status, d_dli, n
        );
    }
    
    // Compute temperature differential
    if (d_outside_temp) {
        compute_temp_differential_kernel<<<gridSize, blockSize, 2 * blockSize * sizeof(float)>>>(
            d_temperature, d_outside_temp, d_temp_diff_mean, d_temp_diff_count, n
        );
    }
    
    // Compute energy features
    if (d_energy_price && d_power_consumption) {
        compute_energy_features_kernel<<<gridSize, blockSize, 3 * blockSize * sizeof(float)>>>(
            d_energy_price, d_power_consumption, 
            d_total_cost, d_peak_hours, d_off_peak_hours,
            price_threshold, n
        );
    }
    
    // Compute solar efficiency
    if (d_solar_radiation && d_light_intensity) {
        compute_solar_efficiency_kernel<<<gridSize, blockSize, 2 * blockSize * sizeof(float)>>>(
            d_solar_radiation, d_light_intensity, d_efficiency, n
        );
    }
    
    // Copy results back and compute final values
    cudaMemcpy(gdd, d_gdd, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dli, d_dli, sizeof(float), cudaMemcpyDeviceToHost);
    
    float temp_diff_sum, temp_diff_n;
    cudaMemcpy(&temp_diff_sum, d_temp_diff_mean, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&temp_diff_n, d_temp_diff_count, sizeof(float), cudaMemcpyDeviceToHost);
    
    if (temp_diff_n > 0) {
        *temp_diff_mean = temp_diff_sum / temp_diff_n;
        // For std, would need a second pass with the mean
    }
    
    cudaMemcpy(total_energy_cost, d_total_cost, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(peak_hours, d_peak_hours, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(off_peak_hours, d_off_peak_hours, sizeof(float), cudaMemcpyDeviceToHost);
    
    float efficiency_sum, efficiency_count;
    cudaMemcpy(&efficiency_sum, d_efficiency, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&efficiency_count, d_efficiency + 1, sizeof(float), cudaMemcpyDeviceToHost);
    
    if (efficiency_count > 0) {
        *solar_efficiency = efficiency_sum / efficiency_count;
    }
    
    // Clean up
    cudaFree(d_gdd);
    cudaFree(d_dli);
    cudaFree(d_temp_diff_mean);
    cudaFree(d_temp_diff_count);
    cudaFree(d_total_cost);
    cudaFree(d_peak_hours);
    cudaFree(d_off_peak_hours);
    cudaFree(d_efficiency);
}

} // extern "C"