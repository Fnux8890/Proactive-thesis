
/// Frequency domain features using FFT
pub fn frequency_domain_kernels() -> &'static str {
    r#"
extern "C" {

#include <cufft.h>

// Compute spectral power in frequency bands
__global__ void compute_spectral_bands(
    const float* __restrict__ fft_magnitude,  // |FFT|^2
    float* __restrict__ band_power,           // Output: power in each band
    const int* __restrict__ band_limits,      // Band boundaries in bins
    const int num_bands,
    const int fft_size
) {
    extern __shared__ float shared_power[];
    
    const int tid = threadIdx.x;
    const int band_id = blockIdx.x;
    
    if (band_id >= num_bands) return;
    
    // Get band boundaries
    int start_bin = band_limits[band_id];
    int end_bin = band_limits[band_id + 1];
    
    // Each thread accumulates power for a subset of bins
    float local_power = 0.0f;
    for (int bin = start_bin + tid; bin < end_bin; bin += blockDim.x) {
        if (bin < fft_size / 2) {  // Only positive frequencies
            local_power += fft_magnitude[bin];
        }
    }
    
    // Store in shared memory
    shared_power[tid] = local_power;
    __syncthreads();
    
    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_power[tid] += shared_power[tid + stride];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        band_power[band_id] = shared_power[0];
    }
}

// Compute spectral centroid (center of mass of spectrum)
__global__ void compute_spectral_centroid(
    const float* __restrict__ fft_magnitude,
    float* __restrict__ centroid,
    const float sample_rate,
    const int fft_size
) {
    extern __shared__ float shared_data[];
    float* shared_weighted = shared_data;
    float* shared_magnitude = &shared_data[blockDim.x];
    
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    const int half_fft = fft_size / 2;
    
    float local_weighted = 0.0f;
    float local_magnitude = 0.0f;
    
    // Each thread processes multiple bins
    for (int bin = gid; bin < half_fft; bin += gridDim.x * blockDim.x) {
        float mag = fft_magnitude[bin];
        float freq = (float)bin * sample_rate / fft_size;
        
        local_weighted += freq * mag;
        local_magnitude += mag;
    }
    
    // Store in shared memory
    shared_weighted[tid] = local_weighted;
    shared_magnitude[tid] = local_magnitude;
    __syncthreads();
    
    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_weighted[tid] += shared_weighted[tid + stride];
            shared_magnitude[tid] += shared_magnitude[tid + stride];
        }
        __syncthreads();
    }
    
    // Final result from first block
    if (tid == 0 && blockIdx.x == 0) {
        float total_weighted = shared_weighted[0];
        float total_magnitude = shared_magnitude[0];
        
        if (total_magnitude > 0.0f) {
            centroid[0] = total_weighted / total_magnitude;
        } else {
            centroid[0] = 0.0f;
        }
    }
}

// Find peak frequency and its magnitude
__global__ void find_peak_frequency(
    const float* __restrict__ fft_magnitude,
    float* __restrict__ peak_freq,
    float* __restrict__ peak_magnitude,
    const float sample_rate,
    const int fft_size,
    const float min_freq,  // Ignore DC and very low frequencies
    const float max_freq   // Ignore above Nyquist or specified max
) {
    extern __shared__ float shared_max[];
    extern __shared__ int shared_idx[];
    
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    const int half_fft = fft_size / 2;
    
    // Frequency bounds in bins
    int min_bin = (int)(min_freq * fft_size / sample_rate);
    int max_bin = (int)(max_freq * fft_size / sample_rate);
    max_bin = min(max_bin, half_fft);
    
    float local_max = -1.0f;
    int local_idx = -1;
    
    // Find local maximum
    for (int bin = gid; bin < max_bin; bin += gridDim.x * blockDim.x) {
        if (bin >= min_bin) {
            float mag = fft_magnitude[bin];
            if (mag > local_max) {
                local_max = mag;
                local_idx = bin;
            }
        }
    }
    
    // Store in shared memory
    shared_max[tid] = local_max;
    shared_idx[tid] = local_idx;
    __syncthreads();
    
    // Reduce to find maximum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_max[tid + stride] > shared_max[tid]) {
                shared_max[tid] = shared_max[tid + stride];
                shared_idx[tid] = shared_idx[tid + stride];
            }
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0 && blockIdx.x == 0) {
        if (shared_idx[0] >= 0) {
            peak_freq[0] = (float)shared_idx[0] * sample_rate / fft_size;
            peak_magnitude[0] = shared_max[0];
        } else {
            peak_freq[0] = 0.0f;
            peak_magnitude[0] = 0.0f;
        }
    }
}

// Compute spectral entropy
__global__ void compute_spectral_entropy(
    const float* __restrict__ fft_magnitude,
    float* __restrict__ entropy,
    const int fft_size
) {
    extern __shared__ float shared_entropy[];
    
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    const int half_fft = fft_size / 2;
    
    float local_sum = 0.0f;
    float local_entropy = 0.0f;
    
    // First pass: compute total energy
    for (int bin = gid; bin < half_fft; bin += gridDim.x * blockDim.x) {
        local_sum += fft_magnitude[bin];
    }
    
    // Reduce to get total
    __shared__ float total_energy;
    if (tid == 0) total_energy = 0.0f;
    __syncthreads();
    atomicAdd(&total_energy, local_sum);
    __syncthreads();
    
    // Second pass: compute entropy
    if (total_energy > 0.0f) {
        for (int bin = gid; bin < half_fft; bin += gridDim.x * blockDim.x) {
            float p = fft_magnitude[bin] / total_energy;
            if (p > 1e-10f) {
                local_entropy -= p * log2f(p);
            }
        }
    }
    
    shared_entropy[tid] = local_entropy;
    __syncthreads();
    
    // Reduce entropy
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_entropy[tid] += shared_entropy[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0 && blockIdx.x == 0) {
        entropy[0] = shared_entropy[0];
    }
}

// Compute spectral rolloff (frequency below which X% of energy is contained)
__global__ void compute_spectral_rolloff(
    const float* __restrict__ fft_magnitude,
    float* __restrict__ rolloff_freq,
    const float rolloff_percent,  // e.g., 0.85 for 85%
    const float sample_rate,
    const int fft_size
) {
    const int half_fft = fft_size / 2;
    
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // Sequential scan to find rolloff point
        float total_energy = 0.0f;
        
        // First compute total energy
        for (int i = 0; i < half_fft; i++) {
            total_energy += fft_magnitude[i];
        }
        
        // Find rolloff frequency
        float threshold = total_energy * rolloff_percent;
        float cumulative = 0.0f;
        int rolloff_bin = 0;
        
        for (int i = 0; i < half_fft; i++) {
            cumulative += fft_magnitude[i];
            if (cumulative >= threshold) {
                rolloff_bin = i;
                break;
            }
        }
        
        rolloff_freq[0] = (float)rolloff_bin * sample_rate / fft_size;
    }
}

// Compute spectral flux (change in spectrum over time)
__global__ void compute_spectral_flux(
    const float* __restrict__ fft_magnitude_current,
    const float* __restrict__ fft_magnitude_previous,
    float* __restrict__ flux,
    const int fft_size
) {
    extern __shared__ float shared_flux[];
    
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    const int half_fft = fft_size / 2;
    
    float local_flux = 0.0f;
    
    // Compute flux for assigned bins
    for (int bin = gid; bin < half_fft; bin += gridDim.x * blockDim.x) {
        float diff = fft_magnitude_current[bin] - fft_magnitude_previous[bin];
        // Only consider positive differences (onset detection)
        if (diff > 0.0f) {
            local_flux += diff * diff;
        }
    }
    
    shared_flux[tid] = local_flux;
    __syncthreads();
    
    // Reduce
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_flux[tid] += shared_flux[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0 && blockIdx.x == 0) {
        flux[0] = sqrtf(shared_flux[0]);
    }
}

} // extern "C"
"#
}