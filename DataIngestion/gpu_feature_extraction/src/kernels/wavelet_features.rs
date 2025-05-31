
/// Wavelet transform features for multi-resolution analysis
pub fn wavelet_kernels() -> &'static str {
    r#"
extern "C" {

// Daubechies 4 (db4) wavelet coefficients
#define H0  0.48296291314469025f
#define H1  0.83651630373746899f
#define H2  0.22414386804185735f
#define H3 -0.12940952255092145f

// Discrete Wavelet Transform (DWT) - single level decomposition
__global__ void dwt_decompose_db4(
    const float* __restrict__ signal,
    float* __restrict__ approx,      // Low-frequency coefficients
    float* __restrict__ detail,      // High-frequency coefficients
    const int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int half_n = n / 2;
    
    if (tid < half_n) {
        int idx = 2 * tid;
        
        // Approximation (low-pass filter)
        float a = 0.0f;
        a += H0 * signal[idx];
        a += H1 * signal[(idx + 1) % n];
        a += H2 * signal[(idx + 2) % n];
        a += H3 * signal[(idx + 3) % n];
        approx[tid] = a;
        
        // Detail (high-pass filter)
        float d = 0.0f;
        d += H3 * signal[idx];
        d -= H2 * signal[(idx + 1) % n];
        d += H1 * signal[(idx + 2) % n];
        d -= H0 * signal[(idx + 3) % n];
        detail[tid] = d;
    }
}

// Multi-level DWT energy computation
__global__ void compute_dwt_energy_levels(
    const float* __restrict__ coefficients,
    float* __restrict__ energy,
    const int* __restrict__ level_sizes,
    const int num_levels
) {
    extern __shared__ float shared_energy[];
    
    const int level = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (level >= num_levels) return;
    
    int start_idx = 0;
    for (int i = 0; i < level; i++) {
        start_idx += level_sizes[i];
    }
    
    int level_size = level_sizes[level];
    float local_energy = 0.0f;
    
    // Compute energy for this level
    for (int i = tid; i < level_size; i += blockDim.x) {
        float coeff = coefficients[start_idx + i];
        local_energy += coeff * coeff;
    }
    
    shared_energy[tid] = local_energy;
    __syncthreads();
    
    // Reduce
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_energy[tid] += shared_energy[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        energy[level] = shared_energy[0];
    }
}

// Wavelet packet decomposition energy
__global__ void compute_wavelet_packet_energy(
    const float* __restrict__ wp_coeffs,
    float* __restrict__ node_energy,
    const int* __restrict__ node_info,  // Start index and size for each node
    const int num_nodes
) {
    const int node_id = blockIdx.x;
    if (node_id >= num_nodes) return;
    
    extern __shared__ float shared_sum[];
    const int tid = threadIdx.x;
    
    int start = node_info[2 * node_id];
    int size = node_info[2 * node_id + 1];
    
    float local_sum = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        float val = wp_coeffs[start + i];
        local_sum += val * val;
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
    
    if (tid == 0) {
        node_energy[node_id] = sqrtf(shared_sum[0]);
    }
}

// Continuous Wavelet Transform (CWT) using Morlet wavelet
__global__ void cwt_morlet(
    const float* __restrict__ signal,
    float* __restrict__ cwt_coeffs,
    const float* __restrict__ scales,
    const int num_scales,
    const int n,
    const float omega0  // Morlet parameter (typically 6.0)
) {
    const int scale_idx = blockIdx.x;
    const int time_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (scale_idx >= num_scales || time_idx >= n) return;
    
    float scale = scales[scale_idx];
    float sum_real = 0.0f;
    float sum_imag = 0.0f;
    
    // Convolve signal with scaled wavelet
    for (int tau = 0; tau < n; tau++) {
        float t = (time_idx - tau) / scale;
        
        // Morlet wavelet: exp(i*omega0*t) * exp(-t^2/2)
        float envelope = expf(-0.5f * t * t) / sqrtf(scale);
        float phase = omega0 * t;
        
        float wavelet_real = envelope * cosf(phase);
        float wavelet_imag = envelope * sinf(phase);
        
        sum_real += signal[tau] * wavelet_real;
        sum_imag += signal[tau] * wavelet_imag;
    }
    
    // Store magnitude
    int idx = scale_idx * n + time_idx;
    cwt_coeffs[idx] = sqrtf(sum_real * sum_real + sum_imag * sum_imag);
}

// Wavelet coherence between two signals
__global__ void compute_wavelet_coherence(
    const float* __restrict__ cwt_x,      // CWT of signal X
    const float* __restrict__ cwt_y,      // CWT of signal Y
    const float* __restrict__ phase_x,    // Phase of CWT X
    const float* __restrict__ phase_y,    // Phase of CWT Y
    float* __restrict__ coherence,
    const int window_size,  // Smoothing window
    const int num_scales,
    const int n
) {
    const int scale_idx = blockIdx.x;
    const int time_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (scale_idx >= num_scales || time_idx >= n) return;
    
    // Compute cross-wavelet spectrum
    int idx = scale_idx * n + time_idx;
    float cross_magnitude = cwt_x[idx] * cwt_y[idx];
    float cross_phase = phase_x[idx] - phase_y[idx];
    
    // Smooth cross-spectrum and individual spectra
    float smooth_cross_real = 0.0f;
    float smooth_cross_imag = 0.0f;
    float smooth_x = 0.0f;
    float smooth_y = 0.0f;
    
    int half_window = window_size / 2;
    int count = 0;
    
    for (int t = -half_window; t <= half_window; t++) {
        int t_idx = time_idx + t;
        if (t_idx >= 0 && t_idx < n) {
            int w_idx = scale_idx * n + t_idx;
            
            float phase_diff = phase_x[w_idx] - phase_y[w_idx];
            smooth_cross_real += cwt_x[w_idx] * cwt_y[w_idx] * cosf(phase_diff);
            smooth_cross_imag += cwt_x[w_idx] * cwt_y[w_idx] * sinf(phase_diff);
            smooth_x += cwt_x[w_idx] * cwt_x[w_idx];
            smooth_y += cwt_y[w_idx] * cwt_y[w_idx];
            count++;
        }
    }
    
    if (count > 0 && smooth_x > 0 && smooth_y > 0) {
        float cross_mag = sqrtf(smooth_cross_real * smooth_cross_real + 
                               smooth_cross_imag * smooth_cross_imag);
        coherence[idx] = cross_mag / sqrtf(smooth_x * smooth_y);
    } else {
        coherence[idx] = 0.0f;
    }
}

// Wavelet-based denoising threshold
__global__ void wavelet_soft_threshold(
    float* __restrict__ coefficients,
    const float threshold,
    const int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        float coeff = coefficients[tid];
        float sign = (coeff >= 0) ? 1.0f : -1.0f;
        float abs_coeff = fabsf(coeff);
        
        // Soft thresholding
        if (abs_coeff > threshold) {
            coefficients[tid] = sign * (abs_coeff - threshold);
        } else {
            coefficients[tid] = 0.0f;
        }
    }
}

// Inverse DWT reconstruction
__global__ void idwt_reconstruct_db4(
    const float* __restrict__ approx,
    const float* __restrict__ detail,
    float* __restrict__ signal,
    const int half_n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = 2 * half_n;
    
    if (tid < n) {
        float val = 0.0f;
        
        // Even samples (approximation reconstruction)
        if (tid % 2 == 0) {
            int j = tid / 2;
            val += H2 * approx[(j - 1 + half_n) % half_n];
            val += H0 * approx[j];
            val += H3 * detail[(j - 1 + half_n) % half_n];
            val -= H1 * detail[j];
        }
        // Odd samples
        else {
            int j = (tid - 1) / 2;
            val += H3 * approx[j];
            val += H1 * approx[(j + 1) % half_n];
            val += H2 * detail[j];
            val += H0 * detail[(j + 1) % half_n];
        }
        
        signal[tid] = val;
    }
}

// Wavelet entropy
__global__ void compute_wavelet_entropy(
    const float* __restrict__ wavelet_coeffs,
    float* __restrict__ entropy,
    const int* __restrict__ level_info,
    const int num_levels
) {
    extern __shared__ float shared_probs[];
    
    const int tid = threadIdx.x;
    
    // Compute total energy
    float total_energy = 0.0f;
    for (int level = 0; level < num_levels; level++) {
        int start = level_info[2 * level];
        int size = level_info[2 * level + 1];
        
        for (int i = tid; i < size; i += blockDim.x) {
            float coeff = wavelet_coeffs[start + i];
            total_energy += coeff * coeff;
        }
    }
    
    // Reduce total energy
    __shared__ float shared_total;
    if (tid == 0) shared_total = 0.0f;
    __syncthreads();
    atomicAdd(&shared_total, total_energy);
    __syncthreads();
    
    // Compute entropy
    if (tid == 0) {
        float h = 0.0f;
        
        for (int level = 0; level < num_levels; level++) {
            int start = level_info[2 * level];
            int size = level_info[2 * level + 1];
            
            float level_energy = 0.0f;
            for (int i = 0; i < size; i++) {
                float coeff = wavelet_coeffs[start + i];
                level_energy += coeff * coeff;
            }
            
            if (level_energy > 0 && shared_total > 0) {
                float p = level_energy / shared_total;
                h -= p * log2f(p);
            }
        }
        
        entropy[0] = h;
    }
}

} // extern "C"
"#
}