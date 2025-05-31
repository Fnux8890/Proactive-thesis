
/// Entropy and complexity measures for time series
pub fn entropy_complexity_kernels() -> &'static str {
    r#"
extern "C" {

// Shannon entropy using histogram
__global__ void compute_shannon_entropy(
    const float* __restrict__ signal,
    float* __restrict__ entropy,
    const int n_bins,
    const int n
) {
    extern __shared__ int shared_hist[];
    
    const int tid = threadIdx.x;
    
    // Initialize histogram
    for (int i = tid; i < n_bins; i += blockDim.x) {
        shared_hist[i] = 0;
    }
    __syncthreads();
    
    // Find min/max for binning
    float min_val = CUDART_INF_F, max_val = -CUDART_INF_F;
    for (int i = tid; i < n; i += blockDim.x) {
        float val = signal[i];
        min_val = fminf(min_val, val);
        max_val = fmaxf(max_val, val);
    }
    
    // Reduce min/max (simplified)
    __shared__ float shared_min, shared_max;
    if (tid == 0) {
        shared_min = min_val;
        shared_max = max_val;
    }
    __syncthreads();
    
    atomicMin((int*)&shared_min, __float_as_int(min_val));
    atomicMax((int*)&shared_max, __float_as_int(max_val));
    __syncthreads();
    
    min_val = shared_min;
    max_val = shared_max;
    float range = max_val - min_val + 1e-10f;
    
    // Build histogram
    for (int i = tid; i < n; i += blockDim.x) {
        int bin = (int)((signal[i] - min_val) / range * (n_bins - 1));
        bin = min(max(bin, 0), n_bins - 1);
        atomicAdd(&shared_hist[bin], 1);
    }
    __syncthreads();
    
    // Compute entropy
    if (tid == 0) {
        float h = 0.0f;
        for (int i = 0; i < n_bins; i++) {
            if (shared_hist[i] > 0) {
                float p = (float)shared_hist[i] / n;
                h -= p * log2f(p);
            }
        }
        entropy[0] = h;
    }
}

// Sample entropy (SampEn) - measures complexity/predictability
__global__ void compute_sample_entropy(
    const float* __restrict__ signal,
    float* __restrict__ sampen,
    const int m,        // Embedding dimension (typically 2)
    const float r,      // Tolerance (fraction of std)
    const int n
) {
    // This is a simplified parallel version
    // Full implementation would need careful handling of pattern matching
    
    extern __shared__ int shared_counts[];
    int* count_m = shared_counts;
    int* count_m1 = &shared_counts[blockDim.x];
    
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    
    count_m[tid] = 0;
    count_m1[tid] = 0;
    
    if (gid < n - m) {
        // Count pattern matches for dimension m
        for (int j = 0; j < n - m; j++) {
            if (j != gid) {
                bool match_m = true;
                bool match_m1 = true;
                
                // Check if patterns match within tolerance
                for (int k = 0; k < m; k++) {
                    if (fabsf(signal[gid + k] - signal[j + k]) > r) {
                        match_m = false;
                        match_m1 = false;
                        break;
                    }
                }
                
                // Check m+1 dimension
                if (match_m && j < n - m - 1 && gid < n - m - 1) {
                    if (fabsf(signal[gid + m] - signal[j + m]) > r) {
                        match_m1 = false;
                    }
                }
                
                if (match_m) count_m[tid]++;
                if (match_m1) count_m1[tid]++;
            }
        }
    }
    
    __syncthreads();
    
    // Reduce counts
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            count_m[tid] += count_m[tid + stride];
            count_m1[tid] += count_m1[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0 && blockIdx.x == 0) {
        float phi_m = (float)count_m[0] / ((n - m) * (n - m - 1));
        float phi_m1 = (float)count_m1[0] / ((n - m - 1) * (n - m - 2));
        
        if (phi_m > 0 && phi_m1 > 0) {
            sampen[0] = -logf(phi_m1 / phi_m);
        } else {
            sampen[0] = 0.0f;  // Undefined
        }
    }
}

// Permutation entropy - complexity based on ordinal patterns
__global__ void compute_permutation_entropy(
    const float* __restrict__ signal,
    float* __restrict__ perm_entropy,
    const int order,    // Embedding dimension (3-7 typical)
    const int n
) {
    extern __shared__ int shared_pattern_hist[];
    
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    
    // Maximum patterns for order (order!)
    const int max_patterns = 5040;  // 7! = 5040
    
    // Initialize pattern histogram
    for (int i = tid; i < max_patterns; i += blockDim.x) {
        shared_pattern_hist[i] = 0;
    }
    __syncthreads();
    
    // Count ordinal patterns
    if (gid < n - order + 1) {
        // Extract pattern at position gid
        int ranks[7];  // Max order = 7
        
        // Compute ranks (ordinal pattern)
        for (int i = 0; i < order; i++) {
            ranks[i] = 0;
            for (int j = 0; j < order; j++) {
                if (signal[gid + j] < signal[gid + i]) {
                    ranks[i]++;
                }
            }
        }
        
        // Convert ranks to pattern index (factorial number system)
        int pattern_idx = 0;
        int factorial = 1;
        for (int i = order - 1; i >= 0; i--) {
            pattern_idx += ranks[i] * factorial;
            factorial *= (order - i);
        }
        
        atomicAdd(&shared_pattern_hist[pattern_idx], 1);
    }
    __syncthreads();
    
    // Compute entropy from pattern distribution
    if (tid == 0) {
        float h = 0.0f;
        int total_patterns = n - order + 1;
        int factorial = 1;
        for (int i = 1; i <= order; i++) factorial *= i;
        
        for (int i = 0; i < factorial; i++) {
            if (shared_pattern_hist[i] > 0) {
                float p = (float)shared_pattern_hist[i] / total_patterns;
                h -= p * log2f(p);
            }
        }
        
        // Normalize by maximum possible entropy
        perm_entropy[0] = h / log2f((float)factorial);
    }
}

// Higuchi fractal dimension
__global__ void compute_higuchi_fd(
    const float* __restrict__ signal,
    float* __restrict__ fractal_dim,
    const int k_max,    // Maximum delay (typically 8-10)
    const int n
) {
    extern __shared__ float shared_lengths[];
    
    const int tid = threadIdx.x;
    const int k = blockIdx.x + 1;  // Delay from 1 to k_max
    
    if (k > k_max) return;
    
    float local_length = 0.0f;
    
    // Compute length for delay k and initial time m
    for (int m = tid; m < k; m += blockDim.x) {
        float Lmk = 0.0f;
        int num_segments = (n - m) / k;
        
        for (int i = 0; i < num_segments - 1; i++) {
            float diff = fabsf(signal[m + (i + 1) * k] - signal[m + i * k]);
            Lmk += diff;
        }
        
        if (num_segments > 1) {
            Lmk = Lmk * (n - 1) / ((num_segments - 1) * k * k);
            local_length += Lmk;
        }
    }
    
    shared_lengths[tid] = local_length;
    __syncthreads();
    
    // Reduce to get average length for this k
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_lengths[tid] += shared_lengths[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        // Store log(L(k)) and log(1/k) for linear regression
        float avg_length = shared_lengths[0] / k;
        if (avg_length > 0) {
            // These would be collected and processed to compute slope (FD)
            // For now, store intermediate results
            fractal_dim[k-1] = logf(avg_length);
        }
    }
}

// Approximate entropy (ApEn)
__global__ void compute_approximate_entropy(
    const float* __restrict__ signal,
    float* __restrict__ apen,
    const int m,        // Pattern length
    const float r,      // Tolerance
    const int n
) {
    // Similar structure to SampEn but includes self-matches
    extern __shared__ float shared_phi[];
    
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    
    float phi_m = 0.0f;
    float phi_m1 = 0.0f;
    
    if (gid < n - m + 1) {
        // Count matches for pattern starting at gid
        int count_m = 0;
        int count_m1 = 0;
        
        for (int j = 0; j < n - m + 1; j++) {
            bool match = true;
            
            // Check m-length pattern
            for (int k = 0; k < m; k++) {
                if (fabsf(signal[gid + k] - signal[j + k]) > r) {
                    match = false;
                    break;
                }
            }
            
            if (match) {
                count_m++;
                
                // Check m+1 length pattern
                if (gid < n - m && j < n - m) {
                    if (fabsf(signal[gid + m] - signal[j + m]) <= r) {
                        count_m1++;
                    }
                }
            }
        }
        
        if (count_m > 0) {
            phi_m = logf((float)count_m / (n - m + 1));
        }
        if (count_m1 > 0 && gid < n - m) {
            phi_m1 = logf((float)count_m1 / (n - m));
        }
    }
    
    shared_phi[tid] = phi_m;
    shared_phi[tid + blockDim.x] = phi_m1;
    __syncthreads();
    
    // Reduce
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_phi[tid] += shared_phi[tid + stride];
            shared_phi[tid + blockDim.x] += shared_phi[tid + stride + blockDim.x];
        }
        __syncthreads();
    }
    
    if (tid == 0 && blockIdx.x == 0) {
        float avg_phi_m = shared_phi[0] / (n - m + 1);
        float avg_phi_m1 = shared_phi[blockDim.x] / (n - m);
        apen[0] = avg_phi_m - avg_phi_m1;
    }
}

// Lempel-Ziv complexity
__global__ void compute_lz_complexity(
    const float* __restrict__ signal,
    int* __restrict__ complexity,
    const float threshold,  // For binarization
    const int n
) {
    // Simplified version - full LZ would need sequential processing
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int c = 1;  // Complexity counter
        int i = 0;
        
        while (i < n - 1) {
            int j = i + 1;
            int k = 1;
            
            // Find longest matching substring
            while (j + k <= n) {
                bool match = true;
                
                // Check if pattern starting at j matches any previous
                for (int start = 0; start <= i - k + 1; start++) {
                    match = true;
                    for (int l = 0; l < k; l++) {
                        bool bit_current = signal[j + l] > threshold;
                        bool bit_prev = signal[start + l] > threshold;
                        if (bit_current != bit_prev) {
                            match = false;
                            break;
                        }
                    }
                    if (match) break;
                }
                
                if (!match) {
                    c++;
                    i = j + k - 1;
                    break;
                }
                k++;
                
                if (j + k > n) {
                    i = n;
                    break;
                }
            }
        }
        
        complexity[0] = c;
    }
}

} // extern "C"
"#
}