
/// Actuator dynamics features for control system analysis
pub fn actuator_dynamics_kernels() -> &'static str {
    r#"
extern "C" {

// Count state transitions (edges) in binary signal
__global__ void count_edges(
    const float* __restrict__ signal,
    int* __restrict__ edge_count,
    const float threshold,
    const unsigned int n
) {
    extern __shared__ int shared_counts[];
    
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    
    // Initialize shared memory
    shared_counts[tid] = 0;
    
    if (gid < n - 1) {
        // Detect rising and falling edges
        bool current = signal[gid] > threshold;
        bool next = signal[gid + 1] > threshold;
        
        if (current != next) {
            shared_counts[tid] = 1;
        }
    }
    
    __syncthreads();
    
    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < blockDim.x) {
            shared_counts[tid] += shared_counts[tid + stride];
        }
        __syncthreads();
    }
    
    // Write block result
    if (tid == 0) {
        atomicAdd(edge_count, shared_counts[0]);
    }
}

// Calculate duty cycle (percentage of time signal is ON)
__global__ void compute_duty_cycle(
    const float* __restrict__ signal,
    float* __restrict__ duty_cycle,
    const float threshold,
    const int window_size,
    const unsigned int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n - window_size + 1) {
        int on_count = 0;
        
        // Count samples above threshold in window
        for (int i = 0; i < window_size; i++) {
            if (signal[tid + i] > threshold) {
                on_count++;
            }
        }
        
        duty_cycle[tid] = (float)on_count / window_size * 100.0f;
    }
}

// Calculate rate of change (derivative)
__global__ void compute_ramp_rate(
    const float* __restrict__ signal,
    float* __restrict__ ramp_rate,
    const float time_step,
    const unsigned int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid > 0 && tid < n) {
        // Central difference for interior points
        if (tid < n - 1) {
            ramp_rate[tid] = (signal[tid + 1] - signal[tid - 1]) / (2.0f * time_step);
        } else {
            // Backward difference for last point
            ramp_rate[tid] = (signal[tid] - signal[tid - 1]) / time_step;
        }
    } else if (tid == 0 && n > 1) {
        // Forward difference for first point
        ramp_rate[0] = (signal[1] - signal[0]) / time_step;
    }
}

// Detect overshoot/undershoot relative to setpoint
__global__ void compute_overshoot(
    const float* __restrict__ signal,
    const float* __restrict__ setpoint,
    float* __restrict__ overshoot_magnitude,
    float* __restrict__ undershoot_magnitude,
    const unsigned int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        float error = signal[tid] - setpoint[tid];
        
        if (error > 0) {
            overshoot_magnitude[tid] = error;
            undershoot_magnitude[tid] = 0.0f;
        } else {
            overshoot_magnitude[tid] = 0.0f;
            undershoot_magnitude[tid] = -error;
        }
    }
}

// Calculate settling time (time to reach and stay within tolerance of setpoint)
__global__ void compute_settling_time(
    const float* __restrict__ signal,
    const float* __restrict__ setpoint,
    float* __restrict__ settling_flags,
    const float tolerance_percent,
    const int look_ahead,
    const unsigned int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n - look_ahead) {
        float sp = setpoint[tid];
        float tolerance = sp * tolerance_percent / 100.0f;
        
        // Check if signal stays within tolerance for look_ahead samples
        bool settled = true;
        for (int i = 0; i < look_ahead; i++) {
            if (fabsf(signal[tid + i] - sp) > tolerance) {
                settled = false;
                break;
            }
        }
        
        settling_flags[tid] = settled ? 1.0f : 0.0f;
    }
}

// Compute control effort (sum of absolute changes)
__global__ void compute_control_effort(
    const float* __restrict__ control_signal,
    float* __restrict__ effort,
    const int window_size,
    const unsigned int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n - window_size) {
        float total_effort = 0.0f;
        
        for (int i = 0; i < window_size - 1; i++) {
            total_effort += fabsf(control_signal[tid + i + 1] - control_signal[tid + i]);
        }
        
        effort[tid] = total_effort;
    }
}

// Synchronization coefficient between multiple actuators
__global__ void compute_actuator_sync(
    const float* __restrict__ actuator1,
    const float* __restrict__ actuator2,
    float* __restrict__ sync_coeff,
    const int window_size,
    const unsigned int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n - window_size + 1) {
        float sum_xy = 0.0f, sum_x = 0.0f, sum_y = 0.0f;
        float sum_x2 = 0.0f, sum_y2 = 0.0f;
        
        // Compute correlation coefficient in window
        for (int i = 0; i < window_size; i++) {
            float x = actuator1[tid + i];
            float y = actuator2[tid + i];
            
            sum_xy += x * y;
            sum_x += x;
            sum_y += y;
            sum_x2 += x * x;
            sum_y2 += y * y;
        }
        
        float n_w = (float)window_size;
        float numerator = n_w * sum_xy - sum_x * sum_y;
        float denominator = sqrtf((n_w * sum_x2 - sum_x * sum_x) * 
                                  (n_w * sum_y2 - sum_y * sum_y));
        
        if (denominator > 0.0001f) {
            sync_coeff[tid] = numerator / denominator;
        } else {
            sync_coeff[tid] = 0.0f;
        }
    }
}

// Oscillation detection (counts zero crossings of derivative)
__global__ void detect_oscillations(
    const float* __restrict__ signal,
    int* __restrict__ oscillation_count,
    const float dead_band,
    const unsigned int n
) {
    extern __shared__ int shared_osc[];
    
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    
    shared_osc[tid] = 0;
    
    if (gid > 1 && gid < n - 1) {
        // Compute second derivative (acceleration)
        float d2_signal = signal[gid + 1] - 2.0f * signal[gid] + signal[gid - 1];
        float d2_prev = signal[gid] - 2.0f * signal[gid - 1] + signal[gid - 2];
        
        // Detect sign change in acceleration (ignoring dead band)
        if (fabsf(d2_signal) > dead_band && fabsf(d2_prev) > dead_band) {
            if ((d2_signal > 0 && d2_prev < 0) || (d2_signal < 0 && d2_prev > 0)) {
                shared_osc[tid] = 1;
            }
        }
    }
    
    __syncthreads();
    
    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < blockDim.x) {
            shared_osc[tid] += shared_osc[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(oscillation_count, shared_osc[0]);
    }
}

} // extern "C"
"#
}