
/// Stress counter features for threshold exceedance monitoring
pub fn stress_counter_kernels() -> &'static str {
    r#"
extern "C" {

// Count samples exceeding critical thresholds
__global__ void compute_stress_counts(
    const float* __restrict__ signal,
    int* __restrict__ stress_count,
    const float* __restrict__ thresholds,  // Array of thresholds
    const int num_thresholds,
    const int window_size,
    const int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int threshold_idx = blockIdx.y;
    
    if (tid < n - window_size + 1 && threshold_idx < num_thresholds) {
        float threshold = thresholds[threshold_idx];
        int count = 0;
        
        for (int i = 0; i < window_size; i++) {
            if (signal[tid + i] > threshold) {
                count++;
            }
        }
        
        stress_count[threshold_idx * n + tid] = count;
    }
}

// VPD stress integral (accumulated stress above threshold)
__global__ void compute_vpd_stress_integral(
    const float* __restrict__ vpd,
    float* __restrict__ stress_integral,
    const float vpd_threshold,  // e.g., 1.2 kPa
    const float time_step,      // hours
    const int window_size,
    const int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n - window_size + 1) {
        float integral = 0.0f;
        
        for (int i = 0; i < window_size; i++) {
            float stress = fmaxf(vpd[tid + i] - vpd_threshold, 0.0f);
            integral += stress * time_step;
        }
        
        stress_integral[tid] = integral;
    }
}

// Temperature stress duration
__global__ void compute_temperature_stress_duration(
    const float* __restrict__ temperature,
    float* __restrict__ high_stress_hours,
    float* __restrict__ low_stress_hours,
    const float high_threshold,  // e.g., 35°C
    const float low_threshold,   // e.g., 10°C
    const float time_step,       // hours
    const int window_size,
    const int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n - window_size + 1) {
        float high_duration = 0.0f;
        float low_duration = 0.0f;
        
        for (int i = 0; i < window_size; i++) {
            float temp = temperature[tid + i];
            if (temp > high_threshold) {
                high_duration += time_step;
            }
            if (temp < low_threshold) {
                low_duration += time_step;
            }
        }
        
        high_stress_hours[tid] = high_duration;
        low_stress_hours[tid] = low_duration;
    }
}

// Light stress (too low or too high)
__global__ void compute_light_stress(
    const float* __restrict__ light_intensity,
    float* __restrict__ low_light_hours,
    float* __restrict__ high_light_hours,
    const float low_threshold,   // e.g., 100 μmol/m²/s
    const float high_threshold,  // e.g., 800 μmol/m²/s
    const float time_step,
    const int window_size,
    const int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n - window_size + 1) {
        float low_duration = 0.0f;
        float high_duration = 0.0f;
        
        for (int i = 0; i < window_size; i++) {
            float light = light_intensity[tid + i];
            if (light < low_threshold && light > 0.0f) {  // Exclude darkness
                low_duration += time_step;
            }
            if (light > high_threshold) {
                high_duration += time_step;
            }
        }
        
        low_light_hours[tid] = low_duration;
        high_light_hours[tid] = high_duration;
    }
}

// CO2 deficiency counter
__global__ void compute_co2_deficiency(
    const float* __restrict__ co2_concentration,
    int* __restrict__ deficiency_count,
    float* __restrict__ deficiency_integral,
    const float optimal_co2,     // e.g., 800 ppm
    const float min_co2,         // e.g., 400 ppm
    const int window_size,
    const int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n - window_size + 1) {
        int count = 0;
        float integral = 0.0f;
        
        for (int i = 0; i < window_size; i++) {
            float co2 = co2_concentration[tid + i];
            if (co2 < optimal_co2) {
                count++;
                float deficiency = (optimal_co2 - co2) / (optimal_co2 - min_co2);
                integral += fminf(deficiency, 1.0f);
            }
        }
        
        deficiency_count[tid] = count;
        deficiency_integral[tid] = integral;
    }
}

// Combined stress index
__global__ void compute_combined_stress_index(
    const float* __restrict__ temp_stress,
    const float* __restrict__ vpd_stress,
    const float* __restrict__ light_stress,
    const float* __restrict__ co2_stress,
    float* __restrict__ combined_stress,
    const float* __restrict__ weights,  // 4 weights for each stress type
    const int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        float stress = weights[0] * temp_stress[tid] +
                      weights[1] * vpd_stress[tid] +
                      weights[2] * light_stress[tid] +
                      weights[3] * co2_stress[tid];
        
        combined_stress[tid] = fminf(stress, 1.0f);  // Normalize to 0-1
    }
}

// Recovery time estimation (time since last stress event)
__global__ void compute_recovery_time(
    const float* __restrict__ stress_indicator,  // 0 or 1
    float* __restrict__ recovery_hours,
    const float time_step,
    const int n
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float hours_since_stress = 0.0f;
        
        for (int i = 0; i < n; i++) {
            if (stress_indicator[i] > 0.5f) {
                hours_since_stress = 0.0f;
            } else {
                hours_since_stress += time_step;
            }
            recovery_hours[i] = hours_since_stress;
        }
    }
}

// Stress event clustering (consecutive stress periods)
__global__ void compute_stress_clusters(
    const float* __restrict__ stress_indicator,
    int* __restrict__ cluster_count,
    float* __restrict__ avg_cluster_duration,
    const float time_step,
    const int window_size,
    const int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n - window_size + 1) {
        int clusters = 0;
        float total_duration = 0.0f;
        bool in_cluster = false;
        float current_duration = 0.0f;
        
        for (int i = 0; i < window_size; i++) {
            if (stress_indicator[tid + i] > 0.5f) {
                if (!in_cluster) {
                    clusters++;
                    in_cluster = true;
                }
                current_duration += time_step;
            } else {
                if (in_cluster) {
                    total_duration += current_duration;
                    current_duration = 0.0f;
                    in_cluster = false;
                }
            }
        }
        
        // Handle case where window ends in a cluster
        if (in_cluster) {
            total_duration += current_duration;
        }
        
        cluster_count[tid] = clusters;
        avg_cluster_duration[tid] = (clusters > 0) ? total_duration / clusters : 0.0f;
    }
}

} // extern "C"
"#
}