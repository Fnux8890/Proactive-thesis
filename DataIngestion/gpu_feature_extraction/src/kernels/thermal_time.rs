
/// Thermal time features critical for plant growth modeling
pub fn thermal_time_kernels() -> &'static str {
    r#"
extern "C" {

// Growing Degree Days (GDD) calculation
__global__ void compute_gdd(
    const float* __restrict__ temperature,
    float* __restrict__ gdd,
    const float base_temp,
    const float max_temp,
    const unsigned int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        float temp = temperature[tid];
        
        // Clamp temperature to valid range
        temp = fmaxf(temp, base_temp);
        temp = fminf(temp, max_temp);
        
        // GDD = max(0, temp - base_temp)
        gdd[tid] = fmaxf(0.0f, temp - base_temp);
    }
}

// Photo-Thermal Time (PTT) calculation
__global__ void compute_ptt(
    const float* __restrict__ temperature,
    const float* __restrict__ light_intensity,
    float* __restrict__ ptt,
    const float base_temp,
    const float opt_temp,
    const float max_temp,
    const float light_threshold,
    const unsigned int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        float temp = temperature[tid];
        float light = light_intensity[tid];
        
        // Temperature factor (0-1)
        float temp_factor = 0.0f;
        if (temp > base_temp && temp < max_temp) {
            if (temp <= opt_temp) {
                temp_factor = (temp - base_temp) / (opt_temp - base_temp);
            } else {
                temp_factor = (max_temp - temp) / (max_temp - opt_temp);
            }
        }
        
        // Light factor (0-1)
        float light_factor = fminf(1.0f, light / light_threshold);
        
        // PTT = temperature factor × light factor
        ptt[tid] = temp_factor * light_factor;
    }
}

// Cumulative VPD Index (CVPI) for water stress
__global__ void compute_cvpi(
    const float* __restrict__ vpd,
    float* __restrict__ cvpi,
    const float vpd_threshold,
    const unsigned int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        // Only accumulate VPD above threshold (water stress)
        float stress_vpd = fmaxf(0.0f, vpd[tid] - vpd_threshold);
        
        // Cumulative sum (in-place prefix sum would be better for large arrays)
        if (tid == 0) {
            cvpi[0] = stress_vpd;
        } else {
            // Note: This is simplified - real implementation needs proper scan
            cvpi[tid] = stress_vpd;  // Placeholder for scan result
        }
    }
}

// Accumulated Light Integral (ALI) in mol/m²
__global__ void compute_ali(
    const float* __restrict__ ppfd,  // μmol/m²/s
    float* __restrict__ ali,         // mol/m²
    const float time_step,           // seconds
    const unsigned int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        // Convert μmol to mol and accumulate
        float mol_per_step = ppfd[tid] * time_step * 1e-6f;
        
        if (tid == 0) {
            ali[0] = mol_per_step;
        } else {
            // Note: Placeholder for prefix sum
            ali[tid] = mol_per_step;
        }
    }
}

// Heat stress hours above critical temperature
__global__ void compute_heat_stress_hours(
    const float* __restrict__ temperature,
    float* __restrict__ stress_hours,
    const float critical_temp,
    const float time_step_hours,
    const unsigned int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        float is_stressed = (temperature[tid] > critical_temp) ? 1.0f : 0.0f;
        stress_hours[tid] = is_stressed * time_step_hours;
    }
}

// Thermal efficiency index (0-1)
__global__ void compute_thermal_efficiency(
    const float* __restrict__ temperature,
    float* __restrict__ efficiency,
    const float t_min,
    const float t_opt_low,
    const float t_opt_high,
    const float t_max,
    const unsigned int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        float temp = temperature[tid];
        float eff = 0.0f;
        
        if (temp >= t_min && temp <= t_max) {
            if (temp >= t_opt_low && temp <= t_opt_high) {
                eff = 1.0f;  // Optimal range
            } else if (temp < t_opt_low) {
                eff = (temp - t_min) / (t_opt_low - t_min);
            } else {
                eff = (t_max - temp) / (t_max - t_opt_high);
            }
        }
        
        efficiency[tid] = eff;
    }
}

} // extern "C"
"#
}