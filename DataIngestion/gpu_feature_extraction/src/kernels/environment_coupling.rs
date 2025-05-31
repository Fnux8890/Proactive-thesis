
/// Inside/Outside environmental coupling features
pub fn environment_coupling_kernels() -> &'static str {
    r#"
extern "C" {

// Thermal coupling slope (heat transfer coefficient estimation)
__global__ void compute_thermal_coupling_slope(
    const float* __restrict__ inside_temp,
    const float* __restrict__ outside_temp,
    const float* __restrict__ heating_power,  // kW
    float* __restrict__ coupling_slope,       // kW/°C
    const int window_size,
    const int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n - window_size + 1) {
        // Linear regression: heating_power = slope * (inside - outside) + intercept
        float sum_x = 0.0f, sum_y = 0.0f, sum_xy = 0.0f, sum_x2 = 0.0f;
        int valid_points = 0;
        
        for (int i = 0; i < window_size; i++) {
            float dt = inside_temp[tid + i] - outside_temp[tid + i];
            float power = heating_power[tid + i];
            
            // Only use points where there's a temperature difference
            if (fabsf(dt) > 0.5f) {
                sum_x += dt;
                sum_y += power;
                sum_xy += dt * power;
                sum_x2 += dt * dt;
                valid_points++;
            }
        }
        
        if (valid_points > 10) {  // Need enough points for regression
            float n_points = (float)valid_points;
            float slope = (n_points * sum_xy - sum_x * sum_y) / 
                         (n_points * sum_x2 - sum_x * sum_x);
            coupling_slope[tid] = slope;
        } else {
            coupling_slope[tid] = 0.0f;
        }
    }
}

// Radiation gain efficiency (solar heat gain coefficient)
__global__ void compute_radiation_gain(
    const float* __restrict__ outside_radiation,  // W/m²
    const float* __restrict__ inside_temp,
    const float* __restrict__ inside_temp_prev,   // Temperature change
    const float* __restrict__ greenhouse_area,    // m²
    float* __restrict__ radiation_gain,           // Efficiency 0-1
    const float time_step,                        // hours
    const float thermal_mass,                     // kJ/°C
    const int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n - 1) {
        float radiation = outside_radiation[tid];
        float temp_change = inside_temp[tid + 1] - inside_temp[tid];
        float area = greenhouse_area[tid];
        
        if (radiation > 50.0f && temp_change > 0.0f) {
            // Energy gained from temperature rise
            float energy_gained = thermal_mass * temp_change;  // kJ
            
            // Energy available from radiation
            float energy_available = radiation * area * time_step * 3.6f;  // W/m² * m² * h * 3.6 = kJ
            
            if (energy_available > 0.0f) {
                float efficiency = fminf(energy_gained / energy_available, 1.0f);
                radiation_gain[tid] = efficiency;
            } else {
                radiation_gain[tid] = 0.0f;
            }
        } else {
            radiation_gain[tid] = 0.0f;
        }
    }
}

// Humidity influx rate (moisture exchange with outside)
__global__ void compute_humidity_influx(
    const float* __restrict__ inside_humidity,    // g/m³
    const float* __restrict__ outside_humidity,   // g/m³
    const float* __restrict__ vent_opening,       // % open
    const float* __restrict__ wind_speed,         // m/s
    float* __restrict__ influx_rate,             // g/m³/h
    const float vent_area,                       // m²
    const float greenhouse_volume,               // m³
    const int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        float humidity_diff = outside_humidity[tid] - inside_humidity[tid];
        float vent_percent = vent_opening[tid] / 100.0f;
        float wind = wind_speed[tid];
        
        // Empirical ventilation rate model
        float air_changes_per_hour = vent_percent * vent_area * 
                                    (0.5f + 0.5f * sqrtf(wind)) / greenhouse_volume;
        
        // Humidity influx rate
        influx_rate[tid] = humidity_diff * air_changes_per_hour;
    }
}

// Ventilation efficacy (actual vs theoretical air exchange)
__global__ void compute_vent_efficacy(
    const float* __restrict__ co2_inside,
    const float* __restrict__ co2_outside,
    const float* __restrict__ co2_production,     // ppm/h from plants/heating
    const float* __restrict__ vent_opening,       // %
    float* __restrict__ efficacy,                 // 0-1
    const int window_size,
    const int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n - window_size + 1) {
        float sum_theoretical_reduction = 0.0f;
        float sum_actual_reduction = 0.0f;
        
        for (int i = 1; i < window_size; i++) {
            float vent = vent_opening[tid + i] / 100.0f;
            
            if (vent > 0.1f) {  // Vents are open
                float co2_diff = co2_inside[tid + i - 1] - co2_outside[tid + i];
                float production = co2_production[tid + i];
                
                // Theoretical reduction based on ventilation
                float theoretical = vent * co2_diff * 0.8f;  // 80% theoretical efficiency
                
                // Actual reduction
                float actual = co2_inside[tid + i - 1] + production - co2_inside[tid + i];
                
                if (theoretical > 0.0f) {
                    sum_theoretical_reduction += theoretical;
                    sum_actual_reduction += fmaxf(actual, 0.0f);
                }
            }
        }
        
        if (sum_theoretical_reduction > 0.0f) {
            efficacy[tid] = fminf(sum_actual_reduction / sum_theoretical_reduction, 1.0f);
        } else {
            efficacy[tid] = 0.0f;
        }
    }
}

// Temperature lag between inside and outside
__global__ void compute_temperature_lag(
    const float* __restrict__ inside_temp,
    const float* __restrict__ outside_temp,
    int* __restrict__ lag_hours,
    float* __restrict__ correlation,
    const int max_lag,
    const int window_size,
    const int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n - window_size - max_lag + 1) {
        float best_corr = -1.0f;
        int best_lag = 0;
        
        // Try different lags
        for (int lag = 0; lag <= max_lag; lag++) {
            float sum_xy = 0.0f, sum_x = 0.0f, sum_y = 0.0f;
            float sum_x2 = 0.0f, sum_y2 = 0.0f;
            
            for (int i = 0; i < window_size; i++) {
                float x = outside_temp[tid + i];
                float y = inside_temp[tid + i + lag];
                
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
            
            if (denominator > 0.0f) {
                float corr = numerator / denominator;
                if (corr > best_corr) {
                    best_corr = corr;
                    best_lag = lag;
                }
            }
        }
        
        lag_hours[tid] = best_lag;
        correlation[tid] = best_corr;
    }
}

// Heat transfer coefficient (U-value estimation)
__global__ void compute_heat_transfer_coefficient(
    const float* __restrict__ inside_temp,
    const float* __restrict__ outside_temp,
    const float* __restrict__ heating_power,      // kW
    const float* __restrict__ solar_radiation,    // W/m²
    float* __restrict__ u_value,                  // W/m²K
    const float greenhouse_area,                  // m²
    const float glazing_area,                     // m²
    const int window_size,
    const int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n - window_size + 1) {
        float sum_heat_loss = 0.0f;
        float sum_dt_area = 0.0f;
        int valid_points = 0;
        
        for (int i = 0; i < window_size; i++) {
            float dt = inside_temp[tid + i] - outside_temp[tid + i];
            float heating = heating_power[tid + i] * 1000.0f;  // kW to W
            float solar = solar_radiation[tid + i] * glazing_area * 0.7f;  // 70% transmittance
            
            // Only use night-time data (no solar) with significant temperature difference
            if (solar < 50.0f && fabsf(dt) > 2.0f) {
                float heat_loss = heating;  // Assume steady state at night
                sum_heat_loss += heat_loss;
                sum_dt_area += dt * greenhouse_area;
                valid_points++;
            }
        }
        
        if (valid_points > 5 && sum_dt_area > 0.0f) {
            u_value[tid] = sum_heat_loss / sum_dt_area;
        } else {
            u_value[tid] = 3.0f;  // Default U-value for greenhouse
        }
    }
}

// Infiltration rate estimation
__global__ void compute_infiltration_rate(
    const float* __restrict__ inside_temp,
    const float* __restrict__ outside_temp,
    const float* __restrict__ wind_speed,         // m/s
    const float* __restrict__ pressure_diff,      // Pa
    const float* __restrict__ vent_opening,       // %
    float* __restrict__ infiltration,            // air changes per hour
    const float building_height,                  // m
    const int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        float dt = fabsf(inside_temp[tid] - outside_temp[tid]);
        float wind = wind_speed[tid];
        float dp = pressure_diff[tid];
        float vent_closed = 1.0f - vent_opening[tid] / 100.0f;
        
        // Stack effect
        float stack_effect = 0.0f;
        if (dt > 0.0f) {
            float t_avg = (inside_temp[tid] + outside_temp[tid]) / 2.0f + 273.15f;
            stack_effect = sqrtf(building_height * dt / t_avg) * 0.15f;
        }
        
        // Wind effect
        float wind_effect = 0.05f * sqrtf(wind);
        
        // Pressure effect
        float pressure_effect = 0.0f;
        if (fabsf(dp) > 0.0f) {
            pressure_effect = 0.02f * sqrtf(fabsf(dp));
        }
        
        // Total infiltration (reduced when vents are open)
        infiltration[tid] = vent_closed * (stack_effect + wind_effect + pressure_effect);
    }
}

// Solar heat gain coefficient (SHGC) dynamic estimation
__global__ void compute_dynamic_shgc(
    const float* __restrict__ solar_radiation,    // W/m²
    const float* __restrict__ inside_temp_change, // °C/h
    const float* __restrict__ shading_position,   // % deployed
    float* __restrict__ shgc,                     // 0-1
    const float thermal_mass,                     // kJ/°C
    const float glazing_area,                     // m²
    const int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        float radiation = solar_radiation[tid];
        float temp_rise = inside_temp_change[tid];
        float shading = shading_position[tid] / 100.0f;
        
        if (radiation > 100.0f && temp_rise > 0.0f) {
            // Energy gained from temperature rise (kJ/h)
            float energy_gained = thermal_mass * temp_rise;
            
            // Solar energy available (W/m² * m² * 3.6 = kJ/h)
            float solar_available = radiation * glazing_area * 3.6f;
            
            // SHGC considering shading
            float base_shgc = fminf(energy_gained / solar_available, 0.87f);  // Max SHGC
            shgc[tid] = base_shgc * (1.0f - 0.8f * shading);  // 80% reduction with full shading
        } else {
            shgc[tid] = 0.4f * (1.0f - 0.8f * shading);  // Default with shading
        }
    }
}

} // extern "C"
"#
}