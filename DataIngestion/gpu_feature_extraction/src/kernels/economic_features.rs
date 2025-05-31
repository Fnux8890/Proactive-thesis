
/// Economic and cost-optimization features
pub fn economic_kernels() -> &'static str {
    r#"
extern "C" {

// Compute price gradient (rate of change in energy prices)
__global__ void compute_price_gradient(
    const float* __restrict__ prices,
    float* __restrict__ gradient,
    const float time_step,  // Hours between samples
    const int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid > 0 && tid < n - 1) {
        // Central difference
        gradient[tid] = (prices[tid + 1] - prices[tid - 1]) / (2.0f * time_step);
    } else if (tid == 0 && n > 1) {
        // Forward difference
        gradient[0] = (prices[1] - prices[0]) / time_step;
    } else if (tid == n - 1 && n > 1) {
        // Backward difference
        gradient[n - 1] = (prices[n - 1] - prices[n - 2]) / time_step;
    }
}

// Price volatility (rolling standard deviation)
__global__ void compute_price_volatility(
    const float* __restrict__ prices,
    float* __restrict__ volatility,
    const int window_size,
    const int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n - window_size + 1) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        
        // Compute mean and variance in window
        for (int i = 0; i < window_size; i++) {
            float val = prices[tid + i];
            sum += val;
            sum_sq += val * val;
        }
        
        float mean = sum / window_size;
        float variance = (sum_sq / window_size) - (mean * mean);
        volatility[tid] = sqrtf(fmaxf(variance, 0.0f));
    }
}

// Price-weighted energy consumption
__global__ void compute_price_weighted_energy(
    const float* __restrict__ energy_consumption,  // kWh
    const float* __restrict__ prices,             // $/kWh
    float* __restrict__ weighted_cost,            // $
    const int window_size,
    const int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n - window_size + 1) {
        float total_cost = 0.0f;
        
        for (int i = 0; i < window_size; i++) {
            total_cost += energy_consumption[tid + i] * prices[tid + i];
        }
        
        weighted_cost[tid] = total_cost;
    }
}

// Cost efficiency ratio (output per cost)
__global__ void compute_cost_efficiency(
    const float* __restrict__ production_output,  // Units produced or growth
    const float* __restrict__ energy_cost,        // $ spent on energy
    float* __restrict__ efficiency_ratio,
    const int window_size,
    const int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n - window_size + 1) {
        float total_output = 0.0f;
        float total_cost = 0.0f;
        
        for (int i = 0; i < window_size; i++) {
            total_output += production_output[tid + i];
            total_cost += energy_cost[tid + i];
        }
        
        if (total_cost > 0.0f) {
            efficiency_ratio[tid] = total_output / total_cost;
        } else {
            efficiency_ratio[tid] = 0.0f;
        }
    }
}

// Peak vs off-peak energy ratio
__global__ void compute_peak_offpeak_ratio(
    const float* __restrict__ energy_consumption,
    const int* __restrict__ hour_of_day,  // 0-23
    float* __restrict__ peak_ratio,
    const int peak_start,    // e.g., 7
    const int peak_end,      // e.g., 23
    const int window_size,
    const int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n - window_size + 1) {
        float peak_energy = 0.0f;
        float offpeak_energy = 0.0f;
        
        for (int i = 0; i < window_size; i++) {
            int hour = hour_of_day[tid + i];
            float energy = energy_consumption[tid + i];
            
            if (hour >= peak_start && hour < peak_end) {
                peak_energy += energy;
            } else {
                offpeak_energy += energy;
            }
        }
        
        if (offpeak_energy > 0.0f) {
            peak_ratio[tid] = peak_energy / offpeak_energy;
        } else {
            peak_ratio[tid] = peak_energy;  // All energy is peak
        }
    }
}

// Demand response opportunity (high price + flexible load)
__global__ void compute_demand_response_opportunity(
    const float* __restrict__ prices,
    const float* __restrict__ load_flexibility,  // 0-1 score
    const float* __restrict__ current_load,      // kW
    float* __restrict__ dr_opportunity,          // Potential savings
    const float price_threshold,
    const int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        float price = prices[tid];
        float flexibility = load_flexibility[tid];
        float load = current_load[tid];
        
        // DR opportunity = high price * flexibility * load
        if (price > price_threshold) {
            dr_opportunity[tid] = (price - price_threshold) * flexibility * load;
        } else {
            dr_opportunity[tid] = 0.0f;
        }
    }
}

// Time-of-use cost optimization score
__global__ void compute_tou_optimization_score(
    const float* __restrict__ energy_scheduled,   // Planned energy use
    const float* __restrict__ energy_actual,      // Actual energy use
    const float* __restrict__ tou_prices,         // Time-of-use prices
    float* __restrict__ optimization_score,       // How well optimized
    const int window_size,
    const int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n - window_size + 1) {
        float optimal_cost = 0.0f;
        float actual_cost = 0.0f;
        float worst_cost = 0.0f;
        
        // Find min and max prices in window
        float min_price = CUDART_INF_F;
        float max_price = -CUDART_INF_F;
        float total_energy = 0.0f;
        
        for (int i = 0; i < window_size; i++) {
            float price = tou_prices[tid + i];
            min_price = fminf(min_price, price);
            max_price = fmaxf(max_price, price);
            total_energy += energy_actual[tid + i];
        }
        
        // Calculate costs
        for (int i = 0; i < window_size; i++) {
            actual_cost += energy_actual[tid + i] * tou_prices[tid + i];
        }
        
        optimal_cost = total_energy * min_price;
        worst_cost = total_energy * max_price;
        
        // Optimization score: 1 = perfect, 0 = worst
        if (worst_cost > optimal_cost) {
            optimization_score[tid] = 1.0f - (actual_cost - optimal_cost) / 
                                            (worst_cost - optimal_cost);
        } else {
            optimization_score[tid] = 1.0f;  // No price variation
        }
    }
}

// Carbon intensity weighted energy
__global__ void compute_carbon_weighted_energy(
    const float* __restrict__ energy_consumption,
    const float* __restrict__ carbon_intensity,   // gCO2/kWh
    float* __restrict__ carbon_emissions,         // gCO2
    const int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        carbon_emissions[tid] = energy_consumption[tid] * carbon_intensity[tid];
    }
}

// Return on energy investment (ROI)
__global__ void compute_energy_roi(
    const float* __restrict__ production_value,   // $ value of output
    const float* __restrict__ energy_cost,        // $ cost of energy
    const float* __restrict__ other_costs,        // $ other operational costs
    float* __restrict__ roi,
    const int window_size,
    const int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n - window_size + 1) {
        float total_revenue = 0.0f;
        float total_energy_cost = 0.0f;
        float total_other_costs = 0.0f;
        
        for (int i = 0; i < window_size; i++) {
            total_revenue += production_value[tid + i];
            total_energy_cost += energy_cost[tid + i];
            total_other_costs += other_costs[tid + i];
        }
        
        float total_cost = total_energy_cost + total_other_costs;
        
        if (total_cost > 0.0f) {
            roi[tid] = (total_revenue - total_cost) / total_cost;
        } else {
            roi[tid] = 0.0f;
        }
    }
}

// Forecast error impact on cost
__global__ void compute_forecast_cost_impact(
    const float* __restrict__ demand_forecast,
    const float* __restrict__ demand_actual,
    const float* __restrict__ prices,
    float* __restrict__ cost_impact,
    const float penalty_rate,  // $/kW for imbalance
    const int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        float error = demand_actual[tid] - demand_forecast[tid];
        
        // Cost impact depends on direction of error
        if (error > 0) {
            // Under-forecast: need to buy at spot price + penalty
            cost_impact[tid] = error * (prices[tid] + penalty_rate);
        } else {
            // Over-forecast: lost opportunity or disposal cost
            cost_impact[tid] = -error * prices[tid] * 0.5f;  // Assume 50% recovery
        }
    }
}

} // extern "C"
"#
}