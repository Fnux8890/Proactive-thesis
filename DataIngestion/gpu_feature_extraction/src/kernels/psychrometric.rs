
/// Psychrometric features for greenhouse climate analysis
pub fn psychrometric_kernels() -> &'static str {
    r#"
extern "C" {

// Constants for psychrometric calculations
#define WATER_VAPOR_MW 18.01528f    // Molecular weight of water vapor (g/mol)
#define DRY_AIR_MW 28.9647f         // Molecular weight of dry air (g/mol)
#define RATIO_MW 0.621945f          // WATER_VAPOR_MW / DRY_AIR_MW
#define CP_AIR 1.006f               // Specific heat of air (kJ/kg·K)
#define CP_VAPOR 1.86f              // Specific heat of water vapor (kJ/kg·K)
#define LATENT_HEAT 2501.0f         // Latent heat of vaporization at 0°C (kJ/kg)

// Dew point temperature calculation
__global__ void compute_dew_point(
    const float* __restrict__ temperature,  // °C
    const float* __restrict__ rel_humidity, // %
    float* __restrict__ dew_point,         // °C
    const unsigned int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        float t = temperature[tid];
        float rh = rel_humidity[tid] / 100.0f;  // Convert to fraction
        
        // Magnus formula constants
        const float a = 17.27f;
        const float b = 237.3f;
        
        // Calculate dew point
        float gamma = a * t / (b + t) + logf(rh);
        dew_point[tid] = b * gamma / (a - gamma);
    }
}

// Wet bulb temperature (simplified psychrometric formula)
__global__ void compute_wet_bulb(
    const float* __restrict__ temperature,  // °C
    const float* __restrict__ rel_humidity, // %
    float* __restrict__ wet_bulb,          // °C
    const unsigned int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        float t = temperature[tid];
        float rh = rel_humidity[tid];
        
        // Stull formula (2011) - accurate for typical greenhouse conditions
        float tw = t * atanf(0.151977f * sqrtf(rh + 8.313659f)) +
                   atanf(t + rh) - atanf(rh - 1.676331f) +
                   0.00391838f * powf(rh, 1.5f) * atanf(0.023101f * rh) -
                   4.686035f;
        
        wet_bulb[tid] = tw;
    }
}

// Humidity ratio (mixing ratio)
__global__ void compute_humidity_ratio(
    const float* __restrict__ temperature,  // °C
    const float* __restrict__ rel_humidity, // %
    const float pressure,                   // kPa
    float* __restrict__ humidity_ratio,     // kg_water/kg_dry_air
    const unsigned int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        float t = temperature[tid];
        float rh = rel_humidity[tid] / 100.0f;
        
        // Saturation vapor pressure (kPa)
        float es = 0.6108f * expf(17.27f * t / (t + 237.3f));
        
        // Actual vapor pressure
        float ea = es * rh;
        
        // Humidity ratio
        humidity_ratio[tid] = RATIO_MW * ea / (pressure - ea);
    }
}

// Enthalpy of moist air
__global__ void compute_enthalpy(
    const float* __restrict__ temperature,     // °C
    const float* __restrict__ humidity_ratio,  // kg_water/kg_dry_air
    float* __restrict__ enthalpy,             // kJ/kg_dry_air
    const unsigned int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        float t = temperature[tid];
        float w = humidity_ratio[tid];
        
        // h = cp_air * t + w * (latent_heat + cp_vapor * t)
        enthalpy[tid] = CP_AIR * t + w * (LATENT_HEAT + CP_VAPOR * t);
    }
}

// Specific volume of moist air
__global__ void compute_specific_volume(
    const float* __restrict__ temperature,     // °C
    const float* __restrict__ humidity_ratio,  // kg_water/kg_dry_air
    const float pressure,                      // kPa
    float* __restrict__ specific_volume,       // m³/kg_dry_air
    const unsigned int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        float t_kelvin = temperature[tid] + 273.15f;
        float w = humidity_ratio[tid];
        
        // Specific gas constant for moist air
        const float R_air = 0.287042f;  // kJ/(kg·K)
        
        // v = R * T * (1 + 1.6078 * w) / P
        specific_volume[tid] = R_air * t_kelvin * (1.0f + 1.6078f * w) / pressure;
    }
}

// Latent to sensible heat ratio
__global__ void compute_bowen_ratio(
    const float* __restrict__ temperature,      // °C
    const float* __restrict__ rel_humidity,     // %
    const float* __restrict__ leaf_temperature, // °C
    float* __restrict__ bowen_ratio,           // dimensionless
    const unsigned int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        float t_air = temperature[tid];
        float rh = rel_humidity[tid] / 100.0f;
        float t_leaf = leaf_temperature[tid];
        
        // Psychrometric constant (kPa/°C) at 101.325 kPa
        const float gamma = 0.0665f;
        
        // Saturation vapor pressures
        float es_air = 0.6108f * expf(17.27f * t_air / (t_air + 237.3f));
        float es_leaf = 0.6108f * expf(17.27f * t_leaf / (t_leaf + 237.3f));
        
        // Vapor pressure deficit
        float vpd_air = es_air * (1.0f - rh);
        float vpd_leaf = es_leaf - es_air * rh;
        
        // Bowen ratio = sensible / latent
        if (vpd_leaf > 0.001f) {
            bowen_ratio[tid] = gamma * (t_leaf - t_air) / vpd_leaf;
        } else {
            bowen_ratio[tid] = 0.0f;
        }
    }
}

// Condensation risk index (0-1)
__global__ void compute_condensation_risk(
    const float* __restrict__ surface_temp,    // °C
    const float* __restrict__ air_temp,        // °C
    const float* __restrict__ rel_humidity,    // %
    float* __restrict__ risk_index,           // 0-1
    const unsigned int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        float t_surface = surface_temp[tid];
        float t_air = air_temp[tid];
        float rh = rel_humidity[tid] / 100.0f;
        
        // Calculate dew point
        float gamma = 17.27f * t_air / (237.3f + t_air) + logf(rh);
        float t_dew = 237.3f * gamma / (17.27f - gamma);
        
        // Risk = 1 when surface temp equals dew point
        // Risk = 0 when surface temp is 5°C above dew point
        float margin = t_surface - t_dew;
        
        if (margin <= 0.0f) {
            risk_index[tid] = 1.0f;
        } else if (margin >= 5.0f) {
            risk_index[tid] = 0.0f;
        } else {
            risk_index[tid] = 1.0f - margin / 5.0f;
        }
    }
}

} // extern "C"
"#
}