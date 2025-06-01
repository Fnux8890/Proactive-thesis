use anyhow::Result;

// CPU-only implementations
// GPU kernels have been moved to Python

pub struct ExtendedStatistics {
    pub mean: f32,
    pub std: f32,
    pub skewness: f32,
    pub kurtosis: f32,
    pub percentiles: Vec<f32>,
}

pub struct GrowthEnergyFeatures {
    pub gdd: f32,
    pub dli: f32,
    pub temp_diff_mean: f32,
    pub temp_diff_std: f32,
    pub total_energy_cost: f32,
    pub peak_hours: f32,
    pub off_peak_hours: f32,
    pub solar_efficiency: f32,
}

pub struct MultiResolutionFeatures {
    pub hourly: Vec<f32>,
    pub daily: Vec<f32>,
    pub weekly: Vec<f32>,
}

// Stub implementations - actual GPU computation happens in Python
pub fn compute_extended_stats_gpu(_data: &[f32]) -> Result<ExtendedStatistics> {
    Ok(ExtendedStatistics {
        mean: 0.0,
        std: 0.0,
        skewness: 0.0,
        kurtosis: 0.0,
        percentiles: vec![],
    })
}

pub fn compute_growth_energy_gpu(
    _temperature: &[f32],
    _light_intensity: &[f32],
    _lamp_status: &[i32],
    _outside_temp: &[f32],
    _solar_radiation: &[f32],
    _energy_price: &[f32],
    _power_consumption: &[f32],
    _base_temp: f32,
    _price_threshold: f32,
) -> Result<GrowthEnergyFeatures> {
    Ok(GrowthEnergyFeatures {
        gdd: 0.0,
        dli: 0.0,
        temp_diff_mean: 0.0,
        temp_diff_std: 0.0,
        total_energy_cost: 0.0,
        peak_hours: 0.0,
        off_peak_hours: 0.0,
        solar_efficiency: 0.0,
    })
}

pub fn extract_multi_resolution_features(_data: &[f32]) -> Result<MultiResolutionFeatures> {
    Ok(MultiResolutionFeatures {
        hourly: vec![],
        daily: vec![],
        weekly: vec![],
    })
}