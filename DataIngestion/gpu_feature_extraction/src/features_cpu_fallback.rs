use anyhow::Result;
use chrono::Utc;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::collections::HashMap;
use std::sync::Arc;

use crate::db::{EraData, FeatureSet};
use crate::kernels::KernelManager;

pub struct GpuFeatureExtractor {
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    #[allow(dead_code)]
    stream: Arc<CudaStream>,
    #[allow(dead_code)]
    kernel_manager: KernelManager,
}

impl GpuFeatureExtractor {
    pub fn new(ctx: Arc<CudaContext>, stream: Arc<CudaStream>) -> Result<Self> {
        let kernel_manager = KernelManager::new(ctx.clone())?;
        
        Ok(Self {
            ctx,
            stream,
            kernel_manager,
        })
    }
    
    pub fn extract_batch(&self, era_batch: &[EraData]) -> Result<Vec<FeatureSet>> {
        let mut results = Vec::with_capacity(era_batch.len());
        
        for era_data in era_batch {
            let features = self.extract_era_features(era_data)?;
            results.push(features);
        }
        
        Ok(results)
    }
    
    fn extract_era_features(&self, era_data: &EraData) -> Result<FeatureSet> {
        let mut all_features = HashMap::new();
        
        // Extract features for each sensor
        for (sensor_name, sensor_values) in &era_data.sensor_data {
            if sensor_values.is_empty() {
                continue;
            }
            
            // 1. Basic Statistical features
            let stats = self.compute_statistical_features(sensor_name, sensor_values)?;
            for (feature_name, value) in stats {
                all_features.insert(format!("{}_{}", sensor_name, feature_name), value);
            }
            
            // 2. Rolling window features (multiple window sizes)
            for window_size in &[60, 300, 3600] {
                let rolling = self.compute_rolling_features(
                    sensor_name, 
                    sensor_values, 
                    *window_size
                )?;
                for (feature_name, value) in rolling {
                    all_features.insert(
                        format!("{}_rolling_{}_{}", sensor_name, window_size, feature_name), 
                        value
                    );
                }
            }
            
            // For now, use placeholder values for the remaining features
            // These will be properly implemented once we fix the GPU memory API
            all_features.insert(format!("{}_extended_rolling", sensor_name), 0.0);
            all_features.insert(format!("{}_temporal", sensor_name), 0.0);
            all_features.insert(format!("{}_complexity", sensor_name), 0.0);
            all_features.insert(format!("{}_wavelet", sensor_name), 0.0);
        }
        
        // Cross-sensor features
        all_features.insert("vpd_computed_mean".to_string(), 0.0);
        all_features.insert("energy_efficiency".to_string(), 0.0);
        all_features.insert("thermal_coupling_slope".to_string(), 0.0);
        all_features.insert("light_temp_coupling".to_string(), 0.0);
        all_features.insert("co2_temp_coupling".to_string(), 0.0);
        all_features.insert("thermal_inertia".to_string(), 0.0);
        
        // Actuator dynamics
        all_features.insert("vent_edge_count".to_string(), 0.0);
        all_features.insert("vent_duty_cycle".to_string(), 0.0);
        all_features.insert("vent_oscillations".to_string(), 0.0);
        all_features.insert("vent_response_time".to_string(), 0.0);
        all_features.insert("curtain_edge_count".to_string(), 0.0);
        all_features.insert("curtain_duty_cycle".to_string(), 0.0);
        all_features.insert("curtain_control_effort".to_string(), 0.0);
        
        // Economic features
        all_features.insert("energy_cost_efficiency".to_string(), 0.0);
        all_features.insert("peak_offpeak_ratio".to_string(), 0.0);
        all_features.insert("carbon_intensity".to_string(), 0.0);
        
        // Stress features
        all_features.insert("temp_stress_count".to_string(), 0.0);
        all_features.insert("temp_stress_integral".to_string(), 0.0);
        all_features.insert("temp_rapid_changes".to_string(), 0.0);
        all_features.insert("humidity_stress_duration".to_string(), 0.0);
        
        // Thermal time features
        all_features.insert("growing_degree_days".to_string(), 0.0);
        all_features.insert("heating_degree_hours".to_string(), 0.0);
        all_features.insert("cooling_degree_hours".to_string(), 0.0);
        
        Ok(FeatureSet {
            era_id: era_data.era.era_id,
            era_level: era_data.era.era_level.clone(),
            features: all_features,
            computed_at: Utc::now(),
        })
    }
    
    fn compute_statistical_features(
        &self,
        _sensor_name: &str,
        values: &[f32],
    ) -> Result<HashMap<String, f64>> {
        // For now, compute on CPU until we fix GPU memory API
        let mut features = HashMap::new();
        
        if values.is_empty() {
            return Ok(features);
        }
        
        let n = values.len() as f64;
        let mean = values.iter().sum::<f32>() as f64 / n;
        let variance = values.iter()
            .map(|&x| {
                let diff = x as f64 - mean;
                diff * diff
            })
            .sum::<f64>() / n;
        let std = variance.sqrt();
        
        let min = values.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let max = values.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        
        // Simplified skewness and kurtosis
        let skewness = if std > 0.0 {
            values.iter()
                .map(|&x| {
                    let z = (x as f64 - mean) / std;
                    z * z * z
                })
                .sum::<f64>() / n
        } else {
            0.0
        };
        
        let kurtosis = if std > 0.0 {
            values.iter()
                .map(|&x| {
                    let z = (x as f64 - mean) / std;
                    z * z * z * z
                })
                .sum::<f64>() / n - 3.0
        } else {
            0.0
        };
        
        features.insert("mean".to_string(), mean);
        features.insert("std".to_string(), std);
        features.insert("min".to_string(), *min as f64);
        features.insert("max".to_string(), *max as f64);
        features.insert("skewness".to_string(), skewness);
        features.insert("kurtosis".to_string(), kurtosis);
        
        Ok(features)
    }
    
    fn compute_rolling_features(
        &self,
        _sensor_name: &str,
        values: &[f32],
        window_size: usize,
    ) -> Result<HashMap<String, f64>> {
        let mut features = HashMap::new();
        
        if values.len() < window_size {
            return Ok(features);
        }
        
        // Compute rolling mean and max std on CPU for now
        let mut rolling_means = Vec::new();
        let mut rolling_stds = Vec::new();
        
        for i in 0..=(values.len() - window_size) {
            let window = &values[i..i + window_size];
            let window_mean = window.iter().sum::<f32>() / window_size as f32;
            let window_var = window.iter()
                .map(|&x| {
                    let diff = x - window_mean;
                    diff * diff
                })
                .sum::<f32>() / window_size as f32;
            let window_std = window_var.sqrt();
            
            rolling_means.push(window_mean);
            rolling_stds.push(window_std);
        }
        
        if !rolling_means.is_empty() {
            let mean_of_means = rolling_means.iter().sum::<f32>() as f64 / rolling_means.len() as f64;
            let max_std = rolling_stds.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            
            features.insert("mean".to_string(), mean_of_means);
            features.insert("max_std".to_string(), *max_std as f64);
        }
        
        Ok(features)
    }
}