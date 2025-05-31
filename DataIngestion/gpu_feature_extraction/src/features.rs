use anyhow::Result;
use chrono::Utc;
use cudarc::driver::safe::{CudaContext, CudaStream, LaunchConfig, CudaSlice};
use std::collections::HashMap;
use std::sync::Arc;

use crate::db::{EraData, FeatureSet};
use crate::kernels::{KernelManager, StatisticalFeatures};

pub struct GpuFeatureExtractor {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    kernel_manager: KernelManager,
}

impl GpuFeatureExtractor {
    pub fn new(ctx: Arc<CudaContext>, stream: Arc<CudaStream>) -> Result<Self> {
        let kernel_manager = KernelManager::new(ctx.clone())?;
        
        Ok(Self {
            _ctx: ctx,
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
            
            // 3. Extended rolling statistics (percentiles, skewness, kurtosis, etc.)
            let extended_rolling = self.compute_extended_rolling_features(sensor_name, sensor_values)?;
            all_features.extend(extended_rolling);
            
            // 4. Temporal dependency features (ACF, PACF)
            let temporal = self.compute_temporal_features(sensor_name, sensor_values)?;
            all_features.extend(temporal);
            
            // 5. Entropy and complexity measures
            let complexity = self.compute_complexity_features(sensor_name, sensor_values)?;
            all_features.extend(complexity);
            
            // 6. Wavelet features (multi-scale decomposition)
            let wavelet = self.compute_wavelet_features(sensor_name, sensor_values)?;
            all_features.extend(wavelet);
        }
        
        // 7. Cross-sensor features (VPD, energy efficiency, etc.)
        let cross_features = self.compute_cross_features(era_data)?;
        all_features.extend(cross_features);
        
        // 8. Environment coupling features (if applicable sensors exist)
        let env_coupling = self.compute_environment_coupling_features(era_data)?;
        all_features.extend(env_coupling);
        
        // 9. Actuator dynamics features (for control signals)
        let actuator = self.compute_actuator_dynamics_features(era_data)?;
        all_features.extend(actuator);
        
        // 10. Economic features (if energy/price data available)
        let economic = self.compute_economic_features(era_data)?;
        all_features.extend(economic);
        
        // 11. Stress counter features
        let stress = self.compute_stress_features(era_data)?;
        all_features.extend(stress);
        
        // 12. Thermal time features (growing degree days, etc.)
        let thermal = self.compute_thermal_time_features(era_data)?;
        all_features.extend(thermal);
        
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
        // Safety check: empty array
        if values.is_empty() {
            return Ok(HashMap::new());
        }
        
        // Allocate device memory and copy data using correct API
        let d_input: CudaSlice<f32> = self.stream.memcpy_stod(values)?;
        let d_output: CudaSlice<StatisticalFeatures> = self.stream.alloc_zeros::<StatisticalFeatures>(1)?;
        
        // Launch kernel with safety checks
        let n = values.len() as u32;
        let block_size = 256u32;
        let grid_size = ((n + block_size - 1) / block_size).min(65535); // CUDA max grid dimension
        
        // Calculate shared memory size for the enhanced kernel
        let num_warps = (block_size + 31) / 32; // 8 warps for 256 threads
        
        // Option 1: Basic kernel (96 bytes) - current implementation
        // let shared_mem_bytes = (num_warps * 3 * std::mem::size_of::<f32>() as u32);
        
        // Option 2: Enhanced kernel with extended statistics
        // WarpStats struct: 9 floats + 2 ints = 44 bytes per warp
        let warp_stats_size = 44u32; // 9 * 4 + 2 * 4 = 44 bytes
        let warp_shared_size = num_warps * warp_stats_size; // 352 bytes for 8 warps
        
        // Add histogram for percentile calculation (256 bins * 4 bytes)
        let histogram_size = 256u32 * std::mem::size_of::<u32>() as u32; // 1024 bytes
        
        // Total shared memory for enhanced kernel
        let shared_mem_bytes = warp_shared_size + histogram_size; // 1376 bytes (~1.3 KB)
        
        // This is still only 1.4% of RTX 4070's 99KB limit!
        
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes,  // Fixed: was 0, now 96 bytes
        };
        
        self.kernel_manager.launch_statistical_kernel(
            &self.stream,
            config,
            &d_input,
            &d_output,
            n,
        )?;
        
        // Copy results back
        let results: Vec<StatisticalFeatures> = self.stream.memcpy_dtov(&d_output)?;
        let stats = &results[0];
        
        let mut features = HashMap::new();
        features.insert("mean".to_string(), stats.mean as f64);
        features.insert("std".to_string(), stats.std as f64);
        features.insert("min".to_string(), stats.min as f64);
        features.insert("max".to_string(), stats.max as f64);
        features.insert("skewness".to_string(), stats.skewness as f64);
        features.insert("kurtosis".to_string(), stats.kurtosis as f64);
        
        Ok(features)
    }
    
    fn compute_rolling_features(
        &self,
        _sensor_name: &str,
        values: &[f32],
        window_size: usize,
    ) -> Result<HashMap<String, f64>> {
        if values.len() < window_size || window_size == 0 {
            return Ok(HashMap::new());
        }
        
        // Allocate device memory using correct API
        let d_input: CudaSlice<f32> = self.stream.memcpy_stod(values)?;
        
        let output_size = values.len() - window_size + 1;
        if output_size == 0 {
            return Ok(HashMap::new());
        }
        
        let d_means: CudaSlice<f32> = self.stream.alloc_zeros::<f32>(output_size)?;
        let d_stds: CudaSlice<f32> = self.stream.alloc_zeros::<f32>(output_size)?;
        
        // Launch rolling statistics kernel with safety checks
        let block_size = 256u32;
        let grid_size = ((output_size as u32 + block_size - 1) / block_size).min(65535); // Cap grid size
        
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: ((window_size * std::mem::size_of::<f32>()) as u32).min(48000), // Cap shared memory
        };
        
        self.kernel_manager.launch_rolling_stats_kernel(
            &self.stream,
            config,
            &d_input,
            &d_means,
            &d_stds,
            values.len() as u32,
            window_size as u32,
        )?;
        
        // Copy results back
        let means: Vec<f32> = self.stream.memcpy_dtov(&d_means)?;
        let stds: Vec<f32> = self.stream.memcpy_dtov(&d_stds)?;
        
        // Extract summary statistics from rolling windows
        let mut features = HashMap::new();
        
        if !means.is_empty() {
            let mean_of_means = means.iter().sum::<f32>() as f64 / means.len() as f64;
            let max_std = stds.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            
            features.insert("mean".to_string(), mean_of_means);
            features.insert("max_std".to_string(), *max_std as f64);
        }
        
        Ok(features)
    }
    
    fn compute_cross_features(&self, era_data: &EraData) -> Result<HashMap<String, f64>> {
        let mut features = HashMap::new();
        
        // VPD calculation if temperature and humidity are available
        if let (Some(temp), Some(rh)) = (
            era_data.sensor_data.get("air_temp_c"),
            era_data.sensor_data.get("relative_humidity_percent"),
        ) {
            if !temp.is_empty() && !rh.is_empty() && temp.len() == rh.len() {
                let vpd = self.compute_vpd_gpu(temp, rh)?;
                features.insert("vpd_computed_mean".to_string(), vpd);
            }
        }
        
        // Energy efficiency metric
        if let (Some(light), Some(co2)) = (
            era_data.sensor_data.get("light_intensity_umol"),
            era_data.sensor_data.get("co2_measured_ppm"),
        ) {
            if !light.is_empty() && !co2.is_empty() {
                let efficiency = self.compute_energy_efficiency(light, co2)?;
                features.insert("energy_efficiency".to_string(), efficiency);
            }
        }
        
        Ok(features)
    }
    
    fn compute_vpd_gpu(&self, temp: &[f32], rh: &[f32]) -> Result<f64> {
        let n = temp.len().min(rh.len());
        
        // Safety check
        if n == 0 {
            return Ok(0.0);
        }
        
        // Allocate device memory using correct API
        let d_temp: CudaSlice<f32> = self.stream.memcpy_stod(&temp[..n])?;
        let d_rh: CudaSlice<f32> = self.stream.memcpy_stod(&rh[..n])?;
        let d_vpd: CudaSlice<f32> = self.stream.alloc_zeros::<f32>(n)?;
        
        // Launch VPD kernel with safety checks
        let block_size = 256u32;
        let grid_size = ((n as u32 + block_size - 1) / block_size).min(65535); // Cap grid size
        
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        
        self.kernel_manager.launch_vpd_kernel(
            &self.stream,
            config,
            &d_temp,
            &d_rh,
            &d_vpd,
            n as u32,
        )?;
        
        // Copy results back and compute mean
        let vpd_values: Vec<f32> = self.stream.memcpy_dtov(&d_vpd)?;
        
        // Safety check for valid values
        if vpd_values.is_empty() {
            return Ok(0.0);
        }
        
        let mean_vpd = vpd_values.iter().sum::<f32>() as f64 / vpd_values.len() as f64;
        
        Ok(mean_vpd)
    }
    
    fn compute_energy_efficiency(&self, light: &[f32], co2: &[f32]) -> Result<f64> {
        // Simple efficiency metric: light utilization per CO2 unit
        let mean_light: f64 = light.iter().map(|&x| x as f64).sum::<f64>() / light.len() as f64;
        let mean_co2: f64 = co2.iter().map(|&x| x as f64).sum::<f64>() / co2.len() as f64;
        
        if mean_co2 > 0.0 {
            Ok(mean_light / mean_co2)
        } else {
            Ok(0.0)
        }
    }
    
    fn compute_extended_rolling_features(&self, sensor_name: &str, values: &[f32]) -> Result<HashMap<String, f64>> {
        let mut features = HashMap::new();
        
        if values.is_empty() {
            return Ok(features);
        }
        
        let n = values.len();
        let window_size = 300.min(n); // 5 minutes at 1Hz sampling
        
        if n < window_size {
            return Ok(features);
        }
        
        // Allocate device memory
        let d_input: CudaSlice<f32> = self.stream.memcpy_stod(values)?;
        let output_size = n - window_size + 1;
        
        // Safety check: ensure output size is valid
        if output_size == 0 {
            return Ok(features);
        }
        
        // Compute rolling percentiles
        let percentiles = vec![0.10, 0.25, 0.50, 0.75, 0.90];
        let num_percentiles = percentiles.len();
        let d_percentiles: CudaSlice<f32> = self.stream.memcpy_stod(&percentiles)?;
        let d_percentile_output: CudaSlice<f32> = self.stream.alloc_zeros::<f32>(output_size * num_percentiles)?;
        
        let block_size = 256u32;
        let grid_size = ((output_size as u32 + block_size - 1) / block_size).min(65535); // Cap grid size
        
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: ((window_size * std::mem::size_of::<f32>()) as u32).min(48000), // Cap shared memory
        };
        
        self.kernel_manager.launch_rolling_percentiles_kernel(
            &self.stream,
            config,
            &d_input,
            &d_percentiles,
            &d_percentile_output,
            n as u32,
            window_size as u32,
            num_percentiles as u32,
        )?;
        
        // Copy results back
        let percentile_results: Vec<f32> = self.stream.memcpy_dtov(&d_percentile_output)?;
        
        // Extract summary statistics from percentile results
        for (i, p) in percentiles.iter().enumerate() {
            let offset = i * output_size;
            let percentile_values = &percentile_results[offset..offset + output_size];
            
            if !percentile_values.is_empty() {
                let mean = percentile_values.iter().sum::<f32>() as f64 / percentile_values.len() as f64;
                let max = percentile_values.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
                let min = percentile_values.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
                
                features.insert(format!("{}_rolling_p{}_mean", sensor_name, (p * 100.0) as u32), mean);
                features.insert(format!("{}_rolling_p{}_max", sensor_name, (p * 100.0) as u32), *max as f64);
                features.insert(format!("{}_rolling_p{}_min", sensor_name, (p * 100.0) as u32), *min as f64);
            }
        }
        
        // Compute IQR from p75 - p25
        if output_size > 0 {
            let p25_offset = 1 * output_size;
            let p75_offset = 3 * output_size;
            let mut iqr_values = Vec::with_capacity(output_size);
            
            for i in 0..output_size {
                let iqr = percentile_results[p75_offset + i] - percentile_results[p25_offset + i];
                iqr_values.push(iqr);
            }
            
            let mean_iqr = iqr_values.iter().sum::<f32>() as f64 / iqr_values.len() as f64;
            features.insert(format!("{}_rolling_iqr_mean", sensor_name), mean_iqr);
        }
        
        Ok(features)
    }
    
    fn compute_temporal_features(&self, sensor_name: &str, values: &[f32]) -> Result<HashMap<String, f64>> {
        let mut features = HashMap::new();
        
        if values.len() < 24 {
            return Ok(features);
        }
        
        let n = values.len();
        let max_lag = 24.min(n / 4); // Limit lag to 1/4 of series length
        
        // Safety check: ensure max_lag is valid
        if max_lag == 0 {
            return Ok(features);
        }
        
        // Allocate device memory
        let d_input: CudaSlice<f32> = self.stream.memcpy_stod(values)?;
        let d_acf: CudaSlice<f32> = self.stream.alloc_zeros::<f32>(max_lag)?;
        
        let block_size = 256u32;
        let grid_size = ((max_lag as u32 + block_size - 1) / block_size).min(65535); // Cap grid size
        
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        
        // Compute ACF
        self.kernel_manager.launch_acf_kernel(
            &self.stream,
            config,
            &d_input,
            &d_acf,
            n as u32,
            max_lag as u32,
        )?;
        
        // Copy results back
        let acf_values: Vec<f32> = self.stream.memcpy_dtov(&d_acf)?;
        
        // Extract key lags
        if acf_values.len() > 0 {
            features.insert(format!("{}_acf_lag1", sensor_name), acf_values[0] as f64);
        }
        if acf_values.len() > 23 {
            features.insert(format!("{}_acf_lag24", sensor_name), acf_values[23] as f64);
        }
        if acf_values.len() > 11 {
            features.insert(format!("{}_acf_lag12", sensor_name), acf_values[11] as f64);
        }
        
        // Find first zero crossing (indicates decorrelation time)
        let mut first_zero_crossing = max_lag;
        for (i, &acf) in acf_values.iter().enumerate() {
            if acf <= 0.0 {
                first_zero_crossing = i + 1;
                break;
            }
        }
        features.insert(format!("{}_acf_first_zero", sensor_name), first_zero_crossing as f64);
        
        // TODO: Implement PACF and time-delayed mutual information
        features.insert(format!("{}_pacf_lag1", sensor_name), 0.0);
        
        Ok(features)
    }
    
    fn compute_complexity_features(&self, sensor_name: &str, values: &[f32]) -> Result<HashMap<String, f64>> {
        let mut features = HashMap::new();
        
        if values.is_empty() {
            return Ok(features);
        }
        
        let n = values.len();
        
        // Compute Shannon entropy
        let num_bins = 16.min(n / 10).max(4); // Adaptive binning
        let d_input: CudaSlice<f32> = self.stream.memcpy_stod(values)?;
        let d_entropy: CudaSlice<f32> = self.stream.alloc_zeros::<f32>(1)?;
        
        let block_size = 256u32;
        let grid_size = 1u32; // Single value output
        
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: ((num_bins * std::mem::size_of::<u32>()) as u32).min(48000), // Cap shared memory
        };
        
        self.kernel_manager.launch_shannon_entropy_kernel(
            &self.stream,
            config,
            &d_input,
            &d_entropy,
            n as u32,
            num_bins as u32,
        )?;
        
        // Copy result back
        let entropy_result: Vec<f32> = self.stream.memcpy_dtov(&d_entropy)?;
        features.insert(format!("{}_shannon_entropy", sensor_name), entropy_result[0] as f64);
        
        // TODO: Implement remaining complexity measures
        features.insert(format!("{}_sample_entropy", sensor_name), 0.0);
        features.insert(format!("{}_fractal_dimension", sensor_name), 0.0);
        
        Ok(features)
    }
    
    fn compute_wavelet_features(&self, sensor_name: &str, values: &[f32]) -> Result<HashMap<String, f64>> {
        let mut features = HashMap::new();
        
        if values.len() < 8 { // Minimum for db4 wavelet
            return Ok(features);
        }
        
        let n = values.len();
        // Ensure power of 2 for DWT
        let padded_len = n.next_power_of_two();
        
        // Safety check: cap maximum size to prevent overflow
        let padded_len = padded_len.min(1 << 20); // Max 1M elements
        
        // Pad input if needed
        let mut padded_values = values.to_vec();
        padded_values.resize(padded_len, values[n-1]); // Pad with last value
        
        // Allocate device memory
        let d_input: CudaSlice<f32> = self.stream.memcpy_stod(&padded_values)?;
        let d_approx: CudaSlice<f32> = self.stream.alloc_zeros::<f32>(padded_len / 2)?;
        let d_detail: CudaSlice<f32> = self.stream.alloc_zeros::<f32>(padded_len / 2)?;
        
        let block_size = 256u32;
        let grid_size = (((padded_len / 2) as u32 + block_size - 1) / block_size).min(65535); // Cap grid size
        
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        
        // Level 1 decomposition
        self.kernel_manager.launch_dwt_decomposition_kernel(
            &self.stream,
            config,
            &d_input,
            &d_approx,
            &d_detail,
            padded_len as u32,
        )?;
        
        // Copy results back
        let _approx1: Vec<f32> = self.stream.memcpy_dtov(&d_approx)?;
        let detail1: Vec<f32> = self.stream.memcpy_dtov(&d_detail)?;
        
        // Compute energy at level 1
        let energy_l1: f64 = detail1.iter()
            .take(n / 2) // Only use actual data, not padding
            .map(|&x| (x * x) as f64)
            .sum::<f64>();
        features.insert(format!("{}_wavelet_energy_l1", sensor_name), energy_l1.sqrt());
        
        // Level 2 decomposition (if enough data)
        if padded_len >= 16 {
            let d_approx2: CudaSlice<f32> = self.stream.alloc_zeros::<f32>(padded_len / 4)?;
            let d_detail2: CudaSlice<f32> = self.stream.alloc_zeros::<f32>(padded_len / 4)?;
            
            let config2 = LaunchConfig {
                grid_dim: ((((padded_len / 4) as u32 + block_size - 1) / block_size).min(65535), 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };
            
            self.kernel_manager.launch_dwt_decomposition_kernel(
                &self.stream,
                config2,
                &d_approx,
                &d_approx2,
                &d_detail2,
                (padded_len / 2) as u32,
            )?;
            
            let detail2: Vec<f32> = self.stream.memcpy_dtov(&d_detail2)?;
            let energy_l2: f64 = detail2.iter()
                .take(n / 4)
                .map(|&x| (x * x) as f64)
                .sum::<f64>();
            features.insert(format!("{}_wavelet_energy_l2", sensor_name), energy_l2.sqrt());
        } else {
            features.insert(format!("{}_wavelet_energy_l2", sensor_name), 0.0);
        }
        
        // TODO: Level 3 and wavelet packet decomposition
        features.insert(format!("{}_wavelet_energy_l3", sensor_name), 0.0);
        
        Ok(features)
    }
    
    fn compute_environment_coupling_features(&self, era_data: &EraData) -> Result<HashMap<String, f64>> {
        let mut features = HashMap::new();
        
        // Thermal coupling between inside and outside
        if let (Some(temp_in), Some(temp_out), Some(radiation)) = (
            era_data.sensor_data.get("air_temp_c"),
            era_data.sensor_data.get("outside_temp_c"),
            era_data.sensor_data.get("radiation_w_m2"),
        ) {
            let min_len = temp_in.len().min(temp_out.len()).min(radiation.len());
            if min_len > 0 {
                let d_temp_in: CudaSlice<f32> = self.stream.memcpy_stod(&temp_in[..min_len])?;
                let d_temp_out: CudaSlice<f32> = self.stream.memcpy_stod(&temp_out[..min_len])?;
                let d_radiation: CudaSlice<f32> = self.stream.memcpy_stod(&radiation[..min_len])?;
                let d_coupling: CudaSlice<f32> = self.stream.alloc_zeros::<f32>(1)?;
                
                let config = LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 3 * std::mem::size_of::<f32>() as u32,
                };
                
                self.kernel_manager.launch_thermal_coupling_kernel(
                    &self.stream,
                    config,
                    &d_temp_in,
                    &d_temp_out,
                    &d_radiation,
                    &d_coupling,
                    min_len as u32,
                )?;
                
                let coupling_result: Vec<f32> = self.stream.memcpy_dtov(&d_coupling)?;
                features.insert("thermal_coupling_slope".to_string(), coupling_result[0] as f64);
            }
        } else {
            // Fallback: simple correlation between light and temperature
            if let (Some(light), Some(temp)) = (
                era_data.sensor_data.get("light_intensity_umol"),
                era_data.sensor_data.get("air_temp_c"),
            ) {
                let min_len = light.len().min(temp.len());
                if min_len > 0 {
                    // Simple Pearson correlation (computed on CPU for simplicity)
                    let light_mean = light[..min_len].iter().sum::<f32>() / min_len as f32;
                    let temp_mean = temp[..min_len].iter().sum::<f32>() / min_len as f32;
                    
                    let mut cov = 0.0;
                    let mut light_var = 0.0;
                    let mut temp_var = 0.0;
                    
                    for i in 0..min_len {
                        let light_diff = light[i] - light_mean;
                        let temp_diff = temp[i] - temp_mean;
                        cov += light_diff * temp_diff;
                        light_var += light_diff * light_diff;
                        temp_var += temp_diff * temp_diff;
                    }
                    
                    let correlation = if light_var > 0.0 && temp_var > 0.0 {
                        cov / (light_var * temp_var).sqrt()
                    } else {
                        0.0
                    };
                    
                    features.insert("light_temp_coupling".to_string(), correlation as f64);
                }
            }
        }
        
        // CO2-temperature coupling
        if let (Some(co2), Some(temp)) = (
            era_data.sensor_data.get("co2_measured_ppm"),
            era_data.sensor_data.get("air_temp_c"),
        ) {
            let min_len = co2.len().min(temp.len());
            if min_len > 0 {
                let co2_mean = co2[..min_len].iter().sum::<f32>() / min_len as f32;
                let temp_mean = temp[..min_len].iter().sum::<f32>() / min_len as f32;
                
                let covariance: f32 = co2[..min_len].iter().zip(temp[..min_len].iter())
                    .map(|(&c, &t)| (c - co2_mean) * (t - temp_mean))
                    .sum::<f32>() / min_len as f32;
                
                features.insert("co2_temp_coupling".to_string(), covariance as f64);
            }
        }
        
        // Thermal inertia estimate
        if let Some(temp) = era_data.sensor_data.get("air_temp_c") {
            if temp.len() > 60 {
                let mut max_change_rate = 0.0f32;
                for i in 60..temp.len() {
                    let change_rate = (temp[i] - temp[i-60]).abs() / 60.0;
                    max_change_rate = max_change_rate.max(change_rate);
                }
                features.insert("thermal_inertia".to_string(), (1.0 / (1.0 + max_change_rate)) as f64);
            }
        }
        
        Ok(features)
    }
    
    fn compute_actuator_dynamics_features(&self, era_data: &EraData) -> Result<HashMap<String, f64>> {
        let mut features = HashMap::new();
        
        // Vent actuator analysis
        if let Some(vent) = era_data.sensor_data.get("vent_pos_1_percent") {
            if !vent.is_empty() {
                let n = vent.len();
                let d_signal: CudaSlice<f32> = self.stream.memcpy_stod(vent)?;
                let d_edge_count: CudaSlice<f32> = self.stream.alloc_zeros::<f32>(1)?;
                let d_duty_cycle: CudaSlice<f32> = self.stream.alloc_zeros::<f32>(1)?;
                
                let config = LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: std::mem::size_of::<f32>() as u32,
                };
                
                self.kernel_manager.launch_actuator_response_kernel(
                    &self.stream,
                    config,
                    &d_signal,
                    &d_edge_count,
                    &d_duty_cycle,
                    n as u32,
                )?;
                
                let edge_count: Vec<f32> = self.stream.memcpy_dtov(&d_edge_count)?;
                let duty_cycle: Vec<f32> = self.stream.memcpy_dtov(&d_duty_cycle)?;
                
                features.insert("vent_edge_count".to_string(), edge_count[0] as f64);
                features.insert("vent_duty_cycle".to_string(), duty_cycle[0] as f64);
                
                // Estimate oscillations from edge count relative to time
                let oscillation_freq = edge_count[0] as f64 / (n as f64 / 3600.0); // edges per hour
                features.insert("vent_oscillations".to_string(), oscillation_freq);
                
                // Simple response time estimate
                let mut response_time = 0.0;
                for i in 1..vent.len().min(300) {
                    if (vent[i] - vent[0]).abs() > 0.9 * (vent.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() - vent[0]).abs() {
                        response_time = i as f64;
                        break;
                    }
                }
                features.insert("vent_response_time".to_string(), response_time);
            }
        }
        
        // Curtain actuator analysis
        if let Some(curtain) = era_data.sensor_data.get("curtain_1_percent") {
            if !curtain.is_empty() {
                let n = curtain.len();
                let d_signal: CudaSlice<f32> = self.stream.memcpy_stod(curtain)?;
                let d_edge_count: CudaSlice<f32> = self.stream.alloc_zeros::<f32>(1)?;
                let d_duty_cycle: CudaSlice<f32> = self.stream.alloc_zeros::<f32>(1)?;
                
                let config = LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: std::mem::size_of::<f32>() as u32,
                };
                
                self.kernel_manager.launch_actuator_response_kernel(
                    &self.stream,
                    config,
                    &d_signal,
                    &d_edge_count,
                    &d_duty_cycle,
                    n as u32,
                )?;
                
                let edge_count: Vec<f32> = self.stream.memcpy_dtov(&d_edge_count)?;
                let duty_cycle: Vec<f32> = self.stream.memcpy_dtov(&d_duty_cycle)?;
                
                features.insert("curtain_edge_count".to_string(), edge_count[0] as f64);
                features.insert("curtain_duty_cycle".to_string(), duty_cycle[0] as f64);
                
                // Control effort (sum of absolute changes)
                let mut control_effort = 0.0;
                for i in 1..curtain.len() {
                    control_effort += (curtain[i] - curtain[i-1]).abs();
                }
                features.insert("curtain_control_effort".to_string(), control_effort as f64);
            }
        }
        
        Ok(features)
    }
    
    fn compute_economic_features(&self, era_data: &EraData) -> Result<HashMap<String, f64>> {
        let mut features = HashMap::new();
        
        // Estimate energy consumption based on actuators and lighting
        let mut estimated_consumption = Vec::new();
        
        // Heating/cooling based on flow temperature
        if let Some(flow_temp) = era_data.sensor_data.get("flow_temp_1_c") {
            for &temp in flow_temp {
                // Simple model: power proportional to temperature difference from ambient
                let power = (temp - 20.0).abs() * 100.0; // Watts
                estimated_consumption.push(power);
            }
        }
        
        // Lighting power
        if let Some(light) = era_data.sensor_data.get("light_intensity_umol") {
            if estimated_consumption.len() < light.len() {
                estimated_consumption.resize(light.len(), 0.0);
            }
            for (i, &intensity) in light.iter().enumerate() {
                if i < estimated_consumption.len() {
                    // Assume 1W per µmol/m²/s for LED efficiency
                    estimated_consumption[i] += intensity;
                }
            }
        }
        
        if !estimated_consumption.is_empty() {
            // Mock energy prices (would come from external data in production)
            let peak_hours = vec![7, 8, 9, 17, 18, 19, 20]; // Peak hours
            let peak_price = 0.25; // $/kWh
            let offpeak_price = 0.10; // $/kWh
            
            let mut peak_consumption = 0.0;
            let mut offpeak_consumption = 0.0;
            let mut total_cost = 0.0;
            
            // Assume data starts at midnight
            for (i, &power) in estimated_consumption.iter().enumerate() {
                let hour = (i / 3600) % 24; // Hour of day
                let energy_kwh = power / 1000.0 / 3600.0; // Convert W to kWh
                
                if peak_hours.contains(&hour) {
                    peak_consumption += energy_kwh;
                    total_cost += energy_kwh * peak_price;
                } else {
                    offpeak_consumption += energy_kwh;
                    total_cost += energy_kwh * offpeak_price;
                }
            }
            
            let total_consumption = peak_consumption + offpeak_consumption;
            
            if total_consumption > 0.0 {
                features.insert("energy_cost_efficiency".to_string(), (total_cost / total_consumption) as f64);
                features.insert("peak_offpeak_ratio".to_string(), 
                    if offpeak_consumption > 0.0 { (peak_consumption / offpeak_consumption) as f64 } else { 0.0 });
            }
            
            // Carbon intensity (mock value - would vary by grid mix)
            let carbon_kg_per_kwh = 0.5;
            features.insert("carbon_intensity".to_string(), (total_consumption * carbon_kg_per_kwh) as f64);
        }
        
        Ok(features)
    }
    
    fn compute_stress_features(&self, era_data: &EraData) -> Result<HashMap<String, f64>> {
        let mut features = HashMap::new();
        
        // Temperature stress
        if let Some(temp) = era_data.sensor_data.get("air_temp_c") {
            if !temp.is_empty() {
                let n = temp.len();
                let low_threshold = 15.0;
                let high_threshold = 28.0;
                
                let d_temp: CudaSlice<f32> = self.stream.memcpy_stod(temp)?;
                let d_count: CudaSlice<f32> = self.stream.alloc_zeros::<f32>(1)?;
                let d_integral: CudaSlice<f32> = self.stream.alloc_zeros::<f32>(1)?;
                
                let config = LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 2 * std::mem::size_of::<f32>() as u32,
                };
                
                self.kernel_manager.launch_stress_count_kernel(
                    &self.stream,
                    config,
                    &d_temp,
                    &d_count,
                    &d_integral,
                    n as u32,
                    low_threshold,
                    high_threshold,
                )?;
                
                let count_result: Vec<f32> = self.stream.memcpy_dtov(&d_count)?;
                let integral_result: Vec<f32> = self.stream.memcpy_dtov(&d_integral)?;
                
                features.insert("temp_stress_count".to_string(), count_result[0] as f64);
                features.insert("temp_stress_integral".to_string(), integral_result[0] as f64);
                
                // Rapid temperature changes (>2°C in 5 minutes)
                let mut rapid_changes = 0;
                let window = 300.min(temp.len());
                for i in window..temp.len() {
                    let change = (temp[i] - temp[i - window]).abs();
                    if change > 2.0 {
                        rapid_changes += 1;
                    }
                }
                features.insert("temp_rapid_changes".to_string(), rapid_changes as f64);
            }
        }
        
        // Humidity stress
        if let Some(rh) = era_data.sensor_data.get("relative_humidity_percent") {
            if !rh.is_empty() {
                let low_rh = 40.0;
                let high_rh = 80.0;
                
                let stress_duration: f64 = rh.iter()
                    .filter(|&&h| h < low_rh || h > high_rh)
                    .count() as f64 / 3600.0; // Convert to hours
                
                features.insert("humidity_stress_duration".to_string(), stress_duration);
            }
        }
        
        Ok(features)
    }
    
    fn compute_thermal_time_features(&self, era_data: &EraData) -> Result<HashMap<String, f64>> {
        let mut features = HashMap::new();
        
        if let Some(temp) = era_data.sensor_data.get("air_temp_c") {
            if temp.is_empty() {
                return Ok(features);
            }
            
            let n = temp.len();
            
            // Growing Degree Days (GDD)
            let base_temp = 10.0;
            let upper_temp = 30.0;
            
            let d_temp: CudaSlice<f32> = self.stream.memcpy_stod(temp)?;
            let d_gdd: CudaSlice<f32> = self.stream.alloc_zeros::<f32>(n)?;
            
            let block_size = 256u32;
            let grid_size = ((n as u32 + block_size - 1) / block_size).min(65535); // Cap grid size
            
            let config = LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };
            
            self.kernel_manager.launch_gdd_kernel(
                &self.stream,
                config,
                &d_temp,
                &d_gdd,
                n as u32,
                base_temp,
                upper_temp,
            )?;
            
            // Copy results back
            let gdd_values: Vec<f32> = self.stream.memcpy_dtov(&d_gdd)?;
            let total_gdd: f64 = gdd_values.iter().map(|&x| x as f64).sum();
            
            // Convert to daily units (assuming 1Hz data)
            let gdd_daily = total_gdd / (3600.0 * 24.0);
            features.insert("growing_degree_days".to_string(), gdd_daily);
            
            // Heating degree hours (below 18°C)
            let heating_base = 18.0;
            let heating_hours: f64 = temp.iter()
                .filter(|&&t| t < heating_base)
                .map(|&t| (heating_base - t) as f64)
                .sum::<f64>() / 3600.0;
            features.insert("heating_degree_hours".to_string(), heating_hours);
            
            // Cooling degree hours (above 24°C)
            let cooling_base = 24.0;
            let cooling_hours: f64 = temp.iter()
                .filter(|&&t| t > cooling_base)
                .map(|&t| (t - cooling_base) as f64)
                .sum::<f64>() / 3600.0;
            features.insert("cooling_degree_hours".to_string(), cooling_hours);
        }
        
        Ok(features)
    }
}