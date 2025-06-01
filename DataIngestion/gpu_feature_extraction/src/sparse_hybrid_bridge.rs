use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::process::{Command, Stdio};
use std::io::Write;
use tracing::{info, warn};

use crate::sparse_features::{SparseFeatures, GreenhouseSparseFeatures};

/// Bridge between Rust sparse features and Python GPU sparse features
pub struct SparseHybridBridge {
    python_path: String,
    script_path: String,
}

#[derive(Serialize)]
struct SparseGPURequest {
    timestamps: Vec<String>,
    sensors: HashMap<String, Vec<Option<f32>>>,
    energy_prices: Vec<(String, f32)>,
    window_configs: WindowConfigs,
    use_gpu: bool,
}

#[derive(Serialize)]
struct WindowConfigs {
    gap_analysis: Vec<i32>,
    event_detection: Vec<i32>,
    pattern_windows: Vec<i32>,
}

#[derive(Deserialize)]
struct SparseGPUResponse {
    status: String,
    features: Option<HashMap<String, f64>>,
    metadata: Option<HashMap<String, serde_json::Value>>,
    error: Option<String>,
    traceback: Option<String>,
}

impl SparseHybridBridge {
    pub fn new() -> Self {
        Self {
            python_path: "python".to_string(),
            script_path: "/app/sparse_gpu_features.py".to_string(),
        }
    }

    /// Extract sparse features using hybrid Rust+Python approach
    pub async fn extract_hybrid_features(
        &self,
        sensor_data: HashMap<String, (Vec<DateTime<Utc>>, Vec<Option<f32>>)>,
        energy_prices: Option<Vec<(DateTime<Utc>, f32)>>,
    ) -> Result<HashMap<String, serde_json::Value>> {
        // 1. Extract CPU-bound features using Rust
        let mut all_features = HashMap::new();
        
        // Extract basic sparse features for each sensor
        for (sensor_name, (timestamps, values)) in &sensor_data {
            let sparse_features = crate::sparse_features::extract_sparse_features(
                timestamps,
                values,
                sensor_name,
            );
            
            // Convert to JSON values
            self.add_sparse_features_to_map(&mut all_features, sensor_name, &sparse_features);
        }
        
        // Extract greenhouse-specific features
        let greenhouse_features = crate::sparse_features::extract_greenhouse_sparse_features(
            &sensor_data,
            energy_prices.as_deref(),
        )?;
        
        self.add_greenhouse_features_to_map(&mut all_features, &greenhouse_features);
        
        // 2. Extract GPU-accelerated features using Python
        let gpu_features = self.call_python_gpu_features(&sensor_data, &energy_prices).await?;
        
        // Merge GPU features
        for (key, value) in gpu_features {
            all_features.insert(key, serde_json::Value::Number(
                serde_json::Number::from_f64(value).unwrap_or(serde_json::Number::from(0))
            ));
        }
        
        info!("Extracted {} total sparse features using hybrid approach", all_features.len());
        
        Ok(all_features)
    }
    
    /// Call Python script for GPU-accelerated sparse feature extraction
    async fn call_python_gpu_features(
        &self,
        sensor_data: &HashMap<String, (Vec<DateTime<Utc>>, Vec<Option<f32>>)>,
        energy_prices: &Option<Vec<(DateTime<Utc>, f32)>>,
    ) -> Result<HashMap<String, f64>> {
        // Prepare request
        let request = self.prepare_gpu_request(sensor_data, energy_prices)?;
        let request_json = serde_json::to_string(&request)?;
        
        // Call Python script
        let mut child = Command::new(&self.python_path)
            .arg(&self.script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        
        // Send data to Python
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(request_json.as_bytes())?;
            stdin.flush()?;
        }
        
        // Wait for response
        let output = child.wait_with_output()?;
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            warn!("Python GPU feature extraction failed: {}", stderr);
            return Ok(HashMap::new());
        }
        
        // Parse response
        let response: SparseGPUResponse = serde_json::from_slice(&output.stdout)?;
        
        if response.status == "error" {
            warn!("GPU feature extraction error: {:?}", response.error);
            return Ok(HashMap::new());
        }
        
        Ok(response.features.unwrap_or_default())
    }
    
    /// Prepare request for Python GPU processing
    fn prepare_gpu_request(
        &self,
        sensor_data: &HashMap<String, (Vec<DateTime<Utc>>, Vec<Option<f32>>)>,
        energy_prices: &Option<Vec<(DateTime<Utc>, f32)>>,
    ) -> Result<SparseGPURequest> {
        // Find common timestamps across all sensors
        let first_sensor = sensor_data.iter().next()
            .ok_or_else(|| anyhow::anyhow!("No sensor data available"))?;
        
        let timestamps: Vec<String> = first_sensor.1.0.iter()
            .map(|ts| ts.to_rfc3339())
            .collect();
        
        // Prepare sensor data
        let mut sensors = HashMap::new();
        for (name, (_, values)) in sensor_data {
            sensors.insert(name.clone(), values.clone());
        }
        
        // Prepare energy prices
        let energy_prices_formatted = energy_prices.as_ref()
            .map(|prices| {
                prices.iter()
                    .map(|(ts, price)| (ts.to_rfc3339(), *price))
                    .collect()
            })
            .unwrap_or_default();
        
        Ok(SparseGPURequest {
            timestamps,
            sensors,
            energy_prices: energy_prices_formatted,
            window_configs: WindowConfigs {
                gap_analysis: vec![60, 180, 360],
                event_detection: vec![30, 120],
                pattern_windows: vec![1440, 10080],
            },
            use_gpu: true,
        })
    }
    
    /// Add Rust sparse features to the feature map
    fn add_sparse_features_to_map(
        &self,
        map: &mut HashMap<String, serde_json::Value>,
        sensor_name: &str,
        features: &SparseFeatures,
    ) {
        map.insert(format!("{}_coverage_ratio", sensor_name), 
            serde_json::Value::Number(serde_json::Number::from_f64(features.coverage_ratio as f64).unwrap()));
        map.insert(format!("{}_longest_gap_hours", sensor_name), 
            serde_json::Value::Number(serde_json::Number::from_f64(features.longest_gap_hours as f64).unwrap()));
        map.insert(format!("{}_mean_gap_hours", sensor_name), 
            serde_json::Value::Number(serde_json::Number::from_f64(features.mean_gap_hours as f64).unwrap()));
        map.insert(format!("{}_data_points_count", sensor_name), 
            serde_json::Value::Number(serde_json::Number::from(features.data_points_count)));
        
        if let Some(mean) = features.sparse_mean {
            map.insert(format!("{}_sparse_mean", sensor_name), 
                serde_json::Value::Number(serde_json::Number::from_f64(mean as f64).unwrap()));
        }
        if let Some(std) = features.sparse_std {
            map.insert(format!("{}_sparse_std", sensor_name), 
                serde_json::Value::Number(serde_json::Number::from_f64(std as f64).unwrap()));
        }
        
        map.insert(format!("{}_change_count", sensor_name), 
            serde_json::Value::Number(serde_json::Number::from(features.change_count)));
        map.insert(format!("{}_large_change_count", sensor_name), 
            serde_json::Value::Number(serde_json::Number::from(features.large_change_count)));
        map.insert(format!("{}_extreme_high_count", sensor_name), 
            serde_json::Value::Number(serde_json::Number::from(features.extreme_high_count)));
        map.insert(format!("{}_extreme_low_count", sensor_name), 
            serde_json::Value::Number(serde_json::Number::from(features.extreme_low_count)));
    }
    
    /// Add greenhouse-specific features to the map
    fn add_greenhouse_features_to_map(
        &self,
        map: &mut HashMap<String, serde_json::Value>,
        features: &GreenhouseSparseFeatures,
    ) {
        map.insert("lamp_on_hours".to_string(), 
            serde_json::Value::Number(serde_json::Number::from_f64(features.lamp_on_hours as f64).unwrap()));
        map.insert("lamp_switches".to_string(), 
            serde_json::Value::Number(serde_json::Number::from(features.lamp_switches)));
        map.insert("heating_active_hours".to_string(), 
            serde_json::Value::Number(serde_json::Number::from_f64(features.heating_active_hours as f64).unwrap()));
        map.insert("ventilation_active_hours".to_string(), 
            serde_json::Value::Number(serde_json::Number::from_f64(features.ventilation_active_hours as f64).unwrap()));
        map.insert("curtain_movements".to_string(), 
            serde_json::Value::Number(serde_json::Number::from(features.curtain_movements)));
        
        if let Some(gdd) = features.gdd_accumulated {
            map.insert("gdd_accumulated".to_string(), 
                serde_json::Value::Number(serde_json::Number::from_f64(gdd as f64).unwrap()));
        }
        if let Some(dli) = features.dli_accumulated {
            map.insert("dli_accumulated".to_string(), 
                serde_json::Value::Number(serde_json::Number::from_f64(dli as f64).unwrap()));
        }
        
        map.insert("lamp_efficiency_proxy".to_string(), 
            serde_json::Value::Number(serde_json::Number::from_f64(features.lamp_efficiency_proxy as f64).unwrap()));
    }
}