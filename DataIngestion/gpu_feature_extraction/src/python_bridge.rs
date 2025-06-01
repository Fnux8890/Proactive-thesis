use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::process::{Command, Stdio};
use std::io::Write;
use tracing::{info, warn, error};
use chrono::{DateTime, Utc};

/// Request structure for Python GPU feature extraction
#[derive(Debug, Serialize)]
pub struct FeatureExtractionRequest {
    pub timestamps: Vec<String>,
    pub sensors: HashMap<String, Vec<f32>>,
    pub window_sizes: Vec<i32>,
    pub use_gpu: bool,
}

/// Response structure from Python GPU feature extraction
#[derive(Debug, Deserialize)]
pub struct FeatureExtractionResponse {
    pub status: String,
    pub features: Option<HashMap<String, f64>>,
    pub error: Option<String>,
    pub metadata: Option<ResponseMetadata>,
}

#[derive(Debug, Deserialize)]
pub struct ResponseMetadata {
    pub num_samples: usize,
    pub num_features: usize,
    pub gpu_used: bool,
}

/// Python GPU feature extraction bridge
pub struct PythonGpuBridge {
    python_script: String,
    use_docker: bool,
}

impl PythonGpuBridge {
    /// Create a new Python GPU bridge
    pub fn new(use_docker: bool) -> Self {
        Self {
            python_script: "/app/minimal_gpu_features.py".to_string(),
            use_docker,
        }
    }

    /// Extract features using Python GPU service
    pub fn extract_features(
        &self,
        timestamps: Vec<DateTime<Utc>>,
        sensor_data: HashMap<String, Vec<f32>>,
        window_sizes: Vec<i32>,
    ) -> Result<HashMap<String, f64>> {
        // Prepare request
        let request = FeatureExtractionRequest {
            timestamps: timestamps.iter().map(|t| t.to_rfc3339()).collect(),
            sensors: sensor_data,
            window_sizes,
            use_gpu: true,
        };

        // Serialize request to JSON
        let request_json = serde_json::to_string(&request)?;
        
        info!("Sending feature extraction request with {} timestamps", timestamps.len());

        // Call Python service
        let response = if self.use_docker {
            self.call_docker_service(&request_json)?
        } else {
            self.call_local_python(&request_json)?
        };

        // Parse response
        let result: FeatureExtractionResponse = serde_json::from_str(&response)
            .context("Failed to parse Python response")?;

        match result.status.as_str() {
            "success" => {
                if let Some(metadata) = &result.metadata {
                    info!(
                        "Feature extraction successful: {} features from {} samples (GPU: {})",
                        metadata.num_features, metadata.num_samples, metadata.gpu_used
                    );
                }
                Ok(result.features.unwrap_or_default())
            }
            "error" => {
                let error_msg = result.error.unwrap_or_else(|| "Unknown error".to_string());
                error!("Python feature extraction failed: {}", error_msg);
                Err(anyhow::anyhow!("Feature extraction failed: {}", error_msg))
            }
            _ => {
                warn!("Unexpected response status: {}", result.status);
                Err(anyhow::anyhow!("Unexpected response status: {}", result.status))
            }
        }
    }

    /// Call Python script in Docker container
    fn call_docker_service(&self, input_json: &str) -> Result<String> {
        let mut child = Command::new("docker")
            .args(&[
                "run",
                "--rm",
                "-i",
                "--gpus", "all",
                "gpu-feature-python:latest",
                "python", "/app/minimal_gpu_features.py"
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("Failed to spawn Docker container")?;

        // Write input to stdin
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(input_json.as_bytes())?;
            stdin.flush()?;
        }

        // Wait for completion and get output
        let output = child.wait_with_output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            error!("Docker command failed: {}", stderr);
            return Err(anyhow::anyhow!("Docker command failed: {}", stderr));
        }

        Ok(String::from_utf8(output.stdout)?)
    }

    /// Call local Python script directly
    fn call_local_python(&self, input_json: &str) -> Result<String> {
        let mut child = Command::new("python3")
            .arg(&self.python_script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("Failed to spawn Python process")?;

        // Write input to stdin
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(input_json.as_bytes())?;
            stdin.flush()?;
        }

        // Wait for completion and get output
        let output = child.wait_with_output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            error!("Python script failed: {}", stderr);
            return Err(anyhow::anyhow!("Python script failed: {}", stderr));
        }

        Ok(String::from_utf8(output.stdout)?)
    }

    /// Extract features for a batch of windows
    pub async fn extract_features_batch(
        &self,
        windows: Vec<(DateTime<Utc>, DateTime<Utc>, HashMap<String, Vec<f32>>)>,
    ) -> Result<Vec<HashMap<String, f64>>> {
        let mut results = Vec::new();

        for (start, end, data) in windows {
            // Generate timestamps for the window
            let duration = end - start;
            let num_samples = data.values().next().map(|v| v.len()).unwrap_or(0);
            
            let mut timestamps = Vec::new();
            for i in 0..num_samples {
                let fraction = i as f64 / (num_samples.max(1) - 1) as f64;
                let timestamp = start + chrono::Duration::seconds((duration.num_seconds() as f64 * fraction) as i64);
                timestamps.push(timestamp);
            }

            // Extract features
            match self.extract_features(timestamps, data, vec![30, 120]) {
                Ok(features) => results.push(features),
                Err(e) => {
                    warn!("Failed to extract features for window {}-{}: {}", start, end, e);
                    // Return empty features for this window
                    results.push(HashMap::new());
                }
            }
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_extraction_request_serialization() {
        let mut sensors = HashMap::new();
        sensors.insert("temperature".to_string(), vec![20.0, 21.0, 22.0]);
        
        let request = FeatureExtractionRequest {
            timestamps: vec!["2024-01-01T00:00:00Z".to_string()],
            sensors,
            window_sizes: vec![30, 120],
            use_gpu: true,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("temperature"));
        assert!(json.contains("window_sizes"));
    }

    #[test]
    fn test_feature_extraction_response_parsing() {
        let json = r#"{
            "status": "success",
            "features": {
                "temperature_mean": 21.0,
                "temperature_std": 1.0
            },
            "metadata": {
                "num_samples": 100,
                "num_features": 2,
                "gpu_used": true
            }
        }"#;

        let response: FeatureExtractionResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.status, "success");
        assert!(response.features.is_some());
        assert_eq!(response.features.unwrap().len(), 2);
    }
}