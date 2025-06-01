use anyhow::Result;
use chrono::{DateTime, Utc, Duration};
use polars::prelude::*;
use sqlx::postgres::PgPool;
use std::collections::HashMap;
use std::path::PathBuf;
use tracing::{info, warn};

use crate::db::{create_features_table_if_not_exists, write_features, FeatureSet};
use crate::python_bridge::PythonGpuBridge;
use crate::sparse_pipeline::{SparsePipeline, SparsePipelineConfig, Era};

/// Hybrid pipeline configuration
pub struct HybridPipelineConfig {
    pub sparse_config: SparsePipelineConfig,
    pub use_docker_python: bool,
    pub python_batch_size: usize,
}

impl Default for HybridPipelineConfig {
    fn default() -> Self {
        Self {
            sparse_config: SparsePipelineConfig::default(),
            use_docker_python: true,
            python_batch_size: 1000,
        }
    }
}

/// Hybrid pipeline that uses Rust for data handling and Python for GPU features
pub struct HybridPipeline {
    pool: PgPool,
    config: HybridPipelineConfig,
    sparse_pipeline: SparsePipeline,
    python_bridge: PythonGpuBridge,
}

impl HybridPipeline {
    pub fn new(pool: PgPool, config: HybridPipelineConfig) -> Self {
        let sparse_pipeline = SparsePipeline::new(pool.clone(), config.sparse_config.clone());
        let python_bridge = PythonGpuBridge::new(config.use_docker_python);
        
        Self {
            pool,
            config,
            sparse_pipeline,
            python_bridge,
        }
    }

    /// Run the hybrid pipeline
    pub async fn run_pipeline(
        &mut self,
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
    ) -> Result<PipelineResults> {
        info!("Starting hybrid pipeline from {} to {}", start_date, end_date);

        // Stage 1: Load and prepare data using Rust sparse pipeline
        let hourly_df = self.sparse_pipeline.stage1_aggregate_hourly(start_date, end_date).await?;
        
        // Stage 2: Fill missing values
        let filled_df = self.sparse_pipeline.stage2_conservative_fill(hourly_df).await?;
        
        // Stage 3: Extract features using Python GPU
        let feature_results = self.stage3_python_gpu_features(&filled_df).await?;
        
        // Stage 4: Era detection (using Rust) - we'll use empty features for now
        let empty_features = Vec::new();
        let eras = self.sparse_pipeline.stage4_create_eras(empty_features).await?;
        
        // Stage 5: Write results to database
        self.write_results_to_db(&feature_results, &eras).await?;
        
        Ok(PipelineResults {
            num_hours_processed: filled_df.height(),
            num_features_extracted: feature_results.len(),
            num_eras_detected: eras.len(),
        })
    }

    /// Stage 3: Extract features using Python GPU service
    async fn stage3_python_gpu_features(
        &self,
        df: &DataFrame,
    ) -> Result<Vec<HashMap<String, f64>>> {
        info!("Starting Python GPU feature extraction");
        
        // Get sensor columns
        let sensor_columns: Vec<String> = df.get_column_names()
            .iter()
            .filter(|name| !matches!(name.as_ref(), "timestamp" | "hour" | "month"))
            .map(|s| s.to_string())
            .collect();
        
        info!("Processing {} sensor columns", sensor_columns.len());
        
        // Convert DataFrame to format suitable for Python
        let timestamps = df.column("timestamp")?.datetime()?
            .as_datetime_iter()
            .filter_map(|dt| dt.map(|ts| {
                // ts is timestamp in milliseconds since epoch
                let timestamp_ms = ts.and_utc().timestamp_millis();
                let secs = timestamp_ms / 1000;
                let nanos = ((timestamp_ms % 1000) * 1_000_000) as u32;
                DateTime::from_timestamp(secs, nanos).unwrap()
            }))
            .collect::<Vec<_>>();
        
        // Prepare sensor data
        let mut sensor_data = HashMap::new();
        for col_name in &sensor_columns {
            if let Ok(column) = df.column(col_name) {
                if let Ok(values) = column.cast(&DataType::Float32) {
                    let float_values: Vec<f32> = values.f32()?
                        .into_iter()
                        .map(|v| v.unwrap_or(0.0))
                        .collect();
                    sensor_data.insert(col_name.clone(), float_values);
                }
            }
        }
        
        // Process in sliding windows
        let window_size = self.config.sparse_config.window_hours;
        let slide_size = self.config.sparse_config.slide_hours;
        let mut all_features = Vec::new();
        
        for start_idx in (0..timestamps.len()).step_by(slide_size) {
            let end_idx = (start_idx + window_size).min(timestamps.len());
            if end_idx <= start_idx {
                break;
            }
            
            // Extract window data
            let window_timestamps = timestamps[start_idx..end_idx].to_vec();
            let mut window_data = HashMap::new();
            
            for (sensor, values) in &sensor_data {
                window_data.insert(
                    sensor.clone(),
                    values[start_idx..end_idx].to_vec()
                );
            }
            
            // Call Python for feature extraction
            match self.python_bridge.extract_features(
                window_timestamps,
                window_data,
                vec![30, 120], // 30 min and 2 hour windows
            ) {
                Ok(features) => {
                    info!("Extracted {} features for window starting at index {}", 
                          features.len(), start_idx);
                    all_features.push(features);
                }
                Err(e) => {
                    warn!("Failed to extract features for window at {}: {}", start_idx, e);
                    // Continue with next window
                }
            }
        }
        
        info!("Completed Python GPU feature extraction: {} feature sets", all_features.len());
        Ok(all_features)
    }

    /// Write results to database
    async fn write_results_to_db(
        &self,
        features: &[HashMap<String, f64>],
        eras: &[Era],
    ) -> Result<()> {
        info!("Writing {} feature sets and {} eras to database", features.len(), eras.len());
        
        // Ensure features table exists
        create_features_table_if_not_exists(&self.pool, "feature_data").await?;
        
        // Convert HashMap features to FeatureSet format
        let feature_sets: Vec<FeatureSet> = features.iter().enumerate().map(|(idx, feature_map)| {
            FeatureSet {
                era_id: idx as i32,
                era_level: "hybrid_window".to_string(),
                features: feature_map.clone(),
                computed_at: Utc::now(),
            }
        }).collect();
        
        // Write features to database
        if !feature_sets.is_empty() {
            write_features(&self.pool, "feature_data", feature_sets).await?;
        }
        
        info!("Successfully wrote results to database");
        Ok(())
    }
}

/// Results from the hybrid pipeline
pub struct PipelineResults {
    pub num_hours_processed: usize,
    pub num_features_extracted: usize,
    pub num_eras_detected: usize,
}