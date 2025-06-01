use anyhow::Result;
use chrono::Utc;
use std::collections::HashMap;

use crate::db::{EraData, FeatureSet};
use crate::external_data::PhenotypeData;

/// CPU-only feature extractor
/// GPU operations have been moved to Python
pub struct GpuFeatureExtractor {
    // No CUDA context needed anymore
}

impl GpuFeatureExtractor {
    pub fn new() -> Result<Self> {
        Ok(Self {})
    }
    
    pub fn extract_batch(&self, era_batch: &[EraData]) -> Result<Vec<FeatureSet>> {
        let mut results = Vec::with_capacity(era_batch.len());
        
        for era_data in era_batch {
            let features = self.extract_era_features(era_data)?;
            results.push(features);
        }
        
        Ok(results)
    }
    
    pub fn extract_batch_enhanced(
        &self,
        era_batch: &[EraData],
        _resolutions: &[(&str, i64)],
        _phenotype_data: Option<&PhenotypeData>,
    ) -> Result<Vec<FeatureSet>> {
        // For now, just use the basic extraction
        // GPU-accelerated features are handled in Python
        self.extract_batch(era_batch)
    }
    
    fn extract_era_features(&self, era_data: &EraData) -> Result<FeatureSet> {
        let mut all_features = HashMap::new();
        
        // Extract basic CPU features for each sensor
        for (sensor_name, sensor_values) in &era_data.sensor_data {
            if sensor_values.is_empty() {
                continue;
            }
            
            // Basic statistical features (CPU implementation)
            let stats = self.compute_statistical_features_cpu(sensor_name, sensor_values)?;
            for (feature_name, value) in stats {
                all_features.insert(format!("{}_{}", sensor_name, feature_name), value);
            }
        }
        
        Ok(FeatureSet {
            era_id: era_data.era.era_id,
            era_level: era_data.era.era_level.clone(),
            features: all_features,
            computed_at: Utc::now(),
        })
    }
    
    fn compute_statistical_features_cpu(
        &self,
        _sensor_name: &str,
        values: &[f32],
    ) -> Result<HashMap<String, f64>> {
        if values.is_empty() {
            return Ok(HashMap::new());
        }
        
        let mut features = HashMap::new();
        
        // Basic statistics
        let sum: f32 = values.iter().sum();
        let mean = sum / values.len() as f32;
        features.insert("mean".to_string(), mean as f64);
        
        let variance: f32 = values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / values.len() as f32;
        let std = variance.sqrt();
        features.insert("std".to_string(), std as f64);
        
        let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        features.insert("min".to_string(), min as f64);
        features.insert("max".to_string(), max as f64);
        features.insert("range".to_string(), (max - min) as f64);
        
        Ok(features)
    }
}