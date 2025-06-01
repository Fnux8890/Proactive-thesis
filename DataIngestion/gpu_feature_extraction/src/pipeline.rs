use anyhow::Result;
use chrono::{DateTime, NaiveDate, Utc};
use sqlx::postgres::PgPool;
use std::sync::Arc;
use std::time::Instant;
use tracing::{info, warn};

use crate::config::Config;
use crate::db;
use crate::features::GpuFeatureExtractor;

#[allow(dead_code)]
pub struct FeaturePipeline {
    config: Config,
    batch_size: usize,
}

#[allow(dead_code)]
impl FeaturePipeline {
    pub fn new(config: Config, batch_size: usize) -> Result<Self> {
        Ok(Self {
            config,
            batch_size,
        })
    }
    
    pub async fn run(
        &self,
        start_date: Option<&str>,
        end_date: Option<&str>,
        era_level: &str,
    ) -> Result<()> {
        info!("Starting feature extraction pipeline (CPU-only)");
        
        // Parse dates if provided
        let start = start_date
            .map(|d| NaiveDate::parse_from_str(d, "%Y-%m-%d"))
            .transpose()?
            .map(|d| d.and_hms_opt(0, 0, 0).unwrap().and_utc());
        
        let end = end_date
            .map(|d| NaiveDate::parse_from_str(d, "%Y-%m-%d"))
            .transpose()?
            .map(|d| d.and_hms_opt(23, 59, 59).unwrap().and_utc());
        
        // Connect to database
        let pool = PgPool::connect(&self.config.database_url).await?;
        
        // Create feature extractor (CPU-only)
        let extractor = GpuFeatureExtractor::new()?;
        
        // Fetch eras within date range
        let eras = if let (Some(s), Some(e)) = (start, end) {
            db::fetch_eras_in_range(&pool, era_level, s, e, 100).await?
        } else {
            db::fetch_eras(&pool, era_level, 100).await?
        };
        
        info!("Found {} eras to process", eras.len());
        
        // Process in batches
        let total_start = Instant::now();
        let mut processed = 0;
        
        for (batch_idx, era_batch) in eras.chunks(self.batch_size).enumerate() {
            let batch_start = Instant::now();
            
            // Fetch data for each era
            let mut era_data_batch = Vec::new();
            for era in era_batch {
                match db::fetch_era_data(&pool, era).await {
                    Ok(data) => era_data_batch.push(data),
                    Err(e) => {
                        warn!("Failed to fetch data for era {}: {}", era.era_id, e);
                        continue;
                    }
                }
            }
            
            if era_data_batch.is_empty() {
                continue;
            }
            
            // Extract features
            let features = extractor.extract_batch(&era_data_batch)?;
            
            // Store features
            db::store_features(&pool, &features, "feature_data").await?;
            
            processed += features.len();
            
            info!(
                "Batch {} completed: {} features in {:.2}s",
                batch_idx,
                features.len(),
                batch_start.elapsed().as_secs_f32()
            );
        }
        
        info!(
            "Pipeline completed: {} features processed in {:.2}s",
            processed,
            total_start.elapsed().as_secs_f32()
        );
        
        Ok(())
    }
}