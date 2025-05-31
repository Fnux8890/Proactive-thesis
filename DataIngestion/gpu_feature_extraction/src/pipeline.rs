use anyhow::Result;
use chrono::{DateTime, NaiveDate, Utc};
use cudarc::driver::safe::{CudaContext, CudaStream};
use sqlx::postgres::PgPool;
use std::sync::Arc;
use std::time::Instant;
use tracing::{info, warn};

use crate::config::Config;
use crate::db;
use crate::features::GpuFeatureExtractor;

#[allow(dead_code)]
pub struct FeaturePipeline {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    config: Config,
    batch_size: usize,
}

#[allow(dead_code)]
impl FeaturePipeline {
    pub fn new(ctx: Arc<CudaContext>, config: Config, batch_size: usize) -> Result<Self> {
        let stream = ctx.default_stream();
        
        Ok(Self {
            ctx,
            stream,
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
        info!("Starting feature extraction pipeline");
        
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
        
        // Create feature extractor
        let extractor = GpuFeatureExtractor::new(self.ctx.clone(), self.stream.clone())?;
        
        // Determine which eras to process
        let eras = self.fetch_eras_for_period(&pool, era_level, start, end).await?;
        
        info!("Found {} eras to process", eras.len());
        
        // Process in batches
        let start_time = Instant::now();
        let mut processed = 0;
        
        for batch in eras.chunks(self.batch_size) {
            let batch_features = self.process_era_batch(&pool, &extractor, batch).await?;
            
            // Write features to database
            db::write_features(&pool, "feature_data", batch_features).await?;
            
            processed += batch.len();
            info!("Processed {}/{} eras", processed, eras.len());
        }
        
        let elapsed = start_time.elapsed();
        info!(
            "Feature extraction complete: {} eras in {:.2}s ({:.0} eras/sec)",
            processed,
            elapsed.as_secs_f32(),
            processed as f32 / elapsed.as_secs_f32()
        );
        
        Ok(())
    }
    
    pub async fn run_benchmark(&self) -> Result<()> {
        info!("Running GPU feature extraction benchmark");
        
        // Connect to database
        let pool = PgPool::connect(&self.config.database_url).await?;
        
        // Create feature extractor
        let extractor = GpuFeatureExtractor::new(self.ctx.clone(), self.stream.clone())?;
        
        // Get sample eras for benchmarking
        let eras = db::fetch_eras(&pool, "A", 100).await?;
        let sample_eras = &eras[..eras.len().min(10)];
        
        info!("Benchmarking with {} sample eras", sample_eras.len());
        
        // Warm up
        for _ in 0..3 {
            let _ = self.process_era_batch(&pool, &extractor, sample_eras).await?;
        }
        
        // Benchmark different configurations
        for &batch_size in &[1, 5, 10] {
            if batch_size > sample_eras.len() {
                break;
            }
            
            let batch = &sample_eras[..batch_size];
            let start = Instant::now();
            
            let _ = self.process_era_batch(&pool, &extractor, batch).await?;
            
            let elapsed = start.elapsed();
            info!(
                "Batch size {}: {:.2}ms total, {:.2}ms per era",
                batch_size,
                elapsed.as_secs_f64() * 1000.0,
                elapsed.as_secs_f64() * 1000.0 / batch_size as f64
            );
        }
        
        Ok(())
    }
    
    async fn fetch_eras_for_period(
        &self,
        pool: &PgPool,
        era_level: &str,
        start: Option<DateTime<Utc>>,
        end: Option<DateTime<Utc>>,
    ) -> Result<Vec<db::Era>> {
        let base_query = r#"
            SELECT 
                era_id,
                era_level,
                start_time,
                end_time,
                row_count
            FROM era_detection_results
            WHERE era_level = $1 
            AND row_count >= 100"#;

        let eras = match (start, end) {
            (Some(start_time), Some(end_time)) => {
                let query = format!("{} AND start_time >= $2 AND end_time <= $3", base_query);
                sqlx::query_as::<_, db::Era>(&query)
                    .bind(era_level)
                    .bind(start_time)
                    .bind(end_time)
                    .fetch_all(pool)
                    .await?
            }
            (Some(start_time), None) => {
                let query = format!("{} AND start_time >= $2", base_query);
                sqlx::query_as::<_, db::Era>(&query)
                    .bind(era_level)
                    .bind(start_time)
                    .fetch_all(pool)
                    .await?
            }
            _ => {
                sqlx::query_as::<_, db::Era>(base_query)
                    .bind(era_level)
                    .fetch_all(pool)
                    .await?
            }
        };
        
        Ok(eras)
    }
    
    async fn process_era_batch(
        &self,
        pool: &PgPool,
        extractor: &GpuFeatureExtractor,
        eras: &[db::Era],
    ) -> Result<Vec<db::FeatureSet>> {
        let mut era_data_batch = Vec::new();
        
        for era in eras {
            match db::fetch_era_data(pool, era).await {
                Ok(data) => era_data_batch.push(data),
                Err(e) => warn!("Failed to fetch data for era {}: {}", era.era_id, e),
            }
        }
        
        if era_data_batch.is_empty() {
            return Ok(Vec::new());
        }
        
        extractor.extract_batch(&era_data_batch)
    }
}