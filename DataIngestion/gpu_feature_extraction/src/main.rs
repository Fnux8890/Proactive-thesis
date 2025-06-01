use anyhow::Result;
use clap::Parser;
use sqlx::postgres::PgPoolOptions;
use std::time::Instant;
use tracing::{info, warn};
use chrono::{DateTime, Utc};

mod config;
mod db;
mod features;
mod kernels;
mod pipeline;
mod sparse_pipeline;
mod enhanced_sparse_pipeline;
mod enhanced_features;
mod data_quality;
mod external_data;
mod python_bridge;
mod hybrid_pipeline;
mod sparse_features;

use crate::features::GpuFeatureExtractor;
use crate::sparse_pipeline::{SparsePipeline, SparsePipelineConfig};
use crate::enhanced_sparse_pipeline::{EnhancedSparsePipeline, EnhancedPipelineConfig};
use crate::hybrid_pipeline::{HybridPipeline, HybridPipelineConfig};

#[derive(Parser, Debug)]
#[command(name = "gpu_feature_extraction")]
#[command(about = "Feature extraction for greenhouse data (GPU processing moved to Python)", long_about = None)]
struct Args {
    /// Database connection string
    #[arg(long, env = "DATABASE_URL")]
    database_url: String,

    /// Enable sparse pipeline mode (handles sparse data)
    #[arg(long)]
    sparse_mode: bool,

    /// Enable enhanced sparse pipeline with external data integration
    #[arg(long)]
    enhanced_mode: bool,

    /// Enable hybrid mode (Rust data handling + Python GPU features)
    #[arg(long)]
    hybrid_mode: bool,

    /// Start date for sparse pipeline (YYYY-MM-DD)
    #[arg(long)]
    start_date: Option<String>,

    /// End date for sparse pipeline (YYYY-MM-DD)
    #[arg(long)]
    end_date: Option<String>,

    /// Era level to process (A, B, or C)
    #[arg(long, default_value = "B")]
    era_level: String,

    /// Minimum era size in rows
    #[arg(long, default_value = "100")]
    min_era_rows: usize,

    /// Batch size for processing (also used as window size in sparse mode)
    #[arg(long, default_value = "24")]
    batch_size: usize,

    /// Features table name
    #[arg(long, default_value = "feature_data")]
    features_table: String,

    /// Maximum eras to process (for testing)
    #[arg(long)]
    max_eras: Option<usize>,

    /// Enable benchmark mode
    #[arg(long)]
    benchmark: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("gpu_feature_extraction=info".parse()?)
        )
        .init();

    // Load environment variables
    dotenv::dotenv().ok();
    
    let args = Args::parse();

    // Initialize database pool
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&args.database_url)
        .await?;

    if args.enhanced_mode {
        // Run enhanced sparse pipeline mode
        info!("Starting Enhanced sparse pipeline mode (CPU-only)");
        
        let start_date = args.start_date.as_ref()
            .map(|s| DateTime::parse_from_rfc3339(&format!("{}T00:00:00Z", s))
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| panic!("Invalid start date format")))
            .unwrap_or_else(|| Utc::now() - chrono::Duration::days(365));
            
        let end_date = args.end_date.as_ref()
            .map(|s| DateTime::parse_from_rfc3339(&format!("{}T00:00:00Z", s))
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| panic!("Invalid end date format")))
            .unwrap_or_else(|| Utc::now());
        
        // CPU-only pipeline
        let config = EnhancedPipelineConfig {
            window_hours: args.batch_size.max(24),
            slide_hours: args.batch_size.max(24) / 4,
            enable_weather_features: true,
            enable_energy_features: true,
            enable_growth_features: true,
            enable_multiresolution: true,
            enable_extended_statistics: true,
            enable_coupling_features: true,
            enable_temporal_features: true,
            ..EnhancedPipelineConfig::default()
        };
        
        let mut enhanced_pipeline = EnhancedSparsePipeline::new(pool.clone(), config);
        
        // Run the pipeline
        let total_start = Instant::now();
        let _results = enhanced_pipeline.run_enhanced_pipeline(start_date, end_date).await?;
        
        info!("Enhanced sparse pipeline completed in {:.2} seconds", 
              total_start.elapsed().as_secs_f32());
              
    } else if args.sparse_mode {
        // Run sparse pipeline mode (CPU-only with efficient sparse handling)
        info!("Starting sparse pipeline mode");
        
        let start_date = args.start_date.as_ref()
            .map(|s| DateTime::parse_from_rfc3339(&format!("{}T00:00:00Z", s))
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| panic!("Invalid start date format")))
            .unwrap_or_else(|| Utc::now() - chrono::Duration::days(365));
            
        let end_date = args.end_date.as_ref()
            .map(|s| DateTime::parse_from_rfc3339(&format!("{}T00:00:00Z", s))
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| panic!("Invalid end date format")))
            .unwrap_or_else(|| Utc::now());
        
        let mut config = SparsePipelineConfig::default();
        config.window_hours = args.batch_size.max(24);
        config.min_hourly_coverage = 0.05;  // 5% minimum data coverage
        
        let mut sparse_pipeline = SparsePipeline::new(pool.clone(), config);
        
        // Run the pipeline
        let total_start = Instant::now();
        let _results = sparse_pipeline.run_pipeline(start_date, end_date).await?;
        
        info!("Sparse pipeline completed in {:.2} seconds", 
              total_start.elapsed().as_secs_f32());
              
    } else if args.hybrid_mode {
        // Run hybrid mode (Rust preprocessing + Python GPU features)
        info!("Starting hybrid pipeline mode");
        
        let start_date = args.start_date.as_ref()
            .map(|s| DateTime::parse_from_rfc3339(&format!("{}T00:00:00Z", s))
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| panic!("Invalid start date format")))
            .unwrap_or_else(|| Utc::now() - chrono::Duration::days(365));
            
        let end_date = args.end_date.as_ref()
            .map(|s| DateTime::parse_from_rfc3339(&format!("{}T00:00:00Z", s))
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| panic!("Invalid end date format")))
            .unwrap_or_else(|| Utc::now());
        
        let mut sparse_config = SparsePipelineConfig::default();
        sparse_config.window_hours = args.batch_size.max(24);
        
        let config = HybridPipelineConfig {
            sparse_config,
            use_docker_python: false,  // Use direct Python for now
            python_batch_size: 1000,
        };
        
        let mut hybrid_pipeline = HybridPipeline::new(pool.clone(), config);
        
        // Run the pipeline
        let total_start = Instant::now();
        let _results = hybrid_pipeline.run_pipeline(start_date, end_date).await?;
        
        info!("Hybrid pipeline completed in {:.2} seconds", 
              total_start.elapsed().as_secs_f32());
              
    } else {
        // Run standard mode (CPU-only)
        info!("Starting standard feature extraction mode (CPU-only)");
        
        // Create CPU feature extractor
        let extractor = GpuFeatureExtractor::new()?;
        
        // Fetch eras from database
        let eras = db::fetch_eras(&pool, &args.era_level, args.min_era_rows).await?;
        info!("Found {} eras to process", eras.len());
        
        let eras_to_process = if let Some(max) = args.max_eras {
            &eras[..max.min(eras.len())]
        } else {
            &eras
        };
        
        // Process in batches
        let batch_start = Instant::now();
        for (batch_idx, era_batch) in eras_to_process.chunks(args.batch_size).enumerate() {
            info!("Processing batch {} with {} eras", batch_idx, era_batch.len());
            
            // Fetch data for each era in the batch
            let mut era_data_batch = Vec::new();
            for era in era_batch {
                let data = db::fetch_era_data(&pool, era).await?;
                era_data_batch.push(data);
            }
            
            // Extract features
            let features = extractor.extract_batch(&era_data_batch)?;
            
            // Store features
            db::store_features(&pool, &features, &args.features_table).await?;
            
            info!("Batch {} completed", batch_idx);
        }
        
        info!("All batches completed in {:.2} seconds", 
              batch_start.elapsed().as_secs_f32());
    }

    Ok(())
}