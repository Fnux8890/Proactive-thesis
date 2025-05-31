use anyhow::Result;
use clap::Parser;
use cudarc::driver::safe::CudaContext;
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
mod data_quality;

use crate::features::GpuFeatureExtractor;
use crate::sparse_pipeline::{SparsePipeline, SparsePipelineConfig};

#[derive(Parser, Debug)]
#[command(name = "gpu_feature_extraction")]
#[command(about = "GPU-accelerated feature extraction for greenhouse data", long_about = None)]
struct Args {
    /// Database connection string
    #[arg(long, env = "DATABASE_URL")]
    database_url: String,

    /// Enable sparse pipeline mode (handles sparse data)
    #[arg(long)]
    sparse_mode: bool,

    /// Start date for sparse pipeline (YYYY-MM-DD)
    #[arg(long, requires = "sparse_mode")]
    start_date: Option<String>,

    /// End date for sparse pipeline (YYYY-MM-DD)
    #[arg(long, requires = "sparse_mode")]
    end_date: Option<String>,

    /// Era level to process (A, B, or C)
    #[arg(long, default_value = "B")]
    era_level: String,

    /// Minimum era size in rows
    #[arg(long, default_value = "100")]
    min_era_rows: usize,

    /// Batch size for GPU processing (also used as window size in sparse mode)
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

    if args.sparse_mode {
        // Run sparse pipeline mode
        info!("Starting GPU sparse pipeline mode");
        
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
        
        // Initialize GPU if available
        let disable_gpu = std::env::var("DISABLE_GPU")
            .unwrap_or_else(|_| "false".to_string())
            .to_lowercase();
        
        let pipeline = if disable_gpu != "true" {
            match CudaContext::new(0) {
                Ok(ctx) => {
                    info!("CUDA context initialized for sparse pipeline");
                    let stream = ctx.default_stream();
                    let extractor = GpuFeatureExtractor::new(ctx, stream)?;
                    
                    let config = SparsePipelineConfig {
                        window_hours: args.batch_size.max(24),  // Use batch_size as window_hours
                        slide_hours: args.batch_size.max(24) / 4,  // 25% overlap
                        ..SparsePipelineConfig::default()
                    };
                    
                    SparsePipeline::new(pool.clone(), config)
                        .with_gpu_extractor(extractor)
                },
                Err(e) => {
                    warn!("GPU initialization failed: {}. Using CPU fallback.", e);
                    let config = SparsePipelineConfig::default();
                    SparsePipeline::new(pool.clone(), config)
                }
            }
        } else {
            info!("GPU disabled by environment variable (DISABLE_GPU={})", disable_gpu);
            let config = SparsePipelineConfig::default();
            SparsePipeline::new(pool.clone(), config)
        };
        
        let start_time = Instant::now();
        let results = pipeline.run_pipeline(start_date, end_date).await?;
        let elapsed = start_time.elapsed();
        
        info!("Sparse pipeline complete in {:.2}s:", elapsed.as_secs_f32());
        info!("  - Hourly data points: {}", results.filled_hourly_data.height());
        info!("  - Window features: {}", results.daily_features.len());
        info!("  - Monthly eras: {}", results.monthly_eras.len());
        
        // Print performance metrics
        if !results.daily_features.is_empty() {
            let features_per_sec = results.daily_features.len() as f32 / elapsed.as_secs_f32();
            info!("  - Performance: {:.1} features/second", features_per_sec);
        }
        
    } else {
        // Original era-based mode
        info!("Starting GPU feature extraction for level {}", args.era_level);

        // Create features table if it doesn't exist
        db::create_features_table_if_not_exists(&pool, &args.features_table).await?;
        info!("Ensured features table '{}' exists", args.features_table);

        // Initialize CUDA context
        let ctx = CudaContext::new(0)?;
        info!("CUDA context initialized for device 0");
        
        // Get default stream
        let stream = ctx.default_stream();

        // Create feature extractor with cudarc
        let extractor = GpuFeatureExtractor::new(ctx, stream)?;

        // Fetch eras to process
        let eras = db::fetch_eras(&pool, &args.era_level, args.min_era_rows).await?;
        let total_eras = eras.len();
        
        let eras_to_process = if let Some(max) = args.max_eras {
            &eras[..max.min(total_eras)]
        } else {
            &eras
        };
        
        info!("Processing {} eras (out of {} total)", eras_to_process.len(), total_eras);

        if args.benchmark {
            info!("Running in benchmark mode");
            run_benchmark(&pool, &extractor, eras_to_process, args.batch_size).await?;
        } else {
            run_extraction(&pool, &extractor, eras_to_process, &args).await?;
        }
    }

    Ok(())
}

async fn run_extraction(
    pool: &sqlx::PgPool,
    extractor: &GpuFeatureExtractor,
    eras: &[db::Era],
    args: &Args,
) -> Result<()> {
    let start_time = Instant::now();
    let mut processed = 0;
    
    for batch in eras.chunks(args.batch_size) {
        let batch_start = Instant::now();
        
        // Fetch data for all eras in batch
        let mut batch_data = Vec::new();
        for era in batch {
            info!("Fetching data for era {} ({} to {})", 
                era.era_id, era.start_time, era.end_time);
            
            match db::fetch_era_data(pool, era).await {
                Ok(data) => {
                    let total_points: usize = data.sensor_data.values()
                        .map(|v| v.len())
                        .next()
                        .unwrap_or(0);
                    info!("Fetched {} data points for era {}", total_points, era.era_id);
                    
                    if total_points > 500_000 {
                        warn!("Era {} has {} data points - this may cause memory issues!", 
                            era.era_id, total_points);
                    }
                    
                    batch_data.push(data);
                },
                Err(e) => warn!("Failed to fetch data for era {}: {}", era.era_id, e),
            }
        }

        if batch_data.is_empty() {
            continue;
        }

        // Extract features on GPU
        info!("Extracting features for {} eras on GPU", batch_data.len());
        let features = match extractor.extract_batch(&batch_data) {
            Ok(f) => f,
            Err(e) => {
                warn!("GPU feature extraction failed: {}", e);
                warn!("This often indicates GPU memory exhaustion or CUDA errors");
                return Err(e);
            }
        };
        info!("Successfully extracted {} feature sets", features.len());

        // Write to database
        db::write_features(pool, &args.features_table, features).await?;

        processed += batch_data.len();
        let batch_time = batch_start.elapsed();
        
        info!(
            "Processed batch: {}/{} eras in {:.2}s ({:.0} eras/sec)",
            processed,
            eras.len(),
            batch_time.as_secs_f32(),
            batch_data.len() as f32 / batch_time.as_secs_f32()
        );
    }

    let total_time = start_time.elapsed();
    info!(
        "Feature extraction complete: {} eras in {:.2}s ({:.0} eras/sec)",
        processed,
        total_time.as_secs_f32(),
        processed as f32 / total_time.as_secs_f32()
    );

    Ok(())
}

async fn run_benchmark(
    pool: &sqlx::PgPool,
    extractor: &GpuFeatureExtractor,
    eras: &[db::Era],
    _batch_size: usize,
) -> Result<()> {
    info!("Running benchmark with {} eras", eras.len().min(100));
    
    // Use first 100 eras for benchmark
    let benchmark_eras = &eras[..eras.len().min(100)];
    
    // Warm up
    info!("Warming up GPU...");
    for _ in 0..3 {
        if let Some(era) = benchmark_eras.first() {
            let data = db::fetch_era_data(pool, era).await?;
            let _ = extractor.extract_batch(&[data])?;
        }
    }
    
    // Benchmark different batch sizes
    for &size in &[1, 10, 50, 100, 500, 1000] {
        if size > benchmark_eras.len() {
            break;
        }
        
        let batch = &benchmark_eras[..size];
        let mut batch_data = Vec::new();
        
        for era in batch {
            let data = db::fetch_era_data(pool, era).await?;
            batch_data.push(data);
        }
        
        let start = Instant::now();
        let _ = extractor.extract_batch(&batch_data)?;
        let elapsed = start.elapsed();
        
        info!(
            "Batch size {}: {:.2}ms total, {:.2}ms per era",
            size,
            elapsed.as_secs_f64() * 1000.0,
            elapsed.as_secs_f64() * 1000.0 / size as f64
        );
    }
    
    Ok(())
}