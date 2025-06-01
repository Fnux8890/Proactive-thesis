use anyhow::Result;
use clap::Parser;
use sqlx::postgres::PgPoolOptions;
use std::time::Instant;
use tracing::{info, warn};

mod config;
mod db;
mod features;
mod sparse_features;

use crate::features::GpuFeatureExtractor;

#[derive(Parser, Debug)]
#[command(name = "gpu_feature_extraction")]
#[command(about = "Feature extraction for greenhouse data (CPU-only, GPU processing moved to Python)", long_about = None)]
struct Args {
    /// Database connection string
    #[arg(long, env = "DATABASE_URL")]
    database_url: String,

    /// Era level to process (A, B, or C)
    #[arg(long, default_value = "B")]
    era_level: String,

    /// Minimum era size in rows
    #[arg(long, default_value = "100")]
    min_era_rows: usize,

    /// Batch size for processing
    #[arg(long, default_value = "24")]
    batch_size: usize,

    /// Features table name
    #[arg(long, default_value = "feature_data")]
    features_table: String,

    /// Maximum eras to process (for testing)
    #[arg(long)]
    max_eras: Option<usize>,
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

    info!("Starting feature extraction mode (CPU-only)");
    info!("GPU features are available via Python using gpu_features_pytorch.py");
    
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
            match db::fetch_era_data(&pool, era).await {
                Ok(data) => era_data_batch.push(data),
                Err(e) => {
                    warn!("Failed to fetch data for era {}: {}", era.era_id, e);
                    continue;
                }
            }
        }
        
        if era_data_batch.is_empty() {
            warn!("No data for batch {}, skipping", batch_idx);
            continue;
        }
        
        // Extract features
        let features = extractor.extract_batch(&era_data_batch)?;
        
        // Store features
        db::store_features(&pool, &features, &args.features_table).await?;
        
        info!("Batch {} completed with {} features", batch_idx, features.len());
    }
    
    info!("All batches completed in {:.2} seconds", 
          batch_start.elapsed().as_secs_f32());

    Ok(())
}