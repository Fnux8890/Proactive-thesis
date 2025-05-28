use crate::db::create_pool;
use crate::db_operations::DbOperations;
use crate::errors::PipelineError;
use crate::metrics::METRICS;
use crate::parallel::{expand_globs_parallel, ParallelProcessor};
use csv::WriterBuilder;
use env_logger;
use log::{error, info, warn};
use std::fs::{self, File};
use std::sync::Arc;
use std::time::Instant;

// Modules
mod config;
mod data_models;
mod db;
mod db_operations;
mod errors;
mod file_processor;
mod metrics;
mod models;
mod parallel;
mod parsers;
mod retry;
mod validation;

use config::load_config;

// Target columns for sensor_data table
const TARGET_COLUMNS: [&str; 62] = [
    "time", "source_system", "source_file", "format_type", "uuid", "lamp_group",
    "air_temp_c", "air_temp_middle_c", "outside_temp_c",
    "relative_humidity_percent", "humidity_deficit_g_m3",
    "radiation_w_m2", "light_intensity_lux", "light_intensity_umol",
    "outside_light_w_m2", "co2_measured_ppm", "co2_required_ppm",
    "co2_dosing_status", "co2_status", "rain_status",
    "vent_pos_1_percent", "vent_pos_2_percent", "vent_lee_afd3_percent",
    "vent_wind_afd3_percent", "vent_lee_afd4_percent", "vent_wind_afd4_percent",
    "curtain_1_percent", "curtain_2_percent", "curtain_3_percent",
    "curtain_4_percent", "window_1_percent", "window_2_percent",
    "lamp_grp1_no3_status", "lamp_grp2_no3_status", "lamp_grp3_no3_status",
    "lamp_grp4_no3_status", "lamp_grp1_no4_status", "lamp_grp2_no4_status",
    "measured_status_bool",
    "heating_setpoint_c", "pipe_temp_1_c", "pipe_temp_2_c",
    "flow_temp_1_c", "flow_temp_2_c",
    "temperature_forecast_c", "sun_radiation_forecast_w_m2",
    "temperature_actual_c", "sun_radiation_actual_w_m2",
    "vpd_hpa", "humidity_deficit_afd3_g_m3", "relative_humidity_afd3_percent",
    "humidity_deficit_afd4_g_m3", "relative_humidity_afd4_percent",
    "behov", "status_str", "timer_on", "timer_off", "dli_sum",
    "oenske_ekstra_lys", "lampe_timer_on", "lampe_timer_off",
    "value"
];

#[tokio::main]
async fn main() -> Result<(), PipelineError> {
    // Initialize logger
    env_logger::init();
    let pipeline_start = Instant::now();
    
    info!("🚀 Starting Enhanced Parallel Data Pipeline");
    
    // Step 1: Create database pool
    info!("Creating database connection pool...");
    let pool = create_pool()?;
    let db_ops = DbOperations::new(pool.clone());
    
    // Step 2: Flush existing data (optional - can be removed in production)
    info!("Flushing existing sensor data...");
    if let Err(e) = db_ops.flush_sensor_data().await {
        error!("Failed to flush sensor data: {}", e);
    }
    
    // Step 3: Load configuration
    info!("Loading configuration...");
    let config_path = "/app/config/data_files.json";
    let file_configs = load_config(config_path).map_err(PipelineError::ConfigParse)?;
    info!("Loaded {} file configurations", file_configs.len());
    
    // Step 4: Setup skipped rows logger
    let skipped_log_path = "/app/logs/skipped_timestamp_rows.csv";
    setup_skipped_logger(skipped_log_path)?;
    
    // Create thread-safe writer for skipped records
    let skipped_log_file = File::create(skipped_log_path)?;
    let mut csv_writer = WriterBuilder::new().has_headers(false).from_writer(skipped_log_file);
    csv_writer.write_record(&["SourceFile", "BatchIndex", "RecordIndex", "UUID", "Reason"])?;
    let skipped_writer = Arc::new(tokio::sync::Mutex::new(csv_writer));
    
    // Step 5: Expand globs in parallel
    info!("Expanding file globs...");
    let start_expand = Instant::now();
    let expanded_configs = expand_globs_parallel(&file_configs);
    let expand_duration = start_expand.elapsed();
    info!("Expanded to {} files in {:?}", expanded_configs.len(), expand_duration);
    
    // Update metrics
    METRICS.lock().record_processing_time("glob_expansion".to_string(), expand_duration);
    
    // Step 6: Process files in parallel
    info!("Processing files in parallel...");
    let processor = ParallelProcessor::new();
    let start_process = Instant::now();
    let results = processor.process_files(expanded_configs);
    let process_duration = start_process.elapsed();
    
    // Update metrics
    METRICS.lock().record_processing_time("file_processing".to_string(), process_duration);
    
    // Step 7: Collect all records and update metrics
    let mut all_records = Vec::new();
    let mut failed_files = Vec::new();
    
    for result in results {
        METRICS.lock().record_file_attempt();
        
        if let Some(error) = result.error {
            METRICS.lock().record_file_failure();
            failed_files.push((result.file_path, error));
        } else {
            let record_count = result.records.len() as u64;
            METRICS.lock().record_file_success(record_count);
            all_records.extend(result.records);
        }
    }
    
    info!("Collected {} total records from successful files", all_records.len());
    
    // Step 8: Insert records to database in parallel batches
    if !all_records.is_empty() {
        info!("Inserting records to database...");
        let start_insert = Instant::now();
        
        match db_ops.insert_records_parallel(all_records, skipped_writer).await {
            Ok(inserted) => {
                let insert_duration = start_insert.elapsed();
                info!("Successfully inserted {} records in {:?}", inserted, insert_duration);
                METRICS.lock().record_processing_time("db_insertion".to_string(), insert_duration);
            }
            Err(e) => {
                error!("Failed to insert records: {}", e);
                return Err(e);
            }
        }
    }
    
    // Step 9: Validate and merge
    info!("Performing validation checks...");
    let validation_result = validation::check_schema_and_data_integrity(
        &pool,
        "public",
        "sensor_data",
        &TARGET_COLUMNS,
    ).await;
    
    match validation_result {
        Ok(_) => {
            info!("Validation passed. Running merge script...");
            let start_merge = Instant::now();
            
            match db_ops.run_merge_script().await {
                Ok(_) => {
                    let merge_duration = start_merge.elapsed();
                    info!("Merge completed in {:?}", merge_duration);
                    METRICS.lock().record_processing_time("merge_script".to_string(), merge_duration);
                }
                Err(e) => {
                    error!("Merge script failed: {}", e);
                }
            }
        }
        Err(e) => {
            error!("Validation failed: {}. Skipping merge.", e);
        }
    }
    
    // Step 10: Print final metrics
    let total_duration = pipeline_start.elapsed();
    METRICS.lock().record_processing_time("total_pipeline".to_string(), total_duration);
    METRICS.lock().print_summary();
    
    // Print failed files if any
    if !failed_files.is_empty() {
        warn!("⚠️  Failed to process {} files:", failed_files.len());
        for (path, error) in &failed_files {
            warn!("  - {}: {}", path, error);
        }
    }
    
    info!("✅ Pipeline completed successfully in {:?}", total_duration);
    Ok(())
}

/// Setup the skipped rows logger
fn setup_skipped_logger(path: &str) -> Result<(), PipelineError> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        fs::create_dir_all(parent).map_err(|e| {
            PipelineError::Config(format!("Failed to create log directory: {}", e))
        })?;
    }
    Ok(())
}

impl From<std::io::Error> for PipelineError {
    fn from(error: std::io::Error) -> Self {
        PipelineError::Config(format!("IO error: {}", error))
    }
}

impl From<csv::Error> for PipelineError {
    fn from(error: csv::Error) -> Self {
        PipelineError::Config(format!("CSV error: {}", error))
    }
}