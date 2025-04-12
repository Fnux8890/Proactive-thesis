// use std::path::Path; // REMOVED unused import
// use walkdir::WalkDir; // REMOVED unused import
// use std::env; // Removed unused import
// use std::error::Error; // REMOVED unused import AGAIN
// use std::fs::File; // REMOVED unused import
// use serde::{Deserialize, Deserializer}; // Now handled in models
// use std::collections::HashMap; // REMOVED unused import
// use serde_json::Value; // Removed unused import
// use std::fmt; // Now handled in models
// use serde::de::{self, Visitor, Error as SerdeError}; // Now handled in models
// use serde::ser; // Removed unused import
// use std::io; // REMOVED unused import
// use std::path::PathBuf; // REMOVED unused import
// use std::fs; // REMOVED unused import
use glob::glob; // Import the glob function
use data_models::ParsedRecord;
use db::{create_pool, DbPool};
use tokio_postgres::binary_copy::BinaryCopyInWriter; // ADD back
use tokio_postgres::types::{ToSql, Type}; // Keep ToSql and Type
// use bytes::{BytesMut, BufMut}; // REMOVE BytesMut/BufMut
use pin_utils::pin_mut; // ADD back
// use byteorder::{NetworkEndian, WriteBytesExt}; // REMOVE byteorder
use log::{info, warn, error}; // REMOVE unused debug
use env_logger; // Import env_logger
// use futures_util::sink::SinkExt; // REMOVE unused import

mod models; // Declare the models module
// mod validation; // Declare the validation module
// use models::*; // REMOVED unused import

// Declare modules
mod config;
mod errors;
mod data_models;
mod parsers;
mod file_processor;
mod db;

// Declare new modules
// mod data_models; // Declare later when needed
// mod parsers; // Declare later when needed
// mod file_processor; // Declare later when needed

use config::load_config;
use errors::PipelineError;

/* // COMMENT OUT unused struct definition AGAIN
struct DataFileEntry {
    workspace_path: String,
    container_path: String,
    status: String,
}
*/

// Struct/Enum definitions removed - they are now in models.rs

// parse_comma_decimal function removed - it is now in models.rs

// The list of columns in the target TimescaleDB table `sensor_data`
// This order MUST EXACTLY MATCH init.sql and the construction of row_values and get_column_types
const TARGET_COLUMNS: [&str; 61] = [
    // Core Identification & Timing (5)
    "time", "source_system", "source_file", "format_type", "uuid",
    // Common Environmental Measurements (13)
    "air_temp_c", "air_temp_middle_c", "outside_temp_c",
    "relative_humidity_percent", "humidity_deficit_g_m3",
    "radiation_w_m2", "light_intensity_lux", "light_intensity_umol",
    "outside_light_w_m2", "co2_measured_ppm", "co2_required_ppm",
    "co2_dosing_status", "co2_status", "rain_status",
    // Control System State (18)
    "vent_pos_1_percent", "vent_pos_2_percent", "vent_lee_afd3_percent",
    "vent_wind_afd3_percent", "vent_lee_afd4_percent", "vent_wind_afd4_percent",
    "curtain_1_percent", "curtain_2_percent", "curtain_3_percent",
    "curtain_4_percent", "window_1_percent", "window_2_percent",
    "lamp_grp1_no3_status", "lamp_grp2_no3_status", "lamp_grp3_no3_status",
    "lamp_grp4_no3_status", "lamp_grp1_no4_status", "lamp_grp2_no4_status",
    "measured_status_bool",
    // Heating & Flow (5)
    "heating_setpoint_c", "pipe_temp_1_c", "pipe_temp_2_c",
    "flow_temp_1_c", "flow_temp_2_c",
    // Forecasts (4)
    "temperature_forecast_c", "sun_radiation_forecast_w_m2",
    "temperature_actual_c", "sun_radiation_actual_w_m2",
    // Knudjepsen Specific / Others (13)
    "vpd_hpa", "humidity_deficit_afd3_g_m3", "relative_humidity_afd3_percent",
    "humidity_deficit_afd4_g_m3", "relative_humidity_afd4_percent",
    "behov", "status_str", "timer_on", "timer_off", "dli_sum",
    "oenske_ekstra_lys", "lampe_timer_on", "lampe_timer_off",
    // Generic Value (1)
    "value"
];

#[tokio::main]
async fn main() -> Result<(), PipelineError> { // Make main async
    env_logger::init(); // Initialize logger
    info!("Starting Data Pipeline...");

    // Create database pool
    let pool = create_pool()?;

    // Define the path to the configuration file copied into the image
    let config_path = "/app/config/data_files.json";
    info!("Attempting to load configuration from: {}", config_path);

    // Load configuration
    let file_configs = load_config(config_path).map_err(PipelineError::ConfigParse)?;
    info!("Successfully loaded {} file configurations.", file_configs.len());

    // --- Process all file configurations ---
    let mut total_processed_files = 0;
    let mut total_processed_records = 0;
    let mut total_inserted_records = 0;
    let mut total_errors = 0;

    for config in file_configs {
        let path_str = config.container_path.to_string_lossy();
        let process_result = if path_str.contains('*') || path_str.contains('?') {
            info!("Glob pattern detected: {}", path_str);
            let mut files_found_in_glob = 0;
            let mut errors_in_glob = 0;

            match glob(&path_str) {
                Ok(paths) => {
                    for entry in paths {
                        match entry {
                            Ok(actual_path) => {
                                files_found_in_glob += 1;
                                info!("Processing matched file: {}", actual_path.display());
                                let mut specific_config = config.clone();
                                specific_config.container_path = actual_path.clone();

                                match file_processor::process_file(&specific_config) {
                                    Ok(parsed_records) => {
                                        info!("Successfully parsed {} records from {}. Attempting insertion...",
                                            parsed_records.len(),
                                            actual_path.display());
                                        total_processed_records += parsed_records.len() as u64;
                                        match insert_records(&pool, &parsed_records).await {
                                            Ok(inserted_count) => {
                                                info!("Successfully inserted {} records from {}.", inserted_count, actual_path.display());
                                                total_inserted_records += inserted_count;
                                                total_processed_files += 1;
                                            }
                                            Err(e) => {
                                                error!("Database insertion failed for {}: {}", actual_path.display(), e);
                                                errors_in_glob += 1;
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        error!("ERROR parsing file {}: {}", actual_path.display(), e);
                                        errors_in_glob += 1;
                                    }
                                }
                            }
                            Err(e) => {
                                error!("ERROR iterating glob result for pattern {}: {}", path_str, e);
                                errors_in_glob += 1;
                            }
                        }
                    }
                }
                Err(e) => {
                    error!("ERROR: Invalid glob pattern '{}': {}", path_str, e);
                    errors_in_glob += 1;
                }
            }
            if files_found_in_glob == 0 {
                warn!("Glob pattern '{}' did not match any files.", path_str);
            }
            errors_in_glob // Return errors from glob processing
        } else {
            // --- No wildcard, process the single file ---
            let actual_path = config.container_path.clone(); // Use the specific path
            info!("Processing single file: {}", actual_path.display());
            match file_processor::process_file(&config) {
                Ok(parsed_records) => {
                    info!("Successfully parsed {} records from {}. Attempting insertion...",
                             parsed_records.len(),
                             actual_path.display());
                    total_processed_records += parsed_records.len() as u64;
                    match insert_records(&pool, &parsed_records).await {
                        Ok(inserted_count) => {
                             info!("Successfully inserted {} records from {}.", inserted_count, actual_path.display());
                             total_inserted_records += inserted_count;
                             total_processed_files += 1;
                             0 // No errors for this file
                        }
                        Err(e) => {
                             error!("Database insertion failed for {}: {}", actual_path.display(), e);
                             1 // 1 error for this file
                        }
                    }
                }
                Err(e) => {
                    error!("ERROR parsing file {}: {}", actual_path.display(), e);
                    1 // 1 error for this file
                }
            }
        };
        total_errors += process_result;
    }
    // --- End processing loop ---

    info!(
        "\nData Pipeline finished processing. Files Processed: {}, Records Parsed: {}, Records Inserted: {}, Errors: {}",
        total_processed_files, total_processed_records, total_inserted_records, total_errors
    );
    Ok(())
}

/// Inserts a batch of ParsedRecord into the TimescaleDB table using COPY FROM STDIN BINARY.
async fn insert_records(pool: &DbPool, records: &[ParsedRecord]) -> Result<u64, PipelineError> {
    if records.is_empty() {
        return Ok(0);
    }

    let mut client = pool.get().await.map_err(PipelineError::DbConnectionError)?;
    let transaction = client.transaction().await.map_err(PipelineError::DbQueryError)?;
    let copy_sql = format!("COPY sensor_data ({}) FROM STDIN BINARY", TARGET_COLUMNS.join(", "));
    let sink = transaction.copy_in(&copy_sql).await.map_err(PipelineError::DbQueryError)?;
    let writer = BinaryCopyInWriter::new(sink, &get_column_types());
    pin_mut!(writer);

    for record in records {
        if record.timestamp_utc.is_none() {
            warn!("Skipping record insertion due to missing timestamp (source: {:?})", record.source_file);
            continue;
        }

        // Build the row vector in the EXACT order defined in TARGET_COLUMNS and init.sql
        let row_values: Vec<&(dyn ToSql + Sync)> = vec![
            // Core Identification & Timing
            &record.timestamp_utc, &record.source_system, &record.source_file,
            &record.format_type, &record.uuid,
            // Common Environmental Measurements
            &record.air_temp_c, &record.air_temp_middle_c, &record.outside_temp_c,
            &record.relative_humidity_percent, &record.humidity_deficit_g_m3,
            &record.radiation_w_m2, &record.light_intensity_lux, &record.light_intensity_umol,
            &record.outside_light_w_m2, &record.co2_measured_ppm, &record.co2_required_ppm,
            &record.co2_dosing_status, &record.co2_status, &record.rain_status,
            // Control System State
            &record.vent_pos_1_percent, &record.vent_pos_2_percent, &record.vent_lee_afd3_percent,
            &record.vent_wind_afd3_percent, &record.vent_lee_afd4_percent, &record.vent_wind_afd4_percent,
            &record.curtain_1_percent, &record.curtain_2_percent, &record.curtain_3_percent,
            &record.curtain_4_percent, &record.window_1_percent, &record.window_2_percent,
            &record.lamp_grp1_no3_status, &record.lamp_grp2_no3_status, &record.lamp_grp3_no3_status,
            &record.lamp_grp4_no3_status, &record.lamp_grp1_no4_status, &record.lamp_grp2_no4_status,
            &record.measured_status_bool,
            // Heating & Flow
            &record.heating_setpoint_c, &record.pipe_temp_1_c, &record.pipe_temp_2_c,
            &record.flow_temp_1_c, &record.flow_temp_2_c,
            // Forecasts
            &record.temperature_forecast_c, &record.sun_radiation_forecast_w_m2,
            &record.temperature_actual_c, &record.sun_radiation_actual_w_m2,
            // Knudjepsen Specific / Others
            &record.vpd_hpa, &record.humidity_deficit_afd3_g_m3, &record.relative_humidity_afd3_percent,
            &record.humidity_deficit_afd4_g_m3, &record.relative_humidity_afd4_percent,
            &record.behov, &record.status_str, &record.timer_on, &record.timer_off, &record.dli_sum,
            &record.oenske_ekstra_lys, &record.lampe_timer_on, &record.lampe_timer_off,
            // Generic Value
            &record.value
        ];

        // Write the row
        writer.as_mut().write(&row_values).await.map_err(|e| PipelineError::DbQueryError(e))?;
    }

    // Finish the COPY
    let inserted_count = writer.as_mut().finish().await.map_err(PipelineError::DbQueryError)?;
    transaction.commit().await.map_err(PipelineError::DbQueryError)?;
    Ok(inserted_count)
}

/// Returns a static list of PostgreSQL types corresponding to TARGET_COLUMNS.
fn get_column_types() -> Vec<Type> {
    // This order MUST EXACTLY MATCH TARGET_COLUMNS, init.sql, and row_values construction
    vec![
        // Core Identification & Timing
        Type::TIMESTAMPTZ, Type::TEXT, Type::TEXT, Type::TEXT, Type::TEXT,
        // Common Environmental Measurements
        Type::FLOAT8, Type::FLOAT8, Type::FLOAT8, Type::FLOAT8, Type::FLOAT8,
        Type::FLOAT8, Type::FLOAT8, Type::FLOAT8, Type::FLOAT8, Type::FLOAT8,
        Type::FLOAT8, Type::FLOAT8, Type::FLOAT8, Type::BOOL,
        // Control System State
        Type::FLOAT8, Type::FLOAT8, Type::FLOAT8, Type::FLOAT8, Type::FLOAT8,
        Type::FLOAT8, Type::FLOAT8, Type::FLOAT8, Type::FLOAT8, Type::FLOAT8,
        Type::FLOAT8, Type::FLOAT8, Type::BOOL, Type::BOOL, Type::BOOL, Type::BOOL,
        Type::BOOL, Type::BOOL, Type::BOOL,
        // Heating & Flow
        Type::FLOAT8, Type::FLOAT8, Type::FLOAT8, Type::FLOAT8, Type::FLOAT8,
        // Forecasts
        Type::FLOAT8, Type::FLOAT8, Type::FLOAT8, Type::FLOAT8,
        // Knudjepsen Specific / Others
        Type::FLOAT8, Type::FLOAT8, Type::FLOAT8, Type::FLOAT8, Type::FLOAT8,
        Type::INT4, Type::TEXT, Type::INT4, Type::INT4, Type::FLOAT8,
        Type::TEXT, Type::INT8, Type::INT8,
        // Generic Value
        Type::FLOAT8
    ]
}

// End of Main Function ///
// Removed old walk_dir_* and process_data_file functions 
// Removed old walk_dir_* and process_data_file functions 