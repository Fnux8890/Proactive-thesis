// Restore necessary imports
use glob::glob;
use crate::data_models::ParsedRecord;
use crate::db::{create_pool, DbPool};
use tokio_postgres::binary_copy::BinaryCopyInWriter;
use tokio_postgres::types::{ToSql, Type};
use pin_utils::pin_mut;
use log::{info, warn, error};
use env_logger;
use std::fs::{self, File};
use csv::WriterBuilder;
// use std::fs; // <-- Remove unused import
// use tokio_postgres::Row; // Still unused
// use std::path::PathBuf; // Still unused in this simplified main
use std::collections::HashSet;
use std::io::Write;

// Restore modules
mod config;
mod errors;
mod data_models;
mod parsers;
mod file_processor;
mod db;
mod models;
mod validation; // <-- Add module declaration

use config::load_config; // <-- Remove unused FileConfig
use errors::PipelineError;

// Restore TARGET_COLUMNS
const TARGET_COLUMNS: [&str; 62] = [ /* ... all column names ... */
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

#[derive(Debug, Clone, Default)]
struct MergedRecord {
    time: Option<chrono::DateTime<chrono::Utc>>,
    source_system: Option<String>,
    source_file: Option<String>,
    format_type: Option<String>,
    uuid: Option<String>,
    lamp_group: Option<String>,
    air_temp_c: Option<f64>,
    air_temp_middle_c: Option<f64>,
    outside_temp_c: Option<f64>,
    relative_humidity_percent: Option<f64>,
    humidity_deficit_g_m3: Option<f64>,
    radiation_w_m2: Option<f64>,
    light_intensity_lux: Option<f64>,
    light_intensity_umol: Option<f64>,
    outside_light_w_m2: Option<f64>,
    co2_measured_ppm: Option<f64>,
    co2_required_ppm: Option<f64>,
    co2_dosing_status: Option<f64>,
    co2_status: Option<f64>,
    rain_status: Option<bool>,
    vent_pos_1_percent: Option<f64>,
    vent_pos_2_percent: Option<f64>,
    vent_lee_afd3_percent: Option<f64>,
    vent_wind_afd3_percent: Option<f64>,
    vent_lee_afd4_percent: Option<f64>,
    vent_wind_afd4_percent: Option<f64>,
    curtain_1_percent: Option<f64>,
    curtain_2_percent: Option<f64>,
    curtain_3_percent: Option<f64>,
    curtain_4_percent: Option<f64>,
    window_1_percent: Option<f64>,
    window_2_percent: Option<f64>,
    lamp_grp1_no3_status: Option<bool>,
    lamp_grp2_no3_status: Option<bool>,
    lamp_grp3_no3_status: Option<bool>,
    lamp_grp4_no3_status: Option<bool>,
    lamp_grp1_no4_status: Option<bool>,
    lamp_grp2_no4_status: Option<bool>,
    measured_status_bool: Option<bool>,
    heating_setpoint_c: Option<f64>,
    pipe_temp_1_c: Option<f64>,
    pipe_temp_2_c: Option<f64>,
    flow_temp_1_c: Option<f64>,
    flow_temp_2_c: Option<f64>,
    temperature_forecast_c: Option<f64>,
    sun_radiation_forecast_w_m2: Option<f64>,
    temperature_actual_c: Option<f64>,
    sun_radiation_actual_w_m2: Option<f64>,
    vpd_hpa: Option<f64>,
    humidity_deficit_afd3_g_m3: Option<f64>,
    relative_humidity_afd3_percent: Option<f64>,
    humidity_deficit_afd4_g_m3: Option<f64>,
    relative_humidity_afd4_percent: Option<f64>,
    behov: Option<i32>,
    status_str: Option<String>,
    timer_on: Option<i32>,
    timer_off: Option<i32>,
    dli_sum: Option<f64>,
    oenske_ekstra_lys: Option<String>,
    lampe_timer_on: Option<i64>,
    lampe_timer_off: Option<i64>,
    value: Option<f64>,
}

impl MergedRecord {
    fn from_row(row: &tokio_postgres::Row) -> Result<Self, PipelineError> {
        // Helper to get value by index, converting error
        fn get_opt<T: tokio_postgres::types::FromSqlOwned>(row: &tokio_postgres::Row, idx: usize) -> Option<T> {
            match row.try_get(idx) {
                Ok(val) => Some(val),
                Err(_) => None, // Could log error here if specific handling is needed
            }
        }

        Ok(MergedRecord {
            time: get_opt(row, 0),
            source_system: get_opt(row, 1),
            source_file: get_opt(row, 2),
            format_type: get_opt(row, 3),
            uuid: get_opt(row, 4),
            lamp_group: get_opt(row, 5),
            air_temp_c: get_opt(row, 6),
            air_temp_middle_c: get_opt(row, 7),
            outside_temp_c: get_opt(row, 8),
            relative_humidity_percent: get_opt(row, 9),
            humidity_deficit_g_m3: get_opt(row, 10),
            radiation_w_m2: get_opt(row, 11),
            light_intensity_lux: get_opt(row, 12),
            light_intensity_umol: get_opt(row, 13),
            outside_light_w_m2: get_opt(row, 14),
            co2_measured_ppm: get_opt(row, 15),
            co2_required_ppm: get_opt(row, 16),
            co2_dosing_status: get_opt(row, 17),
            co2_status: get_opt(row, 18),
            rain_status: get_opt(row, 19),
            vent_pos_1_percent: get_opt(row, 20),
            vent_pos_2_percent: get_opt(row, 21),
            vent_lee_afd3_percent: get_opt(row, 22),
            vent_wind_afd3_percent: get_opt(row, 23),
            vent_lee_afd4_percent: get_opt(row, 24),
            vent_wind_afd4_percent: get_opt(row, 25),
            curtain_1_percent: get_opt(row, 26),
            curtain_2_percent: get_opt(row, 27),
            curtain_3_percent: get_opt(row, 28),
            curtain_4_percent: get_opt(row, 29),
            window_1_percent: get_opt(row, 30),
            window_2_percent: get_opt(row, 31),
            lamp_grp1_no3_status: get_opt(row, 32),
            lamp_grp2_no3_status: get_opt(row, 33),
            lamp_grp3_no3_status: get_opt(row, 34),
            lamp_grp4_no3_status: get_opt(row, 35),
            lamp_grp1_no4_status: get_opt(row, 36),
            lamp_grp2_no4_status: get_opt(row, 37),
            measured_status_bool: get_opt(row, 38),
            heating_setpoint_c: get_opt(row, 39),
            pipe_temp_1_c: get_opt(row, 40),
            pipe_temp_2_c: get_opt(row, 41),
            flow_temp_1_c: get_opt(row, 42),
            flow_temp_2_c: get_opt(row, 43),
            temperature_forecast_c: get_opt(row, 44),
            sun_radiation_forecast_w_m2: get_opt(row, 45),
            temperature_actual_c: get_opt(row, 46),
            sun_radiation_actual_w_m2: get_opt(row, 47),
            vpd_hpa: get_opt(row, 48),
            humidity_deficit_afd3_g_m3: get_opt(row, 49),
            relative_humidity_afd3_percent: get_opt(row, 50),
            humidity_deficit_afd4_g_m3: get_opt(row, 51),
            relative_humidity_afd4_percent: get_opt(row, 52),
            behov: get_opt(row, 53),
            status_str: get_opt(row, 54),
            timer_on: get_opt(row, 55),
            timer_off: get_opt(row, 56),
            dli_sum: get_opt(row, 57),
            oenske_ekstra_lys: get_opt(row, 58),
            lampe_timer_on: get_opt(row, 59),
            lampe_timer_off: get_opt(row, 60),
            value: get_opt(row, 61),
        })
    }
}

async fn perform_upsampling(pool: &DbPool) -> Result<(), PipelineError> {
    info!("Starting upsampling process for sensor_data_merged...");
    let mut client = pool.get().await.map_err(PipelineError::DbConnectionError)?;

    let query_string = format!("SELECT {} FROM sensor_data_merged ORDER BY time ASC", TARGET_COLUMNS.join(", "));
    let rows = client
        .query(&query_string, &[])
        .await
        .map_err(PipelineError::DbQueryError)?;

    if rows.is_empty() {
        info!("No data in sensor_data_merged to upsample.");
        return Ok(());
    }

    let mut records: Vec<MergedRecord> = Vec::with_capacity(rows.len());
    for row in rows {
        match MergedRecord::from_row(&row) {
            Ok(record) => records.push(record),
            Err(e) => {
                error!("Failed to map row to MergedRecord: {:?}. Skipping row.", e);
            }
        }
    }
    info!("Fetched {} records from sensor_data_merged for upsampling.", records.len());

    // Initialize LOCF trackers
    let mut last_known_air_temp_c: Option<f64> = None;
    let mut last_known_relative_humidity_percent: Option<f64> = None;
    let mut last_known_co2_measured_ppm: Option<f64> = None;
    let mut last_known_co2_dosing_status: Option<f64> = None; // Stored as f64 (0.0/1.0)
    let mut last_known_co2_status: Option<f64> = None;        // Stored as f64 (0.0/1.0)
    let mut last_known_rain_status: Option<bool> = None;      // Stored as bool
    // Add more trackers for other columns as needed

    for record in records.iter_mut() {
        // Numeric LOCF
        if record.air_temp_c.is_some() { last_known_air_temp_c = record.air_temp_c; }
        else { record.air_temp_c = last_known_air_temp_c; }

        if record.relative_humidity_percent.is_some() { last_known_relative_humidity_percent = record.relative_humidity_percent; }
        else { record.relative_humidity_percent = last_known_relative_humidity_percent; }
        
        if record.co2_measured_ppm.is_some() { last_known_co2_measured_ppm = record.co2_measured_ppm; }
        else { record.co2_measured_ppm = last_known_co2_measured_ppm; }

        // Statuses (numeric f64 for 0.0/1.0) LOCF
        if record.co2_dosing_status.is_some() { last_known_co2_dosing_status = record.co2_dosing_status; }
        else { record.co2_dosing_status = last_known_co2_dosing_status; }

        if record.co2_status.is_some() { last_known_co2_status = record.co2_status; }
        else { record.co2_status = last_known_co2_status; }

        // Statuses (boolean) LOCF
        if record.rain_status.is_some() { last_known_rain_status = record.rain_status; }
        else { record.rain_status = last_known_rain_status; }

        // Fill light_intensity_umol with 0.0 if None
        if record.light_intensity_umol.is_none() {
            record.light_intensity_umol = Some(0.0);
        }
    }
    info!("LOCF and fillna(0) applied to records in memory.");

    info!("Logging first 10 (or fewer) upsampled records to demonstrate effect:");
    for (i, record) in records.iter().take(10).enumerate() {
        info!("Upsampled Record {}: time={:?}, air_temp_c={:?}, light_intensity_umol={:?}, rain_status={:?}, co2_dosing_status={:?}",
            i, record.time, record.air_temp_c, record.light_intensity_umol, record.rain_status, record.co2_dosing_status);
    }
    
    let upsampled_csv_path = "/app/output/sensor_data_upsampled.csv";
    info!("Attempting to write upsampled data to CSV: {}", upsampled_csv_path);
    if let Some(parent) = std::path::Path::new(upsampled_csv_path).parent() {
        if let Err(e) = fs::create_dir_all(parent) {
            warn!("Could not create output directory for upsampled CSV: {}. Skipping CSV write.", e);
            return Ok(()); 
        }
    }

    match File::create(upsampled_csv_path) {
        Ok(file) => {
            let mut wtr = WriterBuilder::new().from_writer(file);
            if let Err(e) = wtr.write_record(TARGET_COLUMNS.iter()) {
                 error!("Failed to write header to upsampled CSV: {}", e);
            }
            for record in records.iter() {
                // Simplified CSV row writing for demonstration
                let mut csv_row: Vec<String> = Vec::with_capacity(TARGET_COLUMNS.len());
                csv_row.push(record.time.map_or(String::new(), |t| t.to_rfc3339()));
                csv_row.push(record.source_system.clone().unwrap_or_default());
                csv_row.push(record.source_file.clone().unwrap_or_default());
                csv_row.push(record.format_type.clone().unwrap_or_default());
                csv_row.push(record.uuid.clone().unwrap_or_default());
                csv_row.push(record.lamp_group.clone().unwrap_or_default());
                csv_row.push(record.air_temp_c.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.air_temp_middle_c.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.outside_temp_c.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.relative_humidity_percent.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.humidity_deficit_g_m3.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.radiation_w_m2.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.light_intensity_lux.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.light_intensity_umol.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.outside_light_w_m2.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.co2_measured_ppm.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.co2_required_ppm.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.co2_dosing_status.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.co2_status.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.rain_status.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.vent_pos_1_percent.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.vent_pos_2_percent.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.vent_lee_afd3_percent.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.vent_wind_afd3_percent.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.vent_lee_afd4_percent.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.vent_wind_afd4_percent.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.curtain_1_percent.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.curtain_2_percent.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.curtain_3_percent.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.curtain_4_percent.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.window_1_percent.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.window_2_percent.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.lamp_grp1_no3_status.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.lamp_grp2_no3_status.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.lamp_grp3_no3_status.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.lamp_grp4_no3_status.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.lamp_grp1_no4_status.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.lamp_grp2_no4_status.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.measured_status_bool.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.heating_setpoint_c.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.pipe_temp_1_c.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.pipe_temp_2_c.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.flow_temp_1_c.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.flow_temp_2_c.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.temperature_forecast_c.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.sun_radiation_forecast_w_m2.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.temperature_actual_c.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.sun_radiation_actual_w_m2.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.vpd_hpa.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.humidity_deficit_afd3_g_m3.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.relative_humidity_afd3_percent.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.humidity_deficit_afd4_g_m3.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.relative_humidity_afd4_percent.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.behov.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.status_str.clone().unwrap_or_default());
                csv_row.push(record.timer_on.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.timer_off.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.dli_sum.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.oenske_ekstra_lys.clone().unwrap_or_default());
                csv_row.push(record.lampe_timer_on.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.lampe_timer_off.map_or(String::new(), |v| v.to_string()));
                csv_row.push(record.value.map_or(String::new(), |v| v.to_string()));

                 if let Err(e) = wtr.write_record(&csv_row) {
                     error!("Failed to write record to upsampled CSV: {}", e);
                 }
            }
            if let Err(e) = wtr.flush() { error!("Failed to flush upsampled CSV: {}", e); }
            else { info!("Successfully wrote {} upsampled records to {}", records.len(), upsampled_csv_path); }
        }
        Err(e) => {
            error!("Failed to create upsampled CSV file '{}': {}", upsampled_csv_path, e);
        }
    }

    info!("Upsampling process finished.");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), PipelineError> {
    env_logger::init();
    println!(">>> RESTORED MAIN: Step 1 - Logger Init");
    let _ = std::io::stdout().flush();

    println!(">>> RESTORED MAIN: Step 2 - Creating DB Pool...");
    let pool = create_pool()?;
    println!(">>> RESTORED MAIN: Step 3 - DB Pool Created.");
    let _ = std::io::stdout().flush();

        // ... after pool creation ...
    println!(">>> MAIN: ABOUT TO CALL FLUSH <<<"); // <-- ADD THIS
    let _ = std::io::stdout().flush();
    let flush_result = flush_data_in_table(&pool).await;
    println!(">>> MAIN: FLUSH CALL FINISHED <<<"); // <-- ADD THIS
    let _ = std::io::stdout().flush();
    if let Err(e) = flush_result {
        error!("Failed to flush data in table: {}", e);
    } else {
        info!("Data flushed successfully."); // This existing log should appear between the two above
    }
    // ... rest of the code ...

    println!(">>> RESTORED MAIN: Step 4 - Loading Config...");
    let config_path = "/app/config/data_files.json";
    let file_configs = load_config(config_path).map_err(PipelineError::ConfigParse)?;
    println!(">>> RESTORED MAIN: Step 5 - Config Loaded ({} entries).", file_configs.len());
    let _ = std::io::stdout().flush();

    // === Setup Skipped Rows Logger ===
    let skipped_log_path = "/app/logs/skipped_timestamp_rows.csv";
    info!("Attempting to set up skipped rows log at: {}", skipped_log_path);
    if let Some(parent) = std::path::Path::new(skipped_log_path).parent() {
        fs::create_dir_all(parent).map_err(|e| PipelineError::Config(format!("Failed to create log directory '{}': {}", parent.display(), e)))?;
        info!("Log directory '{}' ensured.", parent.display());
    } else {
         warn!("Could not determine parent directory for skipped log file.");
    }
    let skipped_log_file = File::create(skipped_log_path)
        .map_err(|e| PipelineError::Config(format!("Failed to create skipped log file '{}': {}", skipped_log_path, e)))?;
    let mut skipped_writer = WriterBuilder::new().has_headers(false).from_writer(skipped_log_file);
    // Write header explicitly
    if let Err(e) = skipped_writer.write_record(&["SourceFile", "BatchIndex", "UUID", "Reason"]) {
         error!("Failed to write header to skipped log: {}", e);
         // Continue anyway, just won't have headers
    }
     if let Err(e) = skipped_writer.flush() { error!("Failed to flush skipped log header: {}", e); }
     info!("Skipped rows logger initialized.");
    // =================================

    let mut _total_processed_files = 0; // <-- Add underscore
    let mut _total_processed_records = 0; // <-- Add underscore
    let _total_inserted_records = 0; // <-- Add underscore, remove mut
    let mut _total_errors = 0; // <-- Add underscore
    let mut unique_expected_columns: HashSet<String> = HashSet::new();
    let mut total_attempted_files: u64 = 0; // counts every file we try to parse (includes failures)
    let mut failed_files: Vec<(String, String)> = Vec::new(); // (file path, error)

    println!(">>> RESTORED MAIN: Step 6 - Entering Main Loop...");
    let _ = std::io::stdout().flush();

    for (config_index, config) in file_configs.iter().enumerate() {
        let path_str = config.container_path.to_string_lossy();
        println!("---> Loop Start: Idx {}, Path: {} <---", config_index, path_str);
        let mut current_file_errors = 0;
        let mut file_processed_successfully = false;
        let mut parsed_records: Vec<ParsedRecord> = Vec::new();

        if path_str.contains('*') || path_str.contains('?') {
            match glob(&path_str) {
                Ok(paths) => {
                    let mut files_in_glob_processed_successfully = 0;
                    for entry in paths {
                        match entry {
                            Ok(actual_path) => {
                                println!("Processing matched file: {}", actual_path.display());
                                let mut specific_config = config.clone();
                                specific_config.container_path = actual_path.clone();

                                total_attempted_files += 1;

                                let process_result = file_processor::process_file(&specific_config);
                                let _ = std::io::stdout().flush();

                                match process_result {
                                    Ok(records) => {
                                        parsed_records = records;
                                        let _ = std::io::stdout().flush();
                                        _total_processed_records += parsed_records.len() as u64; // <-- Add underscore
                                        _total_processed_files += 1; // <-- Add underscore
                                        file_processed_successfully = true;
                                        files_in_glob_processed_successfully += 1;
                                    }
                                    Err(e) => {
                                        error!("---- PARSE ERR for {}: {} ----", actual_path.display(), e);
                                        current_file_errors += 1;
                                        _total_errors += 1; // <-- Add underscore
                                        failed_files.push((actual_path.display().to_string(), e.to_string()));
                                    }
                                }
                            }
                            Err(e) => { /* handle glob entry error */ error!("Glob entry err: {}", e); current_file_errors += 1; }
                        }
                    }
                    if files_in_glob_processed_successfully > 0 {
                        println!("Glob Summary for '{}': {} files parsed successfully.", path_str, files_in_glob_processed_successfully);
                    } else { warn!("Glob Summary for '{}': No files parsed successfully (Errors: {}).", path_str, current_file_errors); }
                }
                Err(e) => { /* handle invalid glob pattern */ error!("Invalid glob pattern: {}", e); current_file_errors += 1; _total_errors += 1; } // <-- Add underscore
            }
        } else {
            let actual_path = config.container_path.clone();
            println!("Processing single file: {}", actual_path.display());
            total_attempted_files += 1;

            let process_result = file_processor::process_file(&config);
            let _ = std::io::stdout().flush();

            match process_result {
                Ok(records) => {
                    parsed_records = records;
                    let _ = std::io::stdout().flush();
                    _total_processed_records += parsed_records.len() as u64; // <-- Add underscore
                    _total_processed_files += 1; // <-- Add underscore
                    file_processed_successfully = true;
                }
                Err(e) => {
                    error!("---- PARSE ERR for {}: {} ----", actual_path.display(), e);
                    current_file_errors += 1;
                    _total_errors += 1; // <-- Add underscore
                    failed_files.push((actual_path.display().to_string(), e.to_string()));
                }
            }
        }

        _total_errors += current_file_errors; // <-- Reassign to unused variable if needed, or remove line

        if file_processed_successfully {
            let expected_columns_for_config: HashSet<String> = config.column_map.iter().flatten().map(|cm| cm.target.clone()).filter(|tf| tf != "timestamp_utc").collect();
            if !expected_columns_for_config.is_empty() {
                println!("Config Idx {}: Parse OK. Adding columns: {:?}", config_index, expected_columns_for_config);
                unique_expected_columns.extend(expected_columns_for_config);
            } else { println!("Config Idx {}: Parse OK, no columns in map.", config_index); }
        } else { warn!("Config Idx {}: Parse FAILED. No columns added.", config_index); }
        println!("---> Loop End: Idx {} <----", config_index);
        let _ = std::io::stdout().flush();

        if file_processed_successfully && !parsed_records.is_empty() { // Added !parsed_records.is_empty()
            match insert_records(&pool, &parsed_records, &mut skipped_writer).await {
                Ok(_inserted_count) => {
                    info!("Successfully inserted records for file: {}.", config.container_path.display());
                }
                Err(e) => {
                    error!("Failed to insert records for file '{}': {}. Skipping validation and merge.", config.container_path.display(), e);
                    continue;
                }
            }
        }
    }

    println!(">>> RESTORED MAIN: Step 8 - Finished Main Loop.");
    let _ = std::io::stdout().flush();

    println!(">>> RESTORED MAIN: Step 9 - Reached Post-Loop Section (Validation/Merge)...", );
    let _ = std::io::stdout().flush();

    println!("Performing final validation checks...");
    let validation_result = validation::check_schema_and_data_integrity(&pool, "public", "sensor_data", &TARGET_COLUMNS).await;

    let mut proceed_with_merge = false;
    match validation_result {
        Ok(_) => {
            println!("Final validation checks PASSED.");
            proceed_with_merge = true;
        }
        Err(e @ PipelineError::SchemaMismatch { .. }) => {
            error!("Final validation FAILED (Schema Mismatch): {}. Merge will be skipped.", e);
        }
        Err(e @ PipelineError::DataIntegrityError { .. }) => {
            error!("Final validation FAILED (Data Integrity): {}. Merge will be skipped.", e);
        }
        Err(e) => {
            error!("Final validation FAILED (Unexpected Error): {}. Merge will be skipped.", e);
        }
    };

    if proceed_with_merge {
        info!("Final validation passed. Attempting to run merge script...");
        match run_merge_script(&pool).await {
            Ok(_) => {
                info!("Merge script executed successfully.");
                // Upsampling step is now disabled as per user request.
                info!("Skipping upsampling process (perform_upsampling function call disabled).");
                /*  // Original upsampling call -  COMMENTED OUT
                info!("Attempting to perform upsampling on merged data...");
                if let Err(e) = perform_upsampling(&pool).await {
                    error!("Upsampling process failed: {}", e);
                } else {
                    info!("Upsampling process completed (results logged/written to CSV).");
                }
                */
            }
            Err(e) => {
                error!("Failed to execute merge script: {}", e);
            }
        }
    } else {
        // Ensure this log also reflects that upsampling is part of what's skipped.
        println!("Skipping database merge (and subsequent upsampling) due to failed validation checks.");
    }

    println!(">>> RESTORED MAIN: Step 10 - Finished Script");
    let _ = std::io::stdout().flush();

    println!(">>> SUMMARY: Attempted files: {}, Successfully processed: {}, Failed: {} <<<", total_attempted_files, _total_processed_files, failed_files.len());
    if !failed_files.is_empty() {
        println!(">>> FAILED FILES LIST <<<");
        for (path, err) in &failed_files {
            println!("- {} -> {}", path, err);
        }
    }

    Ok(())
}

async fn insert_records(pool: &DbPool, records: &[ParsedRecord], skipped_writer: &mut csv::Writer<File>) -> Result<u64, PipelineError> {
    let source_file_hint = records.first().and_then(|r| r.source_file.as_deref()).unwrap_or("Unknown");
    println!("++++ insert_records: STARTING for batch of {} records (source hint: {}) ++++", records.len(), source_file_hint);
    if records.is_empty() { println!("++++ insert_records: Batch empty, Ok(0) ++++"); return Ok(0); }
    println!("++++ insert_records: Getting connection... ++++");
    let mut client = pool.get().await.map_err(|e| { error!("!!! FAILED get conn: {} !!!", e); PipelineError::DbConnectionError(e) })?;
    println!("++++ insert_records: Got connection. Starting tx... ++++");
    let transaction = client.transaction().await.map_err(|e| { error!("!!! FAILED start tx: {} !!!", e); PipelineError::DbQueryError(e) })?;
    println!("++++ insert_records: Tx started. Preparing COPY... ++++");
    let copy_sql = format!("COPY sensor_data ({}) FROM STDIN BINARY", TARGET_COLUMNS.join(", "));
    println!("++++ insert_records: SQL: `{}`. Starting COPY IN... ++++", copy_sql);
    let sink = transaction.copy_in(&copy_sql).await.map_err(|e| { error!("!!! FAILED copy_in: {} !!!", e); PipelineError::DbQueryError(e) })?;
    println!("++++ insert_records: COPY IN started. Creating writer... ++++");
    let writer = BinaryCopyInWriter::new(sink, &get_column_types());
    pin_mut!(writer);
    println!("++++ insert_records: Writer created. Looping {} records... ++++", records.len());
    for (i, record) in records.iter().enumerate() {
        // === Check and Log Skipped Timestamp Rows ===
        if record.timestamp_utc.is_none() {
            warn!("... skipping record index {} due to missing timestamp (source: {}) ...", i, record.source_file.as_deref().unwrap_or("N/A"));
            // Write to skipped log file
            let source_file_str = record.source_file.as_deref().unwrap_or("Unknown").to_string();
            let uuid_str = record.uuid.as_deref().unwrap_or("N/A").to_string();
            if let Err(e) = skipped_writer.write_record(&[
                &source_file_str,
                &i.to_string(),
                &uuid_str,
                "MissingTimestamp",
            ]) {
                error!("Failed to write skipped row to log: {}", e);
                // Don't stop processing, just log the failure
            }
             if let Err(e) = skipped_writer.flush() { error!("Failed to flush skipped log write: {}", e); }
            continue; // Skip this record
        }
        // ==========================================

        let row_values: Vec<&(dyn ToSql + Sync)> = vec![
            &record.timestamp_utc, &record.source_system, &record.source_file,
            &record.format_type, &record.uuid, &record.lamp_group,
            &record.air_temp_c, &record.air_temp_middle_c, &record.outside_temp_c,
            &record.relative_humidity_percent, &record.humidity_deficit_g_m3,
            &record.radiation_w_m2, &record.light_intensity_lux, &record.light_intensity_umol,
            &record.outside_light_w_m2, &record.co2_measured_ppm, &record.co2_required_ppm,
            &record.co2_dosing_status, &record.co2_status, &record.rain_status,
            &record.vent_pos_1_percent, &record.vent_pos_2_percent, &record.vent_lee_afd3_percent,
            &record.vent_wind_afd3_percent, &record.vent_lee_afd4_percent, &record.vent_wind_afd4_percent,
            &record.curtain_1_percent, &record.curtain_2_percent, &record.curtain_3_percent,
            &record.curtain_4_percent, &record.window_1_percent, &record.window_2_percent,
            &record.lamp_grp1_no3_status, &record.lamp_grp2_no3_status, &record.lamp_grp3_no3_status,
            &record.lamp_grp4_no3_status, &record.lamp_grp1_no4_status, &record.lamp_grp2_no4_status,
            &record.measured_status_bool,
            &record.heating_setpoint_c, &record.pipe_temp_1_c, &record.pipe_temp_2_c,
            &record.flow_temp_1_c, &record.flow_temp_2_c,
            &record.temperature_forecast_c, &record.sun_radiation_forecast_w_m2,
            &record.temperature_actual_c, &record.sun_radiation_actual_w_m2,
            &record.vpd_hpa, &record.humidity_deficit_afd3_g_m3, &record.relative_humidity_afd3_percent,
            &record.humidity_deficit_afd4_g_m3, &record.relative_humidity_afd4_percent,
            &record.behov, &record.status_str, &record.timer_on, &record.timer_off, &record.dli_sum,
            &record.oenske_ekstra_lys, &record.lampe_timer_on, &record.lampe_timer_off,
            &record.value
        ];
        writer.as_mut().write(&row_values).await.map_err(|e| { error!("!!! FAILED write idx {}: {} !!!", i, e); PipelineError::DbQueryError(e) })?;
    }
    println!("++++ insert_records: Loop finished. Finishing writer... ++++");
    let inserted_count = writer.as_mut().finish().await.map_err(|e| { error!("!!! FAILED finish: {} !!!", e); PipelineError::DbQueryError(e) })?;
    println!("++++ insert_records: Writer finished ({} rows). Committing... ++++", inserted_count);
    transaction.commit().await.map_err(|e| { error!("!!! FAILED commit: {} !!!", e); PipelineError::DbQueryError(e) })?;
    println!("++++ insert_records: Committed. FINISHED batch (hint: {}). Ok({}) ++++", source_file_hint, inserted_count);
    Ok(inserted_count)
}

fn get_column_types() -> Vec<Type> {
    vec![
        Type::TIMESTAMPTZ, Type::TEXT, Type::TEXT, Type::TEXT, Type::TEXT, Type::TEXT,
        Type::FLOAT8, Type::FLOAT8, Type::FLOAT8, Type::FLOAT8, Type::FLOAT8,
        Type::FLOAT8, Type::FLOAT8, Type::FLOAT8, Type::FLOAT8, Type::FLOAT8,
        Type::FLOAT8, Type::FLOAT8, Type::FLOAT8, Type::BOOL, // co2_dosing_status (idx 17), co2_status (idx 18), rain_status (idx 19)
        Type::FLOAT8, Type::FLOAT8, Type::FLOAT8, Type::FLOAT8, Type::FLOAT8,
        Type::FLOAT8, Type::FLOAT8, Type::FLOAT8, Type::FLOAT8, Type::FLOAT8,
        Type::FLOAT8, Type::FLOAT8, Type::BOOL, Type::BOOL, Type::BOOL, Type::BOOL,
        Type::BOOL, Type::BOOL, Type::BOOL,
        Type::FLOAT8, Type::FLOAT8, Type::FLOAT8, Type::FLOAT8, Type::FLOAT8,
        Type::FLOAT8, Type::FLOAT8, Type::FLOAT8, Type::FLOAT8,
        Type::FLOAT8, Type::FLOAT8, Type::FLOAT8, Type::FLOAT8, Type::FLOAT8,
        Type::INT4, Type::TEXT, Type::INT4, Type::INT4, Type::FLOAT8,
        Type::TEXT, Type::INT8, Type::INT8,
        Type::FLOAT8
    ]
}

// === ADD Merge Script Function ===
async fn run_merge_script(pool: &DbPool) -> Result<(), PipelineError> {
    let script_path = "/app/sql_scripts/merge_sensor_data.sql";
    info!("Attempting to execute merge script: {}", script_path);

    // 1. Read the SQL script file
    let sql_script = fs::read_to_string(script_path).map_err(|e| {
        error!("Failed to read merge script file '{}': {}", script_path, e);
        PipelineError::MergeScriptError(format!("Failed to read merge script '{}': {}", script_path, e))
    })?;

    // Check if script is empty
    if sql_script.trim().is_empty() {
         error!("Merge script file '{}' is empty.", script_path);
        return Err(PipelineError::MergeScriptError(format!("Merge script '{}' is empty.", script_path)));
    }
    info!("Successfully read merge script ({} bytes)", sql_script.len());

    // 2. Get a database client
    let client = pool.get().await.map_err(PipelineError::DbConnectionError)?;
    info!("Acquired DB connection for merge script.");

    // 3. Execute the script
    // batch_execute runs the entire string, handling multiple statements separated by ';',
    // all within an implicit transaction.
    client.batch_execute(&sql_script).await.map_err(|e| {
        error!("Failed to execute merge script: {}", e);
        PipelineError::MergeScriptError(format!("Merge script execution failed: {}", e))
    })?;

    info!("Successfully executed merge script: {}", script_path);
    Ok(())
}
// ================================

async fn flush_data_in_table(pool: &DbPool) -> Result<(), PipelineError> {
    // Get a client from the connection pool
    let mut client = pool.get().await.map_err(|e| {
        error!("!!! FAILED get conn: {} !!!", e);
        PipelineError::DbConnectionError(e)
    })?;
    
    // Start a transaction
    let transaction = client.transaction().await.map_err(|e| {
        error!("!!! FAILED start tx: {} !!!", e);
        PipelineError::DbQueryError(e)
    })?;

    // SQL to delete all data from the sensor_data table
    let flush_sql = "DELETE FROM sensor_data";

    // Execute the delete statement
    transaction.execute(flush_sql, &[]).await.map_err(|e| {
        error!("!!! FAILED flush: {} !!!", e);
        PipelineError::DbQueryError(e)
    })?;
    
    // Commit the transaction
    transaction.commit().await.map_err(|e| {
        error!("!!! FAILED commit: {} !!!", e);
        PipelineError::DbQueryError(e)
    })?;

    info!("Successfully flushed all data from sensor_data table");
    Ok(())
}

    