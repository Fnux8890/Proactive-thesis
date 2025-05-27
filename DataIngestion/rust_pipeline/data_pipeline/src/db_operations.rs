use crate::data_models::ParsedRecord;
use crate::db::DbPool;
use crate::errors::PipelineError;
use crate::metrics::METRICS;
use crate::retry::{db_retry_config, retry_with_backoff};
use crate::TARGET_COLUMNS;
use csv::Writer;
use futures_util::stream::FuturesUnordered;
use futures_util::StreamExt;
use log::{error, info, warn};
use pin_utils::pin_mut;
use std::fs::File;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio_postgres::binary_copy::BinaryCopyInWriter;
use tokio_postgres::types::{ToSql, Type};

/// Maximum concurrent database operations
const MAX_CONCURRENT_DB_OPS: usize = 4;

/// Batch size for database inserts
const DB_BATCH_SIZE: usize = 5000;

/// Check if null value logging is enabled via environment variable
fn is_null_logging_enabled() -> bool {
    std::env::var("ENABLE_NULL_LOGGING")
        .unwrap_or_else(|_| "false".to_string())
        .to_lowercase() == "true"
}

/// Database operations handler
pub struct DbOperations {
    pool: DbPool,
    semaphore: Arc<Semaphore>,
}

impl DbOperations {
    pub fn new(pool: DbPool) -> Self {
        Self {
            pool,
            semaphore: Arc::new(Semaphore::new(MAX_CONCURRENT_DB_OPS)),
        }
    }

    /// Insert records in parallel batches
    pub async fn insert_records_parallel(
        &self,
        all_records: Vec<ParsedRecord>,
        skipped_writer: Arc<tokio::sync::Mutex<Writer<File>>>,
    ) -> Result<u64, PipelineError> {
        if all_records.is_empty() {
            return Ok(0);
        }

        info!(
            "Starting parallel insertion of {} records in batches of {}",
            all_records.len(),
            DB_BATCH_SIZE
        );

        // Split records into batches
        let batches: Vec<Vec<ParsedRecord>> = all_records
            .chunks(DB_BATCH_SIZE)
            .map(|chunk| chunk.to_vec())
            .collect();

        let num_batches = batches.len();
        info!("Created {} batches for parallel processing", num_batches);

        // Process batches concurrently
        let mut futures = FuturesUnordered::new();

        for (batch_idx, batch) in batches.into_iter().enumerate() {
            let pool = self.pool.clone();
            let semaphore = self.semaphore.clone();
            let writer = skipped_writer.clone();

            futures.push(async move {
                let _permit = semaphore.acquire().await.unwrap();
                let result = Self::insert_batch(&pool, batch, batch_idx, writer).await;
                drop(_permit);
                result
            });
        }

        // Collect results
        let mut total_inserted = 0u64;
        let mut failed_batches = 0;

        while let Some(result) = futures.next().await {
            match result {
                Ok(count) => total_inserted += count,
                Err(e) => {
                    error!("Batch insertion failed: {}", e);
                    failed_batches += 1;
                }
            }
        }

        if failed_batches > 0 {
            warn!(
                "Completed with {} failed batches out of {}",
                failed_batches, num_batches
            );
        }

        METRICS.lock().record_insertion(total_inserted);
        Ok(total_inserted)
    }

    /// Insert a single batch of records
    async fn insert_batch(
        pool: &DbPool,
        records: Vec<ParsedRecord>,
        batch_idx: usize,
        skipped_writer: Arc<tokio::sync::Mutex<Writer<File>>>,
    ) -> Result<u64, PipelineError> {
        let source_file_hint = records
            .first()
            .and_then(|r| r.source_file.as_deref())
            .unwrap_or("Unknown");

        info!(
            "Processing batch {} with {} records (source: {})",
            batch_idx,
            records.len(),
            source_file_hint
        );

        // Use retry logic for database operations
        let retry_config = db_retry_config();
        let pool_clone = pool.clone();
        let records_clone = records.clone();
        let skipped_writer_clone = skipped_writer.clone();

        retry_with_backoff(
            &retry_config,
            &format!("insert_batch_{}", batch_idx),
            move || {
                let pool = pool_clone.clone();
                let records = records_clone.clone();
                let writer = skipped_writer_clone.clone();
                
                async move {
                    Self::insert_batch_inner(&pool, records, batch_idx, writer).await
                }
            },
        )
        .await
    }

    /// Inner batch insertion logic (for retry wrapper)
    async fn insert_batch_inner(
        pool: &DbPool,
        records: Vec<ParsedRecord>,
        batch_idx: usize,
        skipped_writer: Arc<tokio::sync::Mutex<Writer<File>>>,
    ) -> Result<u64, PipelineError> {
        let mut client = pool
            .get()
            .await
            .map_err(PipelineError::DbConnectionError)?;

        let transaction = client
            .transaction()
            .await
            .map_err(PipelineError::DbQueryError)?;

        let copy_sql = format!(
            "COPY sensor_data ({}) FROM STDIN BINARY",
            TARGET_COLUMNS.join(", ")
        );

        let sink = transaction
            .copy_in(&copy_sql)
            .await
            .map_err(PipelineError::DbQueryError)?;

        let writer = BinaryCopyInWriter::new(sink, &get_column_types());
        pin_mut!(writer);

        let mut skipped_count = 0;
        let mut inserted_count = 0;
        let mut null_columns_log = Vec::new();

        for (i, record) in records.iter().enumerate() {
            if record.timestamp_utc.is_none() {
                skipped_count += 1;
                // Log to skipped file
                if let Ok(mut csv_writer) = skipped_writer.try_lock() {
                    let _ = csv_writer.write_record(&[
                        record.source_file.as_deref().unwrap_or("Unknown"),
                        &batch_idx.to_string(),
                        &i.to_string(),
                        record.uuid.as_deref().unwrap_or("N/A"),
                        "MissingTimestamp",
                    ]);
                    let _ = csv_writer.flush();
                }
                continue;
            }
            
            // Track null values for each column (only if logging is enabled)
            if is_null_logging_enabled() {
                let mut null_fields = Vec::new();
                
                // Check each field for null values
                if record.air_temp_c.is_none() { null_fields.push("air_temp_c"); }
                if record.air_temp_middle_c.is_none() { null_fields.push("air_temp_middle_c"); }
                if record.outside_temp_c.is_none() { null_fields.push("outside_temp_c"); }
                if record.relative_humidity_percent.is_none() { null_fields.push("relative_humidity_percent"); }
                if record.humidity_deficit_g_m3.is_none() { null_fields.push("humidity_deficit_g_m3"); }
                if record.radiation_w_m2.is_none() { null_fields.push("radiation_w_m2"); }
                if record.light_intensity_lux.is_none() { null_fields.push("light_intensity_lux"); }
                if record.light_intensity_umol.is_none() { null_fields.push("light_intensity_umol"); }
                if record.outside_light_w_m2.is_none() { null_fields.push("outside_light_w_m2"); }
                if record.co2_measured_ppm.is_none() { null_fields.push("co2_measured_ppm"); }
                if record.co2_required_ppm.is_none() { null_fields.push("co2_required_ppm"); }
                if record.co2_dosing_status.is_none() { null_fields.push("co2_dosing_status"); }
                if record.co2_status.is_none() { null_fields.push("co2_status"); }
                if record.rain_status.is_none() { null_fields.push("rain_status"); }
                if record.vent_pos_1_percent.is_none() { null_fields.push("vent_pos_1_percent"); }
                if record.vent_pos_2_percent.is_none() { null_fields.push("vent_pos_2_percent"); }
                if record.vent_lee_afd3_percent.is_none() { null_fields.push("vent_lee_afd3_percent"); }
                if record.vent_wind_afd3_percent.is_none() { null_fields.push("vent_wind_afd3_percent"); }
                if record.vent_lee_afd4_percent.is_none() { null_fields.push("vent_lee_afd4_percent"); }
                if record.vent_wind_afd4_percent.is_none() { null_fields.push("vent_wind_afd4_percent"); }
                if record.curtain_1_percent.is_none() { null_fields.push("curtain_1_percent"); }
                if record.curtain_2_percent.is_none() { null_fields.push("curtain_2_percent"); }
                if record.curtain_3_percent.is_none() { null_fields.push("curtain_3_percent"); }
                if record.curtain_4_percent.is_none() { null_fields.push("curtain_4_percent"); }
                if record.window_1_percent.is_none() { null_fields.push("window_1_percent"); }
                if record.window_2_percent.is_none() { null_fields.push("window_2_percent"); }
                if record.lamp_grp1_no3_status.is_none() { null_fields.push("lamp_grp1_no3_status"); }
                if record.lamp_grp2_no3_status.is_none() { null_fields.push("lamp_grp2_no3_status"); }
                if record.lamp_grp3_no3_status.is_none() { null_fields.push("lamp_grp3_no3_status"); }
                if record.lamp_grp4_no3_status.is_none() { null_fields.push("lamp_grp4_no3_status"); }
                if record.lamp_grp1_no4_status.is_none() { null_fields.push("lamp_grp1_no4_status"); }
                if record.lamp_grp2_no4_status.is_none() { null_fields.push("lamp_grp2_no4_status"); }
                if record.measured_status_bool.is_none() { null_fields.push("measured_status_bool"); }
                if record.heating_setpoint_c.is_none() { null_fields.push("heating_setpoint_c"); }
                if record.pipe_temp_1_c.is_none() { null_fields.push("pipe_temp_1_c"); }
                if record.pipe_temp_2_c.is_none() { null_fields.push("pipe_temp_2_c"); }
                if record.flow_temp_1_c.is_none() { null_fields.push("flow_temp_1_c"); }
                if record.flow_temp_2_c.is_none() { null_fields.push("flow_temp_2_c"); }
                if record.temperature_forecast_c.is_none() { null_fields.push("temperature_forecast_c"); }
                if record.sun_radiation_forecast_w_m2.is_none() { null_fields.push("sun_radiation_forecast_w_m2"); }
                if record.temperature_actual_c.is_none() { null_fields.push("temperature_actual_c"); }
                if record.sun_radiation_actual_w_m2.is_none() { null_fields.push("sun_radiation_actual_w_m2"); }
                if record.vpd_hpa.is_none() { null_fields.push("vpd_hpa"); }
                if record.humidity_deficit_afd3_g_m3.is_none() { null_fields.push("humidity_deficit_afd3_g_m3"); }
                if record.relative_humidity_afd3_percent.is_none() { null_fields.push("relative_humidity_afd3_percent"); }
                if record.humidity_deficit_afd4_g_m3.is_none() { null_fields.push("humidity_deficit_afd4_g_m3"); }
                if record.relative_humidity_afd4_percent.is_none() { null_fields.push("relative_humidity_afd4_percent"); }
                
                // Log null values if any exist
                if !null_fields.is_empty() {
                    null_columns_log.push((
                        record.source_file.as_deref().unwrap_or("Unknown").to_string(),
                        batch_idx,
                        i,
                        record.timestamp_utc.as_ref().map(|t| t.to_string()).unwrap_or_default(),
                        null_fields.join(",")
                    ));
                }
            }

            let row_values: Vec<&(dyn ToSql + Sync)> = vec![
                &record.timestamp_utc,
                &record.source_system,
                &record.source_file,
                &record.format_type,
                &record.uuid,
                &record.lamp_group,
                &record.air_temp_c,
                &record.air_temp_middle_c,
                &record.outside_temp_c,
                &record.relative_humidity_percent,
                &record.humidity_deficit_g_m3,
                &record.radiation_w_m2,
                &record.light_intensity_lux,
                &record.light_intensity_umol,
                &record.outside_light_w_m2,
                &record.co2_measured_ppm,
                &record.co2_required_ppm,
                &record.co2_dosing_status,
                &record.co2_status,
                &record.rain_status,
                &record.vent_pos_1_percent,
                &record.vent_pos_2_percent,
                &record.vent_lee_afd3_percent,
                &record.vent_wind_afd3_percent,
                &record.vent_lee_afd4_percent,
                &record.vent_wind_afd4_percent,
                &record.curtain_1_percent,
                &record.curtain_2_percent,
                &record.curtain_3_percent,
                &record.curtain_4_percent,
                &record.window_1_percent,
                &record.window_2_percent,
                &record.lamp_grp1_no3_status,
                &record.lamp_grp2_no3_status,
                &record.lamp_grp3_no3_status,
                &record.lamp_grp4_no3_status,
                &record.lamp_grp1_no4_status,
                &record.lamp_grp2_no4_status,
                &record.measured_status_bool,
                &record.heating_setpoint_c,
                &record.pipe_temp_1_c,
                &record.pipe_temp_2_c,
                &record.flow_temp_1_c,
                &record.flow_temp_2_c,
                &record.temperature_forecast_c,
                &record.sun_radiation_forecast_w_m2,
                &record.temperature_actual_c,
                &record.sun_radiation_actual_w_m2,
                &record.vpd_hpa,
                &record.humidity_deficit_afd3_g_m3,
                &record.relative_humidity_afd3_percent,
                &record.humidity_deficit_afd4_g_m3,
                &record.relative_humidity_afd4_percent,
                &record.behov,
                &record.status_str,
                &record.timer_on,
                &record.timer_off,
                &record.dli_sum,
                &record.oenske_ekstra_lys,
                &record.lampe_timer_on,
                &record.lampe_timer_off,
                &record.value,
            ];

            writer
                .as_mut()
                .write(&row_values)
                .await
                .map_err(PipelineError::DbQueryError)?;
            inserted_count += 1;
        }

        let final_count = writer
            .as_mut()
            .finish()
            .await
            .map_err(PipelineError::DbQueryError)?;

        transaction
            .commit()
            .await
            .map_err(PipelineError::DbQueryError)?;

        if skipped_count > 0 {
            warn!(
                "Batch {}: Skipped {} records with missing timestamps",
                batch_idx, skipped_count
            );
        }
        
        // Write null column log to a separate file (only if logging is enabled)
        if is_null_logging_enabled() && !null_columns_log.is_empty() {
            let null_log_path = format!("/app/logs/null_columns_batch_{}.txt", batch_idx);
            if let Ok(mut file) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&null_log_path)
            {
                use std::io::Write;
                writeln!(&mut file, "=== Batch {} Null Columns Report ===", batch_idx).ok();
                writeln!(&mut file, "Total records with nulls: {}", null_columns_log.len()).ok();
                writeln!(&mut file, "Format: SourceFile | BatchIdx | RecordIdx | Timestamp | NullColumns").ok();
                writeln!(&mut file, "========================================").ok();
                
                for (source_file, batch, idx, timestamp, null_cols) in &null_columns_log {
                    writeln!(&mut file, "{} | {} | {} | {} | {}", 
                        source_file, batch, idx, timestamp, null_cols).ok();
                }
                
                info!(
                    "Batch {}: Logged {} records with null columns to {}",
                    batch_idx, null_columns_log.len(), null_log_path
                );
            }
        }

        info!(
            "Batch {}: Successfully inserted {} records",
            batch_idx, inserted_count
        );

        Ok(final_count)
    }

    /// Flush all data from sensor_data table
    pub async fn flush_sensor_data(&self) -> Result<(), PipelineError> {
        let mut client = self
            .pool
            .get()
            .await
            .map_err(PipelineError::DbConnectionError)?;

        let transaction = client
            .transaction()
            .await
            .map_err(PipelineError::DbQueryError)?;

        transaction
            .execute("DELETE FROM sensor_data", &[])
            .await
            .map_err(PipelineError::DbQueryError)?;

        transaction
            .commit()
            .await
            .map_err(PipelineError::DbQueryError)?;

        info!("Successfully flushed all data from sensor_data table");
        Ok(())
    }

    /// Run merge script
    pub async fn run_merge_script(&self) -> Result<(), PipelineError> {
        let script_path = "/app/sql_scripts/merge_sensor_data.sql";
        info!("Executing merge script: {}", script_path);

        let sql_script = std::fs::read_to_string(script_path).map_err(|e| {
            PipelineError::MergeScriptError(format!("Failed to read merge script '{}': {}", script_path, e))
        })?;

        if sql_script.trim().is_empty() {
            return Err(PipelineError::MergeScriptError(
                format!("Merge script '{}' is empty.", script_path),
            ));
        }

        let client = self
            .pool
            .get()
            .await
            .map_err(PipelineError::DbConnectionError)?;

        client
            .batch_execute(&sql_script)
            .await
            .map_err(|e| PipelineError::MergeScriptError(format!("Merge script execution failed: {}", e)))?;

        info!("Successfully executed merge script");
        Ok(())
    }
}

/// Get column types for the sensor_data table
fn get_column_types() -> Vec<Type> {
    vec![
        Type::TIMESTAMPTZ,          // time
        Type::TEXT,                 // source_system
        Type::TEXT,                 // source_file
        Type::TEXT,                 // format_type
        Type::TEXT,                 // uuid
        Type::VARCHAR,              // lamp_group
        Type::FLOAT8,               // air_temp_c
        Type::FLOAT8,               // air_temp_middle_c
        Type::FLOAT8,               // outside_temp_c
        Type::FLOAT8,               // relative_humidity_percent
        Type::FLOAT8,               // humidity_deficit_g_m3
        Type::FLOAT8,               // radiation_w_m2
        Type::FLOAT8,               // light_intensity_lux
        Type::FLOAT8,               // light_intensity_umol
        Type::FLOAT8,               // outside_light_w_m2
        Type::FLOAT8,               // co2_measured_ppm
        Type::FLOAT8,               // co2_required_ppm
        Type::FLOAT8,               // co2_dosing_status
        Type::FLOAT8,               // co2_status
        Type::BOOL,                 // rain_status
        Type::FLOAT8,               // vent_pos_1_percent
        Type::FLOAT8,               // vent_pos_2_percent
        Type::FLOAT8,               // vent_lee_afd3_percent
        Type::FLOAT8,               // vent_wind_afd3_percent
        Type::FLOAT8,               // vent_lee_afd4_percent
        Type::FLOAT8,               // vent_wind_afd4_percent
        Type::FLOAT8,               // curtain_1_percent
        Type::FLOAT8,               // curtain_2_percent
        Type::FLOAT8,               // curtain_3_percent
        Type::FLOAT8,               // curtain_4_percent
        Type::FLOAT8,               // window_1_percent
        Type::FLOAT8,               // window_2_percent
        Type::BOOL,                 // lamp_grp1_no3_status
        Type::BOOL,                 // lamp_grp2_no3_status
        Type::BOOL,                 // lamp_grp3_no3_status
        Type::BOOL,                 // lamp_grp4_no3_status
        Type::BOOL,                 // lamp_grp1_no4_status
        Type::BOOL,                 // lamp_grp2_no4_status
        Type::BOOL,                 // measured_status_bool
        Type::FLOAT8,               // heating_setpoint_c
        Type::FLOAT8,               // pipe_temp_1_c
        Type::FLOAT8,               // pipe_temp_2_c
        Type::FLOAT8,               // flow_temp_1_c
        Type::FLOAT8,               // flow_temp_2_c
        Type::FLOAT8,               // temperature_forecast_c
        Type::FLOAT8,               // sun_radiation_forecast_w_m2
        Type::FLOAT8,               // temperature_actual_c
        Type::FLOAT8,               // sun_radiation_actual_w_m2
        Type::FLOAT8,               // vpd_hpa
        Type::FLOAT8,               // humidity_deficit_afd3_g_m3
        Type::FLOAT8,               // relative_humidity_afd3_percent
        Type::FLOAT8,               // humidity_deficit_afd4_g_m3
        Type::FLOAT8,               // relative_humidity_afd4_percent
        Type::INT4,                 // behov
        Type::TEXT,                 // status_str
        Type::INT4,                 // timer_on
        Type::INT4,                 // timer_off
        Type::FLOAT8,               // dli_sum
        Type::TEXT,                 // oenske_ekstra_lys
        Type::INT8,                 // lampe_timer_on
        Type::INT8,                 // lampe_timer_off
        Type::FLOAT8,               // value
    ]
}