use crate::data_models::ParsedRecord;
use crate::db::DbPool;
use crate::errors::PipelineError;
use crate::metrics::METRICS;
use crate::retry::{db_retry_config, retry_with_backoff};
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

// Column definitions
const TARGET_COLUMNS: [&str; 62] = [
    "time",
    "source_system",
    "source_file",
    "format_type",
    "uuid",
    "lamp_group",
    "air_temp_c",
    "air_temp_middle_c",
    "outside_temp_c",
    "relative_humidity_percent",
    "humidity_deficit_g_m3",
    "radiation_w_m2",
    "light_intensity_lux",
    "light_intensity_umol",
    "outside_light_w_m2",
    "co2_measured_ppm",
    "co2_required_ppm",
    "co2_dosing_status",
    "co2_status",
    "rain_status",
    "vent_pos_1_percent",
    "vent_pos_2_percent",
    "vent_lee_afd3_percent",
    "vent_wind_afd3_percent",
    "vent_lee_afd4_percent",
    "vent_wind_afd4_percent",
    "curtain_1_percent",
    "curtain_2_percent",
    "curtain_3_percent",
    "curtain_4_percent",
    "window_1_percent",
    "window_2_percent",
    "lamp_grp1_no3_status",
    "lamp_grp2_no3_status",
    "lamp_grp3_no3_status",
    "lamp_grp4_no3_status",
    "lamp_grp1_no4_status",
    "lamp_grp2_no4_status",
    "measured_status_bool",
    "heating_setpoint_c",
    "pipe_temp_1_c",
    "pipe_temp_2_c",
    "flow_temp_1_c",
    "flow_temp_2_c",
    "temperature_forecast_c",
    "sun_radiation_forecast_w_m2",
    "temperature_actual_c",
    "sun_radiation_actual_w_m2",
    "vpd_hpa",
    "humidity_deficit_afd3_g_m3",
    "relative_humidity_afd3_percent",
    "humidity_deficit_afd4_g_m3",
    "relative_humidity_afd4_percent",
    "behov",
    "status_str",
    "timer_on",
    "timer_off",
    "dli_sum",
    "oenske_ekstra_lys",
    "lampe_timer_on",
    "lampe_timer_off",
    "value",
];

fn get_column_types() -> Vec<Type> {
    vec![
        Type::TIMESTAMPTZ,
        Type::TEXT,
        Type::TEXT,
        Type::TEXT,
        Type::TEXT,
        Type::TEXT,
        Type::FLOAT8,
        Type::FLOAT8,
        Type::FLOAT8,
        Type::FLOAT8,
        Type::FLOAT8,
        Type::FLOAT8,
        Type::FLOAT8,
        Type::FLOAT8,
        Type::FLOAT8,
        Type::FLOAT8,
        Type::FLOAT8,
        Type::FLOAT8,
        Type::FLOAT8,
        Type::BOOL,
        Type::FLOAT8,
        Type::FLOAT8,
        Type::FLOAT8,
        Type::FLOAT8,
        Type::FLOAT8,
        Type::FLOAT8,
        Type::FLOAT8,
        Type::FLOAT8,
        Type::FLOAT8,
        Type::FLOAT8,
        Type::FLOAT8,
        Type::FLOAT8,
        Type::BOOL,
        Type::BOOL,
        Type::BOOL,
        Type::BOOL,
        Type::BOOL,
        Type::BOOL,
        Type::BOOL,
        Type::FLOAT8,
        Type::FLOAT8,
        Type::FLOAT8,
        Type::FLOAT8,
        Type::FLOAT8,
        Type::FLOAT8,
        Type::FLOAT8,
        Type::FLOAT8,
        Type::FLOAT8,
        Type::FLOAT8,
        Type::FLOAT8,
        Type::FLOAT8,
        Type::FLOAT8,
        Type::FLOAT8,
        Type::INT4,
        Type::TEXT,
        Type::INT4,
        Type::INT4,
        Type::FLOAT8,
        Type::TEXT,
        Type::INT8,
        Type::INT8,
        Type::FLOAT8,
    ]
}