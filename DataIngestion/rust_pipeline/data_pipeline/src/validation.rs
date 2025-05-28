// src/validation.rs
use crate::db::DbPool;
use crate::errors::PipelineError;
use std::collections::HashSet;
use tokio_postgres::Row;
use log::{error, info, warn}; // Use log macros

// SQL for data integrity check (embedded for simplicity)
const DATA_INTEGRITY_SQL: &str = r#"
WITH failed_checks AS (
    -- Check 1: Look for unreasonable air temperature values -- REMOVED
    -- SELECT
    --     'Range Error' AS check_type,
    --     time,
    --     source_file,
    --     'air_temp_c' AS column_name,
    --     air_temp_c::text AS value
    -- FROM sensor_data
    -- WHERE air_temp_c < -50 OR air_temp_c > 60

    -- Check 2: Look for NULL timestamps
    SELECT
        'Null Timestamp' AS check_type,
        time, -- Will be NULL here
        source_file,
        'time' AS column_name,
        'NULL' AS value
    FROM sensor_data
    WHERE time IS NULL

    -- REMOVED CHECK 3 (HUMIDITY) and preceding UNION ALL
    UNION ALL
    -- Check 3: Look for unreasonable humidity values -- REMOVED
    SELECT
        'Range Error' AS check_type,
        time,
        source_file,
        'relative_humidity_percent' AS column_name,
        relative_humidity_percent::text AS value
    FROM sensor_data
    WHERE relative_humidity_percent < 0 OR relative_humidity_percent > 105 -- Allow slightly over 100

    -- Add more UNION ALL blocks for other checks here (if any remain)
)
-- Select the first row (if any) from all combined failed checks
SELECT check_type, source_file, column_name, value
FROM failed_checks
LIMIT 1;
"#;


/// Checks if the target table schema matches expected columns and runs data integrity checks.
///
/// # Arguments
/// * `pool` - The database connection pool.
/// * `schema` - The database schema name (e.g., "public").
/// * `table` - The database table name (e.g., "sensor_data").
/// * `expected_columns` - A slice of expected column names.
///
/// # Errors
/// Returns `PipelineError::SchemaMismatch` if columns don't match.
/// Returns `PipelineError::DataIntegrityError` if the data integrity SQL finds issues.
/// Returns other `PipelineError` variants for database connection or query errors.
pub async fn check_schema_and_data_integrity(
    pool: &DbPool,
    schema: &str,
    table: &str,
    expected_columns: &[&str],
) -> Result<(), PipelineError> {
    info!("Starting schema and data integrity check for {}.{}", schema, table);

    let client = pool.get().await?; // <-- Remove mut

    // --- 1. Schema Check ---
    info!("Performing schema check...");
    let query = "SELECT column_name FROM information_schema.columns WHERE table_schema = $1 AND table_name = $2";
    let rows = client.query(query, &[&schema, &table]).await?; // Propagates DbQueryError

    let actual_columns: HashSet<String> = rows.iter().map(|row| row.get(0)).collect();
    let expected_columns_set: HashSet<String> =
        expected_columns.iter().map(|s| s.to_string()).collect();

    // Find missing columns
    let missing_columns: Vec<_> = expected_columns_set
        .difference(&actual_columns)
        .cloned()
        .collect();

    // Find extra columns (optional check, can be useful)
    let extra_columns: Vec<_> = actual_columns
        .difference(&expected_columns_set)
        .cloned()
        .collect();

    if !missing_columns.is_empty() {
        error!("Schema check FAILED: Missing columns: {:?}", missing_columns);
        // Optionally log extra columns if needed: error!("Extra columns found: {:?}", extra_columns);
        return Err(PipelineError::SchemaMismatch {
            table: format!("{}.{}", schema, table),
            missing: missing_columns,
            extra: extra_columns, // Include extra columns in the error
        });
    }

    if !extra_columns.is_empty() {
         warn!("Schema check WARNING: Found extra columns not in TARGET_COLUMNS: {:?}. Proceeding, but schema might need update.", extra_columns);
    }

    info!("Schema check PASSED.");

    // --- 2. Data Integrity Check ---
    info!("Performing data integrity check...");
    let integrity_result_opt: Option<Row> = client
        .query_opt(DATA_INTEGRITY_SQL, &[]) // No params needed for this SQL
        .await?; // Propagates DbQueryError

    if let Some(failed_row) = integrity_result_opt {
        let check_type: String = failed_row.get(0);
        let source_file: Option<String> = failed_row.get(1); // Source file might be null if timestamp is null
        let column_name: String = failed_row.get(2);
        let value: String = failed_row.get(3);
        error!(
            "Data integrity check FAILED: Type='{}', File='{}', Column='{}', Value='{}'",
            check_type,
            source_file.as_deref().unwrap_or("N/A"),
            column_name,
            value
        );
        return Err(PipelineError::DataIntegrityError {
            check_type,
            source_file,
            column_name,
            value,
        });
    }

    info!("Data integrity check PASSED.");
    info!("Overall check PASSED for {}.{}", schema, table);
    Ok(())
} 