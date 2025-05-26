// src/db.rs
use anyhow::{Context, Result};
use std::io::{Write, Cursor, Read};

use polars::prelude::*;
use polars::lazy::dsl::{col, lit};
use r2d2_postgres::{PostgresConnectionManager, r2d2};
use postgres::NoTls;
use regex::Regex;
use lazy_static::lazy_static;
use std::env;

// Type alias for the R2D2 connection pool

lazy_static! {
    static ref VALID_IDENTIFIER_REGEX: Regex = Regex::new(r"^[a-zA-Z_][a-zA-Z0-9_]*$").unwrap();
}

fn quote_identifier(name: &str) -> Result<String, anyhow::Error> {
    if VALID_IDENTIFIER_REGEX.is_match(name) {
        Ok(format!("\"{}\"", name))
    } else {
        Err(anyhow::anyhow!("Invalid identifier name: '{}'. Must match regex: ^[a-zA-Z_][a-zA-Z0-9_]*$", name))
    }
}

pub type DbPool = r2d2::Pool<PostgresConnectionManager<NoTls>>;

// /// 🔒  DSN was hard-coded. Now read from environment.
// const DB_DSN: &str =
//     "host=127.0.0.1 port=5432 dbname=mydb user=myuser password=supersecret";

/// Database access structure, now holding a connection pool.
pub struct EraDb {
    pool: DbPool,
}

impl EraDb {
    pub fn new(dsn_override: Option<&str>) -> Result<Self> {
        let db_dsn_string = match dsn_override {
            Some(dsn) => dsn.to_string(),
            None => env::var("DB_DSN")
                .with_context(|| "DB_DSN environment variable not set, and no DSN override provided")?,
        };
        // Ensure to use db_dsn_string below instead of db_dsn
        
        let manager = PostgresConnectionManager::new(
            db_dsn_string.parse().with_context(|| format!("Failed to parse DSN: {}", db_dsn_string))?,
            NoTls,
        );
        
        let pool = r2d2::Pool::builder()
            .build(manager)
            .with_context(|| "Failed to create R2D2 connection pool")?;

        // Initialize schema on first connection from the pool
        let mut conn = pool.get().with_context(|| "Failed to get connection from pool for schema initialization")?;
        conn.batch_execute(
            r#"
            CREATE TABLE IF NOT EXISTS era_labels (
                signal_name TEXT        NOT NULL,
                level       CHAR(1)     NOT NULL,   -- A | B | C
                stage       TEXT        NOT NULL,   -- PELT | BOCPD | HMM
                era_id      INT         NOT NULL,
                start_time  TIMESTAMPTZ NOT NULL,
                end_time    TIMESTAMPTZ NULL,
                rows        INT         NULL,
                PRIMARY KEY (signal_name, level, stage, era_id, start_time) -- Adjusted PK for hypertable
            );
            SELECT create_hypertable('era_labels','start_time',
                                     if_not_exists => TRUE);
            "#,
        ).with_context(|| "Failed to create era_labels table or hypertable")?;

        Ok(Self { pool })
    }

    /// Ensures a table with the standard era_labels schema exists and is a hypertable.
    /// Ensures a table with the standard era_labels schema exists and is a hypertable.
    /// The `table_name` argument should be the raw, unquoted name.
    pub fn ensure_era_table_exists(&self, table_name_raw: &str) -> Result<()> {
        let table_name = quote_identifier(table_name_raw)?;
        let mut conn = self.pool.get().with_context(|| format!("Failed to get connection from pool for ensuring table: {}", table_name_raw))?;
        let create_table_sql = format!(
            r#"
            CREATE TABLE IF NOT EXISTS {} (
                signal_name TEXT        NOT NULL,
                level       CHAR(1)     NOT NULL,   -- A | B | C
                stage       TEXT        NOT NULL,   -- PELT | BOCPD | HMM
                era_id      INT         NOT NULL,
                start_time  TIMESTAMPTZ NOT NULL,
                end_time    TIMESTAMPTZ NULL,
                rows        INT         NULL,
                PRIMARY KEY (signal_name, level, stage, era_id, start_time) -- Adjusted PK for hypertable
            );
            SELECT create_hypertable('{}', 'start_time', if_not_exists => TRUE, migrate_data => TRUE);
            "#,
            table_name, table_name // Here table_name is already quoted and sanitized
        );
        conn.batch_execute(&create_table_sql)
            .with_context(|| format!("Failed to create table or hypertable: {}. SQL: \n{}", table_name_raw, create_table_sql))?;
        Ok(())
    }

    /// Saves era labels from a DataFrame to a specified target table in the database.
    /// Ensures the target table exists and is a hypertable before copying data.
    pub fn save_era_labels(
        &self,
        df_source: &DataFrame,
        original_era_col_name: &str,
        signal_name: &str,
        level_char: char,
        stage_str: &str,
        target_table_name_raw: &str, // Renamed to indicate it's the raw name
    ) -> Result<()> {
        // Sanitize and quote the table name for SQL statements
        let target_table_name = quote_identifier(target_table_name_raw)?;

        self.ensure_era_table_exists(target_table_name_raw)
            .with_context(|| format!("Failed to ensure target table '{}' exists.", target_table_name_raw))?;

        let mut df_to_persist = df_source.clone();
        df_to_persist
            .rename(original_era_col_name, "era_id".into())
            .with_context(|| {
                format!(
                    "Failed to rename column '{}' to 'era_id' for signal '{}', level '{}', stage '{}', target table '{}'",
                    original_era_col_name, signal_name, level_char, stage_str, target_table_name
                )
            })?;

        let csv_data = Self::build_segments_csv(&df_to_persist, signal_name, level_char, stage_str)
            .with_context(|| {
                format!(
                    "Failed to build CSV for signal '{}', level '{}', stage '{}', target table '{}'",
                    signal_name, level_char, stage_str, target_table_name_raw
                )
            })?;

        if csv_data.is_empty() {
            log::debug!(
                "No CSV data to persist for signal '{}', level '{}', stage '{}', target table '{}'. Skipping COPY.",
                signal_name,
                level_char,
                stage_str,
                target_table_name_raw
            );
            return Ok(());
        }

        // The target_table_name is already quoted. The column names in the COPY statement
        // should be standard and not require dynamic quoting here.
        let copy_sql = format!(
            "COPY {} (signal_name, level, stage, era_id, start_time, end_time, rows) FROM STDIN CSV",
            target_table_name // This is the quoted version from earlier in this function
        );
        
        let data_len = csv_data.len();
        let mut conn = self.pool.get().with_context(|| format!("Failed to get connection from pool for saving to table: {}", target_table_name))?;
        let mut writer = conn.copy_in(&copy_sql)
            .with_context(|| format!("Failed to start COPY for table '{}' (data size: {} bytes)", target_table_name_raw, data_len))?;
        writer.write_all(&csv_data[..])
            .with_context(|| format!("Failed to write data for COPY to table '{}' (data size: {} bytes)", target_table_name_raw, data_len))?;
        writer.finish()
            .with_context(|| format!("COPY to table '{}' failed during finish (data size: {} bytes)", target_table_name_raw, data_len))?;
        
        Ok(())
    }

    pub fn load_feature_df(&self, table: &str, row_limit: Option<usize>) -> Result<DataFrame> {
        // use postgres::{Client, NoTls}; // Not strictly needed here if using pool directly
        // Polars CsvReader is already in scope via `use polars::prelude::*;`
        // Cursor and Read are brought in by the `use` statement at the top of the file.

        let mut conn = self.pool.get().with_context(|| format!("Failed to get connection from pool for loading table: {}", table))?;
        // Stream the query result out as CSV so we can hand the bytes
        // straight to Polars without an intermediate file.
        let quoted_table_name = quote_identifier(table)
            .with_context(|| format!("Failed to sanitize table name '{}' for load_feature_df", table))?;

        // Check if this is the preprocessed_features table with JSONB
        let base_query = if table == "preprocessed_features" {
            // For preprocessed_features, extract JSONB fields as columns
            log::info!("Loading preprocessed_features table with JSONB expansion");
            
            // Simplified approach - extract key sensor fields and let nulls be nulls
            // Using safe casting that returns NULL on error
            // Extract time column and JSONB fields
            // Time is stored as a separate column, not in JSONB
            format!(
                r#"SELECT 
                    time,
                    (features->>'air_temp_c')::float AS air_temp_c,
                    (features->>'relative_humidity_percent')::float AS relative_humidity_percent,
                    (features->>'co2_measured_ppm')::float AS co2_measured_ppm,
                    (features->>'radiation_w_m2')::float AS radiation_w_m2,
                    (features->>'light_intensity_umol')::float AS light_intensity_umol,
                    (features->>'heating_setpoint_c')::float AS heating_setpoint_c,
                    (features->>'pipe_temp_1_c')::float AS pipe_temp_1_c,
                    (features->>'pipe_temp_2_c')::float AS pipe_temp_2_c,
                    (features->>'flow_temp_1_c')::float AS flow_temp_1_c,
                    (features->>'flow_temp_2_c')::float AS flow_temp_2_c,
                    (features->>'vent_lee_afd3_percent')::float AS vent_lee_afd3_percent,
                    (features->>'vent_wind_afd3_percent')::float AS vent_wind_afd3_percent,
                    (features->>'total_lamps_on')::float AS total_lamps_on,
                    (features->>'dli_sum')::float AS dli_sum,
                    (features->>'vpd_hpa')::float AS vpd_hpa,
                    (features->>'humidity_deficit_g_m3')::float AS humidity_deficit_g_m3
                FROM {}
                WHERE time IS NOT NULL AND features IS NOT NULL
                ORDER BY time"#,
                quoted_table_name
            )
        } else {
            format!("SELECT * FROM {}", quoted_table_name)
        };
        
        let final_query = match row_limit {
            Some(limit) => format!("{} LIMIT {}", base_query, limit),
            None => base_query,
        };
        let copy_sql = format!("COPY ({}) TO STDOUT WITH CSV HEADER", final_query);

        let mut raw_csv: Vec<u8> = Vec::new();
        conn.copy_out(copy_sql.as_str())
            .with_context(|| format!("Failed to execute COPY OUT for table '{}' (sanitized: '{}'). SQL: {}", table, quoted_table_name, copy_sql))?
            .read_to_end(&mut raw_csv)
            .with_context(|| format!("Failed to read CSV data from COPY OUT for table '{}' (sanitized: '{}')", table, quoted_table_name))?;

        let mut df = if table == "preprocessed_features" {
            // For preprocessed_features, infer schema from the entire file to handle mixed types
            let opts = CsvReadOptions::default()
                .with_has_header(true)
                .with_infer_schema_length(None); // Infer from entire file
            
            CsvReader::new(Cursor::new(raw_csv))
                .with_options(opts)
                .finish()
                .with_context(|| format!("Polars CsvReader failed to parse data for table '{}'", table))?
        } else {
            // For other tables, use schema inference
            let opts = CsvReadOptions::default()
                .with_has_header(true)
                .with_infer_schema_length(Some(10000));
            
            CsvReader::new(Cursor::new(raw_csv))
                .with_options(opts)
                .finish()
                .with_context(|| format!("Polars CsvReader failed to parse data for table '{}' (sanitized: '{}')", table, quoted_table_name))?
        };

        // Check if 'time' column exists and is not datetime type - if so, parse it
        if let Ok(time_col) = df.column("time") {
            match time_col.dtype() {
                DataType::String => {
                    log::info!("Time column detected as String type, converting to DateTime");
                    // For Polars 0.48, use simplified strptime
                    df = df.lazy()
                        .with_column(
                            col("time").str().to_datetime(
                                Some(TimeUnit::Microseconds),
                                None,  // format - None for auto-detection
                                StrptimeOptions::default(),
                                lit("raise"),  // ambiguous handling
                            ).alias("time")
                        )
                        .collect()
                        .with_context(|| "Failed to parse time column from string to datetime")?;
                }
                DataType::Datetime(_, _) => {
                    log::debug!("Time column already in DateTime format");
                }
                dt => {
                    log::warn!("Time column has unexpected data type: {:?}", dt);
                }
            }
        }

        Ok(df)
    }

    /// Builds a CSV byte vector from DataFrame segments.
    /// `df` must have columns: time, era_id.
    pub fn build_segments_csv(
        df: &DataFrame,
        signal: &str,
        level: char,
        stage: &str,
    ) -> Result<Vec<u8>> {
        use std::io::Write;

        let mut csv_data = Vec::<u8>::with_capacity(df.height() * 64);

        let era = df.column("era_id")?.i32()?;
        let time = df.column("time")?.datetime()?;

        let mut cur = match era.get(0) {
            Some(val) => val,
            None => {
                // Handle empty DataFrame case: if no rows, no CSV data to build.
                if df.height() == 0 {
                    log::info!("DataFrame is empty for signal '{}', level '{}', stage '{}'. No CSV data generated.", signal, level, stage);
                    return Ok(Vec::new()); // Return empty CSV data
                }
                // If not empty but first era_id is null, this is an actual error.
                return Err(anyhow::anyhow!("Era series contains NULL at index 0 for signal '{}', level '{}', stage '{}'", signal, level, stage));
            }
        };
        let mut seg_start = match time.get(0) {
            Some(val) => val,
            None => {
                 // This case should be caught by the era.get(0) check if df.height() > 0
                return Err(anyhow::anyhow!("Time series contains NULL at index 0 despite era_id being present for signal '{}', level '{}', stage '{}'", signal, level, stage));
            }
        };
        let mut seg_len: i32 = 0;
        let mut prev_ts = seg_start; // Initialize prev_ts

        for idx in 0..df.height() {
            let era_id = era.get(idx).context(format!("NULL detected in `era_id` column at index {}", idx))?;
            let ts = time.get(idx).context(format!("NULL detected in `time` column at index {}", idx))?;

            if era_id != cur {
                // flush previous segment only if seg_len > 0 (i.e., it's not the very first iteration after an empty df check)
                if seg_len > 0 {
                    writeln!(
                        csv_data,
                        "{},{},{},{},{},{},{}",
                        signal, level, stage, cur, seg_start, prev_ts, seg_len // Use prev_ts
                    )?;
                }
                cur = era_id;
                seg_start = ts;
                seg_len = 0; // Reset for the new segment
            } // Removed else block, seg_len is incremented below
            seg_len += 1;
            prev_ts = ts;
        }
        // last open segment
        if df.height() > 0 { // Ensure there was at least one row to form a segment
            writeln!(
                csv_data,
                "{},{},{},{},{},{},{}",
                signal, level, stage, cur, seg_start, prev_ts, seg_len // Use prev_ts for the end time of the last segment
            )?;
        }

        Ok(csv_data)
    }

    /// Bulk-COPY pre-built CSV data for segments into the database.
    /// `stage` = "PELT" | "BOCPD" | "HMM"
    #[allow(dead_code)]
    pub fn copy_segments_from_csv(
        &self, // Changed from &mut self
        csv_data: &[u8],
    ) -> Result<()> {
        let stmt = "COPY era_labels \
                    (signal_name,level,stage,era_id,start_time,end_time,rows) \
                    FROM STDIN CSV";
        let data_len = csv_data.len();
        let mut conn = self.pool.get().with_context(|| "Failed to get connection from pool for copy_segments_from_csv")?;
        let mut writer = conn.copy_in(stmt)
            .with_context(|| format!("Failed to start COPY for era_segments (data size: {} bytes)", data_len))?;
        writer.write_all(&csv_data[..])
            .with_context(|| format!("Failed to write data for COPY era_segments (data size: {} bytes)", data_len))?;
        writer.finish()
            .with_context(|| format!("COPY era_segments failed during finish (data size: {} bytes)", data_len))?;
        Ok(())
    }
}
