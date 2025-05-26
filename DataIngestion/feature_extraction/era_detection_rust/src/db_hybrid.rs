// src/db_hybrid.rs - Updated to support both JSONB and hybrid table structures
use anyhow::{Context, Result};
use std::io::{Write, Cursor, Read};

use polars::prelude::*;
use r2d2_postgres::{PostgresConnectionManager, r2d2};
use postgres::NoTls;
use regex::Regex;
use lazy_static::lazy_static;
use std::env;

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
                PRIMARY KEY (signal_name, level, stage, era_id, start_time)
            );
            SELECT create_hypertable('era_labels','start_time',
                                     if_not_exists => TRUE);
            "#,
        ).with_context(|| "Failed to create era_labels table or hypertable")?;

        Ok(Self { pool })
    }

    /// Check if a table uses the hybrid schema
    fn is_hybrid_table(&self, table_name: &str) -> Result<bool> {
        let mut conn = self.pool.get()?;
        let query = r#"
            SELECT EXISTS (
                SELECT 1 
                FROM information_schema.columns 
                WHERE table_name = $1 
                AND column_name = 'air_temp_c'
                AND table_schema = 'public'
            )
        "#;
        
        let row = conn.query_one(query, &[&table_name])?;
        Ok(row.get::<_, bool>(0))
    }

    pub fn load_feature_df(&self, table: &str, row_limit: Option<usize>) -> Result<DataFrame> {
        let mut conn = self.pool.get().with_context(|| format!("Failed to get connection from pool for loading table: {}", table))?;
        let quoted_table_name = quote_identifier(table)
            .with_context(|| format!("Failed to sanitize table name '{}' for load_feature_df", table))?;

        // Check if this is a preprocessed_features table (JSONB or hybrid)
        let base_query = if table.starts_with("preprocessed_features") {
            // Check if it's the hybrid version
            let is_hybrid = self.is_hybrid_table(table)?;
            
            if is_hybrid {
                log::info!("Loading {} table with hybrid schema", table);
                
                // Direct column access for hybrid table
                format!(
                    r#"SELECT 
                        time,
                        air_temp_c,
                        relative_humidity_percent,
                        co2_measured_ppm,
                        radiation_w_m2,
                        light_intensity_umol,
                        heating_setpoint_c,
                        total_lamps_on,
                        dli_sum,
                        vpd_hpa,
                        -- Extract additional features from extended_features if needed
                        (extended_features->>'outside_temp_c')::float AS outside_temp_c,
                        (extended_features->>'outside_light_w_m2')::float AS outside_light_w_m2
                    FROM {}
                    WHERE time IS NOT NULL
                    ORDER BY time"#,
                    quoted_table_name
                )
            } else {
                log::info!("Loading {} table with JSONB schema", table);
                
                // JSONB extraction for original table
                format!(
                    r#"SELECT 
                        time,
                        (features->>'air_temp_c')::float AS air_temp_c,
                        (features->>'relative_humidity_percent')::float AS relative_humidity_percent,
                        (features->>'co2_measured_ppm')::float AS co2_measured_ppm,
                        (features->>'radiation_w_m2')::float AS radiation_w_m2,
                        (features->>'light_intensity_umol')::float AS light_intensity_umol,
                        (features->>'heating_setpoint_c')::float AS heating_setpoint_c,
                        (features->>'total_lamps_on')::float AS total_lamps_on,
                        (features->>'dli_sum')::float AS dli_sum,
                        (features->>'vpd_hpa')::float AS vpd_hpa
                    FROM {}
                    WHERE time IS NOT NULL AND features IS NOT NULL
                    ORDER BY time"#,
                    quoted_table_name
                )
            }
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
            .with_context(|| format!("Failed to execute COPY OUT for table '{}'. SQL: {}", table, copy_sql))?
            .read_to_end(&mut raw_csv)
            .with_context(|| format!("Failed to read CSV data from COPY OUT for table '{}'", table))?;

        let df = if table.starts_with("preprocessed_features") {
            // For preprocessed_features tables, parse with full schema inference
            let opts = CsvReadOptions::default()
                .with_has_header(true)
                .with_infer_schema_length(None);
            
            let mut df = CsvReader::new(Cursor::new(raw_csv))
                .with_options(opts)
                .finish()
                .with_context(|| format!("Polars CsvReader failed to parse data for table '{}'", table))?;
                
            // Parse time column if it's a string (from PostgreSQL CSV export)
            if df.column("time").is_ok() {
                let time_dtype = df.column("time")?.dtype();
                if matches!(time_dtype, DataType::String) {
                    log::info!("Time column is String type, parsing to DateTime");
                    // For Polars 0.48, use simplified to_datetime
                    df = df.lazy()
                        .with_column(
                            col("time").str()
                            // Replace "+00" (commonly at the end of timestamps from some DBs for UTC)
                            // with "+0000" to make it compatible with the %z strptime format specifier.
                            .replace_all(polars::prelude::lit("+00"), polars::prelude::lit("+0000"), true)
                            .str()
                            .strptime(
                                polars::prelude::DataType::Datetime(polars::prelude::TimeUnit::Microseconds, Some(polars::prelude::TimeZone::UTC)),
                                polars::prelude::StrptimeOptions {
                                    format: Some("%Y-%m-%d %H:%M:%S%z".to_string().into()), // Removed %.f
                                    strict: false, // Be somewhat lenient with parsing overall
                                    exact: false,   // Changed to false for more flexibility
                                    cache: true,
                                },
                                polars::prelude::lit("raise")
                            ).alias("time")
                        )
                        .collect()
                        .with_context(|| "Failed to parse time column")?;
                }
            }
            
            df
        } else {
            // For other tables, use schema inference
            let opts = CsvReadOptions::default()
                .with_has_header(true)
                .with_infer_schema_length(Some(10000));
            
            CsvReader::new(Cursor::new(raw_csv))
                .with_options(opts)
                .finish()
                .with_context(|| format!("Polars CsvReader failed to parse data for table '{}'", table))?
        };

        Ok(df)
    }

    // Rest of the methods remain the same...
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
                PRIMARY KEY (signal_name, level, stage, era_id, start_time)
            );
            SELECT create_hypertable('{}', 'start_time', if_not_exists => TRUE, migrate_data => TRUE);
            "#,
            table_name, table_name
        );
        conn.batch_execute(&create_table_sql)
            .with_context(|| format!("Failed to create table or hypertable: {}", table_name_raw))?;
        Ok(())
    }

    pub fn save_era_labels(
        &self,
        df_source: &DataFrame,
        original_era_col_name: &str,
        signal_name: &str,
        level_char: char,
        stage_str: &str,
        target_table_name_raw: &str,
    ) -> Result<()> {
        let target_table_name = quote_identifier(target_table_name_raw)?;

        self.ensure_era_table_exists(target_table_name_raw)
            .with_context(|| format!("Failed to ensure target table '{}' exists.", target_table_name_raw))?;

        let mut df_to_persist = df_source.clone();
        df_to_persist
            .rename(original_era_col_name, "era_id".into())
            .with_context(|| {
                format!(
                    "Failed to rename column '{}' to 'era_id' for signal '{}', level '{}', stage '{}'",
                    original_era_col_name, signal_name, level_char, stage_str
                )
            })?;

        let csv_data = Self::build_segments_csv(&df_to_persist, signal_name, level_char, stage_str)
            .with_context(|| {
                format!(
                    "Failed to build CSV for signal '{}', level '{}', stage '{}'",
                    signal_name, level_char, stage_str
                )
            })?;

        if csv_data.is_empty() {
            log::debug!(
                "No CSV data to persist for signal '{}', level '{}', stage '{}'. Skipping COPY.",
                signal_name,
                level_char,
                stage_str
            );
            return Ok(());
        }

        let copy_sql = format!(
            "COPY {} (signal_name, level, stage, era_id, start_time, end_time, rows) FROM STDIN CSV",
            target_table_name
        );
        
        let _data_len = csv_data.len();
        let mut conn = self.pool.get().with_context(|| format!("Failed to get connection from pool for saving"))?;
        let mut writer = conn.copy_in(&copy_sql)
            .with_context(|| format!("Failed to start COPY for table '{}'", target_table_name_raw))?;
        writer.write_all(&csv_data[..])
            .with_context(|| format!("Failed to write data for COPY to table '{}'", target_table_name_raw))?;
        writer.finish()
            .with_context(|| format!("COPY to table '{}' failed during finish", target_table_name_raw))?;
        
        Ok(())
    }

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
                if df.height() == 0 {
                    log::info!("DataFrame is empty for signal '{}', level '{}', stage '{}'.", signal, level, stage);
                    return Ok(Vec::new());
                }
                return Err(anyhow::anyhow!("Era series contains NULL at index 0"));
            }
        };
        let mut seg_start = match time.get(0) {
            Some(val) => val,
            None => {
                return Err(anyhow::anyhow!("Time series contains NULL at index 0"));
            }
        };
        let mut seg_len: i32 = 0;
        let mut prev_ts = seg_start;

        for idx in 0..df.height() {
            let era_id = era.get(idx).context(format!("NULL detected in `era_id` column at index {}", idx))?;
            let ts = time.get(idx).context(format!("NULL detected in `time` column at index {}", idx))?;

            if era_id != cur {
                if seg_len > 0 {
                    writeln!(
                        csv_data,
                        "{},{},{},{},{},{},{}",
                        signal, level, stage, cur, seg_start, prev_ts, seg_len
                    )?;
                }
                cur = era_id;
                seg_start = ts;
                seg_len = 0;
            }
            seg_len += 1;
            prev_ts = ts;
        }
        
        if df.height() > 0 {
            writeln!(
                csv_data,
                "{},{},{},{},{},{},{}",
                signal, level, stage, cur, seg_start, prev_ts, seg_len
            )?;
        }

        Ok(csv_data)
    }

    #[allow(dead_code)]
    pub fn copy_segments_from_csv(
        &self,
        csv_data: &[u8],
    ) -> Result<()> {
        let stmt = "COPY era_labels \
                    (signal_name,level,stage,era_id,start_time,end_time,rows) \
                    FROM STDIN CSV";
        let _data_len = csv_data.len();
        let mut conn = self.pool.get().with_context(|| "Failed to get connection from pool")?;
        let mut writer = conn.copy_in(stmt)
            .with_context(|| format!("Failed to start COPY for era_segments"))?;
        writer.write_all(&csv_data[..])
            .with_context(|| format!("Failed to write data for COPY era_segments"))?;
        writer.finish()
            .with_context(|| format!("COPY era_segments failed during finish"))?;
        Ok(())
    }
}