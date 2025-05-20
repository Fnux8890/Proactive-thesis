// src/db.rs
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use polars::prelude::*;
use r2d2_postgres::{PostgresConnectionManager, r2d2};
use postgres::NoTls;
use std::env;

// Type alias for the R2D2 connection pool
pub type DbPool = r2d2::Pool<PostgresConnectionManager<NoTls>>;

// /// 🔒  DSN was hard-coded. Now read from environment.
// const DB_DSN: &str =
//     "host=127.0.0.1 port=5432 dbname=mydb user=myuser password=supersecret";

/// Database access structure, now holding a connection pool.
pub struct EraDb {
    pool: DbPool,
}

impl EraDb {
    pub fn new() -> Result<Self> {
        let db_dsn = env::var("DB_DSN")
            .with_context(|| "DB_DSN environment variable not set")?;
        
        let manager = PostgresConnectionManager::new(
            db_dsn.parse().with_context(|| format!("Failed to parse DSN: {}", db_dsn))?,
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
                PRIMARY KEY (signal_name, level, stage, era_id)
            );
            SELECT create_hypertable('era_labels','start_time',
                                     if_not_exists => TRUE);
            "#,
        ).with_context(|| "Failed to create era_labels table or hypertable")?;

        Ok(Self { pool })
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

        let mut cur = era.get(0).context("Era series is empty or contains NULL at index 0")?;
        let mut seg_start = time.get(0).context("Time series is empty or contains NULL at index 0")?;
        let mut seg_len: i32 = 0;

        for idx in 0..df.height() {
            let era_id = era.get(idx).context(format!("NULL detected in `era_id` column at index {}", idx))?;
            let ts = time.get(idx).context(format!("NULL detected in `time` column at index {}", idx))?;
            if era_id != cur {
                // flush previous segment
                writeln!(
                    csv_data,
                    "{},{},{},{},{},{},{}",
                    signal, level, stage, cur, seg_start, ts, seg_len
                )?;
                cur = era_id;
                seg_start = ts;
                seg_len = 1;
            } else {
                seg_len += 1;
            }
        }
        // last open segment → end_time NULL
        writeln!(
            csv_data,
            "{},{},{},{},{},{},{}",
            signal, level, stage, cur, seg_start, "", seg_len
        )?;

        Ok(csv_data)
    }

    /// Bulk-COPY pre-built CSV data for segments into the database.
    /// `stage` = "PELT" | "BOCPD" | "HMM"
    pub fn copy_segments_from_csv(
        &self, // Changed from &mut self
        csv_data: &[u8],
    ) -> Result<()> {
        let stmt = "COPY era_labels \
                    (signal_name,level,stage,era_id,start_time,end_time,rows) \
                    FROM STDIN CSV";
        let mut conn = self.pool.get().with_context(|| "Failed to get connection from pool for copy_segments_from_csv")?;
        conn.copy_in(stmt, &mut &csv_data[..])
            .with_context(|| "COPY era_segments failed")?;
        Ok(())
    }
}
