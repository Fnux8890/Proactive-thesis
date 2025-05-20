// src/db.rs
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use postgres::{Client, NoTls};
use polars::prelude::*;

const DB_DSN: &str =
    "host=db port=5432 dbname=postgres user=postgres password=postgres";

/// Connection wrapper – open once per Rayon thread or share via Arc.
pub struct EraDb {
    client: Client,
}

impl EraDb {
    pub fn connect() -> Result<Self> {
        let mut client = Client::connect(DB_DSN, NoTls)
            .with_context(|| "connecting to TimescaleDB failed")?;

        // 1️⃣ ensure table + hypertable exactly once
        client.batch_execute(
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
        )?;

        Ok(Self { client })
    }

    /// Bulk-COPY one DataFrame’s segments (`df` must have columns: time, era_id).
    /// `stage` = "PELT" | "BOCPD" | "HMM"
    pub fn copy_segments(
        &mut self,
        df: &DataFrame,
        signal: &str,
        level: char,
        stage: &str,
    ) -> Result<()> {
        use std::io::Write;

        let mut csv = Vec::<u8>::with_capacity(df.height() * 64);

        let era = df.column("era_id")?.i32()?;
        let time = df.column("time")?.datetime()?;

        let mut cur = era.get(0).unwrap();
        let mut seg_start = time.get(0).unwrap();
        let mut seg_len: i32 = 0;

        for idx in 0..df.height() {
            let era_id = era.get(idx).unwrap();
            let ts = time.get(idx).unwrap();
            if era_id != cur {
                // flush previous segment
                writeln!(
                    csv,
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
            csv,
            "{},{},{},{},{},{},{}",
            signal, level, stage, cur, seg_start, "", seg_len
        )?;

        let stmt = "COPY era_labels \
                    (signal_name,level,stage,era_id,start_time,end_time,rows) \
                    FROM STDIN CSV";
        self.client
            .copy_in(stmt, &mut &csv[..])
            .with_context(|| "COPY era_segments failed")?;
        Ok(())
    }
}
