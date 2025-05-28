// src/db_hybrid.rs - Fixed version with proper transaction handling
use anyhow::{Context, Result};
use std::io::{Write, Cursor, Read};
use chrono::{TimeZone, Utc};

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
    // ... [keep all existing methods up to save_era_labels] ...

    /// Save era labels with proper transaction handling to avoid duplicate key violations
    pub fn save_era_labels_transactional(
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

        // Get a connection and start a transaction
        let mut conn = self.pool.get()
            .with_context(|| "Failed to get connection from pool")?;
        
        let transaction = conn.transaction()
            .with_context(|| "Failed to start transaction")?;

        // Delete old data within the transaction
        let delete_sql = format!(
            "DELETE FROM {} WHERE signal_name = $1 AND level = $2 AND stage = $3",
            target_table_name
        );
        let level_str = level_char.to_string();
        
        let rows_deleted = transaction.execute(&delete_sql, &[&signal_name, &level_str, &stage_str])
            .with_context(|| {
                format!(
                    "Failed to delete old era labels for signal '{}', level '{}', stage '{}' from table '{}'",
                    signal_name, level_char, stage_str, target_table_name_raw
                )
            })?;

        if rows_deleted > 0 {
            log::debug!(
                "Deleted {} old era labels for signal '{}', level '{}', stage '{}' from table '{}'",
                rows_deleted, signal_name, level_char, stage_str, target_table_name_raw
            );
        }

        // Copy new data within the same transaction
        let copy_sql = format!(
            "COPY {} (signal_name, level, stage, era_id, start_time, end_time, rows) FROM STDIN CSV",
            target_table_name
        );
        
        let mut writer = transaction.copy_in(&copy_sql)
            .with_context(|| format!("Failed to start COPY for table '{}'", target_table_name_raw))?;
        writer.write_all(&csv_data[..])
            .with_context(|| format!("Failed to write data for COPY to table '{}'", target_table_name_raw))?;
        writer.finish()
            .with_context(|| format!("COPY to table '{}' failed during finish", target_table_name_raw))?;

        // Commit the transaction
        transaction.commit()
            .with_context(|| format!("Failed to commit transaction for table '{}'", target_table_name_raw))?;
        
        Ok(())
    }

    // Keep the original save_era_labels for backward compatibility, but have it call the transactional version
    pub fn save_era_labels(
        &self,
        df_source: &DataFrame,
        original_era_col_name: &str,
        signal_name: &str,
        level_char: char,
        stage_str: &str,
        target_table_name_raw: &str,
    ) -> Result<()> {
        self.save_era_labels_transactional(
            df_source,
            original_era_col_name,
            signal_name,
            level_char,
            stage_str,
            target_table_name_raw,
        )
    }

    // Remove the separate delete_era_labels_for_signal calls from main.rs
    // since deletion is now part of the transaction
}