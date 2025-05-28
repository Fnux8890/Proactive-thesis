// src/db.rs
use deadpool_postgres::{Config, Pool, Runtime};
use tokio_postgres::NoTls;
use crate::errors::PipelineError; // Assuming DB errors will be wrapped in PipelineError
use std::env;

pub type DbPool = Pool;

/// Creates and returns a Deadpool PostgreSQL connection pool.
///
/// Reads the database connection URL from the `DATABASE_URL` environment variable.
/// Panics if the environment variable is not set or if the pool cannot be created.
pub fn create_pool() -> Result<DbPool, PipelineError> {
    let database_url = env::var("DATABASE_URL")
        .map_err(|_| PipelineError::Config("DATABASE_URL environment variable not set".to_string()))?;

    let mut cfg = Config::new();
    cfg.url = Some(database_url);
    // Add other configurations like pool size if needed
    // cfg.pool = Some(PoolConfig::new(10)); // Example: max_size = 10

    let pool = cfg.create_pool(Some(Runtime::Tokio1), NoTls)
        .map_err(|e| PipelineError::DbPoolError(format!("Failed to create database pool: {}", e)))?;

    println!("Successfully created database connection pool.");
    Ok(pool)
} 