use sqlx::postgres::PgPoolOptions;
use sqlx::FromRow;
use dotenvy::dotenv;
use std::env;
use chrono::{DateTime, Utc};

// Define a struct to map the row data for sample rows
#[derive(Debug, FromRow)]
struct SensorDataSample {
    time: DateTime<Utc>,
    air_temp_c: Option<f64>,
    relative_humidity_percent: Option<f64>,
    // Add other columns you want to see in the sample
}

#[tokio::main]
async fn main() -> Result<(), sqlx::Error> {
    // Load environment variables from .env file in the current directory (db_checker)
    dotenv().expect("Failed to read .env file in db_checker directory");

    let database_url = env::var("DATABASE_URL")
        .expect("DATABASE_URL must be set in .env file");

    println!("Connecting to database at {}...", database_url);

    // Create a connection pool
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&database_url)
        .await?;

    println!("Connected! Querying data range and sample from sensor_data table...");

    // --- Query 1: Get Min/Max Timestamps ---
    let time_range: Option<(DateTime<Utc>, DateTime<Utc>)> = sqlx::query_as("
        SELECT MIN(time), MAX(time) FROM sensor_data
    ")
    .fetch_optional(&pool)
    .await?;

    match time_range {
        Some((min_time, max_time)) => {
            println!("-----------------------------------------------------");
            println!("Data Time Range in sensor_data:");
            println!("  Min Time: {}", min_time);
            println!("  Max Time: {}", max_time);
            println!("-----------------------------------------------------");
        }
        None => {
            println!("-----------------------------------------------------");
            println!("Could not determine time range. Is the table empty?");
            println!("-----------------------------------------------------");
        }
    }

    // --- Query 2: Get Row Count ---
    let row_count_result: Result<(i64,), sqlx::Error> = sqlx::query_as("SELECT COUNT(*) FROM sensor_data")
        .fetch_one(&pool)
        .await;

    match row_count_result {
        Ok(row_count) => {
            println!("Total rows in sensor_data: {}", row_count.0);
        }
        Err(e) => {
            println!("Error fetching row count: {}", e);
        }
    }
     println!("-----------------------------------------------------");


    // --- Query 3: Get Sample Rows ---
    let sample_rows: Vec<SensorDataSample> = sqlx::query_as("
        SELECT time, air_temp_c, relative_humidity_percent
        FROM sensor_data
        ORDER BY time ASC
        LIMIT 5
    ")
    .fetch_all(&pool)
    .await?;

    println!("Sample Rows (first 5 by time):");
    if sample_rows.is_empty() {
        println!("  No rows found.");
    } else {
        for row in sample_rows {
            println!("  {:?}", row);
        }
    }
    println!("-----------------------------------------------------");

    pool.close().await;
    println!("Connection closed.");

    Ok(())
} 