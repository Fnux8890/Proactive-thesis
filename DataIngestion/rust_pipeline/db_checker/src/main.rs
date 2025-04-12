use sqlx::postgres::PgPoolOptions;
use dotenvy::dotenv;
use std::env;

#[tokio::main]
async fn main() -> Result<(), sqlx::Error> {
    // Load environment variables from .env file
    dotenv().expect("Failed to read .env file");

    let database_url = env::var("DATABASE_URL")
        .expect("DATABASE_URL must be set in .env file");

    println!("Connecting to database...");

    // Create a connection pool
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&database_url)
        .await?;

    println!("Connected! Querying row count from sensor_data table...");

    // Execute the count query
    let row_count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM sensor_data")
        .fetch_one(&pool)
        .await?;

    println!("Total rows in sensor_data: {}", row_count.0);

    Ok(())
} 