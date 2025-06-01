use anyhow::Result;
use chrono::{DateTime, Utc};
use polars::prelude::*;
use reqwest;
use serde::{Deserialize, Serialize};
use sqlx::{postgres::PgPool, Row};
use std::collections::HashMap;
use tracing::{info, warn, debug};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeatherData {
    pub timestamp: DateTime<Utc>,
    pub temperature_2m: f32,
    pub relative_humidity_2m: f32,
    pub precipitation: f32,
    pub shortwave_radiation: f32,
    pub wind_speed_10m: f32,
    pub cloud_cover: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyPriceData {
    pub timestamp: DateTime<Utc>,
    pub price_area: String,
    pub spot_price_dkk: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhenotypeData {
    pub species: String,
    pub cultivar: String,
    pub base_temperature_c: Option<f64>,
    pub optimal_temperature_c: Option<f64>,
    pub max_temperature_c: Option<f64>,
    pub photoperiod_critical_h: Option<f64>,
    pub dli_optimal_mol_m2_d: Option<f64>,
    pub vpd_optimal_kpa: Option<f64>,
}

pub struct ExternalDataFetcher {
    pool: PgPool,
    http_client: reqwest::Client,
}

impl ExternalDataFetcher {
    pub fn new(pool: PgPool) -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .unwrap();
            
        Self {
            pool,
            http_client,
        }
    }
    
    /// Fetch weather data from database or API
    pub async fn fetch_weather_data(
        &self,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
        lat: f64,
        lon: f64,
    ) -> Result<DataFrame> {
        info!("Fetching weather data for lat={}, lon={} from {} to {}", lat, lon, start_time, end_time);
        
        // First try to fetch from database
        match self.fetch_weather_from_db(start_time, end_time).await {
            Ok(df) if df.height() > 0 => {
                info!("Found {} hours of weather data in database", df.height());
                return Ok(df);
            }
            Ok(_) => {
                info!("No weather data found in database, fetching from Open-Meteo API");
            }
            Err(e) => {
                warn!("Error fetching weather data from database: {}", e);
            }
        }
        
        // If not in database, fetch from Open-Meteo API
        self.fetch_weather_from_api(start_time, end_time, lat, lon).await
    }
    
    /// Fetch weather data from database
    async fn fetch_weather_from_db(
        &self,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Result<DataFrame> {
        let query = r#"
        SELECT 
            time as hour,
            temperature_2m,
            relative_humidity_2m,
            precipitation,
            shortwave_radiation,
            wind_speed_10m,
            cloud_cover
        FROM external_weather_aarhus
        WHERE time >= $1 AND time < $2
        ORDER BY time
        "#;
        
        let rows = sqlx::query(query)
            .bind(start_time)
            .bind(end_time)
            .fetch_all(&self.pool)
            .await?;
        
        let mut timestamps = Vec::new();
        let mut temp_data = Vec::new();
        let mut humidity_data = Vec::new();
        let mut precip_data = Vec::new();
        let mut radiation_data = Vec::new();
        let mut wind_data = Vec::new();
        let mut cloud_data = Vec::new();
        
        for row in rows {
            timestamps.push(row.get::<DateTime<Utc>, _>("hour").timestamp());
            temp_data.push(row.get::<Option<f32>, _>("temperature_2m"));
            humidity_data.push(row.get::<Option<f32>, _>("relative_humidity_2m"));
            precip_data.push(row.get::<Option<f32>, _>("precipitation"));
            radiation_data.push(row.get::<Option<f32>, _>("shortwave_radiation"));
            wind_data.push(row.get::<Option<f32>, _>("wind_speed_10m"));
            cloud_data.push(row.get::<Option<f32>, _>("cloud_cover"));
        }
        
        let df = df![
            "hour" => timestamps,
            "temperature_2m" => temp_data,
            "relative_humidity_2m" => humidity_data,
            "precipitation" => precip_data,
            "shortwave_radiation" => radiation_data,
            "wind_speed_10m" => wind_data,
            "cloud_cover" => cloud_data,
        ]?;
        
        Ok(df)
    }
    
    /// Fetch weather data from Open-Meteo API
    async fn fetch_weather_from_api(
        &self,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
        lat: f64,
        lon: f64,
    ) -> Result<DataFrame> {
        let start_date = start_time.format("%Y-%m-%d").to_string();
        let end_date = end_time.format("%Y-%m-%d").to_string();
        
        let params = [
            ("latitude", lat.to_string()),
            ("longitude", lon.to_string()),
            ("start_date", start_date),
            ("end_date", end_date),
            ("hourly", "temperature_2m,relative_humidity_2m,precipitation,shortwave_radiation,wind_speed_10m,cloud_cover".to_string()),
            ("timezone", "GMT".to_string()),
        ];
        
        let response = self.http_client
            .get("https://archive-api.open-meteo.com/v1/archive")
            .query(&params)
            .send()
            .await?;
            
        let data: serde_json::Value = response.json().await?;
        
        // Parse the JSON response into a DataFrame
        if let Some(hourly) = data.get("hourly") {
            let time_array = hourly.get("time")
                .and_then(|v| v.as_array())
                .ok_or_else(|| anyhow::anyhow!("Missing time array in weather response"))?;
                
            let mut timestamps = Vec::new();
            let mut temp_data = Vec::new();
            let mut humidity_data = Vec::new();
            let mut precip_data = Vec::new();
            let mut radiation_data = Vec::new();
            let mut wind_data = Vec::new();
            let mut cloud_data = Vec::new();
            
            for (i, time_val) in time_array.iter().enumerate() {
                if let Some(time_str) = time_val.as_str() {
                    if let Ok(dt) = DateTime::parse_from_rfc3339(time_str) {
                        timestamps.push(dt.with_timezone(&Utc).timestamp());
                        
                        // Extract values for each parameter
                        temp_data.push(self.extract_float_value(hourly, "temperature_2m", i));
                        humidity_data.push(self.extract_float_value(hourly, "relative_humidity_2m", i));
                        precip_data.push(self.extract_float_value(hourly, "precipitation", i));
                        radiation_data.push(self.extract_float_value(hourly, "shortwave_radiation", i));
                        wind_data.push(self.extract_float_value(hourly, "wind_speed_10m", i));
                        cloud_data.push(self.extract_float_value(hourly, "cloud_cover", i));
                    }
                }
            }
            
            let df = df![
                "hour" => timestamps,
                "temperature_2m" => temp_data,
                "relative_humidity_2m" => humidity_data,
                "precipitation" => precip_data,
                "shortwave_radiation" => radiation_data,
                "wind_speed_10m" => wind_data,
                "cloud_cover" => cloud_data,
            ]?;
            
            // Save to database for future use
            if df.height() > 0 {
                self.save_weather_to_db(&df).await.ok();
            }
            
            Ok(df)
        } else {
            Err(anyhow::anyhow!("No hourly data in weather API response"))
        }
    }
    
    fn extract_float_value(&self, hourly: &serde_json::Value, key: &str, index: usize) -> Option<f32> {
        hourly.get(key)
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.get(index))
            .and_then(|v| v.as_f64())
            .map(|f| f as f32)
    }
    
    /// Save weather data to database
    async fn save_weather_to_db(&self, df: &DataFrame) -> Result<()> {
        // Implementation would save the data to external_weather_aarhus table
        // This is a placeholder - actual implementation would use batch inserts
        info!("Saving {} weather records to database", df.height());
        Ok(())
    }
    
    /// Fetch energy price data
    pub async fn fetch_energy_prices(
        &self,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
        price_area: &str,
    ) -> Result<DataFrame> {
        info!("Fetching energy prices for area {} from {} to {}", price_area, start_time, end_time);
        
        let query = r#"
        SELECT 
            "HourUTC" as hour,
            "SpotPriceDKK"
        FROM external_energy_prices_dk
        WHERE "HourUTC" >= $1 AND "HourUTC" < $2 AND "PriceArea" = $3
        ORDER BY "HourUTC"
        "#;
        
        let rows = sqlx::query(query)
            .bind(start_time)
            .bind(end_time)
            .bind(price_area)
            .fetch_all(&self.pool)
            .await?;
        
        let mut timestamps = Vec::new();
        let mut prices = Vec::new();
        
        for row in rows {
            timestamps.push(row.get::<DateTime<Utc>, _>("hour").timestamp());
            prices.push(row.get::<Option<f32>, _>("SpotPriceDKK"));
        }
        
        let df = df![
            "hour" => timestamps,
            "SpotPriceDKK" => prices,
        ]?;
        
        Ok(df)
    }
    
    /// Load phenotype data for a species
    pub async fn load_phenotype_data(&self, species: &str) -> Result<PhenotypeData> {
        info!("Loading phenotype data for species: {}", species);
        
        // For now, return hardcoded data for Kalanchoe blossfeldiana
        // In a real implementation, this would query from a phenotype database
        if species == "Kalanchoe blossfeldiana" {
            Ok(PhenotypeData {
                species: species.to_string(),
                cultivar: "Molly".to_string(),
                base_temperature_c: Some(10.0),
                optimal_temperature_c: Some(22.0),
                max_temperature_c: Some(30.0),
                photoperiod_critical_h: Some(12.0),
                dli_optimal_mol_m2_d: Some(12.0),
                vpd_optimal_kpa: Some(0.8),
            })
        } else {
            Err(anyhow::anyhow!("Phenotype data not available for species: {}", species))
        }
    }
}