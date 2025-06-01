use anyhow::Result;
use chrono::{DateTime, Utc, Duration};
use polars::prelude::*;
use sqlx::{postgres::PgPool, Row};
use std::collections::HashMap;
use std::path::PathBuf;
use tracing::{info, warn, debug};
use serde::{Serialize, Deserialize};

use crate::db::{create_features_table_if_not_exists, write_features, FeatureSet, EraData, Era as DbEra};
use crate::data_quality::{DataQualityAnalyzer, AdaptiveWindowConfig};

// Enhanced configuration for multi-domain feature extraction
#[derive(Debug, Clone)]
pub struct EnhancedPipelineConfig {
    pub min_hourly_coverage: f32,
    pub max_interpolation_gap: i64,
    pub enable_parquet_checkpoints: bool,
    pub checkpoint_dir: PathBuf,
    pub window_hours: usize,
    pub slide_hours: usize,
    
    // External data integration
    pub enable_weather_features: bool,
    pub enable_energy_features: bool,
    pub enable_growth_features: bool,
    
    // Multi-resolution processing
    pub enable_multiresolution: bool,
    pub resolution_windows: Vec<Duration>,
    
    // Extended feature categories
    pub enable_extended_statistics: bool,
    pub enable_coupling_features: bool,
    pub enable_temporal_features: bool,
}

impl Default for EnhancedPipelineConfig {
    fn default() -> Self {
        Self {
            min_hourly_coverage: 0.1,
            max_interpolation_gap: 2,
            enable_parquet_checkpoints: true,
            checkpoint_dir: PathBuf::from("/tmp/gpu_sparse_pipeline"),
            window_hours: 24,
            slide_hours: 6,
            
            enable_weather_features: true,
            enable_energy_features: true,
            enable_growth_features: true,
            
            enable_multiresolution: true,
            resolution_windows: vec![
                Duration::minutes(15),
                Duration::hours(1),
                Duration::hours(4),
                Duration::hours(12),
                Duration::days(1),
            ],
            
            enable_extended_statistics: true,
            enable_coupling_features: true,
            enable_temporal_features: true,
        }
    }
}

// External data structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeatherData {
    pub timestamp: DateTime<Utc>,
    pub temperature_2m: Option<f32>,
    pub relative_humidity_2m: Option<f32>,
    pub precipitation: Option<f32>,
    pub shortwave_radiation: Option<f32>,
    pub wind_speed_10m: Option<f32>,
    pub pressure_msl: Option<f32>,
    pub cloud_cover: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyPriceData {
    pub timestamp: DateTime<Utc>,
    pub price_area: String,
    pub spot_price_dkk: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhenotypeData {
    pub species: String,
    pub cultivar: String,
    pub base_temperature: f32,
    pub optimal_temp_day: f32,
    pub optimal_temp_night: f32,
    pub light_requirement_dli: f32,
    pub photoperiod_requirement: f32,
    pub flowering_photoperiod: f32,
}

// Enhanced feature set structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedFeatureSet {
    pub era_id: i64,
    pub computed_at: DateTime<Utc>,
    pub resolution: String,
    
    // Core sensor features
    pub sensor_features: HashMap<String, f64>,
    
    // Extended statistical features
    pub extended_stats: HashMap<String, f64>,
    
    // Weather coupling features
    pub weather_features: HashMap<String, f64>,
    
    // Energy optimization features
    pub energy_features: HashMap<String, f64>,
    
    // Plant growth features
    pub growth_features: HashMap<String, f64>,
    
    // Temporal pattern features
    pub temporal_features: HashMap<String, f64>,
    
    // Multi-objective optimization metrics
    pub optimization_metrics: HashMap<String, f64>,
}

// Results structure
#[derive(Debug)]
pub struct EnhancedPipelineResults {
    pub filled_hourly_data: DataFrame,
    pub multiresolution_features: HashMap<String, Vec<EnhancedFeatureSet>>,
    pub monthly_eras: Vec<Era>,
    pub data_quality_metrics: HashMap<String, f64>,
    pub feature_importance_scores: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Era {
    pub era_id: i64,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub feature_count: usize,
    pub avg_temperature: f64,
    pub avg_photoperiod: f64,
    pub energy_cost: f64,
    pub growth_score: f64,
}

pub struct EnhancedSparsePipeline {
    pool: PgPool,
    config: EnhancedPipelineConfig,
    quality_analyzer: DataQualityAnalyzer,
    phenotype_data: Option<PhenotypeData>,
}

impl EnhancedSparsePipeline {
    pub fn new(pool: PgPool, config: EnhancedPipelineConfig) -> Self {
        // Create checkpoint directory if needed
        if config.enable_parquet_checkpoints {
            std::fs::create_dir_all(&config.checkpoint_dir).ok();
        }
        
        let quality_analyzer = DataQualityAnalyzer::new(
            config.min_hourly_coverage as f64,
            0.5,
        );
        
        Self { 
            pool, 
            config,
            quality_analyzer,
            phenotype_data: None,
        }
    }
    
    pub async fn load_phenotype_data(&mut self) -> Result<()> {
        // Load Kalanchoe phenotype data from JSON
        let phenotype_path = self.config.checkpoint_dir.parent()
            .unwrap_or(&self.config.checkpoint_dir)
            .join("../pre_process/phenotype.json");
            
        if phenotype_path.exists() {
            let content = std::fs::read_to_string(&phenotype_path)?;
            let phenotypes: serde_json::Value = serde_json::from_str(&content)?;
            
            // Extract Kalanchoe blossfeldiana data
            if let Some(phenotype_array) = phenotypes["phenotype"].as_array() {
                for entry in phenotype_array {
                    if let Some(species) = entry["species"].as_str() {
                        if species == "Kalanchoe blossfeldiana" {
                            self.phenotype_data = Some(PhenotypeData {
                                species: species.to_string(),
                                cultivar: entry["cultivar_or_line"].as_str()
                                    .unwrap_or("Unknown").to_string(),
                                base_temperature: 10.0, // Typical for Kalanchoe
                                optimal_temp_day: entry["environment_temp_day_C"]
                                    .as_f64().unwrap_or(22.0) as f32,
                                optimal_temp_night: entry["environment_temp_night_C"]
                                    .as_f64().unwrap_or(18.0) as f32,
                                light_requirement_dli: 12.0, // mol/m²/day for Kalanchoe
                                photoperiod_requirement: entry["environment_photoperiod_h"]
                                    .as_f64().unwrap_or(8.0) as f32,
                                flowering_photoperiod: 8.0, // Short day plant
                            });
                            break;
                        }
                    }
                }
            }
            
            if self.phenotype_data.is_some() {
                info!("Loaded Kalanchoe phenotype data successfully");
            } else {
                warn!("No Kalanchoe phenotype data found in JSON");
            }
        } else {
            warn!("Phenotype data file not found: {:?}", phenotype_path);
        }
        
        Ok(())
    }
    
    /// Load external weather data from database
    pub async fn load_weather_data(
        &self,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Result<Vec<WeatherData>> {
        let query = r#"
        SELECT 
            time as timestamp,
            temperature_2m,
            relative_humidity_2m,
            precipitation,
            shortwave_radiation,
            wind_speed_10m,
            pressure_msl,
            cloud_cover
        FROM external_weather_aarhus 
        WHERE time BETWEEN $1 AND $2
        ORDER BY time
        "#;
        
        let rows = sqlx::query(query)
            .bind(start_time)
            .bind(end_time)
            .fetch_all(&self.pool)
            .await?;
        
        let mut weather_data = Vec::new();
        for row in rows {
            weather_data.push(WeatherData {
                timestamp: row.get("timestamp"),
                temperature_2m: row.get("temperature_2m"),
                relative_humidity_2m: row.get("relative_humidity_2m"),
                precipitation: row.get("precipitation"),
                shortwave_radiation: row.get("shortwave_radiation"),
                wind_speed_10m: row.get("wind_speed_10m"),
                pressure_msl: row.get("pressure_msl"),
                cloud_cover: row.get("cloud_cover"),
            });
        }
        
        info!("Loaded {} weather data points", weather_data.len());
        Ok(weather_data)
    }
    
    /// Load energy price data from database
    pub async fn load_energy_data(
        &self,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Result<Vec<EnergyPriceData>> {
        let query = r#"
        SELECT 
            "HourUTC" as timestamp,
            "PriceArea" as price_area,
            "SpotPriceDKK" as spot_price_dkk
        FROM external_energy_prices_dk 
        WHERE "HourUTC" BETWEEN $1 AND $2
        AND "PriceArea" IN ('DK1', 'DK2')
        ORDER BY "HourUTC"
        "#;
        
        let rows = sqlx::query(query)
            .bind(start_time)
            .bind(end_time)
            .fetch_all(&self.pool)
            .await?;
        
        let mut energy_data = Vec::new();
        for row in rows {
            energy_data.push(EnergyPriceData {
                timestamp: row.get("timestamp"),
                price_area: row.get("price_area"),
                spot_price_dkk: row.get("spot_price_dkk"),
            });
        }
        
        info!("Loaded {} energy price data points", energy_data.len());
        Ok(energy_data)
    }
    
    /// Enhanced stage 1: Aggregate with external data alignment
    pub async fn stage1_enhanced_aggregation(
        &self,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Result<(DataFrame, Vec<WeatherData>, Vec<EnergyPriceData>)> {
        info!("Stage 1: Enhanced aggregation with external data...");
        
        // Load sensor data (existing query)
        let sensor_query = r#"
        SELECT 
            DATE_TRUNC('hour', time) as hour,
            AVG(air_temp_c) as air_temp_c_mean,
            COUNT(air_temp_c) as air_temp_c_count,
            STDDEV(air_temp_c) as air_temp_c_std,
            MIN(air_temp_c) as air_temp_c_min,
            MAX(air_temp_c) as air_temp_c_max,
            
            AVG(co2_measured_ppm) as co2_mean,
            COUNT(co2_measured_ppm) as co2_count,
            
            AVG(relative_humidity_percent) as humidity_mean,
            COUNT(relative_humidity_percent) as humidity_count,
            
            AVG(radiation_w_m2) as radiation_mean,
            COUNT(radiation_w_m2) as radiation_count,
            
            AVG(light_intensity_umol) as light_intensity_mean,
            COUNT(light_intensity_umol) as light_count,
            
            AVG(vpd_hpa) as vpd_mean,
            COUNT(vpd_hpa) as vpd_count,
            
            -- Lamp status aggregation
            AVG(CASE WHEN lamp_grp1_no3_status OR lamp_grp1_no4_status OR 
                          lamp_grp2_no3_status OR lamp_grp2_no4_status OR
                          lamp_grp3_no3_status OR lamp_grp4_no3_status
                     THEN 1.0 ELSE 0.0 END) as lamp_status_ratio,
            
            -- Heating signals
            AVG(heating_setpoint_c) as heating_setpoint_mean,
            
            -- Ventilation
            AVG(vent_lee_afd3_percent + vent_wind_afd3_percent) / 2.0 as ventilation_mean
            
        FROM sensor_data 
        WHERE time BETWEEN $1 AND $2
        GROUP BY DATE_TRUNC('hour', time)
        ORDER BY hour
        "#;
        
        let sensor_rows = sqlx::query(sensor_query)
            .bind(start_time)
            .bind(end_time)
            .fetch_all(&self.pool)
            .await?;
        
        // Convert to DataFrame
        let mut hour_vec = Vec::new();
        let mut temp_mean_vec = Vec::new();
        let mut temp_count_vec = Vec::new();
        let mut co2_mean_vec = Vec::new();
        let mut humidity_mean_vec = Vec::new();
        let mut radiation_mean_vec = Vec::new();
        let mut light_mean_vec = Vec::new();
        let mut lamp_ratio_vec = Vec::new();
        let mut heating_mean_vec = Vec::new();
        let mut vent_mean_vec = Vec::new();
        
        for row in sensor_rows {
            hour_vec.push(row.get::<DateTime<Utc>, _>("hour").timestamp());
            temp_mean_vec.push(row.get::<Option<f64>, _>("air_temp_c_mean"));
            temp_count_vec.push(row.get::<Option<i64>, _>("air_temp_c_count").unwrap_or(0));
            co2_mean_vec.push(row.get::<Option<f64>, _>("co2_mean"));
            humidity_mean_vec.push(row.get::<Option<f64>, _>("humidity_mean"));
            radiation_mean_vec.push(row.get::<Option<f64>, _>("radiation_mean"));
            light_mean_vec.push(row.get::<Option<f64>, _>("light_intensity_mean"));
            lamp_ratio_vec.push(row.get::<Option<f64>, _>("lamp_status_ratio"));
            heating_mean_vec.push(row.get::<Option<f64>, _>("heating_setpoint_mean"));
            vent_mean_vec.push(row.get::<Option<f64>, _>("ventilation_mean"));
        }
        
        let sensor_df = df! {
            "hour" => hour_vec,
            "temp_mean" => temp_mean_vec,
            "temp_count" => temp_count_vec,
            "co2_mean" => co2_mean_vec,
            "humidity_mean" => humidity_mean_vec,
            "radiation_mean" => radiation_mean_vec,
            "light_mean" => light_mean_vec,
            "lamp_ratio" => lamp_ratio_vec,
            "heating_mean" => heating_mean_vec,
            "ventilation_mean" => vent_mean_vec,
        }?;
        
        // Load external data in parallel
        let (weather_data, energy_data) = if self.config.enable_weather_features || self.config.enable_energy_features {
            let weather_result = if self.config.enable_weather_features {
                self.load_weather_data(start_time, end_time).await?
            } else {
                Vec::new()
            };
            
            let energy_result = if self.config.enable_energy_features {
                self.load_energy_data(start_time, end_time).await?
            } else {
                Vec::new()
            };
            
            (weather_result, energy_result)
        } else {
            (Vec::new(), Vec::new())
        };
        
        info!("Stage 1 complete: {} sensor hours, {} weather points, {} energy points", 
              sensor_df.height(), weather_data.len(), energy_data.len());
        
        Ok((sensor_df, weather_data, energy_data))
    }
    
    /// Enhanced GPU feature extraction with all domains
    pub async fn stage3_enhanced_gpu_features(
        &self,
        sensor_data: DataFrame,
        weather_data: Vec<WeatherData>,
        energy_data: Vec<EnergyPriceData>,
    ) -> Result<Vec<EnhancedFeatureSet>> {
        info!("Stage 3: Enhanced GPU feature extraction...");
        
        // GPU features are now computed in Python via hybrid pipeline
        // This method now computes CPU-based features only
        let mut all_features = Vec::new();
        
        // Process data in sliding windows
        let window_size = self.config.window_hours;
        let slide_size = self.config.slide_hours;
        let total_hours = sensor_data.height();
        
        for start_idx in (0..total_hours).step_by(slide_size) {
            let end_idx = (start_idx + window_size).min(total_hours);
            if end_idx - start_idx < window_size / 2 {
                break; // Skip small windows at the end
            }
            
            // Extract window data
            let window_data = sensor_data.slice(start_idx as i64, end_idx - start_idx);
            
            // Get corresponding external data
            let window_start_time = DateTime::from_timestamp(
                window_data.column("hour")?.get(0)?.try_extract::<i64>()?,
                0
            ).unwrap();
            let window_end_time = DateTime::from_timestamp(
                window_data.column("hour")?.get(window_data.height() - 1)?.try_extract::<i64>()?,
                0
            ).unwrap() + Duration::hours(1);
            
            let window_weather: Vec<_> = weather_data.iter()
                .filter(|w| w.timestamp >= window_start_time && w.timestamp <= window_end_time)
                .cloned()
                .collect();
                
            let window_energy: Vec<_> = energy_data.iter()
                .filter(|e| e.timestamp >= window_start_time && e.timestamp <= window_end_time)
                .cloned()
                .collect();
            
            // Extract features for each resolution
            for resolution in &self.config.resolution_windows {
                let features = self.extract_multiresolution_features(
                    &window_data,
                    &window_weather,
                    &window_energy,
                    resolution.clone(),
                    start_idx as i64,
                ).await?;
                
                all_features.push(features);
            }
        }
        
        info!("Stage 3 complete: {} enhanced feature sets extracted", all_features.len());
        Ok(all_features)
    }
    
    /// Extract features at specific resolution
    async fn extract_multiresolution_features(
        &self,
        sensor_data: &DataFrame,
        weather_data: &[WeatherData],
        energy_data: &[EnergyPriceData],
        resolution: Duration,
        era_id: i64,
    ) -> Result<EnhancedFeatureSet> {
        let mut feature_set = EnhancedFeatureSet {
            era_id,
            computed_at: Utc::now(),
            resolution: format!("{}min", resolution.num_minutes()),
            sensor_features: HashMap::new(),
            extended_stats: HashMap::new(),
            weather_features: HashMap::new(),
            energy_features: HashMap::new(),
            growth_features: HashMap::new(),
            temporal_features: HashMap::new(),
            optimization_metrics: HashMap::new(),
        };
        
        // Resample data to target resolution
        let resampled_data = self.resample_to_resolution(sensor_data.clone(), resolution)?;
        
        // 1. Extract basic sensor features
        self.extract_sensor_features(&resampled_data, &mut feature_set.sensor_features)?;
        
        // 2. Extended statistical features (if enabled)
        if self.config.enable_extended_statistics {
            self.extract_extended_statistical_features(&resampled_data, &mut feature_set.extended_stats)?;
        }
        
        // 3. Weather coupling features (if data available)
        if !weather_data.is_empty() && self.config.enable_weather_features {
            self.extract_weather_coupling_features(
                &resampled_data, 
                weather_data, 
                &mut feature_set.weather_features
            )?;
        }
        
        // 4. Energy optimization features (if data available)
        if !energy_data.is_empty() && self.config.enable_energy_features {
            self.extract_energy_optimization_features(
                &resampled_data,
                energy_data,
                &mut feature_set.energy_features
            )?;
        }
        
        // 5. Plant growth features (if phenotype data available)
        if self.phenotype_data.is_some() && self.config.enable_growth_features {
            self.extract_plant_growth_features(
                &resampled_data,
                self.phenotype_data.as_ref().unwrap(),
                &mut feature_set.growth_features
            )?;
        }
        
        // 6. Multi-objective optimization metrics
        self.compute_optimization_metrics(&mut feature_set)?;
        
        Ok(feature_set)
    }
    
    /// Extract basic sensor features
    fn extract_sensor_features(
        &self,
        data: &DataFrame,
        features: &mut HashMap<String, f64>,
    ) -> Result<()> {
        let sensors = vec![
            "temp_mean", "co2_mean", "humidity_mean", "radiation_mean",
            "light_mean", "lamp_ratio", "heating_mean", "ventilation_mean"
        ];
        
        for sensor in sensors {
            if let Ok(column) = data.column(sensor) {
                if let Ok(series) = column.cast(&DataType::Float64) {
                    let values = series.f64()?;
                    
                    // Basic statistics
                    if let Some(mean) = values.mean() {
                        features.insert(format!("{}_mean", sensor), mean);
                    }
                    if let Some(std) = values.std(1) {
                        features.insert(format!("{}_std", sensor), std);
                    }
                    if let Some(min) = values.min() {
                        features.insert(format!("{}_min", sensor), min);
                    }
                    if let Some(max) = values.max() {
                        features.insert(format!("{}_max", sensor), max);
                    }
                    
                    // Count of valid values
                    let count = values.len() - values.null_count();
                    features.insert(format!("{}_count", sensor), count as f64);
                }
            }
        }
        
        Ok(())
    }
    
    /// Extract extended statistical features using GPU kernels
    fn extract_extended_statistical_features(
        &self,
        data: &DataFrame,
        features: &mut HashMap<String, f64>,
    ) -> Result<()> {
        // TODO: Implement GPU kernel calls for extended statistics
        // For now, implement CPU versions of percentiles and moments
        
        let sensors = vec!["temp_mean", "co2_mean", "humidity_mean"];
        
        for sensor in sensors {
            if let Ok(column) = data.column(sensor) {
                if let Ok(series) = column.cast(&DataType::Float64) {
                    let values = series.f64()?;
                    let valid_values: Vec<f64> = values.into_iter()
                        .filter_map(|v| v)
                        .collect();
                    
                    if valid_values.len() >= 5 {
                        // Calculate percentiles
                        let mut sorted = valid_values.clone();
                        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        
                        let p5 = percentile(&sorted, 0.05);
                        let p25 = percentile(&sorted, 0.25);
                        let p50 = percentile(&sorted, 0.50);
                        let p75 = percentile(&sorted, 0.75);
                        let p95 = percentile(&sorted, 0.95);
                        
                        features.insert(format!("{}_p5", sensor), p5);
                        features.insert(format!("{}_p25", sensor), p25);
                        features.insert(format!("{}_p50", sensor), p50);
                        features.insert(format!("{}_p75", sensor), p75);
                        features.insert(format!("{}_p95", sensor), p95);
                        features.insert(format!("{}_iqr", sensor), p75 - p25);
                        
                        // Calculate skewness and kurtosis
                        if let (Some(mean), Some(std)) = (values.mean(), values.std(1)) {
                            if std > 0.0 {
                                let skewness = calculate_skewness(&valid_values, mean, std);
                                let kurtosis = calculate_kurtosis(&valid_values, mean, std);
                                
                                features.insert(format!("{}_skewness", sensor), skewness);
                                features.insert(format!("{}_kurtosis", sensor), kurtosis);
                            }
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Extract weather coupling features
    fn extract_weather_coupling_features(
        &self,
        sensor_data: &DataFrame,
        weather_data: &[WeatherData],
        features: &mut HashMap<String, f64>,
    ) -> Result<()> {
        if weather_data.is_empty() {
            return Ok(());
        }
        
        // Get internal temperature
        if let Ok(internal_temp) = sensor_data.column("temp_mean") {
            let internal_temps: Vec<f64> = internal_temp.f64()?
                .into_iter()
                .filter_map(|v| v)
                .collect();
            
            // Get corresponding external temperatures
            let external_temps: Vec<f64> = weather_data.iter()
                .filter_map(|w| w.temperature_2m.map(|t| t as f64))
                .collect();
            
            if !internal_temps.is_empty() && !external_temps.is_empty() {
                // Temperature differential
                let temp_diffs: Vec<f64> = internal_temps.iter()
                    .zip(external_temps.iter())
                    .map(|(i, e)| i - e)
                    .collect();
                
                if !temp_diffs.is_empty() {
                    let diff_mean = temp_diffs.iter().sum::<f64>() / temp_diffs.len() as f64;
                    let diff_std = {
                        let variance = temp_diffs.iter()
                            .map(|d| (d - diff_mean).powi(2))
                            .sum::<f64>() / temp_diffs.len() as f64;
                        variance.sqrt()
                    };
                    
                    features.insert("temp_differential_mean".to_string(), diff_mean);
                    features.insert("temp_differential_std".to_string(), diff_std);
                }
                
                // Solar radiation efficiency
                let solar_rads: Vec<f64> = weather_data.iter()
                    .filter_map(|w| w.shortwave_radiation.map(|r| r as f64))
                    .collect();
                
                if !solar_rads.is_empty() {
                    let total_radiation = solar_rads.iter().sum::<f64>();
                    let temp_gain = temp_diffs.iter()
                        .map(|d| d.max(0.0))
                        .sum::<f64>();
                    
                    let solar_efficiency = if total_radiation > 0.0 {
                        temp_gain / total_radiation
                    } else {
                        0.0
                    };
                    
                    features.insert("solar_efficiency".to_string(), solar_efficiency);
                }
            }
        }
        
        Ok(())
    }
    
    /// Extract energy optimization features
    fn extract_energy_optimization_features(
        &self,
        sensor_data: &DataFrame,
        energy_data: &[EnergyPriceData],
        features: &mut HashMap<String, f64>,
    ) -> Result<()> {
        if energy_data.is_empty() {
            return Ok(());
        }
        
        // Get energy prices (average of DK1 and DK2)
        let mut hourly_prices = HashMap::new();
        for price_point in energy_data {
            if let Some(price) = price_point.spot_price_dkk {
                let hour_key = price_point.timestamp.timestamp() / 3600 * 3600;
                hourly_prices.entry(hour_key)
                    .and_modify(|prices: &mut Vec<f32>| prices.push(price))
                    .or_insert_with(|| vec![price]);
            }
        }
        
        let avg_prices: Vec<f64> = hourly_prices.values()
            .map(|prices| prices.iter().sum::<f32>() as f64 / prices.len() as f64)
            .collect();
        
        if !avg_prices.is_empty() {
            // Get lamp usage
            if let Ok(lamp_column) = sensor_data.column("lamp_ratio") {
                let lamp_usage: Vec<f64> = lamp_column.f64()?
                    .into_iter()
                    .filter_map(|v| v)
                    .collect();
                
                if lamp_usage.len() == avg_prices.len() {
                    // Cost-weighted consumption
                    let cost_weighted: f64 = lamp_usage.iter()
                        .zip(avg_prices.iter())
                        .map(|(usage, price)| usage * price)
                        .sum();
                    
                    features.insert("cost_weighted_consumption".to_string(), cost_weighted);
                    
                    // Peak vs off-peak usage
                    let price_threshold = percentile(&avg_prices, 0.75);
                    let (peak_usage, offpeak_usage) = lamp_usage.iter()
                        .zip(avg_prices.iter())
                        .fold((0.0, 0.0), |(peak, offpeak), (usage, price)| {
                            if *price > price_threshold {
                                (peak + usage, offpeak)
                            } else {
                                (peak, offpeak + usage)
                            }
                        });
                    
                    let peak_ratio = if offpeak_usage > 0.0 {
                        peak_usage / offpeak_usage
                    } else {
                        999.0
                    };
                    
                    features.insert("peak_offpeak_ratio".to_string(), peak_ratio);
                    
                    // Energy efficiency score
                    let total_consumption = lamp_usage.iter().sum::<f64>();
                    let total_cost = cost_weighted;
                    let efficiency = if total_cost > 0.0 {
                        total_consumption / total_cost
                    } else {
                        0.0
                    };
                    
                    features.insert("energy_efficiency_score".to_string(), efficiency);
                }
            }
        }
        
        Ok(())
    }
    
    /// Extract plant growth features
    fn extract_plant_growth_features(
        &self,
        sensor_data: &DataFrame,
        phenotype: &PhenotypeData,
        features: &mut HashMap<String, f64>,
    ) -> Result<()> {
        // Growing Degree Days (GDD)
        if let Ok(temp_column) = sensor_data.column("temp_mean") {
            let temperatures: Vec<f64> = temp_column.f64()?
                .into_iter()
                .filter_map(|v| v)
                .collect();
            
            let gdd: f64 = temperatures.iter()
                .map(|t| (t - phenotype.base_temperature as f64).max(0.0))
                .sum::<f64>() / 24.0; // Daily accumulation
            
            features.insert("growing_degree_days".to_string(), gdd);
            
            // Temperature optimality
            let temp_optimality: f64 = temperatures.iter()
                .map(|t| {
                    let optimal = phenotype.optimal_temp_day as f64;
                    1.0 - (t - optimal).abs() / 10.0
                })
                .map(|opt| opt.max(0.0).min(1.0))
                .sum::<f64>() / temperatures.len() as f64;
            
            features.insert("temperature_optimality".to_string(), temp_optimality);
            
            // Stress degree days
            let stress_dd: f64 = temperatures.iter()
                .map(|t| {
                    let night_thresh = phenotype.optimal_temp_night as f64 - 2.0;
                    let day_thresh = phenotype.optimal_temp_day as f64 + 2.0;
                    if *t < night_thresh || *t > day_thresh {
                        (*t - phenotype.optimal_temp_day as f64).abs()
                    } else {
                        0.0
                    }
                })
                .sum::<f64>() / 24.0;
            
            features.insert("stress_degree_days".to_string(), stress_dd);
        }
        
        // Daily Light Integral (DLI)
        if let Ok(light_column) = sensor_data.column("light_mean") {
            let light_intensities: Vec<f64> = light_column.f64()?
                .into_iter()
                .filter_map(|v| v)
                .collect();
            
            let dli: f64 = light_intensities.iter()
                .sum::<f64>() * 3600.0 / 1_000_000.0 / 24.0; // mol/m²/day
            
            features.insert("daily_light_integral".to_string(), dli);
            
            // Light sufficiency
            let light_sufficiency = dli / phenotype.light_requirement_dli as f64;
            features.insert("light_sufficiency".to_string(), light_sufficiency);
            
            // Photoperiod calculation (hours with light > threshold)
            let photoperiod = light_intensities.iter()
                .filter(|&&light| light > 50.0) // μmol/m²/s threshold
                .count() as f64;
            
            features.insert("photoperiod_hours".to_string(), photoperiod);
            
            // Flowering signal (Kalanchoe is short-day plant)
            let flowering_signal = if photoperiod <= phenotype.flowering_photoperiod as f64 {
                1.0
            } else {
                0.0
            };
            features.insert("flowering_signal".to_string(), flowering_signal);
        }
        
        Ok(())
    }
    
    /// Compute multi-objective optimization metrics
    fn compute_optimization_metrics(
        &self,
        feature_set: &mut EnhancedFeatureSet,
    ) -> Result<()> {
        let metrics = &mut feature_set.optimization_metrics;
        // Growth performance score
        let growth_score = if let (Some(temp_opt), Some(light_suff)) = (
            feature_set.growth_features.get("temperature_optimality"),
            feature_set.growth_features.get("light_sufficiency")
        ) {
            temp_opt * light_suff.min(1.0)
        } else {
            0.0
        };
        metrics.insert("growth_performance_score".to_string(), growth_score);
        
        // Energy cost efficiency
        let cost_efficiency = feature_set.energy_features
            .get("energy_efficiency_score")
            .copied()
            .unwrap_or(0.0);
        metrics.insert("energy_cost_efficiency".to_string(), cost_efficiency);
        
        // Environmental coupling score
        let env_coupling = if let Some(solar_eff) = feature_set.weather_features.get("solar_efficiency") {
            solar_eff.min(1.0)
        } else {
            0.0
        };
        metrics.insert("environmental_coupling_score".to_string(), env_coupling);
        
        // Combined sustainability score (balanced objectives)
        let sustainability_score = (growth_score * 0.4) + 
                                 (cost_efficiency * 0.3) + 
                                 (env_coupling * 0.3);
        metrics.insert("sustainability_score".to_string(), sustainability_score);
        
        // MOEA objectives for optimization
        metrics.insert("obj1_minimize_energy_cost".to_string(), 
                      1.0 - cost_efficiency); // Minimize (invert efficiency)
        metrics.insert("obj2_maximize_growth_rate".to_string(), 
                      growth_score); // Maximize
        metrics.insert("obj3_minimize_stress".to_string(),
                      1.0 - feature_set.growth_features.get("stress_degree_days").unwrap_or(&0.0).min(1.0));
        
        Ok(())
    }
    
    /// Main enhanced pipeline execution
    pub async fn run_enhanced_pipeline(
        &mut self,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Result<EnhancedPipelineResults> {
        info!("Starting Enhanced Sparse GPU Pipeline...");
        info!("Processing period: {} to {}", start_time, end_time);
        
        // Load phenotype data
        self.load_phenotype_data().await?;
        
        // Stage 1: Enhanced aggregation with external data
        let (sensor_data, weather_data, energy_data) = 
            self.stage1_enhanced_aggregation(start_time, end_time).await?;
        
        // Stage 2: Conservative gap filling (from original pipeline)
        let filled_data = self.stage2_conservative_fill(sensor_data).await?;
        
        // Stage 3: Enhanced GPU feature extraction
        let enhanced_features = self.stage3_enhanced_gpu_features(
            filled_data.clone(),
            weather_data,
            energy_data,
        ).await?;
        
        // Stage 4: Create enhanced eras with optimization metrics
        let enhanced_eras = self.stage4_create_enhanced_eras(&enhanced_features).await?;
        
        // Calculate data quality metrics
        let quality_metrics = self.calculate_data_quality_metrics(&filled_data)?;
        
        // Calculate feature importance scores
        let importance_scores = self.calculate_feature_importance(&enhanced_features)?;
        
        // Organize features by resolution
        let mut multiresolution_features = HashMap::new();
        for feature in enhanced_features {
            multiresolution_features
                .entry(feature.resolution.clone())
                .or_insert_with(Vec::new)
                .push(feature);
        }
        
        info!("Enhanced pipeline complete!");
        info!("  - Resolutions: {}", multiresolution_features.len());
        info!("  - Total feature sets: {}", 
              multiresolution_features.values().map(|v| v.len()).sum::<usize>());
        info!("  - Enhanced eras: {}", enhanced_eras.len());
        
        Ok(EnhancedPipelineResults {
            filled_hourly_data: filled_data,
            multiresolution_features,
            monthly_eras: enhanced_eras,
            data_quality_metrics: quality_metrics,
            feature_importance_scores: importance_scores,
        })
    }
    
    // Helper methods (simplified implementations)
    
    async fn stage2_conservative_fill(&self, data: DataFrame) -> Result<DataFrame> {
        // Placeholder - use original conservative fill logic
        Ok(data)
    }
    
    async fn stage4_create_enhanced_eras(&self, features: &[EnhancedFeatureSet]) -> Result<Vec<Era>> {
        let mut eras = Vec::new();
        
        // Group by month and create eras with optimization metrics
        let mut monthly_features: HashMap<String, Vec<&EnhancedFeatureSet>> = HashMap::new();
        
        for feature_set in features {
            let month_key = feature_set.computed_at.format("%Y-%m").to_string();
            monthly_features.entry(month_key)
                .or_insert_with(Vec::new)
                .push(feature_set);
        }
        
        for (month, month_features) in monthly_features {
            if let Some(first_feature) = month_features.first() {
                let start_time = first_feature.computed_at - Duration::days(15);
                let end_time = first_feature.computed_at + Duration::days(15);
                
                // Calculate average optimization metrics
                let avg_growth = month_features.iter()
                    .filter_map(|f| f.optimization_metrics.get("growth_performance_score"))
                    .sum::<f64>() / month_features.len() as f64;
                
                let avg_cost = month_features.iter()
                    .filter_map(|f| f.energy_features.get("cost_weighted_consumption"))
                    .sum::<f64>() / month_features.len() as f64;
                
                let avg_temp = month_features.iter()
                    .filter_map(|f| f.sensor_features.get("temp_mean_mean"))
                    .sum::<f64>() / month_features.len() as f64;
                
                let avg_photoperiod = month_features.iter()
                    .filter_map(|f| f.growth_features.get("photoperiod_hours"))
                    .sum::<f64>() / month_features.len() as f64;
                
                eras.push(Era {
                    era_id: start_time.timestamp(),
                    start_time,
                    end_time,
                    feature_count: month_features.len(),
                    avg_temperature: avg_temp,
                    avg_photoperiod,
                    energy_cost: avg_cost,
                    growth_score: avg_growth,
                });
            }
        }
        
        Ok(eras)
    }
    
    fn resample_to_resolution(&self, data: DataFrame, resolution: Duration) -> Result<DataFrame> {
        // Placeholder - implement resampling logic
        Ok(data)
    }
    
    fn calculate_data_quality_metrics(&self, data: &DataFrame) -> Result<HashMap<String, f64>> {
        let mut metrics = HashMap::new();
        
        // Calculate coverage, completeness, etc.
        let total_rows = data.height() as f64;
        let coverage = total_rows / (24.0 * 30.0); // Rough monthly coverage
        
        metrics.insert("data_coverage".to_string(), coverage);
        metrics.insert("total_hours".to_string(), total_rows);
        
        Ok(metrics)
    }
    
    fn calculate_feature_importance(&self, features: &[EnhancedFeatureSet]) -> Result<HashMap<String, f64>> {
        let mut importance = HashMap::new();
        
        // Placeholder - implement feature importance calculation
        importance.insert("temperature_features".to_string(), 0.8);
        importance.insert("energy_features".to_string(), 0.6);
        importance.insert("growth_features".to_string(), 0.9);
        
        Ok(importance)
    }
}

// Helper functions
fn percentile(sorted_data: &[f64], p: f64) -> f64 {
    let index = p * (sorted_data.len() - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;
    
    if lower == upper {
        sorted_data[lower]
    } else {
        let weight = index - lower as f64;
        sorted_data[lower] * (1.0 - weight) + sorted_data[upper] * weight
    }
}

fn calculate_skewness(values: &[f64], mean: f64, std: f64) -> f64 {
    if std == 0.0 {
        return 0.0;
    }
    
    let n = values.len() as f64;
    let sum_cubed = values.iter()
        .map(|x| ((x - mean) / std).powi(3))
        .sum::<f64>();
    
    sum_cubed / n
}

fn calculate_kurtosis(values: &[f64], mean: f64, std: f64) -> f64 {
    if std == 0.0 {
        return 0.0;
    }
    
    let n = values.len() as f64;
    let sum_fourth = values.iter()
        .map(|x| ((x - mean) / std).powi(4))
        .sum::<f64>();
    
    (sum_fourth / n) - 3.0 // Excess kurtosis
}