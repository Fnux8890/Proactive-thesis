use anyhow::Result;
use chrono::{DateTime, Utc, Duration};
use polars::prelude::*;
use sqlx::{postgres::PgPool, Row};
use std::collections::HashMap;
use std::path::PathBuf;
use tracing::{info, warn, debug};
use serde::{Serialize, Deserialize};

use crate::db::{create_features_table_if_not_exists, write_features, FeatureSet, EraData, Era as DbEra};
use crate::features::GpuFeatureExtractor;
use crate::data_quality::{DataQualityAnalyzer, AdaptiveWindowConfig};

pub struct SparsePipelineConfig {
    pub min_hourly_coverage: f32,  // Minimum coverage to keep an hour
    pub max_interpolation_gap: i64, // Max gap in hours to interpolate
    pub enable_parquet_checkpoints: bool,
    pub checkpoint_dir: PathBuf,
    pub window_hours: usize,        // Sliding window size in hours
    pub slide_hours: usize,         // Window slide step in hours
}

impl Default for SparsePipelineConfig {
    fn default() -> Self {
        Self {
            min_hourly_coverage: 0.1,  // 10% minimum coverage for sparse data
            max_interpolation_gap: 2,
            enable_parquet_checkpoints: true,
            checkpoint_dir: PathBuf::from("/tmp/gpu_sparse_pipeline"),
            window_hours: 24,
            slide_hours: 6,
        }
    }
}

pub struct SparsePipeline {
    pool: PgPool,
    config: SparsePipelineConfig,
    gpu_extractor: Option<GpuFeatureExtractor>,
    quality_analyzer: DataQualityAnalyzer,
}

impl SparsePipeline {
    pub fn new(pool: PgPool, config: SparsePipelineConfig) -> Self {
        // Create checkpoint directory if needed
        if config.enable_parquet_checkpoints {
            std::fs::create_dir_all(&config.checkpoint_dir).ok();
        }
        
        let quality_analyzer = DataQualityAnalyzer::new(
            config.min_hourly_coverage as f64,
            0.5,  // min quality score
        );
        
        Self { 
            pool, 
            config,
            gpu_extractor: None,
            quality_analyzer,
        }
    }
    
    pub fn with_gpu_extractor(mut self, extractor: GpuFeatureExtractor) -> Self {
        self.gpu_extractor = Some(extractor);
        self
    }
    
    
    /// Stage 1: Aggregate raw data to hourly level
    pub async fn stage1_aggregate_hourly(
        &self,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Result<DataFrame> {
        info!("Stage 1: Aggregating sparse data to hourly level...");
        
        let query = r#"
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
            
            AVG(vpd_hpa) as vpd_mean,
            COUNT(vpd_hpa) as vpd_count,
            
            -- Lamp status (only columns that exist)
            MAX(CASE WHEN lamp_grp1_no3_status OR lamp_grp1_no4_status OR 
                          lamp_grp2_no3_status OR lamp_grp2_no4_status OR
                          lamp_grp3_no3_status OR lamp_grp4_no3_status 
                     THEN 1 ELSE 0 END) as any_lamp_on,
            
            COUNT(*) as total_records
        FROM sensor_data_merged
        WHERE time >= $1 AND time < $2
        GROUP BY 1
        ORDER BY 1
        "#;
        
        let rows = sqlx::query(query)
            .bind(start_time)
            .bind(end_time)
            .fetch_all(&self.pool)
            .await?;
        
        // Convert to Polars DataFrame
        let mut hour_vec: Vec<DateTime<Utc>> = Vec::new();
        let mut temp_mean_vec: Vec<Option<f64>> = Vec::new();
        let mut temp_count_vec: Vec<i32> = Vec::new();
        let mut temp_std_vec: Vec<Option<f64>> = Vec::new();
        let mut co2_mean_vec: Vec<Option<f64>> = Vec::new();
        let mut co2_count_vec: Vec<i32> = Vec::new();
        let mut humidity_mean_vec: Vec<Option<f64>> = Vec::new();
        let mut humidity_count_vec: Vec<i32> = Vec::new();
        let mut lamp_on_vec: Vec<i32> = Vec::new();
        let mut total_records_vec: Vec<i64> = Vec::new();
        
        for row in rows {
            hour_vec.push(row.get::<DateTime<Utc>, _>("hour"));
            temp_mean_vec.push(row.get::<Option<f64>, _>("air_temp_c_mean"));
            temp_count_vec.push(row.get::<i64, _>("air_temp_c_count") as i32);
            temp_std_vec.push(row.get::<Option<f64>, _>("air_temp_c_std"));
            co2_mean_vec.push(row.get::<Option<f64>, _>("co2_mean"));
            co2_count_vec.push(row.get::<i64, _>("co2_count") as i32);
            humidity_mean_vec.push(row.get::<Option<f64>, _>("humidity_mean"));
            humidity_count_vec.push(row.get::<i64, _>("humidity_count") as i32);
            lamp_on_vec.push(row.get::<i32, _>("any_lamp_on"));
            total_records_vec.push(row.get::<i64, _>("total_records"));
        }
        
        // Create DataFrame
        let df = df![
            "hour" => hour_vec.into_iter().map(|dt| dt.timestamp()).collect::<Vec<_>>(),
            "temp_mean" => temp_mean_vec,
            "temp_count" => temp_count_vec,
            "temp_std" => temp_std_vec,
            "co2_mean" => co2_mean_vec,
            "co2_count" => co2_count_vec,
            "humidity_mean" => humidity_mean_vec,
            "humidity_count" => humidity_count_vec,
            "lamp_on" => lamp_on_vec,
            "total_records" => total_records_vec,
        ]?;
        
        // Calculate coverage and filter
        let df = self.add_coverage_metrics(df)?;
        let viable_df = self.filter_viable_hours(df)?;
        
        info!("Stage 1 complete: {} viable hours found", viable_df.height());
        
        // Save checkpoint if enabled
        if self.config.enable_parquet_checkpoints {
            let checkpoint_path = self.config.checkpoint_dir.join("stage1_hourly.parquet");
            info!("Saving stage 1 checkpoint to {:?}", checkpoint_path);
            
            viable_df.clone().lazy()
                .sink_parquet(checkpoint_path.clone(), Default::default())
                .map_err(|e| anyhow::anyhow!("Failed to save checkpoint: {}", e))?;
                
            // Also save with timestamp for debugging
            let timestamped_path = self.config.checkpoint_dir.join(
                format!("stage1_hourly_{}.parquet", Utc::now().format("%Y%m%d_%H%M%S"))
            );
            viable_df.clone().lazy()
                .sink_parquet(timestamped_path, Default::default())
                .map_err(|e| anyhow::anyhow!("Failed to save timestamped checkpoint: {}", e))?;
        }
        
        Ok(viable_df)
    }
    
    /// Stage 2: Conservative gap filling with physics constraints
    pub async fn stage2_conservative_fill(&self, hourly_df: DataFrame) -> Result<DataFrame> {
        info!("Stage 2: Conservative gap filling...");
        
        let mut df = hourly_df;
        
        // Define filling rules with physics constraints
        let fill_rules = vec![
            ("temp_mean", 10.0, 40.0, 2.0),    // min, max, max_change/hour
            ("co2_mean", 300.0, 1500.0, 100.0),
            ("humidity_mean", 30.0, 95.0, 10.0),
        ];
        
        let mut fill_stats = HashMap::new();
        
        for (col_name, min_val, max_val, max_change) in fill_rules {
            let (filled_df, stats) = self.fill_column_with_stats(df, col_name, min_val, max_val, max_change)?;
            df = filled_df;
            fill_stats.insert(col_name.to_string(), stats);
        }
        
        info!("Stage 2 complete: Gap filling applied");
        for (col, stats) in &fill_stats {
            info!("  {}: filled {} gaps", col, stats);
        }
        
        // Save checkpoint
        if self.config.enable_parquet_checkpoints {
            let checkpoint_path = self.config.checkpoint_dir.join("stage2_filled.parquet");
            info!("Saving stage 2 checkpoint to {:?}", checkpoint_path);
            
            df.clone().lazy()
                .sink_parquet(checkpoint_path.clone(), Default::default())
                .map_err(|e| anyhow::anyhow!("Failed to save checkpoint: {}", e))?;
                
            let timestamped_path = self.config.checkpoint_dir.join(
                format!("stage2_filled_{}.parquet", Utc::now().format("%Y%m%d_%H%M%S"))
            );
            df.clone().lazy()
                .sink_parquet(timestamped_path, Default::default())
                .map_err(|e| anyhow::anyhow!("Failed to save timestamped checkpoint: {}", e))?;
        }
        
        Ok(df)
    }
    
    /// Stage 3: Extract features using GPU
    pub async fn stage3_gpu_features(&self, filled_df: DataFrame) -> Result<Vec<FeatureSet>> {
        info!("Stage 3: GPU feature extraction on filled data...");
        
        let mut all_features = Vec::new();
        
        // Analyze overall data quality
        let quality_metrics = self.quality_analyzer.analyze_window(&filled_df)?;
        info!("Overall data quality score: {:.2}", quality_metrics.overall_score);
        info!("  Coverage: {:.1}%, Continuity: {:.1}%, Consistency: {:.1}%",
            quality_metrics.coverage * 100.0,
            quality_metrics.continuity * 100.0,
            quality_metrics.consistency * 100.0
        );
        
        // Get adaptive window configuration
        let adaptive_config = AdaptiveWindowConfig::from_quality_metrics(&quality_metrics);
        info!("Adaptive window config: size={} hours, overlap={:.0}%",
            adaptive_config.window_size,
            adaptive_config.overlap_ratio * 100.0
        );
        
        // Process with adaptive windows
        let windows = self.create_adaptive_windows(filled_df, &adaptive_config)?;
        info!("Created {} adaptive windows", windows.len());
        
        for (window_start, window_df) in windows {
            if window_df.height() < self.config.window_hours / 2 {
                debug!("Skipping window with insufficient data: {} hours", window_df.height());
                continue;
            }
            
            // Extract data for GPU processing
            let sensor_data = self.prepare_gpu_data(&window_df)?;
            
            // Use GPU feature extraction if available
            let features = if let Some(ref extractor) = self.gpu_extractor {
                debug!("Using GPU feature extraction for window starting at {}", window_start);
                
                // Create EraData structure for GPU extractor
                let era = DbEra {
                    era_id: window_start as i32,
                    era_level: "window".to_string(),
                    start_time: DateTime::from_timestamp(window_start, 0).unwrap(),
                    end_time: DateTime::from_timestamp(window_start + (self.config.window_hours as i64 * 3600), 0).unwrap(),
                    row_count: window_df.height() as i32,
                };
                
                let era_data = EraData {
                    era,
                    sensor_data,
                    timestamps: vec![],  // Not needed for GPU processing
                };
                
                // Extract features on GPU
                match extractor.extract_batch(&[era_data]) {
                    Ok(mut features) => features.pop().unwrap(),
                    Err(e) => {
                        warn!("GPU extraction failed, falling back to CPU: {}", e);
                        self.compute_window_features_cpu(window_start, &window_df)?
                    }
                }
            } else {
                debug!("Using CPU feature extraction for window starting at {}", window_start);
                self.compute_window_features_cpu(window_start, &window_df)?
            };
            
            all_features.push(features);
        }
        
        info!("Stage 3 complete: {} window feature sets extracted", all_features.len());
        
        // Save checkpoint if enabled
        if self.config.enable_parquet_checkpoints {
            let checkpoint_path = self.config.checkpoint_dir.join("stage3_features.json");
            info!("Saving stage 3 checkpoint to {:?}", checkpoint_path);
            
            let json_data = serde_json::to_string_pretty(&all_features)?;
            std::fs::write(&checkpoint_path, json_data)?;
            
            let timestamped_path = self.config.checkpoint_dir.join(
                format!("stage3_features_{}.json", Utc::now().format("%Y%m%d_%H%M%S"))
            );
            let json_data = serde_json::to_string_pretty(&all_features)?;
            std::fs::write(timestamped_path, json_data)?;
        }
        
        Ok(all_features)
    }
    
    /// Stage 4: Create monthly eras from features
    pub async fn stage4_create_eras(&self, features: Vec<FeatureSet>) -> Result<Vec<Era>> {
        info!("Stage 4: Creating monthly eras from features...");
        
        let mut eras = Vec::new();
        let mut current_month = None;
        let mut month_features = Vec::new();
        
        for feature_set in features {
            let month = feature_set.computed_at.format("%Y-%m").to_string();
            
            if current_month.as_ref() != Some(&month) {
                // Process previous month if exists
                if !month_features.is_empty() {
                    if let Some(era) = self.create_era_from_features(&month_features) {
                        eras.push(era);
                    }
                }
                
                // Start new month
                current_month = Some(month);
                month_features = vec![feature_set];
            } else {
                month_features.push(feature_set);
            }
        }
        
        // Process last month
        if !month_features.is_empty() {
            if let Some(era) = self.create_era_from_features(&month_features) {
                eras.push(era);
            }
        }
        
        info!("Stage 4 complete: {} monthly eras created", eras.len());
        
        // Save checkpoint if enabled
        if self.config.enable_parquet_checkpoints {
            let checkpoint_path = self.config.checkpoint_dir.join("stage4_eras.json");
            info!("Saving stage 4 checkpoint to {:?}", checkpoint_path);
            
            let json_data = serde_json::to_string_pretty(&eras)?;
            std::fs::write(&checkpoint_path, json_data)?;
            
            let timestamped_path = self.config.checkpoint_dir.join(
                format!("stage4_eras_{}.json", Utc::now().format("%Y%m%d_%H%M%S"))
            );
            let json_data = serde_json::to_string_pretty(&eras)?;
            std::fs::write(timestamped_path, json_data)?;
        }
        
        Ok(eras)
    }
    
    /// Main pipeline execution
    pub async fn run_pipeline(
        &self,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Result<PipelineResults> {
        info!("Starting Sparse GPU Pipeline...");
        info!("Processing period: {} to {}", start_time, end_time);
        
        // Stage 1: Aggregate to hourly
        let hourly_df = self.stage1_aggregate_hourly(start_time, end_time).await?;
        
        // Stage 2: Conservative fill
        let filled_df = self.stage2_conservative_fill(hourly_df).await?;
        
        // Stage 3: GPU features
        let features = self.stage3_gpu_features(filled_df.clone()).await?;
        
        // Stage 4: Create eras
        let eras = self.stage4_create_eras(features.clone()).await?;
        
        // Save results to database
        self.save_results(&features, &eras).await?;
        
        Ok(PipelineResults {
            filled_hourly_data: filled_df,
            daily_features: features,
            monthly_eras: eras,
        })
    }
    
    // Helper methods
    
    fn add_coverage_metrics(&self, df: DataFrame) -> Result<DataFrame> {
        let temp_counts = df.column("temp_count")?
            .cast(&DataType::Float32)?
            .f32()?
            .clone();
        // For sparse data, consider 10 samples per hour as 100% coverage
        let temp_coverage = temp_counts / 10.0;
        
        let co2_counts = df.column("co2_count")?
            .cast(&DataType::Float32)?
            .f32()?
            .clone();
        let co2_coverage = co2_counts / 10.0;
        
        // Calculate overall coverage as average of all sensors
        let humidity_counts = df.column("humidity_count")?
            .cast(&DataType::Float32)?
            .f32()?
            .clone();
        let humidity_coverage = humidity_counts / 10.0;
        
        // Convert to series for arithmetic operations
        let temp_series = temp_coverage.into_series();
        let co2_series = co2_coverage.into_series();
        let humidity_series = humidity_coverage.into_series();
        
        // Calculate overall coverage as average of three sensors
        let sum_coverage = (&temp_series + &co2_series)?;
        let sum_coverage_final = (&sum_coverage + &humidity_series)?;
        let overall_coverage = &sum_coverage_final / 3.0;
        
        let mut df = df;
        df.with_column(temp_series.with_name("temp_coverage"))?;
        df.with_column(co2_series.with_name("co2_coverage"))?;
        df.with_column(humidity_series.with_name("humidity_coverage"))?;
        df.with_column(overall_coverage.with_name("overall_coverage"))?;
        
        Ok(df)
    }
    
    fn filter_viable_hours(&self, df: DataFrame) -> Result<DataFrame> {
        let mask = df.column("overall_coverage")?
            .gt(self.config.min_hourly_coverage)?;
        
        Ok(df.filter(&mask)?)
    }
    
    fn fill_column_with_stats(
        &self,
        mut df: DataFrame,
        col_name: &str,
        min_val: f64,
        max_val: f64,
        max_change: f64,
    ) -> Result<(DataFrame, usize)> {
        let col = df.column(col_name)?.clone();
        let _original_nulls = col.null_count();
        
        // Create a mutable series for interpolation
        let mut values: Vec<Option<f64>> = col.f64()?
            .into_iter()
            .collect();
            
        // Linear interpolation with gap limit
        let mut filled_count = 0;
        for i in 1..values.len()-1 {
            if values[i].is_none() {
                // Check gap size
                let mut gap_size = 1;
                let mut j = i + 1;
                while j < values.len() && values[j].is_none() {
                    gap_size += 1;
                    j += 1;
                }
                
                // Only fill if gap is small enough
                if gap_size <= self.config.max_interpolation_gap as usize {
                    // Find previous and next valid values
                    if let (Some(prev_val), Some(next_val)) = 
                        (values[i-1], values.get(j).and_then(|v| *v)) {
                        
                        // Linear interpolation
                        let interpolated = prev_val + (next_val - prev_val) * 
                            (1.0 / (gap_size + 1) as f64);
                        
                        // Check physics constraints
                        if interpolated >= min_val && 
                           interpolated <= max_val &&
                           (interpolated - prev_val).abs() <= max_change {
                            values[i] = Some(interpolated);
                            filled_count += 1;
                        }
                    }
                }
            }
        }
        
        // Convert back to series
        let filled_series = Float64Chunked::from_iter(values.into_iter())
            .into_series()
            .with_name(col_name);
        
        df.replace(col_name, filled_series)?;
        
        Ok((df, filled_count))
    }
    
    fn create_sliding_windows(&self, df: DataFrame) -> Result<Vec<(i64, DataFrame)>> {
        let mut windows = Vec::new();
        
        let timestamps = df.column("hour")?
            .i64()?
            .into_no_null_iter()
            .collect::<Vec<_>>();
            
        if timestamps.is_empty() {
            return Ok(windows);
        }
        
        let window_duration = self.config.window_hours as i64 * 3600;
        let slide_duration = self.config.slide_hours as i64 * 3600;
        
        let mut window_start = timestamps[0];
        let end_time = *timestamps.last().unwrap();
        
        while window_start + window_duration <= end_time {
            let window_end = window_start + window_duration;
            
            // Create mask for this window
            let mask = df.column("hour")?
                .i64()?
                .into_iter()
                .map(|opt_ts| {
                    opt_ts.map_or(false, |ts| ts >= window_start && ts < window_end)
                })
                .collect::<BooleanChunked>()
                .into_series();
            
            let mask_ca = mask.bool()?;
            let window_df = df.filter(mask_ca)?;
            
            if window_df.height() >= self.config.window_hours / 2 {
                // Check window quality before adding
                let quality = self.quality_analyzer.analyze_window(&window_df).ok();
                if let Some(q) = quality {
                    if q.overall_score >= self.config.min_hourly_coverage as f64 {
                        windows.push((window_start, window_df));
                    }
                } else {
                    windows.push((window_start, window_df));
                }
            }
            
            window_start += slide_duration;
        }
        
        info!("Created {} sliding windows", windows.len());
        Ok(windows)
    }
    
    fn create_adaptive_windows(&self, df: DataFrame, config: &AdaptiveWindowConfig) -> Result<Vec<(i64, DataFrame)>> {
        let mut windows = Vec::new();
        
        let timestamps = df.column("hour")?
            .i64()?
            .into_no_null_iter()
            .collect::<Vec<_>>();
            
        if timestamps.is_empty() {
            return Ok(windows);
        }
        
        let window_duration = config.window_size as i64 * 3600;
        let slide_duration = ((config.window_size as f64 * (1.0 - config.overlap_ratio)) as i64).max(1) * 3600;
        
        let mut window_start = timestamps[0];
        let end_time = *timestamps.last().unwrap();
        
        while window_start + window_duration <= end_time {
            let window_end = window_start + window_duration;
            
            // Create mask for this window
            let mask = df.column("hour")?
                .i64()?
                .into_iter()
                .map(|opt_ts| {
                    opt_ts.map_or(false, |ts| ts >= window_start && ts < window_end)
                })
                .collect::<BooleanChunked>()
                .into_series();
            
            let mask_ca = mask.bool()?;
            let window_df = df.filter(mask_ca)?;
            
            // Check window quality
            if window_df.height() >= config.window_size / 2 {
                let quality = self.quality_analyzer.analyze_window(&window_df)?;
                
                if quality.overall_score >= config.quality_threshold {
                    debug!("Window at {} has quality score {:.2}", 
                        DateTime::from_timestamp(window_start, 0).unwrap(), 
                        quality.overall_score
                    );
                    windows.push((window_start, window_df));
                } else {
                    debug!("Skipping window at {} due to low quality: {:.2}",
                        DateTime::from_timestamp(window_start, 0).unwrap(),
                        quality.overall_score
                    );
                }
            }
            
            window_start += slide_duration;
        }
        
        info!("Created {} quality windows from {} candidates", 
            windows.len(), 
            ((end_time - timestamps[0]) / slide_duration) as usize
        );
        Ok(windows)
    }
    
    fn prepare_gpu_data(&self, df: &DataFrame) -> Result<HashMap<String, Vec<f32>>> {
        let mut sensor_data = HashMap::new();
        
        // Extract sensor columns for GPU processing
        let sensors = vec![
            ("temp_mean", "air_temp_c"),
            ("co2_mean", "co2_measured_ppm"),
            ("humidity_mean", "relative_humidity_percent"),
        ];
        
        for (col_name, sensor_name) in sensors {
            if let Ok(col) = df.column(col_name) {
                let data: Vec<f32> = col
                    .cast(&DataType::Float32)?
                    .f32()?
                    .into_iter()
                    .map(|opt| opt.unwrap_or(f32::NAN))
                    .collect();
                
                sensor_data.insert(sensor_name.to_string(), data);
            }
        }
        
        Ok(sensor_data)
    }
    
    fn compute_window_features_cpu(
        &self,
        window_start: i64,
        window_df: &DataFrame,
    ) -> Result<FeatureSet> {
        let mut features = HashMap::new();
        
        // Extract and compute features for each sensor
        let sensors = vec![
            ("temp_mean", "temp"),
            ("co2_mean", "co2"),
            ("humidity_mean", "humidity"),
        ];
        
        for (col_name, feature_prefix) in sensors {
            if let Ok(col) = window_df.column(col_name) {
                let data: Vec<f32> = col
                    .cast(&DataType::Float32)?
                    .f32()?
                    .into_iter()
                    .filter_map(|v| v)
                    .collect();
                
                if !data.is_empty() {
                    features.insert(format!("{}_mean", feature_prefix), 
                        data.iter().sum::<f32>() as f64 / data.len() as f64);
                    features.insert(format!("{}_std", feature_prefix), 
                        self.compute_std(&data));
                    features.insert(format!("{}_min", feature_prefix), 
                        *data.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() as f64);
                    features.insert(format!("{}_max", feature_prefix), 
                        *data.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() as f64);
                }
            }
        }
        
        // Photoperiod
        if let Ok(lamp_col) = window_df.column("lamp_on") {
            let lamp_hours = lamp_col.i32()?
                .into_no_null_iter()
                .filter(|&v| v > 0)
                .count() as f64;
            features.insert("photoperiod_hours".to_string(), lamp_hours);
        }
        
        Ok(FeatureSet {
            era_id: window_start as i32,
            era_level: "window".to_string(),
            features,
            computed_at: DateTime::from_timestamp(window_start, 0).unwrap(),
        })
    }
    
    fn compute_std(&self, data: &[f32]) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }
        
        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let variance = data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / (data.len() - 1) as f32;
        
        (variance as f64).sqrt()
    }
    
    fn create_era_from_features(&self, features: &[FeatureSet]) -> Option<Era> {
        if features.is_empty() {
            return None;
        }
        
        let start_time = features.first().unwrap().computed_at;
        let end_time = features.last().unwrap().computed_at + Duration::hours(self.config.window_hours as i64);
        
        // Compute average metrics
        let mut avg_temp = 0.0;
        let mut avg_photoperiod = 0.0;
        let mut count = 0;
        
        for feature_set in features {
            if let Some(temp) = feature_set.features.get("temp_mean") {
                avg_temp += temp;
                count += 1;
            }
            if let Some(photo) = feature_set.features.get("photoperiod_hours") {
                avg_photoperiod += photo;
            }
        }
        
        if count > 0 {
            avg_temp /= count as f64;
            avg_photoperiod /= features.len() as f64;
        }
        
        Some(Era {
            era_id: start_time.timestamp() as i32,
            start_time,
            end_time,
            feature_count: features.len(),
            avg_temperature: avg_temp,
            avg_photoperiod,
        })
    }
    
    async fn save_results(&self, features: &[FeatureSet], eras: &[Era]) -> Result<()> {
        // Save features
        create_features_table_if_not_exists(&self.pool, "sparse_window_features").await?;
        write_features(&self.pool, "sparse_window_features", features.to_vec()).await?;
        
        // Save eras
        let query = r#"
        CREATE TABLE IF NOT EXISTS sparse_monthly_eras (
            era_id INTEGER PRIMARY KEY,
            start_time TIMESTAMPTZ NOT NULL,
            end_time TIMESTAMPTZ NOT NULL,
            feature_count INTEGER NOT NULL,
            avg_temperature FLOAT,
            avg_photoperiod FLOAT
        )
        "#;
        sqlx::query(query).execute(&self.pool).await?;
        
        for era in eras {
            sqlx::query(
                "INSERT INTO sparse_monthly_eras (era_id, start_time, end_time, feature_count, avg_temperature, avg_photoperiod)
                 VALUES ($1, $2, $3, $4, $5, $6)
                 ON CONFLICT (era_id) DO UPDATE SET
                    end_time = EXCLUDED.end_time,
                    feature_count = EXCLUDED.feature_count,
                    avg_temperature = EXCLUDED.avg_temperature,
                    avg_photoperiod = EXCLUDED.avg_photoperiod"
            )
            .bind(era.era_id)
            .bind(era.start_time)
            .bind(era.end_time)
            .bind(era.feature_count as i32)
            .bind(era.avg_temperature)
            .bind(era.avg_photoperiod)
            .execute(&self.pool)
            .await?;
        }
        
        Ok(())
    }
}

pub struct PipelineResults {
    pub filled_hourly_data: DataFrame,
    pub daily_features: Vec<FeatureSet>,
    pub monthly_eras: Vec<Era>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Era {
    pub era_id: i32,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub feature_count: usize,
    pub avg_temperature: f64,
    pub avg_photoperiod: f64,
}