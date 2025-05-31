use polars::prelude::*;
use std::collections::HashMap;
use anyhow::Result;

/// Data quality metrics for a time window
#[derive(Debug, Clone)]
pub struct DataQualityMetrics {
    pub overall_score: f64,      // 0.0 to 1.0
    pub coverage: f64,           // Percentage of non-null values
    pub continuity: f64,         // How continuous the data is (few gaps)
    pub consistency: f64,        // How consistent values are (low noise)
    pub sensor_availability: f64, // How many sensors have data
}

/// Analyzes data quality and suggests optimal window sizes
pub struct DataQualityAnalyzer {
    _min_coverage_threshold: f64,
    _min_quality_score: f64,
}

impl DataQualityAnalyzer {
    pub fn new(min_coverage: f64, min_quality: f64) -> Self {
        Self {
            _min_coverage_threshold: min_coverage,
            _min_quality_score: min_quality,
        }
    }

    /// Analyze quality of a data window
    pub fn analyze_window(&self, df: &DataFrame) -> Result<DataQualityMetrics> {
        let total_rows = df.height() as f64;
        
        // Key sensors to check
        let sensors = vec![
            "temp_mean", "co2_mean", "humidity_mean", 
            "radiation_mean", "vpd_mean"
        ];
        
        let mut sensor_scores = Vec::new();
        let mut available_sensors = 0;
        
        for sensor in &sensors {
            if let Ok(col) = df.column(sensor) {
                let non_null_count = (total_rows - col.null_count() as f64) as f64;
                let coverage = non_null_count / total_rows;
                
                if coverage > 0.01 {  // At least 1% data (very sparse)
                    available_sensors += 1;
                    
                    // Calculate continuity (inverse of gap ratio)
                    let continuity = self.calculate_continuity(col)?;
                    
                    // Calculate consistency (inverse of noise)
                    let consistency = self.calculate_consistency(col)?;
                    
                    sensor_scores.push((coverage, continuity, consistency));
                }
            }
        }
        
        if sensor_scores.is_empty() {
            return Ok(DataQualityMetrics {
                overall_score: 0.0,
                coverage: 0.0,
                continuity: 0.0,
                consistency: 0.0,
                sensor_availability: 0.0,
            });
        }
        
        // Aggregate metrics
        let avg_coverage = sensor_scores.iter().map(|(c, _, _)| c).sum::<f64>() / sensor_scores.len() as f64;
        let avg_continuity = sensor_scores.iter().map(|(_, cn, _)| cn).sum::<f64>() / sensor_scores.len() as f64;
        let avg_consistency = sensor_scores.iter().map(|(_, _, cs)| cs).sum::<f64>() / sensor_scores.len() as f64;
        let sensor_availability = available_sensors as f64 / sensors.len() as f64;
        
        // Overall quality score (weighted average)
        let overall_score = 0.4 * avg_coverage + 
                           0.3 * avg_continuity + 
                           0.2 * avg_consistency + 
                           0.1 * sensor_availability;
        
        Ok(DataQualityMetrics {
            overall_score,
            coverage: avg_coverage,
            continuity: avg_continuity,
            consistency: avg_consistency,
            sensor_availability,
        })
    }

    /// Calculate continuity score (how continuous the data is)
    fn calculate_continuity(&self, col: &Series) -> Result<f64> {
        let values = col.f64()?;
        let mut gap_count = 0;
        let mut last_valid_idx = None;
        
        for (idx, val) in values.into_iter().enumerate() {
            if val.is_some() {
                if let Some(last_idx) = last_valid_idx {
                    if idx - last_idx > 1 {
                        gap_count += 1;
                    }
                }
                last_valid_idx = Some(idx);
            }
        }
        
        // Continuity is inverse of gap ratio
        let max_gaps = col.len() / 2;  // Theoretical max gaps
        let continuity = 1.0 - (gap_count as f64 / max_gaps as f64);
        
        Ok(continuity.max(0.0).min(1.0))
    }

    /// Calculate consistency score (inverse of noise/variability)
    fn calculate_consistency(&self, col: &Series) -> Result<f64> {
        let values = col.f64()?
            .into_no_null_iter()
            .collect::<Vec<_>>();
        
        if values.len() < 3 {
            return Ok(0.5);  // Not enough data
        }
        
        // Calculate coefficient of variation (CV)
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        
        if mean.abs() < 0.001 {
            return Ok(1.0);  // Very stable (near zero)
        }
        
        let variance = values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / (values.len() - 1) as f64;
        
        let std_dev = variance.sqrt();
        let cv = std_dev / mean.abs();
        
        // Convert CV to consistency score (lower CV = higher consistency)
        let consistency = (-cv).exp();  // Exponential decay
        
        Ok(consistency.max(0.0).min(1.0))
    }

    /// Suggest optimal window size based on data density
    pub fn suggest_window_size(&self, df: &DataFrame, min_hours: usize, max_hours: usize) -> Result<usize> {
        let mut window_scores = HashMap::new();
        
        // Test different window sizes
        for window_size in (min_hours..=max_hours).step_by(6) {
            let mut scores = Vec::new();
            
            // Sample windows at different positions
            let step = window_size / 2;  // 50% overlap for testing
            let mut start = 0;
            
            while start + window_size <= df.height() {
                let window = df.slice(start as i64, window_size);
                let metrics = self.analyze_window(&window)?;
                
                if metrics.overall_score >= 0.1 {  // Use default threshold
                    scores.push(metrics.overall_score);
                }
                
                start += step;
            }
            
            if !scores.is_empty() {
                let avg_score = scores.iter().sum::<f64>() / scores.len() as f64;
                let viable_windows = scores.len();
                
                // Prefer smaller windows if quality is similar
                let size_penalty = (window_size as f64 / max_hours as f64) * 0.1;
                let adjusted_score = avg_score * (1.0 - size_penalty);
                
                window_scores.insert(window_size, (adjusted_score, viable_windows));
            }
        }
        
        // Find best window size
        let best_size = window_scores.iter()
            .max_by(|(_, (score1, _)), (_, (score2, _))| {
                score1.partial_cmp(score2).unwrap()
            })
            .map(|(&size, _)| size)
            .unwrap_or(24);  // Default to 24 hours
        
        Ok(best_size)
    }
}

/// Adaptive window configuration based on data quality
#[derive(Debug, Clone)]
pub struct AdaptiveWindowConfig {
    pub window_size: usize,
    pub overlap_ratio: f64,
    pub quality_threshold: f64,
    pub min_sensors: usize,
}

impl AdaptiveWindowConfig {
    pub fn from_quality_metrics(metrics: &DataQualityMetrics) -> Self {
        // Adjust window size based on data quality
        let window_size = if metrics.coverage > 0.8 {
            12  // High coverage: smaller windows
        } else if metrics.coverage > 0.5 {
            24  // Medium coverage: daily windows
        } else {
            48  // Low coverage: larger windows
        };
        
        // Adjust overlap based on continuity
        let overlap_ratio = if metrics.continuity > 0.8 {
            0.25  // Good continuity: less overlap needed
        } else if metrics.continuity > 0.5 {
            0.50  // Medium continuity: moderate overlap
        } else {
            0.75  // Poor continuity: high overlap
        };
        
        // Quality threshold based on overall score
        let quality_threshold = metrics.overall_score * 0.8;
        
        // Minimum sensors based on availability
        let min_sensors = ((metrics.sensor_availability * 5.0).ceil() as usize).max(2);
        
        Self {
            window_size,
            overlap_ratio,
            quality_threshold,
            min_sensors,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_data_quality_analyzer() {
        // Create test data
        let df = df![
            "temp_mean" => vec![Some(20.0), Some(21.0), None, Some(22.0), Some(23.0)],
            "co2_mean" => vec![Some(400.0), Some(410.0), Some(420.0), None, Some(430.0)],
            "humidity_mean" => vec![Some(60.0), None, None, None, Some(65.0)],
        ].unwrap();
        
        let analyzer = DataQualityAnalyzer::new(0.3, 0.5);
        let metrics = analyzer.analyze_window(&df).unwrap();
        
        assert!(metrics.coverage > 0.0);
        assert!(metrics.overall_score > 0.0);
    }
}