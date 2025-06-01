// Sparse-aware feature extraction for greenhouse data with 91.3% missing values
// CPU-bound features using Rust for efficiency

use anyhow::Result;
use chrono::{DateTime, Utc, Duration, Timelike, Datelike};
use std::collections::HashMap;
use rayon::prelude::*;

/// Features specifically designed for sparse time series data
#[derive(Debug, Clone)]
pub struct SparseFeatures {
    // Coverage metrics
    pub coverage_ratio: f32,
    pub longest_gap_hours: f32,
    pub mean_gap_hours: f32,
    pub data_points_count: u32,
    
    // Sparse statistics (only using available points)
    pub sparse_mean: Option<f32>,
    pub sparse_std: Option<f32>,
    pub sparse_min: Option<f32>,
    pub sparse_max: Option<f32>,
    pub sparse_range: Option<f32>,
    
    // Event-based features
    pub change_count: u32,
    pub large_change_count: u32,  // Changes > 2 std
    pub zero_crossings: u32,
    pub sign_changes: u32,
    
    // Extremes (useful even with sparse data)
    pub extreme_high_count: u32,  // > 95th percentile
    pub extreme_low_count: u32,   // < 5th percentile
    pub extreme_duration_hours: f32,
    
    // Presence patterns
    pub active_hours: Vec<bool>,  // 24-hour pattern of data availability
    pub weekend_vs_weekday_coverage: f32,
    pub night_vs_day_coverage: f32,
}

/// Domain-specific sparse features for greenhouse control
#[derive(Debug, Clone)]
pub struct GreenhouseSparseFeatures {
    // Control action features (binary sensors)
    pub lamp_on_hours: f32,
    pub lamp_switches: u32,
    pub heating_active_hours: f32,
    pub ventilation_active_hours: f32,
    pub curtain_movements: u32,
    
    // Environmental accumulation (when data available)
    pub gdd_accumulated: Option<f32>,      // Growing degree days
    pub dli_accumulated: Option<f32>,      // Daily light integral
    pub vpd_stress_hours: Option<f32>,     // Hours outside optimal VPD
    pub temp_stress_hours: Option<f32>,    // Hours outside optimal temp
    
    // Energy indicators (sparse-aware)
    pub peak_hour_activity: f32,     // Activity during expensive hours
    pub night_heating_hours: f32,    // Heating when it's cold outside
    pub lamp_efficiency_proxy: f32,  // Light provided vs energy used
}

/// Extract sparse-aware features from a time series
pub fn extract_sparse_features(
    timestamps: &[DateTime<Utc>],
    values: &[Option<f32>],
    sensor_name: &str,
) -> SparseFeatures {
    let n = timestamps.len();
    
    // Coverage metrics
    let available_points: Vec<(usize, f32)> = values.iter()
        .enumerate()
        .filter_map(|(i, v)| v.map(|val| (i, val)))
        .collect();
    
    let coverage_ratio = available_points.len() as f32 / n as f32;
    let data_points_count = available_points.len() as u32;
    
    // Gap analysis
    let (longest_gap_hours, mean_gap_hours) = if available_points.len() > 1 {
        let gaps: Vec<f32> = available_points.windows(2)
            .map(|w| {
                let gap = timestamps[w[1].0].signed_duration_since(timestamps[w[0].0]);
                gap.num_hours() as f32
            })
            .filter(|&g| g > 0.0)
            .collect();
        
        let longest = gaps.iter().cloned().fold(0.0, f32::max);
        let mean = if !gaps.is_empty() {
            gaps.iter().sum::<f32>() / gaps.len() as f32
        } else {
            0.0
        };
        (longest, mean)
    } else {
        (0.0, 0.0)
    };
    
    // Sparse statistics
    let (sparse_mean, sparse_std, sparse_min, sparse_max, sparse_range) = if !available_points.is_empty() {
        let values: Vec<f32> = available_points.iter().map(|(_, v)| *v).collect();
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
        let std = variance.sqrt();
        let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        
        (Some(mean), Some(std), Some(min), Some(max), Some(max - min))
    } else {
        (None, None, None, None, None)
    };
    
    // Event-based features
    let (change_count, large_change_count, zero_crossings, sign_changes) = if available_points.len() > 1 {
        let mut changes = 0u32;
        let mut large_changes = 0u32;
        let mut zero_cross = 0u32;
        let mut sign_chg = 0u32;
        
        let threshold = sparse_std.unwrap_or(1.0) * 2.0;
        let mean = sparse_mean.unwrap_or(0.0);
        
        for window in available_points.windows(2) {
            let delta = (window[1].1 - window[0].1).abs();
            if delta > 0.0 {
                changes += 1;
                if delta > threshold {
                    large_changes += 1;
                }
            }
            
            // Zero crossings (around mean)
            if (window[0].1 - mean) * (window[1].1 - mean) < 0.0 {
                zero_cross += 1;
            }
            
            // Sign changes
            if window[0].1.signum() != window[1].1.signum() {
                sign_chg += 1;
            }
        }
        
        (changes, large_changes, zero_cross, sign_chg)
    } else {
        (0, 0, 0, 0)
    };
    
    // Extremes (using percentiles from available data)
    let (extreme_high_count, extreme_low_count, extreme_duration_hours) = if available_points.len() > 10 {
        let mut sorted_values: Vec<f32> = available_points.iter().map(|(_, v)| *v).collect();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let p95 = sorted_values[(sorted_values.len() as f32 * 0.95) as usize];
        let p5 = sorted_values[(sorted_values.len() as f32 * 0.05) as usize];
        
        let high_count = available_points.iter().filter(|(_, v)| *v > p95).count() as u32;
        let low_count = available_points.iter().filter(|(_, v)| *v < p5).count() as u32;
        
        // Estimate duration (rough approximation based on coverage)
        let extreme_ratio = (high_count + low_count) as f32 / available_points.len() as f32;
        let total_hours = timestamps.last()
            .and_then(|end| timestamps.first().map(|start| 
                end.signed_duration_since(*start).num_hours() as f32
            ))
            .unwrap_or(0.0);
        let extreme_hours = total_hours * extreme_ratio * coverage_ratio;
        
        (high_count, low_count, extreme_hours)
    } else {
        (0, 0, 0.0)
    };
    
    // Presence patterns (24-hour profile)
    let mut active_hours = vec![false; 24];
    for (idx, _) in &available_points {
        let hour = timestamps[*idx].hour() as usize;
        active_hours[hour] = true;
    }
    
    // Weekend vs weekday coverage
    let mut weekend_points = Vec::new();
    let mut weekday_points = Vec::new();
    
    for point in available_points.iter() {
        let weekday = timestamps[point.0].weekday();
        if weekday == chrono::Weekday::Sat || weekday == chrono::Weekday::Sun {
            weekend_points.push(point);
        } else {
            weekday_points.push(point);
        }
    }
    
    let weekend_ratio = weekend_points.len() as f32 / weekend_points.len().max(1) as f32;
    let weekday_ratio = weekday_points.len() as f32 / weekday_points.len().max(1) as f32;
    let weekend_vs_weekday_coverage = if weekday_ratio > 0.0 {
        weekend_ratio / weekday_ratio
    } else {
        0.0
    };
    
    // Night vs day coverage (6am-6pm as day)
    let mut night_points = Vec::new();
    let mut day_points = Vec::new();
    
    for point in available_points.iter() {
        let hour = timestamps[point.0].hour();
        if hour < 6 || hour >= 18 {
            night_points.push(point);
        } else {
            day_points.push(point);
        }
    }
    
    let night_ratio = night_points.len() as f32 / night_points.len().max(1) as f32;
    let day_ratio = day_points.len() as f32 / day_points.len().max(1) as f32;
    let night_vs_day_coverage = if day_ratio > 0.0 {
        night_ratio / day_ratio
    } else {
        0.0
    };
    
    SparseFeatures {
        coverage_ratio,
        longest_gap_hours,
        mean_gap_hours,
        data_points_count,
        sparse_mean,
        sparse_std,
        sparse_min,
        sparse_max,
        sparse_range,
        change_count,
        large_change_count,
        zero_crossings,
        sign_changes,
        extreme_high_count,
        extreme_low_count,
        extreme_duration_hours,
        active_hours,
        weekend_vs_weekday_coverage,
        night_vs_day_coverage,
    }
}

/// Extract greenhouse-specific features from multiple sensors
pub fn extract_greenhouse_sparse_features(
    sensor_data: &HashMap<String, (Vec<DateTime<Utc>>, Vec<Option<f32>>)>,
    energy_prices: Option<&[(DateTime<Utc>, f32)]>,
) -> Result<GreenhouseSparseFeatures> {
    // Helper to get sensor data safely
    let get_sensor = |name: &str| -> Option<&(Vec<DateTime<Utc>>, Vec<Option<f32>>)> {
        sensor_data.get(name)
    };
    
    // Lamp features (binary on/off patterns)
    let lamp_features = [
        "lamp_grp1_no3_status",
        "lamp_grp1_no4_status",
        "lamp_grp2_no3_status",
        "lamp_grp2_no4_status",
    ];
    
    let mut lamp_on_hours = 0.0;
    let mut lamp_switches = 0u32;
    
    for lamp in &lamp_features {
        if let Some((timestamps, values)) = get_sensor(lamp) {
            let available: Vec<(usize, bool)> = values.iter()
                .enumerate()
                .filter_map(|(i, v)| v.map(|val| (i, val > 0.5)))
                .collect();
            
            if available.len() > 1 {
                // Count switches
                for window in available.windows(2) {
                    if window[0].1 != window[1].1 {
                        lamp_switches += 1;
                    }
                }
                
                // Estimate on hours
                let on_points = available.iter().filter(|(_, on)| *on).count();
                let coverage = available.len() as f32 / values.len() as f32;
                let total_hours = timestamps.last()
                    .and_then(|end| timestamps.first().map(|start| 
                        end.signed_duration_since(*start).num_hours() as f32
                    ))
                    .unwrap_or(0.0);
                lamp_on_hours += (on_points as f32 / available.len() as f32) * total_hours * coverage;
            }
        }
    }
    
    // Heating features
    let mut heating_active_hours = 0.0;
    if let Some((timestamps, values)) = get_sensor("heating_setpoint_c") {
        if let Some((_, temps)) = get_sensor("air_temp_c") {
            let paired: Vec<(f32, f32)> = values.iter()
                .zip(temps.iter())
                .filter_map(|(sp, t)| match (sp, t) {
                    (Some(setpoint), Some(temp)) => Some((*setpoint, *temp)),
                    _ => None,
                })
                .collect();
            
            let heating_points = paired.iter().filter(|(sp, t)| t < sp).count();
            let coverage = paired.len() as f32 / values.len() as f32;
            let total_hours = timestamps.last()
                .and_then(|end| timestamps.first().map(|start| 
                    end.signed_duration_since(*start).num_hours() as f32
                ))
                .unwrap_or(0.0);
            heating_active_hours = (heating_points as f32 / paired.len().max(1) as f32) * total_hours * coverage;
        }
    }
    
    // Ventilation features
    let mut ventilation_active_hours = 0.0;
    let vent_sensors = ["vent_lee_afd3_percent", "vent_wind_afd3_percent"];
    for vent in &vent_sensors {
        if let Some((timestamps, values)) = get_sensor(vent) {
            let open_points = values.iter()
                .filter_map(|v| *v)
                .filter(|&v| v > 5.0)  // More than 5% open
                .count();
            let available_points = values.iter().filter(|v| v.is_some()).count();
            let coverage = available_points as f32 / values.len() as f32;
            let total_hours = timestamps.last()
                .and_then(|end| timestamps.first().map(|start| 
                    end.signed_duration_since(*start).num_hours() as f32
                ))
                .unwrap_or(0.0);
            ventilation_active_hours += (open_points as f32 / available_points.max(1) as f32) * total_hours * coverage / 2.0;
        }
    }
    
    // Curtain movements
    let mut curtain_movements = 0u32;
    for i in 1..=4 {
        let sensor = format!("curtain_{}_percent", i);
        if let Some((_, values)) = get_sensor(&sensor) {
            let available: Vec<f32> = values.iter().filter_map(|v| *v).collect();
            if available.len() > 1 {
                for window in available.windows(2) {
                    if (window[1] - window[0]).abs() > 5.0 {  // Movement > 5%
                        curtain_movements += 1;
                    }
                }
            }
        }
    }
    
    // Environmental accumulation
    let gdd_accumulated = if let Some((timestamps, values)) = get_sensor("air_temp_c") {
        let base_temp = 10.0;  // Base temperature for Kalanchoe
        let daily_values: Vec<(DateTime<Utc>, Vec<f32>)> = group_by_day(timestamps, values);
        
        let gdd_sum: f32 = daily_values.iter()
            .map(|(_, day_temps)| {
                if !day_temps.is_empty() {
                    let avg_temp = day_temps.iter().sum::<f32>() / day_temps.len() as f32;
                    (avg_temp - base_temp).max(0.0)
                } else {
                    0.0
                }
            })
            .sum();
        
        Some(gdd_sum)
    } else {
        None
    };
    
    // DLI accumulation
    let dli_accumulated = if let Some((_, values)) = get_sensor("dli_sum") {
        values.iter()
            .filter_map(|v| *v)
            .fold(0.0, |acc, v| acc + v)
            .into()
    } else {
        None
    };
    
    // VPD stress hours
    let vpd_stress_hours = if let Some((timestamps, values)) = get_sensor("vpd_hpa") {
        let stress_points = values.iter()
            .filter_map(|v| *v)
            .filter(|&v| v < 0.4 || v > 1.6)  // Outside optimal range
            .count();
        let available_points = values.iter().filter(|v| v.is_some()).count();
        let coverage = available_points as f32 / values.len() as f32;
        let total_hours = timestamps.last()
            .and_then(|end| timestamps.first().map(|start| 
                end.signed_duration_since(*start).num_hours() as f32
            ))
            .unwrap_or(0.0);
        Some((stress_points as f32 / available_points.max(1) as f32) * total_hours * coverage)
    } else {
        None
    };
    
    // Temperature stress hours
    let temp_stress_hours = if let Some((timestamps, values)) = get_sensor("air_temp_c") {
        let stress_points = values.iter()
            .filter_map(|v| *v)
            .filter(|&v| v < 15.0 || v > 30.0)  // Outside optimal range
            .count();
        let available_points = values.iter().filter(|v| v.is_some()).count();
        let coverage = available_points as f32 / values.len() as f32;
        let total_hours = timestamps.last()
            .and_then(|end| timestamps.first().map(|start| 
                end.signed_duration_since(*start).num_hours() as f32
            ))
            .unwrap_or(0.0);
        Some((stress_points as f32 / available_points.max(1) as f32) * total_hours * coverage)
    } else {
        None
    };
    
    // Energy indicators
    let (peak_hour_activity, night_heating_hours) = if let Some(prices) = energy_prices {
        // Identify peak hours (top 25% of prices)
        let mut sorted_prices: Vec<f32> = prices.iter().map(|(_, p)| *p).collect();
        sorted_prices.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p75_price = sorted_prices[(sorted_prices.len() as f32 * 0.75) as usize];
        
        let peak_hours: Vec<u32> = prices.iter()
            .filter(|(_, p)| *p >= p75_price)
            .map(|(t, _)| t.hour())
            .collect();
        
        // Count lamp activity during peak hours
        let mut peak_activity = 0.0;
        for lamp in &lamp_features {
            if let Some((timestamps, values)) = get_sensor(lamp) {
                let peak_on = values.iter()
                    .zip(timestamps.iter())
                    .filter(|(v, t)| v.is_some() && peak_hours.contains(&t.hour()))
                    .filter(|(v, _)| v.unwrap() > 0.5)
                    .count();
                let total_peak = values.iter()
                    .zip(timestamps.iter())
                    .filter(|(v, t)| v.is_some() && peak_hours.contains(&t.hour()))
                    .count();
                peak_activity += peak_on as f32 / total_peak.max(1) as f32 / 4.0;  // Average over 4 lamps
            }
        }
        
        // Night heating (heating when it's cold outside - 10pm to 6am)
        let night_heating = if let Some((timestamps, values)) = get_sensor("heating_setpoint_c") {
            if let Some((_, temps)) = get_sensor("air_temp_c") {
                let night_heating_points = values.iter()
                    .zip(temps.iter())
                    .zip(timestamps.iter())
                    .filter(|((sp, t), time)| {
                        let hour = time.hour();
                        sp.is_some() && t.is_some() && (hour >= 22 || hour < 6)
                    })
                    .filter(|((sp, t), _)| t.unwrap() < sp.unwrap())
                    .count();
                
                let total_night = timestamps.iter()
                    .filter(|t| {
                        let hour = t.hour();
                        hour >= 22 || hour < 6
                    })
                    .count();
                
                let coverage = total_night as f32 / timestamps.len() as f32;
                let total_hours = timestamps.last()
                    .and_then(|end| timestamps.first().map(|start| 
                        end.signed_duration_since(*start).num_hours() as f32
                    ))
                    .unwrap_or(0.0);
                (night_heating_points as f32 / total_night.max(1) as f32) * total_hours * coverage
            } else {
                0.0
            }
        } else {
            0.0
        };
        
        (peak_activity, night_heating)
    } else {
        (0.0, 0.0)
    };
    
    // Lamp efficiency proxy (DLI achieved per hour of lamp operation)
    let lamp_efficiency_proxy = if lamp_on_hours > 0.0 && dli_accumulated.is_some() {
        dli_accumulated.unwrap() / lamp_on_hours
    } else {
        0.0
    };
    
    Ok(GreenhouseSparseFeatures {
        lamp_on_hours,
        lamp_switches,
        heating_active_hours,
        ventilation_active_hours,
        curtain_movements,
        gdd_accumulated,
        dli_accumulated,
        vpd_stress_hours,
        temp_stress_hours,
        peak_hour_activity,
        night_heating_hours,
        lamp_efficiency_proxy,
    })
}

/// Group sparse data by day
fn group_by_day(
    timestamps: &[DateTime<Utc>],
    values: &[Option<f32>],
) -> Vec<(DateTime<Utc>, Vec<f32>)> {
    let mut daily_groups: HashMap<DateTime<Utc>, Vec<f32>> = HashMap::new();
    
    for (ts, val) in timestamps.iter().zip(values.iter()) {
        if let Some(v) = val {
            let day = ts.date_naive().and_hms_opt(0, 0, 0).unwrap().and_utc();
            daily_groups.entry(day).or_insert_with(Vec::new).push(*v);
        }
    }
    
    let mut sorted_days: Vec<_> = daily_groups.into_iter().collect();
    sorted_days.sort_by_key(|(day, _)| *day);
    sorted_days
}

/// Parallel feature extraction for multiple sensors
pub fn extract_all_sparse_features(
    sensor_data: HashMap<String, (Vec<DateTime<Utc>>, Vec<Option<f32>>)>,
    _energy_prices: Option<Vec<(DateTime<Utc>, f32)>>,
) -> Result<HashMap<String, SparseFeatures>> {
    // Extract sparse features for each sensor in parallel
    let sensor_features: HashMap<String, SparseFeatures> = sensor_data
        .par_iter()
        .map(|(name, (timestamps, values))| {
            let features = extract_sparse_features(timestamps, values, name);
            (name.clone(), features)
        })
        .collect();
    
    Ok(sensor_features)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sparse_features_empty_data() {
        let timestamps = vec![];
        let values = vec![];
        let features = extract_sparse_features(&timestamps, &values, "test");
        
        assert_eq!(features.coverage_ratio, 0.0);
        assert_eq!(features.data_points_count, 0);
        assert!(features.sparse_mean.is_none());
    }
    
    #[test]
    fn test_sparse_features_full_coverage() {
        let start = Utc::now();
        let timestamps: Vec<_> = (0..10).map(|i| start + Duration::hours(i)).collect();
        let values: Vec<_> = (0..10).map(|i| Some(i as f32)).collect();
        
        let features = extract_sparse_features(&timestamps, &values, "test");
        
        assert_eq!(features.coverage_ratio, 1.0);
        assert_eq!(features.data_points_count, 10);
        assert!(features.sparse_mean.is_some());
        assert_eq!(features.sparse_mean.unwrap(), 4.5);
    }
    
    #[test]
    fn test_sparse_features_with_gaps() {
        let start = Utc::now();
        let timestamps: Vec<_> = (0..10).map(|i| start + Duration::hours(i)).collect();
        let values = vec![
            Some(1.0), None, None, Some(4.0), None,
            None, Some(7.0), None, None, Some(10.0),
        ];
        
        let features = extract_sparse_features(&timestamps, &values, "test");
        
        assert_eq!(features.coverage_ratio, 0.4);
        assert_eq!(features.data_points_count, 4);
        assert_eq!(features.sparse_mean, Some(5.5));
        assert!(features.longest_gap_hours > 0.0);
    }
}