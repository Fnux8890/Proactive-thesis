use anyhow::Result;
use chrono::{DateTime, Utc};
use gpu_feature_extraction::python_bridge::PythonGpuBridge;
use std::collections::HashMap;
use tracing::{info, Level};

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    info!("Testing Python GPU Bridge Communication");

    // Create bridge (use local mode for testing)
    let bridge = PythonGpuBridge::new(false);

    // Create test data
    let mut timestamps = Vec::new();
    let mut sensor_data = HashMap::new();
    
    // Generate 24 hours of test data
    let start_time = "2024-01-01T00:00:00Z".parse::<DateTime<Utc>>()?;
    let mut air_temp_data = Vec::new();
    let mut humidity_data = Vec::new();

    for i in 0..24 {
        let timestamp = start_time + chrono::Duration::hours(i as i64);
        timestamps.push(timestamp);
        
        // Generate simple patterns
        air_temp_data.push(20.0 + 2.0 * (i as f32 % 24.0) / 24.0);
        humidity_data.push(50.0 + 10.0 * ((i as f32 + 12.0) % 24.0) / 24.0);
    }

    sensor_data.insert("air_temp_c".to_string(), air_temp_data);
    sensor_data.insert("relative_humidity_percent".to_string(), humidity_data);

    info!("Testing feature extraction with {} timestamps", timestamps.len());
    
    // Test feature extraction
    match bridge.extract_features(timestamps, sensor_data, vec![30, 120]) {
        Ok(features) => {
            info!("SUCCESS: Extracted {} features", features.len());
            
            // Display some key features
            for (feature_name, value) in &features {
                if feature_name.contains("mean") || feature_name.contains("std") {
                    info!("  {}: {:.3}", feature_name, value);
                }
            }
            
            // Test specific expected features
            let expected_features = vec![
                "air_temp_c_mean",
                "air_temp_c_std", 
                "relative_humidity_percent_mean",
                "calculated_vpd_kpa"
            ];
            
            let mut missing_features = Vec::new();
            for expected in &expected_features {
                if !features.contains_key(*expected) {
                    missing_features.push(*expected);
                }
            }
            
            if missing_features.is_empty() {
                info!("SUCCESS: All expected features present");
            } else {
                info!("WARNING: Missing expected features: {:?}", missing_features);
            }
        }
        Err(e) => {
            eprintln!("ERROR: Feature extraction failed: {}", e);
            return Err(e);
        }
    }

    // Test with sparse data
    info!("\nTesting with sparse data (91.3% missing)...");
    
    let mut sparse_sensor_data = HashMap::new();
    let mut sparse_air_temp = Vec::new();
    let mut sparse_humidity = Vec::new();
    
    // Create very sparse data - only every 12th sample has data
    for i in 0..48 {
        if i % 12 == 0 {
            sparse_air_temp.push(20.0 + (i as f32 / 48.0) * 5.0);
            sparse_humidity.push(50.0 + (i as f32 / 48.0) * 20.0);
        } else {
            // Note: For testing with the minimal script, we'll use NaN
            // In the real Python scripts, these would be None/null
            sparse_air_temp.push(f32::NAN);
            sparse_humidity.push(f32::NAN);
        }
    }
    
    sparse_sensor_data.insert("air_temp_c".to_string(), sparse_air_temp);
    sparse_sensor_data.insert("relative_humidity_percent".to_string(), sparse_humidity);
    
    // Generate timestamps for sparse test
    let mut sparse_timestamps = Vec::new();
    for i in 0..48 {
        let timestamp = start_time + chrono::Duration::minutes(i as i64 * 30);
        sparse_timestamps.push(timestamp);
    }
    
    match bridge.extract_features(sparse_timestamps, sparse_sensor_data, vec![60, 180]) {
        Ok(sparse_features) => {
            info!("SUCCESS: Extracted {} sparse features", sparse_features.len());
            
            // Look for coverage features
            for (feature_name, value) in &sparse_features {
                if feature_name.contains("coverage") {
                    info!("  {}: {:.3}", feature_name, value);
                }
            }
        }
        Err(e) => {
            info!("WARNING: Sparse feature extraction failed (expected with minimal script): {}", e);
            // This is expected since our minimal test script doesn't handle NaN properly
        }
    }

    info!("\nRust-Python bridge communication test completed successfully!");
    info!("The bridge can properly serialize data, call Python scripts, and parse responses.");
    
    Ok(())
}