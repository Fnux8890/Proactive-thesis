//! Module for data validation logic.

use crate::models::{AarslevCelleOutputRecord, SensorValue}; // Import necessary structs/enums
use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// Validates a parsed Aarslev Celle Output record.
///
/// Checks:
/// - Timestamp is present.
/// - TODO: Sensor data ranges and required fields.
///
/// Returns Ok(()) if valid, otherwise Err(String) with the validation error.
pub fn validate_record(record: &AarslevCelleOutputRecord) -> Result<(), String> {
    // 1. Validate Timestamp
    validate_timestamp(&record.timestamp)?;

    // 2. Validate Sensor Data (Placeholder)
    validate_sensor_data(&record.sensor_data)?;

    // Add more validation checks here...

    Ok(())
}

/// Checks if the timestamp Option contains a valid DateTime.
pub fn validate_timestamp(timestamp: &Option<DateTime<Utc>>) -> Result<(), String> {
    if timestamp.is_none() {
        Err("Missing or invalid timestamp".to_string())
    } else {
        // TODO: Add more sophisticated checks? (e.g., ensure it's within a reasonable range)
        Ok(())
    }
}

/// Placeholder for validating sensor data ranges, units, etc.
pub fn validate_sensor_data(sensor_data: &HashMap<String, SensorValue>) -> Result<(), String> {
    for (key, value) in sensor_data {
        let lower_key = key.to_lowercase();
        
        if let SensorValue::Number(num_val) = value {
            // Temperature Checks (assuming keys contain "temp")
            if lower_key.contains("temp") {
                if !(-20.0..=50.0).contains(num_val) {
                    return Err(format!("Validation Error: Temperature '{}' ({}) out of range (-20 to 50)", key, num_val));
                }
            }
            // Humidity Checks (assuming keys contain "hum" or "rf")
            else if lower_key.contains("hum") || lower_key.contains("rf") { // RF often used for Relative Humidity
                if !(0.0..=100.0).contains(num_val) {
                    return Err(format!("Validation Error: Humidity '{}' ({}) out of range (0 to 100)", key, num_val));
                }
            }
            // CO2 Checks (assuming keys contain "co2")
            else if lower_key.contains("co2") {
                if !(0.0..=5000.0).contains(num_val) {
                    return Err(format!("Validation Error: CO2 '{}' ({}) out of range (0 to 5000)", key, num_val));
                }
            }
            // Add other sensor checks here (e.g., radiation, light levels)
        } 
        // else: Handle SensorValue::Text or SensorValue::Empty if needed 
        // (e.g., check if required fields are empty)
    }
    Ok(())
}
