use crate::config::FileConfig;
use crate::data_models::ParsedRecord;
use crate::errors::ParseError;
use std::path::Path;
use std::fs::File;
use std::io::BufReader;
use serde::Deserialize;
use chrono::{Utc, TimeZone};

// --- Replicate set_field! focusing on fields from stream_map --- 
// NOTE: This should eventually be moved to a shared utils module.
macro_rules! set_field {
    // Match f64
    ($record:expr, $field_name:expr, $value:expr; f64) => {
        match $field_name {
            "temperature_c" => $record.air_temp_c = Some($value), // Map to ParsedRecord field
            "sun_radiation_w_m2" => $record.radiation_w_m2 = Some($value), // Map to ParsedRecord field
            _ => eprintln!("WARN (StreamJSON): Attempted to set unhandled f64 field '{}' in ParsedRecord.", $field_name),
        }
    };
     // Fallback for unhandled types
     ($record:expr, $field_name:expr, $value:expr; $type:ty) => {
         eprintln!("WARN (StreamJSON): set_field! macro called for unhandled type '{}' for field '{}'.", stringify!($type), $field_name);
     };
}

// Temporary struct for deserializing the AarslevStreamJSON structure
#[derive(Deserialize, Debug)]
struct StreamData {
    uuid: String,
    #[serde(rename = "Readings")]
    readings: Vec<[serde_json::Number; 2]>,
}

pub fn parse_aarslev_stream_json(config: &FileConfig, file_path: &Path) -> Result<Vec<ParsedRecord>, ParseError> {
    println!("DEBUG: Entering parse_aarslev_stream_json for {}", file_path.display());
    let file = File::open(file_path).map_err(|e| ParseError::IoError { path: file_path.to_path_buf(), source: e })?;
    let reader = BufReader::new(file);

    // Deserialize the entire JSON into a Vec of StreamData objects
    println!("DEBUG (StreamJSON): Attempting to deserialize {}", file_path.display());
    let stream_data_vec: Vec<StreamData> = serde_json::from_reader(reader).map_err(|e| ParseError::JsonParseError {
        path: file_path.to_path_buf(),
        source: e,
    })?;
    println!("DEBUG (StreamJSON): Successfully deserialized {} stream objects from {}", stream_data_vec.len(), file_path.display());

    let mut parsed_records = Vec::new();

    // Get config details needed for parsing
    println!("DEBUG (StreamJSON): Retrieving stream_map config from file config.");
    let stream_map = config.stream_map.as_ref().ok_or_else(|| ParseError::ConfigError {
        path: file_path.to_path_buf(),
        field: "stream_map".to_string(),
        message: "stream_map is required for AarslevStreamJSON format.".to_string(),
    })?;
     // Note: timestamp_info config isn't directly used here as structure is fixed (unix_ms at index 0)

    // Iterate through each stream object in the deserialized vector
    println!("DEBUG (StreamJSON): Iterating through {} stream objects for {}", stream_data_vec.len(), file_path.display());
    for stream_data in stream_data_vec {
        let uuid = &stream_data.uuid;
        println!("DEBUG (StreamJSON): Processing stream UUID: {}", uuid);

        // Find the mapping for this specific UUID in the config
        if let Some(mapping) = stream_map.get(uuid) {
            let target_field = &mapping.target;
            let data_type = &mapping.data_type;
            println!("DEBUG (StreamJSON): Found config for UUID {}: target='{}', type='{}'", uuid, target_field, data_type);

            // Iterate through the readings for this stream
            println!("DEBUG (StreamJSON): Iterating through {} readings for UUID {}", stream_data.readings.len(), uuid);
            for (reading_index, reading) in stream_data.readings.iter().enumerate() {
                let _file_row_num = reading_index + 1; // Use reading index as pseudo row number
                let mut record = ParsedRecord::default();
                let mut row_has_error = false;

                record.source_file = Some(file_path.to_string_lossy().into_owned());
                record.source_system = config.source_system.clone();
                record.format_type = Some(config.format_type.clone());
                record.uuid = Some(uuid.clone()); // Store the UUID

                // Extract timestamp (index 0, expected i64 for Unix ms)
                let ts_num = &reading[0];
                // Try parsing as f64 first, then cast to i64, to handle numbers like `123.0`
                if let Some(ts_f64) = ts_num.as_f64() {
                    let ts_ms = ts_f64 as i64; // Truncate potential decimal part
                     match Utc.timestamp_millis_opt(ts_ms).single() {
                         Some(ts_utc) => {
                             record.timestamp_utc = Some(ts_utc);
                             // println!("DEBUG (StreamJSON): Parsed timestamp {} -> {:?}", ts_ms, ts_utc); // Optional: Very verbose
                         }
                         None => {
                             eprintln!(
                                "ERROR (StreamJSON): Unix ms timestamp '{}' out of range in {} for UUID {}",
                                ts_ms, file_path.display(), uuid
                             );
                             row_has_error = true;
                         }
                     }
                } else {
                    eprintln!(
                        "ERROR (StreamJSON): Failed to parse timestamp '{:?}' as f64/i64 (Unix ms) in {} for UUID {}",
                        ts_num, file_path.display(), uuid
                    );
                    row_has_error = true;
                }

                // Extract value (index 1)
                let val_num = &reading[1];

                // Parse value based on configured data_type for this UUID
                match data_type.as_str() {
                    "float" => {
                         if let Some(val_f64) = val_num.as_f64() {
                            set_field!(record, target_field.as_str(), val_f64; f64);
                            // println!("DEBUG (StreamJSON): Parsed value {} -> {}", val_num, val_f64); // Optional: Very verbose
                        } else {
                             eprintln!(
                                "ERROR (StreamJSON): Failed to parse value '{:?}' as f64 for field '{}' in {} for UUID {}",
                                val_num, target_field, file_path.display(), uuid
                            );
                             row_has_error = true;
                         }
                    }
                    // Add cases for "int", "boolean", "string" if they appear in stream_map
                    unknown_type => {
                         eprintln!("WARN (StreamJSON): Unsupported data_type '{}' in stream_map for UUID {}. Cannot parse value.", unknown_type, uuid);
                         row_has_error = true;
                    }
                }

                if !row_has_error {
                    parsed_records.push(record);
                } else {
                     eprintln!("INFO (StreamJSON): Skipping reading index {} in {} for UUID {} due to parsing errors.", reading_index, file_path.display(), uuid);
                }
            }
        } else {
             eprintln!("WARN (StreamJSON): UUID '{}' found in file {} but not defined in config.stream_map. Skipping stream.", uuid, file_path.display());
        }
    }

    println!("DEBUG: Finished parse_aarslev_stream_json for {}. Parsed {} records.", file_path.display(), parsed_records.len());
    Ok(parsed_records)
}

// Removed placeholder for parse_aarslev_celle_json as the format is not defined

// Placeholder for AarslevCelleJSON (e.g., celle*/...csv.json)
pub fn parse_aarslev_celle_json(_config: &FileConfig, file_path: &Path) -> Result<Vec<ParsedRecord>, ParseError> {
     println!(
        "WARN: Parser for AarslevCelleJSON format not fully implemented yet. Skipping file: {}",
        file_path.display()
    );
     Ok(Vec::new())
     // TODO: Implement parsing logic - this format seems different, might be column-oriented.
     // 1. Define appropriate structs for serde deserialization.
     // 2. Read and parse JSON.
     // 3. Iterate through data points, potentially needing to align columns by timestamp or index.
     // 4. Create ParsedRecord instances.
} 