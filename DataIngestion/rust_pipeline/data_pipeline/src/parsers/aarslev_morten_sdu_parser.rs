// Placeholder for AarslevMortenSDU specific parsing logic

use crate::config::FileConfig;
use crate::data_models::ParsedRecord;
use crate::errors::ParseError;
use chrono::{DateTime, Utc, NaiveDateTime, TimeZone};
use csv::{ReaderBuilder, StringRecord};
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

// ADDED BACK: set_field! macro
macro_rules! set_field {
    // Match f64
    ($record:expr, $field_name:expr, $value:expr; f64) => {
        match $field_name {
            "air_temp_c" => $record.air_temp_c = Some($value),
            "air_temp_middle_c" => $record.air_temp_middle_c = Some($value),
            "outside_temp_c" => $record.outside_temp_c = Some($value),
            "relative_humidity_percent" => $record.relative_humidity_percent = Some($value),
            "humidity_deficit_g_m3" => $record.humidity_deficit_g_m3 = Some($value),
            "radiation_w_m2" => $record.radiation_w_m2 = Some($value),
            "light_intensity_lux" => $record.light_intensity_lux = Some($value),
            "light_intensity_umol" => $record.light_intensity_umol = Some($value),
            "outside_light_w_m2" => $record.outside_light_w_m2 = Some($value),
            "co2_measured_ppm" => $record.co2_measured_ppm = Some($value),
            "co2_required_ppm" => $record.co2_required_ppm = Some($value),
            "co2_dosing_status" => $record.co2_dosing_status = Some($value),
            "co2_status" => $record.co2_status = Some($value),
            "vent_pos_1_percent" => $record.vent_pos_1_percent = Some($value),
            "vent_pos_2_percent" => $record.vent_pos_2_percent = Some($value),
            "vent_lee_afd3_percent" => $record.vent_lee_afd3_percent = Some($value),
            "vent_wind_afd3_percent" => $record.vent_wind_afd3_percent = Some($value),
            "vent_lee_afd4_percent" => $record.vent_lee_afd4_percent = Some($value),
            "vent_wind_afd4_percent" => $record.vent_wind_afd4_percent = Some($value),
            "curtain_1_percent" => $record.curtain_1_percent = Some($value),
            "curtain_2_percent" => $record.curtain_2_percent = Some($value),
            "curtain_3_percent" => $record.curtain_3_percent = Some($value),
            "curtain_4_percent" => $record.curtain_4_percent = Some($value),
            "window_1_percent" => $record.window_1_percent = Some($value),
            "window_2_percent" => $record.window_2_percent = Some($value),
            "heating_setpoint_c" => $record.heating_setpoint_c = Some($value),
            "pipe_temp_1_c" => $record.pipe_temp_1_c = Some($value),
            "pipe_temp_2_c" => $record.pipe_temp_2_c = Some($value),
            "flow_temp_1_c" => $record.flow_temp_1_c = Some($value),
            "flow_temp_2_c" => $record.flow_temp_2_c = Some($value),
            "temperature_forecast_c" => $record.temperature_forecast_c = Some($value),
            "sun_radiation_forecast_w_m2" => $record.sun_radiation_forecast_w_m2 = Some($value),
            "temperature_actual_c" => $record.temperature_actual_c = Some($value),
            "sun_radiation_actual_w_m2" => $record.sun_radiation_actual_w_m2 = Some($value),
            "vpd_hpa" => $record.vpd_hpa = Some($value),
            "humidity_deficit_afd3_g_m3" => $record.humidity_deficit_afd3_g_m3 = Some($value),
            "relative_humidity_afd3_percent" => $record.relative_humidity_afd3_percent = Some($value),
            "humidity_deficit_afd4_g_m3" => $record.humidity_deficit_afd4_g_m3 = Some($value),
            "relative_humidity_afd4_percent" => $record.relative_humidity_afd4_percent = Some($value),
            "dli_sum" => $record.dli_sum = Some($value),
            "value" => $record.value = Some($value),
            _ => eprintln!(
                "WARN (MortenSDU): Attempted to set unhandled f64 field '{}' in ParsedRecord.",
                $field_name
            ),
        }
    };
    // Match bool
    ($record:expr, $field_name:expr, $value:expr; bool) => {
        match $field_name {
            "rain_status" => $record.rain_status = Some($value),
            "lamp_grp1_no3_status" => $record.lamp_grp1_no3_status = Some($value),
            "lamp_grp2_no3_status" => $record.lamp_grp2_no3_status = Some($value),
            "lamp_grp3_no3_status" => $record.lamp_grp3_no3_status = Some($value),
            "lamp_grp4_no3_status" => $record.lamp_grp4_no3_status = Some($value),
            "lamp_grp1_no4_status" => $record.lamp_grp1_no4_status = Some($value),
            "lamp_grp2_no4_status" => $record.lamp_grp2_no4_status = Some($value),
            "measured_status_bool" => $record.measured_status_bool = Some($value),
            _ => eprintln!(
                "WARN (MortenSDU): Attempted to set unhandled bool field '{}' in ParsedRecord.",
                $field_name
            ),
        }
    };
    // Fallback for unhandled types
    ($record:expr, $field_name:expr, $value:expr; $type:ty) => {
        eprintln!(
            "WARN (MortenSDU): set_field! macro called for unhandled type '{}' for field '{}'.",
            stringify!($type),
            $field_name
        );
    };
}

// ADDED BACK: Helper function parse_datetime_utc
fn parse_datetime_utc(datetime_str: &str, format: &str) -> Result<DateTime<Utc>, String> {
    // use chrono::NaiveDateTime; // Use import from top
    NaiveDateTime::parse_from_str(datetime_str, format)
        .map_err(|e| {
            format!(
                "Failed to parse timestamp '{}' with format '{}': {}",
                datetime_str, format, e
            )
        })
        .map(|naive_dt| Utc.from_utc_datetime(&naive_dt)) // Use import from top
}

pub fn parse_morten_sdu(
    config: &FileConfig,
    file_path: &Path,
) -> Result<Vec<ParsedRecord>, ParseError> {
    // println!("DEBUG: Entering parse_morten_sdu for {}", file_path.display()); // <-- Comment out
    let file = File::open(file_path).map_err(|e| ParseError::IoError {
        path: file_path.to_path_buf(),
        source: e,
    })?;

    // MortenSDU uses comma delimiter and has headers
    let delimiter = b',';
    let header_rows_to_skip = config.header_rows.unwrap_or(1);
    if header_rows_to_skip == 0 {
        return Err(ParseError::ConfigError {
            path: file_path.to_path_buf(),
            field: "header_rows".to_string(),
            message: "AarslevMortenSDU format requires header_rows >= 1.".to_string(),
        });
    }

    // --- CSV Reader Setup ---
    let mut reader_builder = ReaderBuilder::new();
    reader_builder.delimiter(delimiter);
    reader_builder.has_headers(true);

    // Configure Quoting based on Option<serde_json::Value>
    let mut use_quoting = false; // Default for MortenSDU is false based on config
    let mut quote_char = b'"';

    if let Some(quoting_val) = &config.quoting {
        match quoting_val {
            serde_json::Value::Bool(b) => {
                use_quoting = *b;
            }
            serde_json::Value::String(s) if s.len() == 1 => {
                quote_char = s.chars().next().unwrap() as u8;
                use_quoting = true; // Enable quoting if char is specified
            }
            _ => {
                eprintln!(
                    "WARN (MortenSDU): Invalid 'quoting' value {:?} in config for {}. Defaulting to no quoting.",
                    quoting_val,
                    file_path.display()
                );
                use_quoting = false;
            }
        }
    } // else: config.quoting is None, use default (false)

    reader_builder.quoting(use_quoting);
    if use_quoting {
        reader_builder.quote(quote_char);
    }

    let mut reader = reader_builder.from_reader(file);

    // --- Header Processing & Index Mapping ---
    let headers = reader
        .headers()
        .map_err(|e| ParseError::HeaderReadError {
            path: file_path.to_path_buf(),
            source: e,
        })?
        .clone();

    let mut header_map: HashMap<String, usize> = HashMap::new();
    for (index, header) in headers.iter().enumerate() {
        header_map.insert(header.trim().to_string(), index);
    }

    let ts_info = config
        .timestamp_info
        .as_ref()
        .ok_or_else(|| ParseError::ConfigError {
            path: file_path.to_path_buf(),
            field: "timestamp_info".to_string(),
            message: "timestamp_info is required for AarslevMortenSDU format.".to_string(),
        })?;

    let start_col_name =
        ts_info
            .start_col_name
            .as_deref()
            .ok_or_else(|| ParseError::ConfigError {
                path: file_path.to_path_buf(),
                field: "timestamp_info.start_col_name".to_string(),
                message: "start_col_name is required.".to_string(),
            })?;
    let _end_col_name = ts_info
        .end_col_name
        .as_deref()
        .ok_or_else(|| ParseError::ConfigError {
            path: file_path.to_path_buf(),
            field: "timestamp_info.end_col_name".to_string(),
            message: "end_col_name is required.".to_string(),
        })?;
    let timestamp_format = ts_info
        .format
        .as_deref()
        .ok_or_else(|| ParseError::ConfigError {
            path: file_path.to_path_buf(),
            field: "timestamp_info.format".to_string(),
            message: "Timestamp format is required.".to_string(),
        })?;

    let start_index = *header_map
        .get(start_col_name)
        .ok_or_else(|| ParseError::ConfigError {
            path: file_path.to_path_buf(),
            field: format!("timestamp_info.start_col_name: {}", start_col_name),
            message: "Start column name not found in headers.".to_string(),
        })?;

    // Stores (target_field, source_index, data_type)
    let mut column_index_map: Vec<(String, usize, String)> = Vec::new();
    if let Some(map_config) = &config.column_map {
        for mapping in map_config {
            // Use correct field names: source, target
            let source_name = mapping.source.trim();
            let target_field = &mapping.target;
            let data_type = &mapping.data_type;

            let source_index =
                *header_map
                    .get(source_name)
                    .ok_or_else(|| ParseError::ConfigError {
                        path: file_path.to_path_buf(),
                        field: format!("column_map.source: {}", source_name),
                        message: "Column name not found in headers.".to_string(),
                    })?;
            column_index_map.push((target_field.clone(), source_index, data_type.clone()));
        }
    } else {
        return Err(ParseError::ConfigError {
            path: file_path.to_path_buf(),
            field: "column_map".to_string(),
            message: "column_map is required for AarslevMortenSDU format.".to_string(),
        });
    }

    // --- Skip Header Rows (if more than 1 specified) ---
    let remaining_rows_to_skip = if header_rows_to_skip > 1 {
        header_rows_to_skip - 1
    } else {
        0
    };
    for i in 0..remaining_rows_to_skip {
        let mut record = csv::StringRecord::new();
        match reader.read_record(&mut record) {
            Ok(true) => { /* Skipped */ }
            Ok(false) => {
                eprintln!(
                    "WARN (MortenSDU): Reached end of file while skipping header row {}/{} in {}",
                    i + 1,
                    remaining_rows_to_skip,
                    file_path.display()
                );
                break;
            }
            Err(e) => {
                return Err(ParseError::IoError {
                    path: file_path.to_path_buf(),
                    source: std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()),
                })
            }
        }
    }

    // --- Data Row Processing ---
    let mut parsed_records = Vec::new();
    let null_markers = config.null_markers.as_ref().cloned().unwrap_or_default();

    for (row_index, result) in reader.records().enumerate() {
        let record = match result {
            Ok(r) => r,
            Err(e) => {
                let file_row_num = header_rows_to_skip + row_index + 1;
                eprintln!(
                    "ERROR (MortenSDU): Failed to read record at file row {} in {}: {}",
                    file_row_num,
                    file_path.display(),
                    e
                );
                continue;
            }
        };

        let file_row_num = header_rows_to_skip + row_index + 1;
        let mut parsed_record = ParsedRecord::default();
        let mut row_has_error = false;

        parsed_record.source_file = Some(file_path.to_string_lossy().into_owned());
        parsed_record.source_system = config.source_system.clone();
        parsed_record.format_type = Some(config.format_type.clone());

        // --- Timestamp Parsing (Use Start Time) ---
        match get_field_by_index(&record, start_index) {
            // Use helper
            Ok(Some(start_str)) => {
                // Check inner Option
                if !null_markers.iter().any(|m| m == start_str) {
                    // Check for null marker
                    match parse_datetime_utc(start_str, timestamp_format) {
                        Ok(ts) => parsed_record.timestamp_utc = Some(ts),
                        Err(e) => {
                            eprintln!("ERROR (MortenSDU): Timestamp parse failed for Start ('{}') in {} at row {}: {}", start_str, file_path.display(), file_row_num, e);
                            row_has_error = true;
                        }
                    }
                } // else: start time is null marker, timestamp_utc remains None
            }
            Ok(None) => { /* Field was empty/whitespace */ }
            Err(e) => {
                // Error from get_field_by_index
                eprintln!("ERROR (MortenSDU): Missing Start timestamp column (index {}) in {} at row {}: {:?}", start_index, file_path.display(), file_row_num, e);
                row_has_error = true;
            }
        }

        // --- Data Column Processing ---
        for (target_field, source_index, data_type) in &column_index_map {
            match get_field_by_index(&record, *source_index) {
                // Use helper
                Ok(Some(raw_value)) => {
                    // Check inner Option
                    if null_markers.iter().any(|marker| marker == raw_value) {
                        continue; // Null marker
                    }
                    let trimmed_value = raw_value; // Already trimmed by helper

                    // --- Special handling for rain_status boolean ---
                    if *target_field == "rain_status" {
                        // Need to dereference target_field
                        match trimmed_value.parse::<f64>() {
                            Ok(val_f64) => {
                                // Treat any non-zero float as true
                                parsed_record.rain_status = Some(val_f64 > 0.0);
                            }
                            Err(e) => {
                                eprintln!(
                                    "ERROR (MortenSDU): Failed to parse float for field '{}' ('{}') at row {}: {}. Setting row error.",
                                    target_field, trimmed_value, file_row_num, e // Use trimmed_value here
                                );
                                row_has_error = true;
                            }
                        }
                        continue; // Skip the generic match for this handled field
                    }

                    match data_type.as_str() {
                        "float" => match trimmed_value.parse::<f64>() {
                            Ok(val) => {
                                // RE-ADDED: Range check specifically for humidity
                                if *target_field == "relative_humidity_percent"
                                    && (val < -5.0 || val > 105.0)
                                {
                                    eprintln!(
                                         "WARN (MortenSDU): Humidity value {:.2} for field '{}' in {} at row {} is out of range [-5, 105]. Setting to None.",
                                         val, target_field, file_path.display(), file_row_num
                                     );
                                    // Don't set the field, effectively leaving it as None
                                } else {
                                    set_field!(parsed_record, target_field.as_str(), val; f64);
                                    // Set if in range
                                }
                            }
                            Err(e) => {
                                eprintln!("ERROR (MortenSDU): Float parse failed for field '{}' ('{}') in {} at row {}: {}", target_field, raw_value, file_path.display(), file_row_num, e);
                                row_has_error = true;
                            }
                        },
                        "boolean" | "bool" => {
                            // Allow "0", "1", "0.0", "1.0" (case-insensitive matching for true/false might be too broad here)
                            match trimmed_value {
                                "1" | "1.0" => set_field!(parsed_record, target_field.as_str(), true; bool),
                                "0" | "0.0" => {
                                    set_field!(parsed_record, target_field.as_str(), false; bool)
                                }
                                _ => {
                                    eprintln!("ERROR (MortenSDU): Boolean parse failed for field '{}' ('{}') in {} at row {}: Expected 0, 1, 0.0, or 1.0", target_field, raw_value, file_path.display(), file_row_num);
                                    row_has_error = true;
                                }
                            }
                        }
                        unknown_type => {
                            eprintln!("WARN (MortenSDU): Unknown data_type '{}' specified for field '{}'.", unknown_type, target_field);
                        }
                    }
                }
                Ok(None) => {
                    /* Field was empty/whitespace */
                    continue;
                }
                Err(e) => {
                    // Error from get_field_by_index
                    eprintln!("ERROR (MortenSDU): Missing column at index {} for field '{}' in {} at row {}: {:?}", *source_index, target_field, file_path.display(), file_row_num, e);
                    row_has_error = true;
                }
            }
        }

        if !row_has_error {
            parsed_records.push(parsed_record);
        } else {
            eprintln!(
                "INFO (MortenSDU): Skipping row {} in {} due to parsing errors.",
                file_row_num,
                file_path.display()
            );
        }
    }

    // println!("DEBUG: Finished parse_morten_sdu for {}. Parsed {} records.", file_path.display(), parsed_records.len()); // <-- Comment out
    Ok(parsed_records)
}

// ADDED BACK: Helper function get_field_by_index
fn get_field_by_index<'r>(
    record: &'r StringRecord,
    index: usize,
) -> Result<Option<&'r str>, ParseError> {
    match record.get(index) {
        Some(s) => {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                Ok(None)
            } else {
                Ok(Some(trimmed))
            }
        }
        None => Err(ParseError::ConfigError {
            path: PathBuf::new(), // Needs PathBuf
            field: format!("column index {}", index),
            message: format!("Index out of bounds (record len = {})", record.len()),
        }),
    }
}
