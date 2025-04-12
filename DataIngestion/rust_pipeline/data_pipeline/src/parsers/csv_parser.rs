use crate::config::FileConfig;
use crate::data_models::ParsedRecord;
use crate::errors::ParseError;
use std::path::{Path, PathBuf};
use std::fs::File;
use csv::{ReaderBuilder, StringRecord};
use chrono::{NaiveDateTime, DateTime, Utc, TimeZone};
use std::collections::HashMap;

// Helper to set fields in ParsedRecord dynamically (using a macro for less boilerplate)
// The macro now accepts specific types directly.
macro_rules! set_field {
    // Match f64
    ($record:expr, $field_name:expr, $value:expr; f64) => {
        match $field_name.as_str() {
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
            "temperature_c" => $record.air_temp_c = Some($value),
            "sun_radiation_w_m2" => $record.radiation_w_m2 = Some($value),
            // "heating_setpoint_c" => $record.heating_setpoint_c = Some($value),
            // "vpd_hpa" => $record.vpd_hpa = Some($value),
            // "light_intensity_lux" => $record.light_intensity_lux = Some($value),
            // "curtain_1_pos_percent" => $record.curtain_1_percent = Some($value),
            // "solar_radiation_w_m2" => $record.radiation_w_m2 = Some($value),
            _ => eprintln!("WARN: Attempted to set unhandled f64 field '{}' in ParsedRecord.", $field_name),
        }
    };
    // Match i64
    ($record:expr, $field_name:expr, $value:expr; i64) => {
        match $field_name.as_str() {
            "lampe_timer_on" => $record.lampe_timer_on = Some($value),
            "lampe_timer_off" => $record.lampe_timer_off = Some($value),
            "co2_measured_ppm" => $record.co2_measured_ppm = Some($value as f64),
            "radiation_w_m2" => $record.radiation_w_m2 = Some($value as f64),
            _ => eprintln!("WARN: Attempted to set unhandled i64 field '{}' in ParsedRecord.", $field_name),
        }
    };
    // Match i32
     ($record:expr, $field_name:expr, $value:expr; i32) => {
        match $field_name.as_str() {
            "behov" => $record.behov = Some($value),
            "timer_on" => $record.timer_on = Some($value),
            "timer_off" => $record.timer_off = Some($value),
            "vent_lee_afd3_percent" => $record.vent_lee_afd3_percent = Some($value as f64),
            "vent_wind_afd3_percent" => $record.vent_wind_afd3_percent = Some($value as f64),
            "vent_lee_afd4_percent" => $record.vent_lee_afd4_percent = Some($value as f64),
            "vent_wind_afd4_percent" => $record.vent_wind_afd4_percent = Some($value as f64),
            _ => eprintln!("WARN: Attempted to set unhandled i32 field '{}' in ParsedRecord.", $field_name),
        }
    };
    // Match bool
     ($record:expr, $field_name:expr, $value:expr; bool) => {
        match $field_name.as_str() {
            "rain_status" => $record.rain_status = Some($value),
            "lamp_grp1_no3_status" => $record.lamp_grp1_no3_status = Some($value),
            "lamp_grp2_no3_status" => $record.lamp_grp2_no3_status = Some($value),
            "lamp_grp3_no3_status" => $record.lamp_grp3_no3_status = Some($value),
            "lamp_grp4_no3_status" => $record.lamp_grp4_no3_status = Some($value),
            "lamp_grp1_no4_status" => $record.lamp_grp1_no4_status = Some($value),
            "lamp_grp2_no4_status" => $record.lamp_grp2_no4_status = Some($value),
            "measured_status_bool" => $record.measured_status_bool = Some($value),
            _ => eprintln!("WARN: Attempted to set unhandled bool field '{}' in ParsedRecord.", $field_name),
        }
    };
    // Match String
     ($record:expr, $field_name:expr, $value:expr; String) => {
        match $field_name.as_str() {
            "source_file" => $record.source_file = Some($value),
            "source_system" => $record.source_system = Some($value),
            "format_type" => $record.format_type = Some($value),
            "status_str" => $record.status_str = Some($value),
            "oenske_ekstra_lys" => $record.oenske_ekstra_lys = Some($value),
            "uuid" => $record.uuid = Some($value),
            _ => eprintln!("WARN: Attempted to set unhandled String field '{}' in ParsedRecord.", $field_name),
        }
    };
    // Fallback for direct calls if type isn't specified (should not happen with new structure)
    ($record:expr, $field_name:expr, $value:expr) => {
         eprintln!("ERROR: Direct set_field! call without type specifier for field '{}'. This indicates a bug.", $field_name);
    };
}

// Enum to represent the determined timestamp parsing strategy
#[derive(Debug, Clone)]
enum TimestampStrategy {
    NoTimestamp,
    UnixMsByIndex(usize),
    DateTimeByIndex(usize, String), // index, format
    DateIndexTimeIndex(usize, usize, String), // date_idx, time_idx, format
}

pub fn parse_csv(config: &FileConfig, file_path: &Path) -> Result<Vec<ParsedRecord>, ParseError> {
    println!("DEBUG: Entering parse_csv for {}", file_path.display());
    let file = File::open(file_path).map_err(|e| ParseError::IoError { path: file_path.to_path_buf(), source: e })?;

    // Handle Option<String> for delimiter
    let delimiter = config.delimiter.as_deref()
        .unwrap_or(",") // Default to comma if None
        .chars().next() // Get first char
        .unwrap_or(',') // Default to comma if string is empty
        as u8;

    let header_rows_to_skip = config.header_rows.unwrap_or(0);
    let has_headers = header_rows_to_skip > 0;

    // --- CSV Reader Setup ---
    let mut reader_builder = ReaderBuilder::new();
    reader_builder.delimiter(delimiter);
    reader_builder.has_headers(has_headers);

    // --- Configure Quoting ---
    // Default behavior if 'quoting' is None or invalid
    let mut use_quoting = true;
    let mut quote_char = b'"'; // Default quote character

    if let Some(quoting_val) = &config.quoting {
        match quoting_val {
            serde_json::Value::Bool(b) => {
                use_quoting = *b;
            }
            serde_json::Value::String(s) if s.len() == 1 => {
                // If it's a string of length 1, treat it as the quote character
                quote_char = s.chars().next().unwrap() as u8;
                use_quoting = true; // Ensure quoting is enabled if a specific char is given
            }
            _ => {
                // Invalid quoting value, log a warning and use default (true, ")
                eprintln!(
                    "WARN: Invalid 'quoting' value {:?} in config for {}. Defaulting to standard quoting.",
                    quoting_val,
                    file_path.display()
                );
                // Keep use_quoting = true and quote_char = b'"'
            }
        }
    } // else: quoting is None, use default (true, ")

    reader_builder.quoting(use_quoting);
    if use_quoting {
         reader_builder.quote(quote_char);
    }
    // --- End Quoting Configuration ---


    let mut reader = reader_builder.from_reader(file);

    // --- Header Processing & Index Mapping (if headers exist) ---
    let mut header_map: HashMap<String, usize> = HashMap::new();
    let mut timestamp_strategy = TimestampStrategy::NoTimestamp;
    // Stores (target_field, source_index, data_type)
    let mut column_index_map: Vec<(String, usize, String)> = Vec::new();

    if has_headers {
        let headers = reader.headers().map_err(|e| ParseError::HeaderReadError {
            path: file_path.to_path_buf(),
            source: e,
        })?.clone();

        for (index, header) in headers.iter().enumerate() {
            header_map.insert(header.trim().to_string(), index);
        }

        // Determine Timestamp Strategy & Indices
        if let Some(ts_info) = &config.timestamp_info {
             // Format is required if any column strategy is used (name or index)
             // Exception: If NO timestamp info is provided at all, it's okay.
             let format_opt = ts_info.format.as_deref(); // Check if format exists

             // Helper closure to get index or return ConfigError
             let get_index = |name: &str, field_desc: &str| {
                  header_map.get(name.trim()).copied().ok_or_else(|| ParseError::ConfigError {
                      path: file_path.to_path_buf(),
                      field: format!("timestamp_info.{} / '{}'", field_desc, name),
                      message: "Column name not found in headers.".to_string(),
                  })
             };

             if let Some(name) = &ts_info.datetime_col_name {
                 let index = get_index(name, "datetime_col_name")?;
                 timestamp_strategy = TimestampStrategy::DateTimeByIndex(index, format_opt.ok_or_else(|| missing_format_error(file_path, "datetime_col_name"))?.to_string());
             } else if let (Some(date_name), Some(time_name)) = (&ts_info.date_col_name, &ts_info.time_col_name) {
                 let date_index = get_index(date_name, "date_col_name")?;
                 let time_index = get_index(time_name, "time_col_name")?;
                 timestamp_strategy = TimestampStrategy::DateIndexTimeIndex(date_index, time_index, format_opt.ok_or_else(|| missing_format_error(file_path, "date/time_col_name"))?.to_string());
             } else if let Some(name) = &ts_info.unix_ms_col_name {
                  let index = get_index(name, "unix_ms_col_name")?;
                  timestamp_strategy = TimestampStrategy::UnixMsByIndex(index); // Format not needed for unix ms
             } else if let Some(idx) = ts_info.datetime_col { // Fallback to index-based
                  timestamp_strategy = TimestampStrategy::DateTimeByIndex(idx, format_opt.ok_or_else(|| missing_format_error(file_path, "datetime_col"))?.to_string());
             } else if let (Some(date_idx), Some(time_idx)) = (ts_info.date_col, ts_info.time_col) {
                  timestamp_strategy = TimestampStrategy::DateIndexTimeIndex(date_idx, time_idx, format_opt.ok_or_else(|| missing_format_error(file_path, "date/time_col"))?.to_string());
             } else if let Some(idx) = ts_info.unix_ms_col {
                  timestamp_strategy = TimestampStrategy::UnixMsByIndex(idx);
             }
             // No specific strategy found in ts_info, remains NoTimestamp
        }

        // --- Determine Data Column Indices ---
        if let Some(map_config) = &config.column_map {
            for mapping in map_config {
                // Use correct field names: source, target
                let source_identifier = mapping.source.trim();
                let target_field = &mapping.target;
                let data_type = &mapping.data_type;

                let source_index = match source_identifier.parse::<usize>() {
                     Ok(index) => index,
                     Err(_) => { // Not an index, treat as name
                         *header_map.get(source_identifier).ok_or_else(|| ParseError::ConfigError {
                            path: file_path.to_path_buf(),
                            field: format!("column_map.source: {}", source_identifier),
                            message: "Column name/index not found or invalid.".to_string(),
                        })?
                     }
                 };
                column_index_map.push((
                    target_field.clone(), // Use correct field name
                    source_index,
                    data_type.clone(),
                ));
            }
        }
    } else {
        // --- No Headers: Rely on index-based config ONLY ---\
         if let Some(ts_info) = &config.timestamp_info {
             let format_opt = ts_info.format.as_deref();

             // Check for name-based config which is invalid without headers
              if ts_info.datetime_col_name.is_some() || ts_info.date_col_name.is_some() || ts_info.unix_ms_col_name.is_some() {
                  return Err(ParseError::ConfigError {
                      path: file_path.to_path_buf(),
                      field: "timestamp_info.*_col_name".to_string(),
                      message: "Column names specified for timestamp, but file has no headers.".to_string(),
                  });
              }

             // Proceed with index-based config
             if let Some(idx) = ts_info.datetime_col {
                 timestamp_strategy = TimestampStrategy::DateTimeByIndex(idx, format_opt.ok_or_else(|| missing_format_error(file_path, "datetime_col (no headers)"))?.to_string());
             } else if let (Some(date_idx), Some(time_idx)) = (ts_info.date_col, ts_info.time_col) {
                  timestamp_strategy = TimestampStrategy::DateIndexTimeIndex(date_idx, time_idx, format_opt.ok_or_else(|| missing_format_error(file_path, "date/time_col (no headers)"))?.to_string());
             } else if let Some(idx) = ts_info.unix_ms_col {
                  timestamp_strategy = TimestampStrategy::UnixMsByIndex(idx);
             }
             // else remains NoTimestamp
         }

        if let Some(map_config) = &config.column_map {
             for mapping in map_config {
                 // Use correct field names: source, target
                 let source_identifier = mapping.source.trim();
                 let target_field = &mapping.target;
                 let data_type = &mapping.data_type;

                 // Must be an index if no headers
                 let source_index = source_identifier.parse::<usize>().map_err(|_| ParseError::ConfigError {
                     path: file_path.to_path_buf(),
                     field: format!("column_map.source: {}", source_identifier),
                     message: "Source column must be a valid index when file has no headers.".to_string(),
                 })?;
                 column_index_map.push((
                     target_field.clone(),
                     source_index,
                     data_type.clone(),
                 ));
             }
         }
    }

    // --- Skip Header Rows Manually (if necessary AFTER reading headers) ---
    let remaining_rows_to_skip = if has_headers && header_rows_to_skip > 1 {
        header_rows_to_skip - 1
    } else if !has_headers && header_rows_to_skip > 0 {
         header_rows_to_skip
    } else {
        0
    };

    for i in 0..remaining_rows_to_skip {
        let mut record = csv::StringRecord::new();
        match reader.read_record(&mut record) {
            Ok(true) => { /* Successfully skipped a row */ }
            Ok(false) => {
                eprintln!("WARN: Reached end of file while skipping header row {}/{} in {}", i+1, remaining_rows_to_skip, file_path.display());
                return Ok(Vec::new()); // No data rows left
            }
            Err(e) => {
                return Err(ParseError::IoError {
                    path: file_path.to_path_buf(),
                    source: std::io::Error::new(std::io::ErrorKind::InvalidData, format!("Error skipping header row {}: {}", i+1, e))
                });
            }
        }
    }


    // --- Data Row Processing ---\
    let mut parsed_records = Vec::new();
    let null_markers = config.null_markers.as_ref().cloned().unwrap_or_default();

    for (row_index, result) in reader.records().enumerate() {
        let record = match result {
            Ok(r) => r,
            Err(e) => {
                let file_row_num = header_rows_to_skip + row_index + 1;
                eprintln!("ERROR: Failed to read record at file row {} in {}: {}", file_row_num, file_path.display(), e);
                continue;
            }
        };

        let file_row_num = header_rows_to_skip + row_index + 1;
        let mut parsed_record = ParsedRecord::default();
        let mut row_has_error = false;

        parsed_record.source_file = Some(file_path.to_string_lossy().into_owned());
        parsed_record.source_system = config.source_system.clone();
        parsed_record.format_type = Some(config.format_type.clone());

        // Timestamp Parsing
        let timestamp_result = parse_timestamp_from_strategy(&record, &timestamp_strategy, file_path, file_row_num);
        match timestamp_result {
            Ok(ts) => parsed_record.timestamp_utc = ts,
            Err(e) => {
                // Use the specific TimestampParseError now available
                 eprintln!("ERROR: {} processing file {}", e, file_path.display()); // Error includes details
                 // Example of propagating error if desired, though current logic logs and skips row
                 // return Err(e);
                 row_has_error = true;
            }
        }


        // Data Column Processing
        for (target_field, source_index, data_type) in &column_index_map {
            match get_field_by_index(&record, *source_index) { // Use helper here
                Ok(Some(raw_value)) => { // Check inner Option

                    // Check for null markers BEFORE trimming
                    if null_markers.iter().any(|marker| marker == raw_value) {
                        continue; // Skip nulls
                    }

                    let trimmed_value = raw_value.trim(); // Now trim non-null value
                    if trimmed_value.is_empty() {
                        continue; // Skip empty strings after trimming (unless explicitly allowed)
                    }

                    // --- Handle datetime_partial specifically before the generic match ---
                    // This data type indicates the value is part of a timestamp handled by parseTimestampFromStrategy
                    if data_type == "datetime_partial" {
                        continue; // Skip further processing for this column
                    }

                    // Attempt to parse based on data_type
                    match data_type.as_str() {
                        "float" => {
                            let cleaned_value = trimmed_value.replace(',', ".");
                            match cleaned_value.parse::<f64>() {
                                Ok(val) => set_field!(parsed_record, *target_field, val; f64),
                                Err(e) => {
                                    eprintln!("WARN: Float parse failed for field '{}' ('{}' -> '{}') in {} at row {}: {}. Setting field to None.", target_field, raw_value, cleaned_value, file_path.display(), file_row_num, e);
                                    // REMOVED: row_has_error = true;
                                }
                            }
                        }
                        "int" | "integer" => {
                             match trimmed_value.parse::<i64>() {
                                Ok(val) => set_field!(parsed_record, *target_field, val; i64),
                                Err(_) => {
                                     // Try parsing as f64 then casting
                                     match trimmed_value.replace(',', ".").parse::<f64>() {
                                         Ok(f_val) if (f_val.fract() < 1e-9) || (f_val.fract() > 1.0 - 1e-9) => {
                                             set_field!(parsed_record, *target_field, f_val.round() as i64; i64)
                                         }
                                         Ok(f_val) => { // Parsed as float but wasn't whole number
                                             eprintln!("WARN: Integer parse failed (non-integer float: {}) for field '{}' ('{}') in {} at row {}. Setting field to None.", f_val, target_field, raw_value, file_path.display(), file_row_num);
                                             // REMOVED: row_has_error = true;
                                         }
                                         Err(e) => { // Failed to parse as int or float
                                              eprintln!("WARN: Integer parse failed for field '{}' ('{}') in {} at row {}: {}. Setting field to None.", target_field, raw_value, file_path.display(), file_row_num, e);
                                              // REMOVED: row_has_error = true;
                                         }
                                     }
                                }
                            }
                        }
                        "boolean" | "bool" => {
                             match trimmed_value.to_lowercase().as_str() {
                                 "true" | "1" | "yes" | "t" | "y" => set_field!(parsed_record, *target_field, true; bool),
                                 "false" | "0" | "no" | "f" | "n" => set_field!(parsed_record, *target_field, false; bool),
                                 _ => {
                                     eprintln!("WARN: Boolean parse failed for field '{}' ('{}') in {} at row {}: Invalid boolean value. Setting field to None.", target_field, raw_value, file_path.display(), file_row_num);
                                     // REMOVED: row_has_error = true;
                                 }
                             }
                         }
                         "string" => {
                             set_field!(parsed_record, *target_field, trimmed_value.to_string(); String);
                         }
                        unknown_type => { // Catch-all for truly unknown types
                            eprintln!("WARN: Unknown data_type '{}' specified for field '{}' in config for {}. Treating as string.", 
                                      unknown_type, target_field, file_path.display());
                            set_field!(parsed_record, *target_field, trimmed_value.to_string(); String);
                        }
                    }
                }
                 Ok(None) => { /* Field was empty/whitespace after trimming - already None */ continue; }
                 Err(_) => { // Error from get_field_by_index (should indicate index out of bounds)
                     eprintln!("ERROR: Missing column at expected index {} for field '{}' in {} at row {}", *source_index, target_field, file_path.display(), file_row_num);
                     // CRITICAL Error - index mapping is wrong, maybe keep row_has_error?
                     row_has_error = true; // Keep this one as it indicates a config/file mismatch
                }
            }
        }

        // Only skip row if a CRITICAL error occurred (like timestamp or index out of bounds)
        if !row_has_error {
            // Check if timestamp is None, only push if we have a valid timestamp
            if parsed_record.timestamp_utc.is_some() {
                parsed_records.push(parsed_record);
            } else {
                eprintln!("INFO: Skipping row {} in {} due to missing or invalid timestamp.", file_row_num, file_path.display());
            }
        } else {
            eprintln!("INFO: Skipping row {} in {} due to critical parsing errors (index out of bounds or invalid timestamp).", file_row_num, file_path.display());
        }
    }

    println!("DEBUG: Finished parse_csv for {}. Parsed {} records.", file_path.display(), parsed_records.len());
    Ok(parsed_records)
}

// Helper function to create a ConfigError for missing timestamp format
fn missing_format_error(file_path: &Path, context: &str) -> ParseError {
     ParseError::ConfigError {
         path: file_path.to_path_buf(),
         field: format!("timestamp_info.format (needed for {})", context),
         message: "Timestamp format is required.".to_string(),
     }
}


// Helper function to parse timestamp based on the determined strategy
fn parse_timestamp_from_strategy(
    record: &StringRecord,
    strategy: &TimestampStrategy,
    file_path: &Path,
    row_num: usize,
) -> Result<Option<DateTime<Utc>>, ParseError> { // Return ParseError directly
    match strategy {
        TimestampStrategy::NoTimestamp => Ok(None),
        TimestampStrategy::UnixMsByIndex(index) => {
            get_field_by_index(record, *index)?
                .map(|val_str| {
                     parse_unix_ms_utc(val_str) // Returns Result<DateTime<Utc>, String>
                          .map_err(|e| ParseError::TimestampParseError {
                              path: file_path.to_path_buf(),
                              row: row_num,
                              value: val_str.to_string(),
                              format: "unix_ms".to_string(),
                              message: e, // Use 'message' field
                          })
                })
                .transpose() // Convert Option<Result<Option<T>, E>> to Result<Option<T>, E>
        }
        TimestampStrategy::DateTimeByIndex(index, format) => {
            get_field_by_index(record, *index)?
                .map(|val_str| {
                     parse_datetime_utc(val_str, format) // Returns Result<DateTime<Utc>, String>
                          .map_err(|e| ParseError::TimestampParseError {
                              path: file_path.to_path_buf(),
                              row: row_num,
                              value: val_str.to_string(),
                              format: format.clone(),
                              message: e, // Use 'message' field
                          })
                })
                .transpose() // Convert Option<Result<Option<T>, E>> to Result<Option<T>, E>
        }
        TimestampStrategy::DateIndexTimeIndex(date_idx, time_idx, format) => {
            let date_str_opt = get_field_by_index(record, *date_idx)?;
            let time_str_opt = get_field_by_index(record, *time_idx)?;
            if let (Some(date_str), Some(time_str)) = (date_str_opt, time_str_opt) {
                let datetime_str = format!("{} {}", date_str, time_str);
                match parse_datetime_utc(&datetime_str, format) {
                    Ok(dt) => Ok(Some(dt)),
                    Err(e) => Err(ParseError::TimestampParseError {
                        path: file_path.to_path_buf(),
                        row: row_num,
                        value: datetime_str,
                        format: format.clone(),
                        message: e, // Use 'message' field
                    }),
                }
            } else {
                Ok(None)
            }
        }
    }
}


// Helper to safely get a field by index, returning Option<&str>
// Returns outer Result for potential index out of bounds error
// Returns inner Option<&str> which is None if the field is empty/whitespace
fn get_field_by_index<'r>(record: &'r StringRecord, index: usize) -> Result<Option<&'r str>, ParseError> {
     match record.get(index) {
         Some(s) => {
             let trimmed = s.trim();
             if trimmed.is_empty() {
                 Ok(None) // Treat empty/whitespace fields as None
             } else {
                 Ok(Some(trimmed)) // Return the trimmed, non-empty field
             }
         }
         None => Err(ParseError::ConfigError { // Changed to ConfigError as it implies bad index from config/headers
             path: PathBuf::new(), // TODO: Need to pass path here somehow if we want it in the error
             field: format!("column index {}", index),
             message: format!("Index out of bounds (record len = {})", record.len()),
         }),
     }
}


// Parses a datetime string with a given format into UTC DateTime
// Returns Result<DateTime<Utc>, String> where Err contains a descriptive message
fn parse_datetime_utc(datetime_str: &str, format: &str) -> Result<DateTime<Utc>, String> {
    NaiveDateTime::parse_from_str(datetime_str, format)
        .map_err(|e| format!("Failed to parse timestamp '{}' with format '{}': {}", datetime_str, format, e))
        .map(|naive_dt| {
             Utc.from_utc_datetime(&naive_dt)
        })
}

// Parses a Unix millisecond timestamp string into UTC DateTime
// Returns Result<DateTime<Utc>, String> where Err contains a descriptive message
fn parse_unix_ms_utc(ms_str: &str) -> Result<DateTime<Utc>, String> {
    ms_str.parse::<i64>()
        .map_err(|e| format!("Failed to parse '{}' as i64 for Unix ms timestamp: {}", ms_str, e))
        .and_then(|ms| {
             Utc.timestamp_millis_opt(ms).single()
                 .ok_or_else(|| format!("Unix ms timestamp '{}' ({}) is out of range", ms_str, ms))
        })
}