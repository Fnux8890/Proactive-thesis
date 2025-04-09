use walkdir::WalkDir;
use std::path::Path;
// use std::env; // Removed unused import
use std::error::Error;
use std::fs::File;
use serde::{Deserialize, Deserializer}; // Import Deserializer trait
use std::collections::HashMap;
// use serde_json::Value; // Removed unused import
use std::fmt;
use serde::de::{self, Visitor}; // Removed unused import Error as SerdeError
// use serde::ser; // Removed unused import
use std::io;
use std::path::PathBuf;
use std::fs; // Added fs import

// Placeholder struct for initial testing or fallback
#[derive(Debug, Deserialize)]
struct GenericRecord {
    #[serde(flatten)]
    other_fields: HashMap<String, String>,
}

// Struct for knudjepsen/NO{3,4}_LAMPEGRP_1.csv format
#[derive(Debug, Deserialize)]
struct KnudjepsenLampRecord {
    #[serde(rename = "")]
    timestamp_str: String,
    #[serde(rename = "mål temp afd  Mid")]
    maal_temp_afd_mid_str: Option<String>,
    udetemp_str: Option<String>,
    stråling: Option<i64>,
    #[serde(rename = "CO2 målt")]
    co2_maalt_str: Option<String>,
    #[serde(rename = "mål gard 1")]
    maal_gard_1_str: Option<String>,
    #[serde(rename = "mål gard 2")]
    maal_gard_2_str: Option<String>,
    #[serde(rename = "målt status")]
    maalt_status: Option<i64>,
    #[serde(rename = "mål FD")]
    maal_fd_str: Option<String>,
    #[serde(rename = "mål rør 1")]
    maal_roer_1_str: Option<String>,
    #[serde(rename = "mål rør 2")]
    maal_roer_2_str: Option<String>,
    
    // Add fields to hold the parsed numeric values (non-serialized)
    #[serde(skip_deserializing)]
    maal_temp_afd_mid: Option<f64>,
    #[serde(skip_deserializing)]
    udetemp: Option<f64>,
    #[serde(skip_deserializing)]
    co2_maalt: Option<f64>, // Assuming f64 is suitable
    #[serde(skip_deserializing)]
    maal_gard_1: Option<f64>,
    #[serde(skip_deserializing)]
    maal_gard_2: Option<f64>,
    #[serde(skip_deserializing)]
    maal_fd: Option<f64>,
    #[serde(skip_deserializing)]
    maal_roer_1: Option<f64>, // Assuming f64 is suitable
    #[serde(skip_deserializing)]
    maal_roer_2: Option<f64>,
}

// Struct for aarslev/*MortenSDUData*.csv
#[derive(Debug, Deserialize)]
struct AarslevMortenRecord {
    #[serde(rename = "Start")]
    start: Option<String>, // Keep as string for now
    #[serde(rename = "End")]
    end: Option<String>,   // Keep as string for now
    // Use flatten for the dynamic celle columns
    #[serde(flatten)]
    celle_data: HashMap<String, Option<f64>>, // Assume f64, handle errors later if needed
}

// Enum to represent parsed sensor values more explicitly
#[derive(Debug)] // Add Debug trait
enum SensorValue {
    Number(f64),
    Empty,
    Text(String),
}

// Implement Deserialize for SensorValue to handle quotes and parsing
impl<'de> Deserialize<'de> for SensorValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct SensorValueVisitor;

        impl<'de> Visitor<'de> for SensorValueVisitor {
            type Value = SensorValue;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a string (potentially quoted, comma-decimal), number, or empty field representing a sensor value")
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                let trimmed = value.trim_matches('"');
                if trimmed.is_empty() {
                    Ok(SensorValue::Empty)
                } else {
                    match parse_comma_decimal(trimmed) {
                        Ok(num) => Ok(SensorValue::Number(num)),
                        Err(_) => Ok(SensorValue::Text(trimmed.to_string())),
                    }
                }
            }
            
            // Handle cases where the CSV reader directly provides a float
            fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(SensorValue::Number(value))
            }

            // Also handle potential integers if the CSV reader provides them
            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(SensorValue::Number(value as f64))
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(SensorValue::Number(value as f64))
            }

            fn visit_unit<E>(self) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(SensorValue::Empty)
            }

            fn visit_none<E>(self) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(SensorValue::Empty)
            }
        }

        // Use deserialize_any to let the visitor handle the type determined by serde_csv
        deserializer.deserialize_any(SensorValueVisitor)
    }
}

// Struct for aarslev/celle*/output-*.csv
#[derive(Debug, Deserialize)]
struct AarslevCelleOutputRecord {
    #[serde(rename = "Date")]
    date_str: Option<String>,
    #[serde(rename = "Time")]
    time_str: Option<String>,
    #[serde(flatten)]
    sensor_data: HashMap<String, SensorValue>, // Use the new Enum
}

// Struct for aarslev/{weather, data_jan_feb, winter}*.csv types
#[derive(Debug, Deserialize)]
struct AarslevWeatherRecord {
    timestamp: Option<i64>, // Assuming milliseconds epoch
    temperature: Option<f64>,
    sun_radiation: Option<f64>,
    temperature_forecast: Option<f64>,
    sun_radiation_forecast: Option<f64>,
    actual_temperature: Option<f64>, // From winter2014
    actual_radiation: Option<f64>,   // From winter2014
}

// Struct for aarslev/temperature_sunradiation*.json.csv
#[derive(Debug, Deserialize)]
struct AarslevTempSunRecord {
    timestamp: Option<i64>,
    // Use map to capture the UUID columns dynamically
    #[serde(flatten)]
    uuid_data: HashMap<String, Option<f64>>,
}

// Struct for knudjepsen/NO?_EkstraLysStyring.csv
#[derive(Debug, Deserialize)]
struct KnudjepsenEkstraLysRecord {
    #[serde(rename = "")]
    timestamp_str: Option<String>,
    #[serde(rename = "Ønske om ekstra lys")]
    oenske_ekstra_lys: Option<String>,
    #[serde(rename = "lampe timer on")]
    lampe_timer_on: Option<i64>, // Assuming direct parse works
    #[serde(rename = "lampe timer off")]
    lampe_timer_off: Option<i64>, // Assuming direct parse works
}

// Struct for knudjepsen/NO?_BelysningsStyring.csv
#[derive(Debug, Deserialize)]
struct KnudjepsenBelysningRecord {
    #[serde(rename = "")]
    timestamp_str: Option<String>,
    #[serde(rename = "Behov")]
    behov: Option<i64>, // Assuming direct parse works
    #[serde(rename = "Status")]
    status: Option<String>,
    #[serde(rename = "Timer on")]
    timer_on: Option<i64>, // Assuming direct parse works
    #[serde(rename = "Timer off")]
    timer_off: Option<i64>, // Assuming direct parse works
    #[serde(rename = "DLI Sum")]
    dli_sum_str: Option<String>, // Read as string for comma decimal

    // Add field to hold the parsed numeric value (non-serialized)
    #[serde(skip_deserializing)]
    dli_sum: Option<f64>,
}

// Struct for aarslev/celle*/output-*.csv.json files
#[derive(Debug, Deserialize, Clone)]
struct AarslevCelleJsonConfig {
    #[serde(rename = "Configuration")]
    configuration: Option<AarslevCelleJsonConfigDetails>,
    #[serde(rename = "Filename")]
    filename: Option<String>,
    #[serde(rename = "Variables")]
    variables: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, Clone)]
struct AarslevCelleJsonConfigDetails {
    date_format: Option<String>,
    delimiter: Option<String>,
    time_format: Option<String>,
}

// Helper function to parse comma-decimal strings
fn parse_comma_decimal(s: &str) -> Result<f64, std::num::ParseFloatError> {
    s.replace(',', ".").parse::<f64>()
}

fn main() -> Result<(), Box<dyn Error>> {
    let base_dir = PathBuf::from("/app/data"); // Corrected base path to match docker-compose volume
    println!("Searching for files in: {}", base_dir.display());

    let mut processed_file_count = 0;
    let mut total_records_parsed = 0;
    let mut total_parse_errors = 0;
    let mut celle_configs: HashMap<PathBuf, AarslevCelleJsonConfig> = HashMap::new(); // Store configs

    // --- Pre-scan for JSON configs first ---
    println!("\n--- Pre-scanning for JSON configuration files ---");
    walk_dir_for_json(&base_dir, &mut celle_configs)?;
    println!("--- Finished JSON pre-scan ---");


    // --- Process all files, using stored configs ---
    walk_dir_for_data(&base_dir, &mut processed_file_count, &mut total_records_parsed, &mut total_parse_errors, &celle_configs)?;


    println!("\n--- Processing Summary ---");
    println!("Total files processed (CSV/JSON): {}", processed_file_count); // Adjusted count meaning
    println!("Total records parsed (from CSVs): {}", total_records_parsed);
    println!("Total parse errors (CSV/JSON): {}", total_parse_errors);
    println!("--- Rust Pipeline Finished ---");

    Ok(())
}

// Pre-scan function specifically for JSON config files
fn walk_dir_for_json(
    dir: &Path,
    celle_configs: &mut HashMap<PathBuf, AarslevCelleJsonConfig>,
) -> io::Result<()> {
    // println!("DEBUG: walk_dir_for_json scanning dir: {}", dir.display()); // REMOVE DEBUG
    if dir.is_dir() {
        for entry_result in fs::read_dir(dir)? {
            // println!("DEBUG: Processing entry in {}", dir.display()); // REMOVE DEBUG
            match entry_result {
                Ok(entry) => {
                    let path = entry.path();
                    // println!("DEBUG: walk_dir_for_json found path: {}", path.display()); // REMOVE DEBUG
                    if path.is_dir() {
                        walk_dir_for_json(&path, celle_configs)?;
                    } else if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                        if ext.to_lowercase() == "json" {
                             // println!("DEBUG: walk_dir_for_json found JSON file: {}", path.display()); // REMOVE DEBUG
                             // Avoid processing files that aren't the target config type yet
                             process_json_config_file(&path, celle_configs)?;
                        } else {
                            // Skip non-JSON
                        }
                    }
                }
                Err(e) => {
                     eprintln!("ERROR: Failed to read directory entry in {}: {}. Skipping entry.", dir.display(), e);
                }
            }
        }
    }
    Ok(())
}


// Function to process *only* JSON config files during pre-scan
fn process_json_config_file(
    path: &PathBuf,
    celle_configs: &mut HashMap<PathBuf, AarslevCelleJsonConfig>,
) -> io::Result<()> {
     let base_dir = PathBuf::from("/app/data"); // Corrected base path
     let display_path = path.strip_prefix(&base_dir).unwrap_or(path);
     let normalized_path_str = display_path.to_string_lossy().replace("\\", "/");
     let relative_parent_dir = display_path.parent().map(|p| p.to_path_buf());

     // Only process Aarslev Celle JSONs for now
     let is_aarslev_celle_json = normalized_path_str.starts_with("aarslev/celle") && normalized_path_str.ends_with(".csv.json");

     if is_aarslev_celle_json {
         println!("  Processing potential config: {}", display_path.display());
         match File::open(path) {
             Ok(file) => {
                 let reader = std::io::BufReader::new(file);
                 match serde_json::from_reader::<_, AarslevCelleJsonConfig>(reader) {
                     Ok(config) => {
                         println!("    Successfully parsed Aarslev Celle JSON Config");
                         if let Some(parent_dir) = relative_parent_dir {
                             celle_configs.insert(parent_dir.clone(), config.clone()); // Store config
                             println!("    Stored config for directory: {}", parent_dir.display());
                         } else {
                             eprintln!("    Warning: Could not determine parent directory for {}, config not stored.", display_path.display());
                         }
                     },
                     Err(err) => {
                         // Log error but don't increment main error count during pre-scan
                         eprintln!("    Error parsing Aarslev Celle JSON Config: {}", err);
                     }
                 }
             }
             Err(e) => {
                 eprintln!("    Error opening potential config file {}: {}", display_path.display(), e);
             }
         }
     }
     Ok(())
}


// Main data processing walk function (processes CSVs and other JSONs)
fn walk_dir_for_data(
    dir: &Path,
    processed_file_count: &mut i32,
    total_records_parsed: &mut i32,
    total_parse_errors: &mut i32,
    celle_configs: &HashMap<PathBuf, AarslevCelleJsonConfig>,
) -> io::Result<()> {
     // println!("DEBUG: [WDFD] Scanning dir: {}", dir.display()); // REMOVE DEBUG
    if dir.is_dir() {
        // println!("DEBUG: [WDFD] Confirmed {} is a directory.", dir.display()); // REMOVE DEBUG
        for entry_result in fs::read_dir(dir)? {
            // println!("DEBUG: [WDFD] Processing next entry in {}", dir.display()); // REMOVE DEBUG
            match entry_result {
                Ok(entry) => {
                    let path = entry.path();
                    // println!("DEBUG: [WDFD] Found path: {}", path.display()); // REMOVE DEBUG
                    if path.is_dir() {
                         // println!("DEBUG: [WDFD] Path is directory, recursing into: {}", path.display()); // REMOVE DEBUG
                        walk_dir_for_data(&path, processed_file_count, total_records_parsed, total_parse_errors, celle_configs)?;
                         // println!("DEBUG: [WDFD] Finished recursion for: {}", path.display()); // REMOVE DEBUG
                    } else {
                        if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                            let lower_ext = ext.to_lowercase();
                            if lower_ext == "csv" || lower_ext == "json" {
                                 // println!("DEBUG: [WDFD] Path is relevant file ({}): {}", lower_ext, path.display()); // REMOVE DEBUG
                                match process_data_file(&path, processed_file_count, total_records_parsed, total_parse_errors, celle_configs) {
                                    Ok(_) => {
                                         // println!("DEBUG: [WDFD] Successfully processed file: {}", path.display()); // REMOVE DEBUG
                                    }
                                    Err(e) => {
                                        eprintln!("ERROR: [WDFD] Error processing file {}: {}. Continuing walk.", path.display(), e);
                                        *total_parse_errors += 1; 
                                    }
                                }
                            } else {
                                // println!("DEBUG: [WDFD] Skipping file with unhandled extension ({}): {}", lower_ext, path.display()); // REMOVE DEBUG
                            }
                        } else {
                             // println!("DEBUG: [WDFD] Skipping file with no extension: {}", path.display()); // REMOVE DEBUG
                        }
                    }
                    // println!("DEBUG: [WDFD] Done processing entry for {}", path.display()); // REMOVE DEBUG
                }
                Err(e) => {
                    eprintln!("ERROR: [WDFD] Failed to read directory entry in {}: {}. Skipping entry.", dir.display(), e);
                    *total_parse_errors += 1; 
                }
            }
        }
         // println!("DEBUG: [WDFD] Finished iterating directory: {}", dir.display()); // REMOVE DEBUG
    } else {
        // println!("DEBUG: [WDFD] Path {} is not a directory, skipping scan.", dir.display()); // REMOVE DEBUG
    }
    // println!("DEBUG: [WDFD] Exiting function for dir: {}", dir.display()); // REMOVE DEBUG
    Ok(())
}

// Function to process individual files (CSV or non-config JSON)
// This function was previously named process_file, renamed to avoid confusion
fn process_data_file(
    path: &PathBuf,
    processed_file_count: &mut i32,
    total_records_parsed: &mut i32,
    total_parse_errors: &mut i32,
    celle_configs: &HashMap<PathBuf, AarslevCelleJsonConfig>,
) -> io::Result<()> {
    // println!("DEBUG: process_data_file START processing: {}", path.display()); // REMOVE DEBUG
    let base_dir = PathBuf::from("/app/data");
    let display_path = path.strip_prefix(&base_dir).unwrap_or(path);

    let normalized_path_str = display_path.to_string_lossy().replace("\\", "/");
    let lower_path_str = normalized_path_str.to_lowercase();
    let relative_parent_dir = display_path.parent().map(|p| p.to_path_buf());


    if let Some(ext) = path.extension().and_then(std::ffi::OsStr::to_str) {
        let lower_ext = ext.to_lowercase();

        if lower_ext == "csv" {
            *processed_file_count += 1; 
            let mut current_file_errors = 0;

            // Determine specific CSV format properties
            let is_knudjepsen = lower_path_str.starts_with("knudjepsen/");
            let is_knudjepsen_lamp = is_knudjepsen &&
                (lower_path_str.contains("no3_lampegrp_1") || lower_path_str.contains("no4_lampegrp_1"));
             let is_knudjepsen_belysning = is_knudjepsen && lower_path_str.contains("belysningsgrp");
             let is_knudjepsen_ekstra = is_knudjepsen && lower_path_str.contains("ekstralysstyring");

            let is_aarslev_celle = lower_path_str.starts_with("aarslev/celle") && lower_path_str.contains("/output-");
            let is_aarslev_morten = lower_path_str.starts_with("aarslev/") && lower_path_str.contains("mortensdudata");
            let is_aarslev_tempsun = lower_path_str.starts_with("aarslev/temperature_sunradiation") && lower_path_str.ends_with(".json.csv");
            let is_aarslev_winter = lower_path_str.starts_with("aarslev/data/winter/raw.csv") || lower_path_str.starts_with("aarslev/winter2014.csv"); // Adjust path if needed
            let is_aarslev_janfeb = lower_path_str.starts_with("aarslev/data_jan_feb_2014.csv") || lower_path_str.starts_with("aarslev/weather_jan_feb_2014.csv");


            let mut delimiter_byte = b','; // Default
            let mut config_source = "default (',')".to_string();

            // Determine delimiter based on file type
            if is_aarslev_celle {
                 delimiter_byte = b';'; // Default for Celle is semicolon
                 config_source = "default (';')".to_string();
                 if let Some(parent_dir) = &relative_parent_dir {
                     if let Some(config) = celle_configs.get(parent_dir) { // Retrieve config
                        println!("  -> Found JSON config for this directory.");
                        if let Some(details) = &config.configuration {
                            if let Some(delim_str) = &details.delimiter {
                                if delim_str.len() == 1 {
                                    delimiter_byte = delim_str.as_bytes()[0];
                                    config_source = format!("config ('{}')", delim_str);
                                } else {
                                    eprintln!("  Warning: Invalid delimiter '{}' in config for {}, using default ';'.", delim_str, parent_dir.display());
                                    delimiter_byte = b';';
                                    config_source = "default (';') invalid config".to_string();
                                }
                            } else { /* Use default */ }

                            // Print first few variables from config if available
                            // if let Some(vars) = &config.variables { // REMOVE DEBUG
                            //     println!("  -> Config Variables (first 5): {:?}", vars.iter().take(5).collect::<Vec<_>>());
                            // } 
                         } else { /* Use default delimiter if details missing */ }
                     } else { /* Use default */ 
                         println!("  -> WARNING: No JSON config found for this directory, using default delimiter.");
                     }
                 } else { /* Use default */ }
            } else if is_knudjepsen || is_aarslev_winter || is_aarslev_janfeb || is_aarslev_tempsun {
                 delimiter_byte = b';';
                 config_source = "default (';') known type".to_string();
            } else if is_aarslev_morten {
                 delimiter_byte = b','; // Morten files use comma
                 config_source = "default (',') MortenSDU".to_string();
            }

            let headers_to_skip = if is_knudjepsen { 3 } else { 0 };
            let has_headers_setting = !is_knudjepsen;

            println!("\nProcessing CSV file: {}", display_path.display());
            println!("  -> Using {} delimiter", config_source);
             if headers_to_skip > 0 {
                 println!("  -> Skipping {} header rows", headers_to_skip);
            }

            // --- DEBUG PRINTS --- START
            // println!("DEBUG CHECK - Path: {}", lower_path_str); // REMOVE DEBUG
            // println!("  is_knudjepsen_lamp: {}", is_knudjepsen_lamp); // REMOVE DEBUG
            // println!("  is_knudjepsen_belysning: {}", is_knudjepsen_belysning); // REMOVE DEBUG
            // println!("  is_knudjepsen_ekstra: {}", is_knudjepsen_ekstra); // REMOVE DEBUG
            // println!("  is_aarslev_celle: {}", is_aarslev_celle); // REMOVE DEBUG
            // println!("  is_aarslev_morten: {}", is_aarslev_morten); // REMOVE DEBUG
            // println!("  is_aarslev_janfeb || is_aarslev_winter: {}", is_aarslev_janfeb || is_aarslev_winter); // REMOVE DEBUG
            // println!("  is_aarslev_tempsun: {}", is_aarslev_tempsun); // REMOVE DEBUG
            // --- DEBUG PRINTS --- END

            match File::open(path) {
                Ok(file) => {
                    let mut reader = csv::ReaderBuilder::new()
                        .delimiter(delimiter_byte)
                        .has_headers(has_headers_setting)
                        .flexible(true)
                        .from_reader(file);

                    if headers_to_skip > 0 {
                         let mut byte_record = csv::ByteRecord::new();
                         for i in 0..headers_to_skip {
                             match reader.read_byte_record(&mut byte_record) {
                                 Ok(true) => { /* Skipped */ },
                                 Ok(false) => { /* EOF */ break; },
                                 Err(e) => {
                                     eprintln!("  Warning: Error reading header row {} to skip: {}. Stopping processing.", i + 1, e);
                                     current_file_errors += 1;
                                     *total_parse_errors += current_file_errors;
                                     return Ok(());
                                 }
                             }
                         }
                     }

                    println!("  Attempting to read first 3 data records...");
                    let mut records_processed_this_file = 0;

                    // Conditional Deserialization - Reordered Knudjepsen first
                    if is_knudjepsen_lamp {
                        println!("  -> Using KnudjepsenLampRecord parser");
                        // ... loop and parse KnudjepsenLampRecord ...
                        for result in reader.deserialize::<KnudjepsenLampRecord>() {
                            if records_processed_this_file >= 3 { break; }
                            match result {
                                Ok(mut record) => {
                                    record.maal_temp_afd_mid = parse_comma_decimal(record.maal_temp_afd_mid_str.as_deref().unwrap_or("")).ok();
                                    record.udetemp = parse_comma_decimal(record.udetemp_str.as_deref().unwrap_or("")).ok();
                                    record.co2_maalt = parse_comma_decimal(record.co2_maalt_str.as_deref().unwrap_or("")).ok();
                                    record.maal_gard_1 = parse_comma_decimal(record.maal_gard_1_str.as_deref().unwrap_or("")).ok();
                                    record.maal_gard_2 = parse_comma_decimal(record.maal_gard_2_str.as_deref().unwrap_or("")).ok();
                                    record.maal_fd = parse_comma_decimal(record.maal_fd_str.as_deref().unwrap_or("")).ok();
                                    record.maal_roer_1 = parse_comma_decimal(record.maal_roer_1_str.as_deref().unwrap_or("")).ok();
                                    record.maal_roer_2 = parse_comma_decimal(record.maal_roer_2_str.as_deref().unwrap_or("")).ok();
                                    *total_records_parsed += 1;
                                    records_processed_this_file += 1;
                                    println!("    Record (KnudjepsenLamp) {}: {:?}", records_processed_this_file, record);
                                },
                                Err(err) => {
                                    current_file_errors += 1;
                                    println!("    Error deserializing (KnudjepsenLamp) record {}: {}", records_processed_this_file + 1, err);
                                }
                            }
                        }
                    } else if is_knudjepsen_belysning { // Check Belysning next
                        println!("  -> Using KnudjepsenBelysningRecord parser");
                         // ... loop and parse KnudjepsenBelysningRecord ...
                         for result in reader.deserialize::<KnudjepsenBelysningRecord>() {
                            if records_processed_this_file >= 3 { break; }
                            match result {
                                Ok(mut record) => {
                                    record.dli_sum = parse_comma_decimal(record.dli_sum_str.as_deref().unwrap_or("")).ok();
                                    *total_records_parsed += 1;
                                    records_processed_this_file += 1;
                                    println!("    Record (KnudjepsenBelysning) {}: {:?}", records_processed_this_file, record);
                                },
                                Err(err) => {
                                    current_file_errors += 1;
                                    println!("    Error deserializing (KnudjepsenBelysning) record {}: {}", records_processed_this_file + 1, err);
                                }
                            }
                        }
                    } else if is_knudjepsen_ekstra { // Check Ekstra last for Knudjepsen
                        println!("  -> Using KnudjepsenEkstraLysRecord parser");
                        // ... loop and parse KnudjepsenEkstraLysRecord ...
                         for result in reader.deserialize::<KnudjepsenEkstraLysRecord>() {
                            if records_processed_this_file >= 3 { break; }
                            match result {
                                Ok(record) => {
                                    *total_records_parsed += 1;
                                    records_processed_this_file += 1;
                                    println!("    Record (KnudjepsenEkstraLys) {}: {:?}", records_processed_this_file, record);
                                },
                                Err(err) => {
                                    current_file_errors += 1;
                                    println!("    Error deserializing (KnudjepsenEkstraLys) record {}: {}", records_processed_this_file + 1, err);
                                }
                            }
                        }
                    } else if is_aarslev_celle { // Now check Aarslev types
                         println!("  -> Using AarslevCelleOutputRecord parser");
                         // ... loop and parse AarslevCelleOutputRecord ...
                         for result in reader.deserialize::<AarslevCelleOutputRecord>() {
                            if records_processed_this_file >= 3 { break; }
                            match result {
                                Ok(record) => {
                                    *total_records_parsed += 1;
                                    records_processed_this_file += 1;
                                    println!("    Record (AarslevCelle) {}: {:?}", records_processed_this_file, record);
                                },
                                Err(err) => {
                                    current_file_errors += 1;
                                    println!("    Error deserializing (AarslevCelle) record {}: {}", records_processed_this_file + 1, err);
                                }
                            }
                        }
                    } else if is_aarslev_morten {
                        println!("  -> Using AarslevMortenRecord parser");
                        // ... loop and parse AarslevMortenRecord ...
                        for result in reader.deserialize::<AarslevMortenRecord>() {
                            if records_processed_this_file >= 3 { break; }
                            match result {
                                Ok(record) => {
                                    *total_records_parsed += 1;
                                    records_processed_this_file += 1;
                                    println!("    Record (AarslevMorten) {}: {:?}", records_processed_this_file, record);
                                },
                                Err(err) => {
                                    current_file_errors += 1;
                                    println!("    Error deserializing (AarslevMorten) record {}: {}", records_processed_this_file + 1, err);
                                }
                            }
                        }
                    } else if is_aarslev_janfeb || is_aarslev_winter {
                        println!("  -> Using AarslevWeatherRecord parser");
                        // ... loop and parse AarslevWeatherRecord ...
                        for result in reader.deserialize::<AarslevWeatherRecord>() {
                            if records_processed_this_file >= 3 { break; }
                            match result {
                                Ok(record) => {
                                    *total_records_parsed += 1;
                                    records_processed_this_file += 1;
                                    println!("    Record (AarslevWeather) {}: {:?}", records_processed_this_file, record);
                                }
                                Err(err) => {
                                    current_file_errors += 1;
                                    println!("    Error deserializing (AarslevWeather) record {}: {}", records_processed_this_file + 1, err);
                                }
                            }
                        }
                    } else if is_aarslev_tempsun {
                        println!("  -> Using AarslevTempSunRecord parser");
                         // ... loop and parse AarslevTempSunRecord ...
                         for result in reader.deserialize::<AarslevTempSunRecord>() {
                            if records_processed_this_file >= 3 { break; }
                            match result {
                                Ok(record) => {
                                    *total_records_parsed += 1;
                                    records_processed_this_file += 1;
                                    println!("    Record (AarslevTempSun) {}: {:?}", records_processed_this_file, record);
                                }
                                Err(err) => {
                                    current_file_errors += 1;
                                    println!("    Error deserializing (AarslevTempSun) record {}: {}", records_processed_this_file + 1, err);
                                }
                            }
                        }
                    } else {
                        // Fallback for other CSVs: Log a warning and skip parsing for now
                         println!("  -> WARNING: Unrecognized CSV file type. Skipping content processing.");
                         // Don't increment current_file_errors here, as we are just skipping
                    }

                }
                Err(e) => {                    
                     eprintln!("  Error opening file {}: {}", display_path.display(), e);
                     current_file_errors += 1; // Count file open error
                 }
            }
            *total_parse_errors += current_file_errors;

        } else if lower_ext == "json" {
             // Only process non-config JSON files here
             let is_aarslev_celle_json_config = normalized_path_str.starts_with("aarslev/celle") && normalized_path_str.ends_with(".csv.json");
             if !is_aarslev_celle_json_config {
                *processed_file_count += 1; // Count other processed JSONs
                let mut current_file_errors = 0;
                println!("\nProcessing Generic JSON file: {}", display_path.display());
                 match File::open(path) {
                     Ok(file) => {
                          let reader = std::io::BufReader::new(file);
                          match serde_json::from_reader::<_, serde_json::Value>(reader) { // Use serde_json::Value
                             Ok(_value) => {
                                 println!("  Successfully parsed generic JSON (structure not validated yet)");
                             },
                             Err(err) => {
                                 *total_parse_errors += 1; // Increment global error count
                                 println!("    Error parsing generic JSON: {}", err);
                             }
                         }
                     }
                     Err(e) => {
                         *total_parse_errors += 1; // Increment global error count
                         eprintln!("  Error opening file {}: {}", display_path.display(), e);
                     }
                 }
             }
        }
    }
    // println!("DEBUG: process_data_file FINISHED processing: {}", path.display()); // REMOVE DEBUG
    Ok(())
} 