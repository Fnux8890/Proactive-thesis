use std::path::Path;
// use walkdir::WalkDir; // REMOVED unused import
// use std::env; // Removed unused import
// use std::error::Error; // REMOVED unused import AGAIN
// use std::fs::File; // REMOVED unused import
// use serde::{Deserialize, Deserializer}; // Now handled in models
use std::collections::HashMap;
// use serde_json::Value; // Removed unused import
// use std::fmt; // Now handled in models
// use serde::de::{self, Visitor, Error as SerdeError}; // Now handled in models
// use serde::ser; // Removed unused import
use std::io;
use std::path::PathBuf;
use std::fs; // ADD BACK std::fs
use glob::glob; // Import the glob function

mod models; // Declare the models module
mod validation; // Declare the validation module
use models::*; // Bring structs/enums into scope
use validation::*; // Bring validation functions into scope

// Declare modules
mod config;
mod errors;
// mod data_models; // Declare later when needed
// mod parsers; // Declare later when needed
// mod file_processor; // Declare later when needed

// Declare new modules
mod data_models;
mod parsers;
mod file_processor;

use config::load_config;
use errors::PipelineError;

/* // COMMENT OUT unused struct definition AGAIN
struct DataFileEntry {
    workspace_path: String,
    container_path: String,
    status: String,
}
*/

// Struct/Enum definitions removed - they are now in models.rs

// parse_comma_decimal function removed - it is now in models.rs

fn main() -> Result<(), PipelineError> {
    println!("Starting Data Pipeline...");

    // --- START DEBUG OWN EXECUTABLE READ ---
    let executable_path = "/usr/local/bin/data_pipeline";
    println!("DEBUG: Attempting to read own executable at {}", executable_path);
    match std::fs::read(executable_path) {
        Ok(bytes) => {
            println!("DEBUG: Successfully read own executable ({} bytes).", bytes.len());
        }
        Err(e) => {
            eprintln!("FATAL: Failed to read own executable at {}: {}", executable_path, e);
            // Panic here as well if this basic operation fails
            panic!("Failed basic file read for executable '{}': {}", executable_path, e);
        }
    }
    println!("DEBUG: Own executable read attempt finished.");
    // --- END DEBUG OWN EXECUTABLE READ ---

    // Define the path to the configuration file copied into the image
    let config_path = "/app/config/data_files.json"; 
    println!("Attempting to load configuration from: {}", config_path);

    // Load configuration (with existing match block for detailed errors)
    let file_configs = match load_config(config_path) {
        Ok(configs) => configs,
        Err(e) => {
            eprintln!("Error loading configuration: {}", e);
            return Err(PipelineError::Config(e)); 
        }
    };
    println!("Successfully loaded {} file configurations.", file_configs.len());

    // --- Process all file configurations ---
    let mut total_processed = 0;
    let mut total_errors = 0;

    for config in file_configs {
        // Log details (optional, keep commented unless debugging)
        // println!("  Workspace Path: {}", config.workspace_path.display());
        // println!("  Container Path: {}", config.container_path.display());
        // println!("  Format Type: {}", config.format_type);
        // println!("  Status: {}", config.status);

        // --- Handle potential wildcards in container_path ---
        let path_str = config.container_path.to_string_lossy();
        if path_str.contains('*') || path_str.contains('?') {
            println!("DEBUG: Glob pattern detected: {}", path_str);
            match glob(&path_str) {
                Ok(paths) => {
                    let mut found_files = false;
                    for entry in paths {
                        match entry {
                            Ok(actual_path) => {
                                found_files = true;
                                println!("\nProcessing matched file:");
                                // Clone the config and update the path for this specific file
                                let mut specific_config = config.clone();
                                specific_config.container_path = actual_path;

                                // Call the file processor with the specific file config
                                match file_processor::process_file(&specific_config) {
                                    Ok(parsed_records) => {
                                        total_processed += 1;
                                        println!("Successfully processed file: {}. Parsed {} records.",
                                            specific_config.container_path.display(),
                                            parsed_records.len());
                                        // Optional: Print records
                                    }
                                    Err(e) => {
                                        total_errors += 1;
                                        eprintln!("ERROR processing file {}: {}",
                                            specific_config.container_path.display(),
                                            e
                                        );
                                    }
                                }
                            }
                            Err(e) => {
                                eprintln!("ERROR iterating glob result for pattern {}: {}", path_str, e);
                                total_errors += 1; // Count glob iteration error
                            }
                        }
                    }
                    if !found_files {
                         println!("WARN: Glob pattern '{}' did not match any files.", path_str);
                         // Increment total_errors? Or just log? Let's just log for now.
                         // total_errors += 1;
                    }
                }
                Err(e) => {
                    eprintln!("ERROR: Invalid glob pattern '{}': {}", path_str, e);
                    total_errors += 1; // Count invalid pattern as an error
                }
            }
        } else {
            // --- No wildcard, process the single file --- 
            println!("\nProcessing file configuration:");
            match file_processor::process_file(&config) { // Pass reference to config
                Ok(parsed_records) => {
                    total_processed += 1;
                    println!("Successfully processed file: {}. Parsed {} records.",
                        config.container_path.display(),
                        parsed_records.len());
                    // Optional: Print first few records for verification
                    // for (i, record) in parsed_records.iter().take(5).enumerate() {
                    //     println!("  Record {}: {:?}", i + 1, record);
                    // }
                }
                Err(e) => {
                    total_errors += 1;
                    eprintln!("ERROR processing file {}: {}",
                        config.container_path.display(),
                        e
                    );
                    // Continue processing other files even if one fails
                }
            }
        }
    }
    // --- End processing loop ---

    println!(
        "\nData Pipeline finished processing. Processed: {}, Errors: {}",
        total_processed, total_errors
    );
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
                    } else {
                        // Check if it's a file ending with .csv.json specifically for Aarslev Celle configs
                        let filename_str = path.file_name().unwrap_or_default().to_string_lossy(); // Use unwrap_or_default for safety
                        if filename_str.ends_with(".csv.json") {
                            // Potentially add a check here for `aarslev/celle` in the path if needed later,
                            // but for now, just check the suffix as per the immediate error.
                            // println!("DEBUG: Found potential Aarslev Celle config: {}", path.display()); // Keep commented out
                             process_json_config_file(&path, celle_configs)?;
                        } else {
                             // println!("DEBUG: Skipping non-config file during pre-scan: {}", path.display()); // Keep commented out
                            // Skip other files during this specific JSON config pre-scan phase
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
    // println!("DEBUG: process_json_config_file reading: {}", path.display()); // REMOVE DEBUG

    let file_content = match fs::read_to_string(path) {
        Ok(content) => content,
        Err(e) => {
            eprintln!("ERROR: Failed to read JSON config file {}: {}", path.display(), e);
            return Err(e);
        }
    };
    
    // println!("DEBUG: JSON content: {}", file_content); // REMOVE DEBUG

    match serde_json::from_str::<AarslevCelleJsonConfig>(&file_content) { // Use struct from models
        Ok(config) => {
            // Use the directory containing the JSON file as the key
            if let Some(parent_dir) = path.parent() {
                println!("DEBUG: Storing config for key: {:?}", parent_dir); // RE-ADD DEBUG
                celle_configs.insert(parent_dir.to_path_buf(), config);
            }
             // Optionally print stored config for debugging
            // if let Some(stored_config) = celle_configs.get(&path.parent().unwrap().to_path_buf()) {
            //     println!("DEBUG: Stored config: {:?}", stored_config); // REMOVE DEBUG
            // }
        }
        Err(e) => {
            eprintln!("ERROR: Failed to parse JSON config file {}: {}", path.display(), e);
            // Continue processing other files even if one JSON fails to parse
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
    json_data_file_count: &mut i32, // Add counter
    json_data_records_parsed: &mut i64, // Add counter (use i64 for potentially large record counts)
) -> io::Result<()> {
    println!("---> [WDFD] ENTERING directory: {}", dir.display()); // ADDED DEBUG LOG

    if dir.is_dir() {
        println!("---> [WDFD] Starting loop for directory: {}", dir.display()); // ADDED DEBUG LOG
        let read_dir_result = fs::read_dir(dir);

        match read_dir_result {
            Ok(entries) => {
                for entry_result in entries {
                    println!("---> [WDFD] Loop yielded entry: {:?}", &entry_result);

                    // Propagate error if reading the entry failed
                    let entry = entry_result?; // ADDED '?' TO PROPAGATE ERROR
                    
                    let path = entry.path();
                    if path.is_dir() {
                        // Recurse
                        walk_dir_for_data(
                            &path,
                            processed_file_count,
                            total_records_parsed,
                            total_parse_errors,
                            celle_configs,
                            json_data_file_count, // Pass counter down
                            json_data_records_parsed // Pass counter down
                        )?;
                    } else {
                        // Check extension and process file
                        if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                            let lower_ext = ext.to_lowercase();
                            if lower_ext == "csv" || lower_ext == "json" {
                                println!("---> [WDFD] Found relevant file: {}", path.display()); // ADDED: Print path
                                // --- Re-enable processing --- 
                                match process_data_file(
                                    &path,
                                    processed_file_count,
                                    total_records_parsed,
                                    total_parse_errors,
                                    celle_configs,
                                    json_data_file_count, // Pass counter
                                    json_data_records_parsed // Pass counter
                                ) {
                                    Ok(_) => {} 
                                    Err(e) => {
                                        eprintln!("ERROR: [WDFD] Error processing file {}: {}. Continuing walk.", path.display(), e);
                                        *total_parse_errors += 1;
                                    }
                                }
                            } else {
                                // Skipped file (handled extension) 
                            }
                        } else {
                            // Skipped file (no extension)
                        }
                    }
                } // End of for loop
                println!("---> [WDFD] FINISHED loop successfully for directory: {}", dir.display());
            }
            Err(e) => {
                // Error trying to read the directory itself
                eprintln!("ERROR: [WDFD] Failed to read_dir for {}: {}. Skipping directory.", dir.display(), e);
                 // Decide if this should propagate. For now, just log and increment error.
                *total_parse_errors += 1;
                 // Return Ok(()) here means we skip this problematic directory but continue parent walks
                // If we wanted to halt everything, we'd return Err(e) here.
                 // Let's try returning Ok to match previous behavior of continuing.
            }
        }

    } else {
        println!("---> [WDFD] Path {} is not a directory, skipping walk.", dir.display()); // ADDED DEBUG LOG
    }

    println!("---> [WDFD] RETURNING Ok(()) from directory: {}", dir.display()); // ADDED DEBUG LOG
    Ok(())
}

// Function to process individual files (CSV or non-config JSON)
// This function was previously named process_file, renamed to avoid confusion
fn process_data_file(
    path: &Path,
    processed_file_count: &mut i32,
    total_records_parsed: &mut i32,
    total_parse_errors: &mut i32,
    celle_configs: &HashMap<PathBuf, AarslevCelleJsonConfig>,
    json_data_file_count: &mut i32, // Add counter
    json_data_records_parsed: &mut i64, // Add counter
) -> io::Result<()> {
    // println!("DEBUG: process_data_file START processing: {}", path.display()); // REMOVE DEBUG
    let base_dir = PathBuf::from("/app/data");
    let display_path = path.strip_prefix(&base_dir).unwrap_or(path);

    let filename = path.file_name().unwrap_or_default().to_string_lossy();

    // Check if it's a JSON config file (ends with .csv.json) and skip it here
    // as it was handled in the pre-scan
    if filename.ends_with(".csv.json") {
        // println!("DEBUG: Skipping config file during data processing: {}", path.display()); // REMOVE DEBUG
        return Ok(());
    }

    println!("Processing file: {}", display_path.display());

    // Check if it's another JSON file (potential data)
    if filename.ends_with(".json") {
        *json_data_file_count += 1; // Increment JSON data file counter

        let file_content = fs::read_to_string(path)?;

        // Decide which JSON structure to parse based on path or content inspection
        // Example for Knudjepsen JSON data:
        if display_path.starts_with("knudjepsen") {
             // Attempt to parse as SensorDataFile
            match serde_json::from_str::<SensorDataFile>(&file_content) {
                Ok(data) => {
                    let mut records_in_file = 0;
                    for (_, entry) in data {
                         records_in_file += entry.readings.len();
                    }
                    println!("   > Parsed Knudjepsen JSON: {} records", records_in_file);
                    *json_data_records_parsed += records_in_file as i64;
                }
                Err(e) => {
                    eprintln!("   ERROR: Failed to parse Knudjepsen JSON {}: {}", display_path.display(), e);
                    *total_parse_errors += 1;
                }
            }
        // Placeholder for AarslevTempSun JSON (assuming AarslevJsonDataFile)
        } else if display_path.to_string_lossy().contains("aarslev/temperature_sunradiation") && filename.ends_with(".json") {
             match serde_json::from_str::<Vec<AarslevJsonReading>>(&file_content) { // Deserialize directly into Vec<AarslevJsonReading>
                Ok(data) => {
                    let mut records_in_file = 0;
                    for reading_set in data { // Iterate over the Vec directly
                        records_in_file += reading_set.readings.len();
                    }
                    println!("   > Parsed Aarslev TempSun JSON: {} records", records_in_file);
                    *json_data_records_parsed += records_in_file as i64;
                }
                                 Err(e) => {
                    eprintln!("   ERROR: Failed to parse Aarslev TempSun JSON {}: {}", display_path.display(), e);
                    *total_parse_errors += 1;
                }
             }
        } else {
            println!("   > Skipping unrecognized JSON file: {}", display_path.display());
            // Optionally increment errors or just log
        }
        return Ok(()); // Finish processing this JSON file
    }

    // --- Must be a CSV file if we reached here ---
    *processed_file_count += 1; // Increment CSV file counter

    // --- Build the CSV Reader --- 
    // Default delimiter
    let mut delimiter = b';'; // Default to semicolon

    if display_path.starts_with("aarslev/celle") {
        // Attempt to get parent directory to look up config
        if let Some(parent_dir) = path.parent() {
            // Convert parent dir path to string for map lookup - NO, use Path directly
            // if let Some(parent_dir_str) = parent_dir.to_str() { // REMOVED
                // Normalize the key for lookup (match the format used during pre-scan)
                // The pre-scan used the absolute path, so we use the absolute parent_dir directly.
                
                // let lookup_key = parent_dir; // Use parent_dir directly
                 println!(" DEBUG: Looking up config for key: {:?}", parent_dir);
                if let Some(config) = celle_configs.get(parent_dir) { // Use parent_dir directly for lookup
                    // Access the delimiter within the nested structure
                    if let Some(delim_str) = config.configuration.as_ref().and_then(|details| details.delimiter.as_deref()) {
                         if !delim_str.is_empty() {
                            // Take the first char of the string
                            if let Some(delim_char) = delim_str.chars().next() {
                                 // Convert char to u8 byte for the reader
                                // Ensure it's ASCII for simplicity, as CSV crate expects u8
                                if delim_char.is_ascii() {
                                    delimiter = delim_char as u8;
                                    println!("   > Using delimiter '{}' from JSON config for {}", delim_char, display_path.display());
                                } else {
                                     eprintln!("   WARN: Non-ASCII delimiter '{}' found in config for {}, using default ';'", delim_char, display_path.display());
                                }
                            } else { 
                                // This case should ideally not happen if delim_str is not empty, but handle defensively
                                eprintln!("   WARN: Empty delimiter string found in config for {}, using default ';'", display_path.display());
                            }
                        } // else: empty delimiter string in config, use default
                    } // else: no delimiter in config, use default
                } // else: no config found for this dir, use default
                else {
                     println!(" DEBUG: No config found for key: {:?}", parent_dir);
                }
            // } // REMOVED block for parent_dir_str
        }
        // If any step above fails, the default delimiter remains ';'
    }
    // Add logic here for other file types if they might use different delimiters

    let mut rdr = match csv::ReaderBuilder::new()
        .delimiter(delimiter) // Use the determined delimiter
        .has_headers(true)
        .from_path(path)
    {
        Ok(reader) => reader,
        Err(e) => {
            eprintln!(
                "ERROR: Failed to create CSV reader for {}: {}",
                display_path.display(),
                e
            );
            *total_parse_errors += 1;
            return Ok(());
        }
    };

    let mut records_in_file = 0;
    let mut errors_in_file = 0;

    // Determine the correct struct based on the path/filename
    // This is where the routing logic goes.
    let path_str = display_path.to_string_lossy();

    // Example Routing Logic:
    if path_str.starts_with("knudjepsen/NO3_LAMPEGRP_1.csv")
        || path_str.starts_with("knudjepsen/NO4_LAMPEGRP_1.csv") {
        // Use KnudjepsenLampRecord
        for result in rdr.deserialize::<KnudjepsenLampRecord>() {
                            match result {
                Ok(_record) => {
                    //println!("Parsed KnudjepsenLampRecord: {:?}", record); // Optional: Print parsed record
                    records_in_file += 1;
                }
                Err(e) => {
                    eprintln!("   ERROR parsing KnudjepsenLampRecord in {}: {}", display_path.display(), e);
                    errors_in_file += 1;
                }
            }
        }
    } else if path_str.contains("NO3-NO4_belysningsgrp.csv") { // Corrected check for this specific file
         // Use KnudjepsenBelysningGrpRecord (the new struct)
        for result in rdr.deserialize::<KnudjepsenBelysningGrpRecord>() {
                            match result {
                Ok(_record) => {
                    // Skip printing successful records
                    records_in_file += 1;
                }
                Err(e) => {
                    // Log errors, expecting some for the multi-line headers
                    eprintln!("   INFO/ERROR parsing KnudjepsenBelysningGrpRecord in {}: {}", display_path.display(), e);
                    errors_in_file += 1;
                }
            }
        }
    } else if path_str.starts_with("knudjepsen/Ekstra lys.csv") {
         // Use KnudjepsenEkstraLysRecord
        for result in rdr.deserialize::<KnudjepsenEkstraLysRecord>() {
                            match result {
                Ok(_record) => {
                    records_in_file += 1;
                }
                Err(e) => {
                    eprintln!("   ERROR parsing KnudjepsenEkstraLysRecord in {}: {}", display_path.display(), e);
                    errors_in_file += 1;
                }
            }
        }
    } else if path_str.starts_with("knudjepsen/Belysning") { // Match folder - Keep for potential other files
         // Use OLD KnudjepsenBelysningRecord (or decide to remove/refactor this block)
        // **NOTE:** This block might need removal or the original KnudjepsenBelysningRecord struct fixed/removed
        // if NO3-NO4_belysningsgrp.csv was the *only* file it was intended for.
        println!("   > WARNING: Attempting to parse {} with potentially incorrect OLD KnudjepsenBelysningRecord struct.", display_path.display());
        for result in rdr.deserialize::<KnudjepsenBelysningRecord>() {
                            match result {
                Ok(_record) => {
                    records_in_file += 1;
                }
                Err(e) => {
                    eprintln!("   ERROR parsing KnudjepsenBelysningRecord in {}: {}", display_path.display(), e);
                    errors_in_file += 1;
                }
            }
        }
    } else if path_str.ends_with(".extra.csv") { // <<<--- ADDED CHECK for .extra.csv
        // Use KnudjepsenExtraRecord
        for result in rdr.deserialize::<KnudjepsenExtraRecord>() {
                            match result {
                Ok(_record) => {
                    // Skip printing successful records for brevity
                    records_in_file += 1;
                }
                Err(e) => {
                    // Log errors, especially for the multi-line headers we expect to fail initially
                    eprintln!("   INFO/ERROR parsing KnudjepsenExtraRecord in {}: {}", display_path.display(), e);
                    errors_in_file += 1;
                }
            }
        }
    } else if {
        let lower_path = path_str.to_lowercase();
        lower_path.contains("aarslev/") && 
        lower_path.contains("mortensdudata") && 
        lower_path.ends_with(".csv") 
    } {
        // SUPER MANUAL PARSING FOR AarslevMortenRecord (using StringRecord by position)
        println!("   > Using SUPER MANUAL parser for AarslevMortenRecord.");

        // Read headers first, handle error, then proceed
        let headers = match rdr.headers() {
            Ok(hdr) => hdr.iter().map(String::from).collect::<Vec<String>>(),
            Err(e) => {
                eprintln!("   ERROR: Could not read headers for {}: {}", display_path.display(), e);
                errors_in_file += 1;
                *total_parse_errors += errors_in_file; // Increment total errors
                // Return early to skip processing the rest of this file
                println!("   > Skipping file {} due to header error.", display_path.display());
                return Ok(()); 
            }
        };

        let mut record = csv::StringRecord::new();
        while rdr.read_record(&mut record).unwrap_or(false) {
            // Extract by position (index)
            let start_str = record.get(1).map(|s| s.trim().to_string());
            let end_str = record.get(2).map(|s| s.trim().to_string());
            let mut celle_data: HashMap<String, Option<f64>> = HashMap::new();

            // Iterate over remaining fields/headers starting from index 3
            for i in 3..headers.len() {
                if let Some(header_name) = headers.get(i) {
                    if let Some(field_value) = record.get(i) {
                        let trimmed_value = field_value.trim();
                        let float_val = if trimmed_value.is_empty() {
                            None
                        } else {
                            models::parse_comma_decimal(trimmed_value).ok() // Use models:: prefix
                        };
                        celle_data.insert(header_name.clone(), float_val);
                    } else {
                        // If record is shorter than headers, treat missing fields as None 
                        // (or could log warning/error)
                        celle_data.insert(header_name.clone(), None);
                    }
                }
                // No else needed, if headers are shorter than record, extra fields are ignored
            }
            
            // Construct the record instance (optional, useful for validation/later use)
            let _parsed_record = models::AarslevMortenRecord {
                start: start_str,
                end: end_str,
                celle_data,
            };
            // println!(" DEBUG Parsed Morten Record: {:?}", _parsed_record);

            records_in_file += 1;
        }
        // errors_in_file remains 0 unless header reading failed
    } else if path_str.starts_with("aarslev/celle") { // Matches files inside celle* folders
        // Use AarslevCelleOutputRecord (manual timestamp parsing)
        println!("   > Using AarslevCelleOutputRecord parser (with manual timestamp).");

        let headers = match rdr.headers() {
            Ok(hdr) => hdr.clone(), // Clone headers for deserializing the map later
            Err(e) => {
                eprintln!("   ERROR: Could not read headers for {}: {}", display_path.display(), e);
                errors_in_file += 1;
                *total_parse_errors += errors_in_file;
                return Ok(());
            }
        };

        // Get format strings from config (outside the loop for efficiency)
        let mut combined_format_opt: Option<String> = None;

        if let Some(parent_dir) = path.parent() {
            if let Some(config) = celle_configs.get(parent_dir) {
                if let Some(details) = config.configuration.as_ref() {
                    // Directly combine date and time formats if both exist
                    if let (Some(df), Some(tf)) = (&details.date_format, &details.time_format) {
                        combined_format_opt = Some(format!("{} {}", df, tf));
                        println!("   > Using combined format: '{}'", combined_format_opt.as_ref().unwrap());
                    } else {
                        eprintln!("   WARN: Missing date_format or time_format in config for {}. Cannot parse timestamp.", display_path.display());
                    }
                }
            }
        }

        let mut record = csv::StringRecord::new();
        while rdr.read_record(&mut record).unwrap_or(false) {
            let mut parsed_timestamp: Option<chrono::DateTime<chrono::Utc>> = None;

            // Find indices for Date and Time columns (case-insensitive)
            let date_idx = headers.iter().position(|h| h.eq_ignore_ascii_case("Date"));
            let time_idx = headers.iter().position(|h| h.eq_ignore_ascii_case("Time"));

            // Attempt timestamp parsing only if format is available
            if let (Some(date_i), Some(time_i), Some(fmt)) = (date_idx, time_idx, &combined_format_opt) {
                if let (Some(date_str), Some(time_str)) = (record.get(date_i), record.get(time_i)) {
                    let datetime_str = format!("{} {}", date_str.trim(), time_str.trim());
                    match chrono::NaiveDateTime::parse_from_str(&datetime_str, fmt) {
                        Ok(naive_dt) => {
                            // Assume UTC for now, or use timezone from config if available later
                            parsed_timestamp = Some(chrono::DateTime::from_naive_utc_and_offset(naive_dt, chrono::Utc));
                        }
                        Err(e) => {
                            eprintln!(
                                "   ERROR parsing timestamp '{}' with format '{}' in {}: {}",
                                datetime_str,
                                fmt,
                                display_path.display(),
                                e
                            );
                            errors_in_file += 1;
                            // Continue to parse sensor data even if timestamp fails
                        }
                    }
                } else {
                    eprintln!("   WARN: Missing Date or Time field in record in {}. Skipping timestamp parse.", display_path.display());
                    errors_in_file += 1; // Count as error if date/time columns are missing
                }
            } else if combined_format_opt.is_none() {
                // Error already logged about missing format config, maybe increment error count here too?
                // errors_in_file += 1; 
            } else {
                 eprintln!("   WARN: Could not find 'Date' or 'Time' column in {}. Skipping timestamp parse.", display_path.display());
                 // Don't count this as an error maybe, as some files might genuinely lack these?
            }

            // Deserialize the rest into the map, excluding Date/Time by providing only headers for other fields?
            // It's simpler to deserialize the whole record and let serde handle the #[serde(flatten)].
            // We need the headers for flatten to work with StringRecord.
            let sensor_data = match record.deserialize::<HashMap<String, SensorValue>>(Some(&headers)) {
                Ok(mut map_data) => { // Make map_data mutable here
                    // Remove Date and Time keys if they exist, as they are handled separately
                    map_data.remove("Date");
                    map_data.remove("Time"); 
                    map_data.remove("date"); // Case variations
                    map_data.remove("time");
                    map_data // Return the modified map
                }
                Err(e) => {
                    eprintln!("   ERROR deserializing sensor data map in {}: {}", display_path.display(), e);
                    errors_in_file += 1;
                    // Optionally continue with an empty map or skip record
                    continue; // Skip this record if map deserialization fails
                }
            };

            // Construct the final record (even if timestamp failed)
            let final_record = AarslevCelleOutputRecord {
                timestamp: parsed_timestamp,
                timestamp_format: combined_format_opt.clone(), // Store the format used
                sensor_data,
            };
            
            // --- ADD VALIDATION STEP ---
            match validate_record(&final_record) {
                Ok(_) => {
                    // Validation passed, increment parsed count
                    records_in_file += 1;
                    // Optionally print successfully validated record
                    // println!("DEBUG Validated Record: {:?}", final_record); 
                }
                Err(validation_err) => {
                    eprintln!(
                        "   VALIDATION ERROR in record {:?} from {}: {}", 
                        record, // Log the original string record for context
                        display_path.display(), 
                        validation_err
                    );
                    errors_in_file += 1;
                    // Do not increment records_in_file here
                }
            }
            // --- END VALIDATION STEP ---
            
            // REMOVED: Original increment location
            // records_in_file += 1;
        }
        
    } else if path_str.starts_with("aarslev/weather") || path_str.contains("winter2014.csv") || path_str.contains("data_jan_feb_2014.csv") {
        // Use AarslevWeatherRecord
        println!("   > Using AarslevWeatherRecord parser."); // Add log
         for result in rdr.deserialize::<AarslevWeatherRecord>() {
            match result {
                Ok(_record) => {
                    records_in_file += 1;
                }
                Err(e) => {
                    eprintln!("   ERROR parsing AarslevWeatherRecord in {}: {}", display_path.display(), e);
                    errors_in_file += 1;
                }
            }
        }
    // Change condition for TempSun CSV handling
    } else if path_str.contains("aarslev/temperature_sunradiation") && path_str.ends_with(".json.csv") {
        // Use AarslevTempSunRecord
        println!("   > Using AarslevTempSunRecord parser."); // Add log
         for result in rdr.deserialize::<AarslevTempSunRecord>() {
            match result {
                Ok(_record) => {
                    records_in_file += 1;
                     }
                     Err(e) => {
                    eprintln!("   ERROR parsing AarslevTempSunRecord in {}: {}", display_path.display(), e);
                    errors_in_file += 1;
                }
            }
        }
    }
    // --- Fallback for unknown CSV formats ---
    else {
        println!("   > Unrecognized CSV format: {}. Attempting generic parse.", display_path.display());
        for result in rdr.deserialize::<GenericRecord>() { // Use GenericRecord
            match result {
                Ok(_record) => {
                    // Generic record successfully parsed (likely just headers or simple KV)
                    records_in_file += 1;
                }
                Err(e) => {
                    eprintln!("   ERROR parsing GenericRecord in {}: {}", display_path.display(), e);
                    errors_in_file += 1;
                }
            }
        }
    }

    println!(
        "   > Finished processing {}: {} records parsed, {} errors.",
        display_path.display(),
        records_in_file,
        errors_in_file
    );
    *total_records_parsed += records_in_file;
    *total_parse_errors += errors_in_file;

    Ok(())
} 