use crate::config::FileConfig;
use crate::data_models::ParsedRecord;
use crate::errors::PipelineError;
use crate::parsers;

/// Processes a single file based on its configuration.
/// Selects the appropriate parser and returns the parsed data or an error.
pub fn process_file(config_entry: &FileConfig) -> Result<Vec<ParsedRecord>, PipelineError> {
    println!(
        "INFO: Processing file: {} with format type: {}",
        config_entry.container_path.display(),
        config_entry.format_type
    );

    // Match on the format_type string to determine the parser
    let parsed_result = match config_entry.format_type.as_str() {
        // --- CSV Formats ---
        "KnudJepsenNO3Lamp" | 
        "KnudJepsenNO4Lamp" | 
        "KnudJepsenBelysningGrp" | 
        "KnudJepsenEkstraLys" | 
        "KnudJepsenBelysning" | // Might be superseded by Grp, keep for now
        "KnudJepsenExtra" |
        "AarslevWeather" | 
        "AarslevTempSunCSV" | 
        "AarslevCelleCSV" |
        "KnudjepsenNO3NO4Extra" | 
        "KnudJepsenLighting" | 
        "AarslevDataForecast" | 
        "AarslevUUIDJsonCSV" | "CSV" => {
            // Use the enhanced generic CSV parser
            parsers::csv_parser::parse_csv(config_entry, &config_entry.container_path)
                .map_err(|parse_err| PipelineError::Parse(parse_err, config_entry.container_path.clone()))
        }

        "AarslevMortenSDU1" |
        "AarslevMortenSDU2" |
        "AarslevMortenSDU3" => {
            // Use the dedicated MortenSDU parser
            parsers::aarslev_morten_sdu_parser::parse_morten_sdu(config_entry, &config_entry.container_path)
                .map_err(|parse_err| PipelineError::Parse(parse_err, config_entry.container_path.clone()))
        }

        // Generic CSV handler (handled in the first arm)
        // "CSV" => { ... }, // REMOVED Unreachable arm

        // --- JSON Formats ---
        "AarslevStreamJSON" => {
            // Call the specific stream JSON parser
             parsers::json_parser::parse_aarslev_stream_json(config_entry, &config_entry.container_path)
                 .map_err(|parse_err| PipelineError::Parse(parse_err, config_entry.container_path.clone()))
         }
        // Removed AarslevCelleJSON placeholder as it refers to config files
        // Add other JSON format_type matches here...

        // Generic JSON handler (Placeholder for wildcard entries like celle*/output-*.csv.json)
        "JSON" => {
             // Currently, we don't know the structure of these files.
             // Route to the placeholder function which logs a warning.
             parsers::json_parser::parse_aarslev_celle_json(config_entry, &config_entry.container_path)
                 .map_err(|parse_err| PipelineError::Parse(parse_err, config_entry.container_path.clone()))
         }

        // --- Unknown/Unsupported Format ---
        unsupported_format => { // Use variable name for clarity
            eprintln!("ERROR: Unsupported format_type '{}' for file {}. No parser defined.",
                unsupported_format,
                config_entry.container_path.display()
            );
            Err(PipelineError::UnsupportedFormat {
                 format_type: config_entry.format_type.clone(),
                 path: config_entry.container_path.clone(),
            })
        }
    };

    // parsed_result is now Result<Vec<ParsedRecord>, PipelineError>
    parsed_result
}
