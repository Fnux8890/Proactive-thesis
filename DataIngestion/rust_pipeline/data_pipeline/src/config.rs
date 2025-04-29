use serde::{Deserialize, Serialize};
// use std::collections::HashMap; // Remove unused import
use std::fs::File;
use std::io::{/*self,*/ BufReader}; // Remove unused self
use std::path::{/*Path,*/ PathBuf}; // Remove unused Path import
use crate::errors::ConfigError; // Assuming errors.rs is in the same directory
use std::collections::HashMap;

// Define the structure for individual column mapping rules
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ColumnMapping {
    #[serde(rename = "source_column")]
    pub source: String, // Source column name or index (parser handles name vs index)
    #[serde(rename = "target_field")]
    pub target: String, // Target field name in ParsedRecord
    pub data_type: String, // Expected target data type
}

// Define the structure for timestamp parsing information
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct TimestampInfo {
    #[serde(default)]
    pub date_col: Option<usize>,
    #[serde(default)]
    pub time_col: Option<usize>,
    #[serde(default)]
    pub datetime_col: Option<usize>,
    #[serde(default)]
    pub unix_ms_col: Option<usize>,
    #[serde(default)]
    pub date_col_name: Option<String>,
    #[serde(default)]
    pub time_col_name: Option<String>,
    #[serde(default)]
    pub datetime_col_name: Option<String>,
    #[serde(default)]
    pub unix_ms_col_name: Option<String>,
    #[serde(default)]
    pub start_col_name: Option<String>,
    #[serde(default)]
    pub end_col_name: Option<String>,
    pub format: Option<String>, // Format string required for most strategies
}

// Added struct for the value in stream_map
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StreamMapTarget {
    pub target: String, // Target field name in ParsedRecord
    #[serde(rename = "type")] // Match JSON key "type"
    pub data_type: String, // Expected data type ("float", etc.)
}

// Define the main configuration structure for each file entry
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FileConfig {
    pub workspace_path: PathBuf,
    pub container_path: PathBuf,
    #[serde(default = "default_status")]
    pub status: String,
    pub format_type: String,
    #[serde(default)]
    pub source_system: Option<String>,
    #[serde(default)]
    pub delimiter: Option<String>,
    #[serde(default)]
    pub quoting: Option<serde_json::Value>,
    #[serde(default)]
    pub header_rows: Option<usize>,
    #[serde(default)]
    pub timestamp_info: Option<TimestampInfo>,
    #[serde(default)]
    pub column_map: Option<Vec<ColumnMapping>>,
    #[serde(default)]
    pub null_markers: Option<Vec<String>>,
    // Added stream_map field for JSON stream formats
    #[serde(default)]
    pub stream_map: Option<HashMap<String, StreamMapTarget>>, // UUID -> Target/Type mapping

    // ADDED: Optional field to identify the source group/category for this file
    #[serde(default)]
    pub lamp_group_id: Option<String>,
}

// Function to provide a default status if not present in JSON
fn default_status() -> String {
    "pending".to_string()
}

// Function to load the configuration from the JSON file
pub fn load_config(path_str: &str) -> Result<Vec<FileConfig>, ConfigError> {
    let path = PathBuf::from(path_str);
    if !path.exists() {
        // NOTE: This check might not be reliable with container volume mount timing/permissions
        eprintln!("DEBUG: path.exists() check returned false for {}", path.display());
        return Err(ConfigError::NotFound { path });
    }
    eprintln!("DEBUG: path.exists() check returned true for {}", path.display());

    let file = File::open(&path).map_err(|e| {
        eprintln!("DEBUG: File::open failed for {}: {}", path.display(), e); // Add debug print
        ConfigError::IoError { path: path.clone(), source: e }
    })?;
    eprintln!("DEBUG: File::open succeeded for {}", path.display());
    let reader = BufReader::new(file);

    let configs: Vec<FileConfig> = serde_json::from_reader(reader).map_err(|e| {
         eprintln!("DEBUG: serde_json::from_reader failed for {}: {}", path.display(), e); // Add debug print
         ConfigError::JsonParseError {
             path: path.clone(),
             source: e,
         }
    })?;

    Ok(configs)
} 