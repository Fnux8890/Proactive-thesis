use thiserror::Error;
use std::io;
use std::path::PathBuf;

#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("IO error reading config file {path}: {source}")]
    IoError {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("Failed to parse JSON configuration in {path}: {source}")]
    JsonParseError {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("Configuration file not found at {path}")]
    NotFound { path: PathBuf },
}

// Placeholder for future errors
#[derive(Error, Debug)]
pub enum PipelineError {
    #[error("Configuration loading failed: {0}")]
    Config(#[from] ConfigError),
    #[error("Parsing failed for {1}: {0}")]
    Parse(ParseError, PathBuf),
    #[error("Unsupported format type '{format_type}' for file {path}")]
    UnsupportedFormat { format_type: String, path: PathBuf },
    // Add other pipeline-level errors later (e.g., Database)
}

#[derive(Error, Debug)]
pub enum ParseError {
    #[error("IO error reading data file {path}: {source}")]
    IoError {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("Timestamp parsing error in {path} at row {row} for value '{value}' with format '{format}': {message}")]
    TimestampParseError {
        path: PathBuf,
        row: usize,
        value: String,
        format: String,
        message: String,
    },
    #[error("Type conversion error in {path} at row {row}, column '{column}': {details}")]
    TypeConversion {
        path: PathBuf,
        row: usize,
        column: String,
        details: String,
    },
    #[error("JSON parsing error in {path}: {source}")]
    JsonParseError {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
     #[error("Unsupported file format type specified in config for {path}: {format_type}")]
    UnsupportedFormatType {
        path: PathBuf,
        format_type: String,
    },
    #[error("Error reading CSV headers in {path}: {source}")]
    HeaderReadError {
        path: PathBuf,
        #[source]
        source: csv::Error,
    },
    #[error("Configuration error in {path} for field '{field}': {message}")]
    ConfigError {
        path: PathBuf,
        field: String,
        message: String,
    },
    // Add other specific parsing errors as needed
} 