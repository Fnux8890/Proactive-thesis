use std::io;
use std::path::PathBuf;
use thiserror::Error;

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
    Config(String), // Keep simple string for now for DB URL error
    #[error("Configuration parsing failed: {0}")]
    ConfigParse(#[from] ConfigError),
    #[error("Parsing failed for {1}: {0}")]
    Parse(ParseError, PathBuf),
    #[error("Unsupported format type '{format_type}' for file {path}")]
    UnsupportedFormat { format_type: String, path: PathBuf },
    #[error("Database pool creation error: {0}")]
    DbPoolError(String), // From db.rs
    #[error("Database operation failed: {0}")]
    DbQueryError(#[from] tokio_postgres::Error),
    #[error("Failed to get database connection from pool: {0}")]
    DbConnectionError(#[from] deadpool_postgres::PoolError),
    #[error("Schema mismatch for table '{table}': Missing columns: {missing:?}, Extra columns: {extra:?}")]
    SchemaMismatch {
        table: String,
        missing: Vec<String>,
        extra: Vec<String>,
    },
    #[error("Data integrity check failed: Type='{check_type}', File='{source_file:?}', Column='{column_name}', Value='{value}'")]
    DataIntegrityError {
        check_type: String,
        source_file: Option<String>,
        column_name: String,
        value: String,
    },
    #[error("Merge script execution failed: {0}")]
    MergeScriptError(String),
    #[error("Channel communication error: {0}")]
    ChannelError(String),
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
    #[error("JSON parsing error in {path}: {source}")]
    JsonParseError {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
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

// --- Validation Error Enum ---

#[derive(Error, Debug)]
pub enum ValidationError {
    // Add other validation error types here (e.g., RequiredFieldMissing, InvalidTimestampOrder)
}
