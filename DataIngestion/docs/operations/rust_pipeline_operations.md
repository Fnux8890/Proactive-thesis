# Rust Data Ingestion Pipeline Operations Guide

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Sequence Diagrams](#sequence-diagrams)
4. [Core Components](#core-components)
5. [Data Flow](#data-flow)
6. [Database Schema](#database-schema)
7. [Configuration](#configuration)
8. [Error Handling](#error-handling)
9. [Migration Guide to C#](#migration-guide-to-c)
10. [Performance Considerations](#performance-considerations)

## Overview

The Rust data ingestion pipeline is a high-performance system designed to:
- Process multiple file formats (CSV, JSON, custom formats)
- Validate and transform sensor data
- Efficiently load data into TimescaleDB
- Merge and upsample data for analysis

### Key Features
- **Async/concurrent processing** using Tokio runtime
- **Bulk data loading** via PostgreSQL binary COPY
- **Flexible parser architecture** supporting multiple formats
- **Comprehensive validation** with graceful error handling
- **Connection pooling** for database efficiency

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Data Files    │────▶│  File Processor  │────▶│     Parsers     │
│ (CSV/JSON/etc) │     │                  │     │ (Format-based)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                           │
                                                           ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   TimescaleDB   │◀────│  Binary COPY     │◀────│   Validation    │
│  (sensor_data)  │     │   Bulk Insert    │     │   & Transform   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│   Merge Data    │────▶│  Upsampled Data  │
│  (SQL Script)   │     │    (1-minute)    │
└─────────────────┘     └──────────────────┘
```

## Sequence Diagrams

### Main Pipeline Flow

```mermaid
sequenceDiagram
    participant Main
    participant Config
    participant DB
    participant FileProcessor
    participant Parser
    participant Validator
    participant TimescaleDB

    Main->>Config: Load data_files.json
    Config-->>Main: FileConfigurations
    Main->>DB: Create connection pool
    DB-->>Main: Pool<AsyncPgConnection>
    Main->>TimescaleDB: Create sensor_data table
    
    loop For each file configuration
        Main->>FileProcessor: Process files (glob pattern)
        FileProcessor->>Parser: Route by format_type
        Parser->>Parser: Parse records
        Parser-->>FileProcessor: Vec<ParsedRecord>
        FileProcessor->>Validator: Validate records
        Validator-->>FileProcessor: Valid records
        FileProcessor->>Main: Parsed & validated data
        Main->>TimescaleDB: Binary COPY insert
        TimescaleDB-->>Main: Insert confirmation
    end
    
    Main->>TimescaleDB: Execute merge_sensor_data.sql
    TimescaleDB-->>Main: Merged data created
```

### Parser Selection Flow

```mermaid
sequenceDiagram
    participant FileProcessor
    participant CSVParser
    participant JSONParser
    participant AarslevParser
    participant KnudJepsenParser

    FileProcessor->>FileProcessor: Check format_type
    
    alt format_type = "csv"
        FileProcessor->>CSVParser: parse_csv()
        CSVParser-->>FileProcessor: Vec<ParsedRecord>
    else format_type = "json"
        FileProcessor->>JSONParser: parse_json()
        JSONParser-->>FileProcessor: Vec<ParsedRecord>
    else format_type = "aarslev_morten_sdu"
        FileProcessor->>AarslevParser: parse_aarslev()
        AarslevParser-->>FileProcessor: Vec<ParsedRecord>
    else format_type = "knudjepsen"
        FileProcessor->>KnudJepsenParser: parse_knudjepsen()
        KnudJepsenParser-->>FileProcessor: Vec<ParsedRecord>
    end
```

### Database Operations Flow

```mermaid
sequenceDiagram
    participant Main
    participant ConnectionPool
    participant Transaction
    participant BinaryCopy
    participant TimescaleDB

    Main->>ConnectionPool: Get connection
    ConnectionPool-->>Main: AsyncPgConnection
    Main->>Transaction: Begin transaction
    
    Main->>BinaryCopy: Prepare COPY command
    BinaryCopy->>BinaryCopy: Convert records to binary
    
    loop Batch of records
        BinaryCopy->>TimescaleDB: Stream binary data
        TimescaleDB-->>BinaryCopy: Acknowledge
    end
    
    BinaryCopy-->>Main: Copy complete
    Main->>Transaction: Commit
    Transaction->>TimescaleDB: Finalize
    TimescaleDB-->>Main: Success
```

## Core Components

### 1. Main Entry Point (`main.rs`)

```rust
#[tokio::main]
async fn main() -> Result<()> {
    // 1. Initialize logging
    env_logger::init();
    
    // 2. Load configuration
    let configs = load_config("data_files.json")?;
    
    // 3. Create database pool
    let pool = create_pool().await?;
    
    // 4. Create tables
    create_sensor_table(&pool).await?;
    
    // 5. Process each configuration
    for config in configs {
        process_file_configuration(&pool, &config).await?;
    }
    
    // 6. Merge data
    execute_merge_script(&pool).await?;
    
    Ok(())
}
```

### 2. Configuration Loading (`config.rs`)

```rust
#[derive(Debug, Deserialize, Clone)]
pub struct FileConfiguration {
    pub workspace_path: String,
    pub container_path: String,
    pub format_type: String,
    pub source_system: String,
    pub column_mappings: HashMap<String, ColumnMapping>,
    pub timestamp_strategy: TimestampStrategy,
    pub null_markers: Vec<String>,
    pub lamp_group: Option<String>,
}

pub fn load_config(path: &str) -> Result<Vec<FileConfiguration>> {
    let content = fs::read_to_string(path)?;
    let configs: Vec<FileConfiguration> = serde_json::from_str(&content)?;
    Ok(configs)
}
```

### 3. Data Models (`data_models.rs`)

```rust
#[derive(Debug, Default, Clone)]
pub struct ParsedRecord {
    pub time: Option<DateTime<Utc>>,
    pub source_system: Option<String>,
    pub source_file: Option<String>,
    pub format_type: Option<String>,
    pub uuid: Option<String>,
    pub lamp_group: Option<String>,
    
    // Environmental sensors (56 fields)
    pub air_temp_c: Option<f64>,
    pub air_temp_middle_c: Option<f64>,
    pub relative_humidity_percent: Option<f64>,
    pub co2_measured_ppm: Option<f64>,
    pub light_intensity_umol: Option<f64>,
    // ... 51 more fields
}
```

### 4. File Processing (`file_processor.rs`)

```rust
pub async fn process_files(
    config: &FileConfiguration,
    pool: &PgPool
) -> Result<Vec<ParsedRecord>> {
    let mut all_records = Vec::new();
    
    // Find files matching pattern
    let files = glob(&config.container_path)?;
    
    for file_path in files {
        let records = match config.format_type.as_str() {
            "csv" => parse_csv(&file_path, config)?,
            "json" => parse_json(&file_path, config)?,
            "aarslev_morten_sdu" => parse_aarslev(&file_path, config)?,
            "knudjepsen" => parse_knudjepsen(&file_path, config)?,
            _ => return Err(anyhow!("Unknown format type")),
        };
        
        all_records.extend(records);
    }
    
    Ok(all_records)
}
```

### 5. Validation (`validation.rs`)

```rust
pub fn validate_record(record: &ParsedRecord) -> Result<()> {
    // Required field validation
    if record.time.is_none() {
        return Err(anyhow!("Missing timestamp"));
    }
    
    // Range validation (if enabled)
    if let Some(temp) = record.air_temp_c {
        if temp < -50.0 || temp > 100.0 {
            return Err(anyhow!("Temperature out of range"));
        }
    }
    
    // Additional validations...
    Ok(())
}
```

## Data Flow

### 1. Input Processing
- Files are discovered using glob patterns from configuration
- Each file is routed to the appropriate parser based on `format_type`
- Parsers handle format-specific logic and timestamp parsing

### 2. Data Transformation
- Raw data is converted to `ParsedRecord` structs
- Null markers are replaced with `None`
- Timestamps are parsed according to configured strategy
- Source metadata is attached to each record

### 3. Validation & Error Handling
- Records are validated for required fields
- Invalid records are logged to CSV files
- Processing continues despite individual record errors

### 4. Database Loading
- Records are batched for efficient insertion
- PostgreSQL binary COPY protocol is used
- Transactions ensure data consistency
- Connection pooling manages resources

### 5. Post-Processing
- Merge script combines duplicate records
- LOCF upsampling creates regular time series
- Indexes are created for query performance

## Database Schema

### sensor_data Table

```sql
CREATE TABLE IF NOT EXISTS sensor_data (
    time TIMESTAMPTZ NOT NULL,
    source_system TEXT,
    source_file TEXT,
    format_type TEXT,
    uuid TEXT,
    lamp_group TEXT,
    
    -- Environmental measurements
    air_temp_c DOUBLE PRECISION,
    air_temp_middle_c DOUBLE PRECISION,
    relative_humidity_percent DOUBLE PRECISION,
    co2_measured_ppm DOUBLE PRECISION,
    light_intensity_umol DOUBLE PRECISION,
    radiation_w_m2 DOUBLE PRECISION,
    
    -- Control systems
    heating_setpoint_c DOUBLE PRECISION,
    pipe_temp_1_c DOUBLE PRECISION,
    pipe_temp_2_c DOUBLE PRECISION,
    flow_temp_1_c DOUBLE PRECISION,
    flow_temp_2_c DOUBLE PRECISION,
    
    -- Ventilation
    vent_lee_afd3_percent DOUBLE PRECISION,
    vent_lee_afd4_percent DOUBLE PRECISION,
    vent_wind_afd3_percent DOUBLE PRECISION,
    vent_wind_afd4_percent DOUBLE PRECISION,
    
    -- Lighting
    lamp_grp1_no3_status DOUBLE PRECISION,
    lamp_grp1_no4_status DOUBLE PRECISION,
    -- ... more lamp statuses
    
    -- Additional fields (62 total)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('sensor_data', 'time', if_not_exists => TRUE);
```

## Configuration

### data_files.json Structure

```json
[
  {
    "workspace_path": "/data/aarslev/april2014/*.csv",
    "container_path": "/app/data/aarslev/april2014/*.csv",
    "format_type": "aarslev_morten_sdu",
    "source_system": "aarslev",
    "column_mappings": {
      "Time": {
        "target_field": "time",
        "data_type": "timestamp",
        "timestamp_format": "%d-%m-%Y %H:%M:%S"
      },
      "Mål temp afd. mid [°C]": {
        "target_field": "air_temp_middle_c",
        "data_type": "float"
      }
    },
    "timestamp_strategy": {
      "type": "column",
      "column_name": "Time",
      "format": "%d-%m-%Y %H:%M:%S"
    },
    "null_markers": ["", "NULL", "null", "NaN", "na", "n/a"],
    "lamp_group": null
  }
]
```

### Environment Variables

```env
DATABASE_URL=postgresql://postgres:postgres@timescaledb:5432/postgres
REDIS_URL=redis://redis:6379
DATA_SOURCE_PATH=/app/data
RUST_LOG=info
```

## Error Handling

### Error Types

```rust
#[derive(Error, Debug)]
pub enum PipelineError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Database error: {0}")]
    Database(#[from] tokio_postgres::Error),
    
    #[error("Parse error: {0}")]
    Parse(String),
    
    #[error("Validation error: {0}")]
    Validation(String),
    
    #[error("Configuration error: {0}")]
    Config(String),
}
```

### Error Handling Strategy

1. **Graceful Degradation**: Continue processing valid records
2. **Error Logging**: Save failed records to CSV with error details
3. **Transaction Safety**: Rollback on batch failures
4. **Context Preservation**: Include file path, row number in errors

## Migration Guide to C#

### 1. Project Structure

```csharp
// C# equivalent structure
namespace DataIngestion
{
    public class Program
    {
        static async Task Main(string[] args)
        {
            var configs = ConfigLoader.LoadConfigurations("data_files.json");
            using var connectionPool = new NpgsqlConnectionPool(connectionString);
            
            await DatabaseSetup.CreateTablesAsync(connectionPool);
            
            foreach (var config in configs)
            {
                await FileProcessor.ProcessFilesAsync(connectionPool, config);
            }
            
            await DatabaseSetup.ExecuteMergeScriptAsync(connectionPool);
        }
    }
}
```

### 2. Data Models

```csharp
public class ParsedRecord
{
    public DateTime? Time { get; set; }
    public string SourceSystem { get; set; }
    public string SourceFile { get; set; }
    public string FormatType { get; set; }
    public string Uuid { get; set; }
    public string LampGroup { get; set; }
    
    // Environmental measurements
    public double? AirTempC { get; set; }
    public double? AirTempMiddleC { get; set; }
    public double? RelativeHumidityPercent { get; set; }
    public double? Co2MeasuredPpm { get; set; }
    // ... additional fields
}
```

### 3. Parser Architecture

```csharp
public interface IDataParser
{
    Task<List<ParsedRecord>> ParseAsync(string filePath, FileConfiguration config);
}

public class CsvParser : IDataParser
{
    public async Task<List<ParsedRecord>> ParseAsync(string filePath, FileConfiguration config)
    {
        using var reader = new StreamReader(filePath);
        using var csv = new CsvReader(reader, CultureInfo.InvariantCulture);
        
        var records = new List<ParsedRecord>();
        await foreach (var row in csv.GetRecordsAsync<dynamic>())
        {
            var record = MapToRecord(row, config);
            records.Add(record);
        }
        
        return records;
    }
}
```

### 4. Database Operations

```csharp
public class DatabaseService
{
    private readonly NpgsqlConnectionPool _pool;
    
    public async Task BulkInsertAsync(List<ParsedRecord> records)
    {
        using var connection = await _pool.OpenConnectionAsync();
        using var writer = connection.BeginBinaryImport(
            "COPY sensor_data (time, source_system, ...) FROM STDIN (FORMAT BINARY)");
        
        foreach (var record in records)
        {
            writer.StartRow();
            writer.Write(record.Time ?? DateTime.MinValue, NpgsqlDbType.TimestampTz);
            writer.Write(record.SourceSystem ?? string.Empty);
            // ... write other fields
        }
        
        await writer.CompleteAsync();
    }
}
```

### 5. Configuration Management

```csharp
public class FileConfiguration
{
    public string WorkspacePath { get; set; }
    public string ContainerPath { get; set; }
    public string FormatType { get; set; }
    public string SourceSystem { get; set; }
    public Dictionary<string, ColumnMapping> ColumnMappings { get; set; }
    public TimestampStrategy TimestampStrategy { get; set; }
    public List<string> NullMarkers { get; set; }
    public string LampGroup { get; set; }
}

public class ConfigLoader
{
    public static List<FileConfiguration> LoadConfigurations(string path)
    {
        var json = File.ReadAllText(path);
        return JsonSerializer.Deserialize<List<FileConfiguration>>(json);
    }
}
```

### 6. Async/Parallel Processing

```csharp
public class FileProcessor
{
    public static async Task ProcessFilesAsync(
        NpgsqlConnectionPool pool, 
        FileConfiguration config)
    {
        var files = Directory.GetFiles(
            Path.GetDirectoryName(config.ContainerPath), 
            Path.GetFileName(config.ContainerPath));
        
        var tasks = files.Select(file => ProcessSingleFileAsync(pool, file, config));
        await Task.WhenAll(tasks);
    }
}
```

### 7. Key C# Libraries to Use

- **Npgsql**: PostgreSQL driver with binary COPY support
- **CsvHelper**: CSV parsing
- **Newtonsoft.Json**: JSON parsing
- **Serilog**: Structured logging
- **Polly**: Retry policies and resilience
- **System.Threading.Channels**: Producer/consumer patterns
- **System.IO.Pipelines**: High-performance I/O

### 8. Performance Considerations for C#

1. **Use `ValueTask` for hot paths**
2. **Implement `IAsyncEnumerable` for streaming**
3. **Use `ArrayPool<T>` for buffer management**
4. **Leverage `Span<T>` for parsing operations**
5. **Use `Channel<T>` for producer/consumer patterns**
6. **Implement batch processing with configurable sizes**

## Performance Considerations

### Optimization Strategies

1. **Binary COPY Protocol**
   - 10-100x faster than individual INSERTs
   - Minimal parsing overhead
   - Direct binary format

2. **Connection Pooling**
   - Reuse database connections
   - Configurable pool size
   - Automatic connection management

3. **Batch Processing**
   - Process records in configurable batches
   - Balance memory usage vs. performance
   - Transaction scope optimization

4. **Async I/O**
   - Non-blocking file operations
   - Concurrent database operations
   - Efficient resource utilization

5. **Memory Management**
   - Stream large files instead of loading fully
   - Use iterators for record processing
   - Clear collections after batch insert

### Monitoring & Metrics

- Track records processed per second
- Monitor memory usage during processing
- Log failed record counts and reasons
- Measure database insertion rates
- Profile CPU usage for parser bottlenecks

### Scalability Considerations

1. **Horizontal Scaling**
   - Process different file sets on multiple instances
   - Use message queues for work distribution
   - Implement idempotent operations

2. **Vertical Scaling**
   - Increase batch sizes for more memory
   - Use more connection pool slots
   - Leverage multi-core for parallel parsing

3. **Database Optimization**
   - Partition hypertables by time
   - Create appropriate indexes after bulk load
   - Use tablespaces for data distribution
   - Configure autovacuum appropriately