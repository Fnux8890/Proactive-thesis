# Null Value Logging Implementation for Rust Pipeline

## Overview
This document describes the implementation of comprehensive null value logging for the Rust data ingestion pipeline to track database operations and ensure data quality.

## Controlling Null Value Logging

### Enabling/Disabling Logging
The null value logging can be controlled via the `ENABLE_NULL_LOGGING` environment variable:

- **To Enable**: Set `ENABLE_NULL_LOGGING=true`
- **To Disable**: Set `ENABLE_NULL_LOGGING=false` (or leave unset)

### Methods to Control

#### 1. Via docker-compose.yml (Default: Disabled)
```yaml
environment:
  ENABLE_NULL_LOGGING: "false"  # Change to "true" to enable
```

#### 2. Via Command Line Override
```bash
# Enable logging for a single run
ENABLE_NULL_LOGGING=true docker compose up rust_pipeline

# Or using docker compose run
docker compose run -e ENABLE_NULL_LOGGING=true rust_pipeline
```

#### 3. Via .env File
Create a `.env` file in the same directory as docker-compose.yml:
```env
ENABLE_NULL_LOGGING=true
```

Then update docker-compose.yml:
```yaml
environment:
  ENABLE_NULL_LOGGING: ${ENABLE_NULL_LOGGING:-false}
```

## Changes Implemented

### 1. Enhanced Database Operations (`src/db_operations.rs`)

#### Added Null Value Tracking
- Added `null_columns_log` vector to track records with null values
- Implemented comprehensive null field checking for all 45+ data columns
- Each null occurrence is logged with:
  - Source file name
  - Batch index
  - Record index within batch
  - Timestamp (if available)
  - Comma-separated list of null column names

#### Null Column Logging
```rust
// Track null values for each column
let mut null_fields = Vec::new();

// Check each field for null values
if record.air_temp_c.is_none() { null_fields.push("air_temp_c"); }
if record.relative_humidity_percent.is_none() { null_fields.push("relative_humidity_percent"); }
// ... (checks for all columns)

// Log null values if any exist
if !null_fields.is_empty() {
    null_columns_log.push((
        record.source_file.as_deref().unwrap_or("Unknown").to_string(),
        batch_idx,
        i,
        record.timestamp_utc.as_ref().map(|t| t.to_string()).unwrap_or_default(),
        null_fields.join(",")
    ));
}
```

#### Batch-Level Reporting
- Creates separate log files for each batch: `/app/logs/null_columns_batch_{batch_idx}.txt`
- Each log file contains:
  - Batch number
  - Total records with null values
  - Detailed breakdown of each record's null columns
  - Format: `SourceFile | BatchIdx | RecordIdx | Timestamp | NullColumns`

### 2. Updated Parser Support (`src/parsers/aarslev_morten_sdu_parser.rs`)

#### Expanded Field Mapping
Updated the `set_field!` macro to support all database columns:
- **Environmental measurements**: air_temp_c, humidity, CO2, radiation, light intensity
- **Control systems**: ventilation, heating, curtains, windows
- **Operational data**: lamp status, dosing status
- **Forecasts**: temperature and radiation predictions
- **Calculated values**: VPD, humidity deficit, DLI sum

### 3. Log File Structure

#### Skipped Records Log (`/app/logs/skipped_timestamp_rows.csv`)
Tracks records that cannot be inserted due to missing timestamps:
```csv
SourceFile,BatchIndex,RecordIndex,UUID,Reason
/app/data/aarslev/celle5/output-2014-01-01-00-00.csv,0,15,N/A,MissingTimestamp
```

#### Null Columns Logs (`/app/logs/null_columns_batch_*.txt`)
Detailed null value tracking per batch:
```
=== Batch 0 Null Columns Report ===
Total records with nulls: 1234
Format: SourceFile | BatchIdx | RecordIdx | Timestamp | NullColumns
========================================
/app/data/aarslev/celle5/output-2014-01-01-00-00.csv | 0 | 10 | 2014-01-01T10:00:00Z | co2_required_ppm,heating_setpoint_c,vpd_hpa
/app/data/aarslev/celle5/output-2014-01-01-00-00.csv | 0 | 11 | 2014-01-01T10:10:00Z | co2_required_ppm,vpd_hpa
```

## Benefits

1. **Data Quality Monitoring**
   - Identify patterns of missing data across different sources
   - Track sensor failures or data collection issues
   - Validate data completeness before analysis

2. **Debugging Support**
   - Quickly identify which columns are frequently null
   - Trace back issues to specific files and timestamps
   - Understand data gaps for different greenhouse locations

3. **Operational Insights**
   - Detect sensor malfunctions
   - Identify periods of incomplete data collection
   - Support data imputation strategies

## Usage

### Running the Pipeline
The null value logging is automatically enabled when running the pipeline:
```bash
docker compose up data_pipeline
```

### Accessing Logs
Logs are stored in the container at `/app/logs/`:
- `skipped_timestamp_rows.csv` - Records without timestamps
- `null_columns_batch_*.txt` - Null value reports per batch

To extract logs from the container:
```bash
docker cp data_pipeline:/app/logs ./pipeline_logs
```

### Analyzing Null Patterns
Use the logs to:
1. Identify columns that need default values
2. Detect systematic data collection issues
3. Validate parser implementations
4. Guide data cleaning strategies

## Future Enhancements

1. **Summary Statistics**
   - Add overall null percentage per column
   - Generate daily/weekly null value reports
   - Create visualization dashboards

2. **Alerting**
   - Set thresholds for acceptable null percentages
   - Send alerts when null values exceed limits
   - Monitor specific critical columns

3. **Integration**
   - Export null statistics to monitoring systems
   - Include in data quality metrics
   - Support for null value imputation strategies