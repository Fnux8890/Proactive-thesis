# Timestamp Parsing Fix for Era Detector

## Issue
The era detector was reporting all time values as NULL even though the database contained valid timestamps.

## Root Cause
When PostgreSQL exports data via `COPY TO STDOUT WITH CSV`, timestamp columns are exported as strings (e.g., "2013-12-01 00:00:00+00"). Polars' CSV reader doesn't automatically parse these string timestamps into datetime types, causing them to appear as NULL when accessed as datetime values.

## Solution
Added timestamp parsing logic in `db.rs` after loading CSV data:

```rust
// Parse time column if it's a string
let df = if df.column("time").is_ok() {
    let time_dtype = df.column("time")?.dtype();
    if matches!(time_dtype, DataType::String) {
        log::info!("Time column is String type, parsing to DateTime");
        df.lazy()
            .with_column(
                col("time").str().strptime(
                    DataType::Datetime(TimeUnit::Microseconds, Some("UTC".into())),
                    Some(StrptimeOptions {
                        format: Some("%Y-%m-%d %H:%M:%S%:z".into()),
                        ..Default::default()
                    }),
                    lit("raise"),
                )
            )
            .collect()?
    } else {
        df
    }
} else {
    df
};
```

## Table Structure
The `preprocessed_features` table has this structure:
- `time` (TIMESTAMPTZ) - Separate column, NOT in JSONB
- `era_identifier` (TEXT)
- `features` (JSONB) - Contains all sensor data

## Testing
To verify the fix works:
```bash
cd /mnt/d/GitKraken/Proactive-thesis/DataIngestion
docker compose up --build era_detector
```

The era detector should now properly load and process the timestamp data without reporting NULL values.