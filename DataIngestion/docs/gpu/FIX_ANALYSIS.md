# GPU Feature Extraction Fix Analysis

## Problems Identified

### 1. SQL Performance Issue
- Fetching 1.4+ million rows per era (entire dataset!)
- Each query takes >1 second
- No LIMIT clause or pagination
- Level A has only 12 eras, but each era spans the ENTIRE dataset (2013-2016)

### 2. Memory Exhaustion → Segmentation Fault
- 1.4M rows × 13 sensors × 4 bytes = ~73MB per era
- 12 eras × 73MB = ~876MB just for raw data
- Plus GPU feature computation memory
- Causes GPU memory exhaustion → exit 139

### 3. Database Write Strategy
- Writing after each batch is fine
- But the batch size (1000) doesn't matter when each era has 1.4M rows!

## Root Cause
The era table likely has incorrect time ranges, causing each era to span the entire dataset instead of specific periods.

## Solutions

### Quick Fix: Add LIMIT to SQL query
```rust
// In db.rs, modify fetch_era_data query:
let query = r#"
    SELECT ... 
    FROM sensor_data_merged
    WHERE time >= $1 AND time < $2
    ORDER BY time
    LIMIT 100000  -- Cap at 100K rows max
"#;
```

### Proper Fix: Investigate era definitions
```sql
-- Check what's in era_labels_level_a
SELECT era_id, start_time, end_time, rows,
       end_time - start_time as duration
FROM era_labels_level_a
WHERE rows >= 1000
ORDER BY start_time
LIMIT 20;
```

### Memory-Safe Processing
1. Process eras with smaller time windows
2. Stream data instead of loading all at once
3. Increase MIN_ERA_ROWS to filter out huge eras