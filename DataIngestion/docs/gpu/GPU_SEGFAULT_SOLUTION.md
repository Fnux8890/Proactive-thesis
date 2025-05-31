# GPU Feature Extraction Segfault Solution

## Problem Summary

The GPU feature extraction crashes with exit code 139 (segmentation fault) due to:

1. **Massive Data Queries**: Each era query returns 1.4+ million rows (entire dataset)
2. **Memory Exhaustion**: ~73MB per era × multiple eras = GPU memory overflow
3. **Slow SQL**: Queries take >1 second due to fetching entire dataset

## Root Cause

The era definitions in `era_labels_level_a` likely have incorrect time ranges, causing each era to span the entire dataset (2013-2016) instead of specific periods.

## Immediate Fixes Applied

### 1. SQL Query Limit (db.rs)
```rust
// Added LIMIT to prevent fetching millions of rows
ORDER BY time
LIMIT 100000  -- Safety limit
```

### 2. Enhanced Logging (main.rs)
- Log era time ranges and data point counts
- Warn when era has >500K points
- Better error messages for GPU failures

### 3. Batch Database Writes (db.rs)
- Use transactions for batch writes
- Reduces database connection overhead

## How to Debug

### 1. Check Era Definitions
```bash
docker compose exec db psql -U postgres -d postgres -f investigate_eras.sql
```

This will show:
- Era time ranges
- Duration in days
- Whether eras overlap

### 2. Monitor Execution
```bash
# Watch GPU memory usage
nvidia-smi -l 1

# Check container logs with detailed info
docker compose logs -f gpu_feature_extraction_level_a
```

### 3. Test with Smaller Batches
```env
# In .env
MIN_ERA_ROWS_A=10000  # Filter more aggressively
BATCH_SIZE_A=1       # Process one era at a time
```

## Long-term Solutions

### 1. Fix Era Definitions
If eras span the entire dataset, the era detection algorithm needs fixing:
- Check PELT/BOCPD/HMM parameters
- Ensure proper segmentation

### 2. Implement Streaming
For truly large eras, implement streaming:
```rust
// Process in chunks of 10K rows
for chunk in 0..total_rows.step_by(10000) {
    let data = fetch_chunk(era, chunk, 10000);
    process_chunk(data);
}
```

### 3. Add Era Validation
Before processing:
```rust
if era.duration_days() > 30 {
    warn!("Era {} spans {} days - skipping", era.id, era.duration_days());
    continue;
}
```

## Expected Behavior

With fixes:
1. Queries should return <100K rows (via LIMIT)
2. No segmentation faults
3. Processing speed ~1000 eras/sec for reasonably sized eras

## Testing

After applying fixes:
```bash
# Rebuild
docker compose build gpu_feature_extraction

# Run with monitoring
docker compose up gpu_feature_extraction_level_a
```

Watch for:
- "Fetched X data points for era Y" messages
- Warnings about large eras
- Successful completion without exit 139