# CUDA_ERROR_INVALID_VALUE Fix

## Root Causes Found

### 1. Era Detection is Broken
The SQL investigation revealed that eras span **YEARS** instead of hours/days:
- Era 144: **1011 days** (2013-12-01 to 2016-09-08)
- Era 251: **1011 days** 
- All Level A eras end at 2016-09-08 (dataset end)

This is completely wrong - the era detection algorithm failed to properly segment the data.

### 2. Shared Memory Configuration Error
The `compute_statistics` kernel declares `extern __shared__ float shared[];` but we were passing `shared_mem_bytes: 0`, causing CUDA to reject the launch with `CUDA_ERROR_INVALID_VALUE`.

## Fixes Applied

### 1. Fixed Shared Memory (features.rs)
```rust
// Calculate shared memory size for the kernel
let num_warps = (block_size + 31) / 32; // 8 warps for 256 threads
let shared_mem_bytes = (num_warps * 3 * std::mem::size_of::<f32>()) as u32; // 96 bytes

let config = LaunchConfig {
    grid_dim: (grid_size, 1, 1),
    block_dim: (block_size, 1, 1),
    shared_mem_bytes,  // Fixed: was 0, now 96 bytes
};
```

### 2. Added Struct Size Check (kernels.rs)
```rust
// Compile-time check: ensure struct size matches between host and device
const _: () = assert!(std::mem::size_of::<StatisticalFeatures>() == 24);
```

### 3. SQL LIMIT for Safety (db.rs)
```sql
LIMIT 100000  -- Prevents fetching millions of rows
```

## Testing Steps

1. **Rebuild the container**:
```bash
docker compose build gpu_feature_extraction
```

2. **Test with single era**:
```bash
docker compose run --rm gpu_feature_extraction_level_a \
  --max-eras 1 --batch-size 1 --min-era-rows 10
```

3. **Enable debug mode**:
```bash
CUDA_LAUNCH_BLOCKING=1 RUST_BACKTRACE=1 docker compose up gpu_feature_extraction_level_a
```

## Next Steps

### Fix Era Detection
The real problem is that era detection created spans of years. Need to:

1. **Check era detection parameters**:
```bash
# Look at era_detector command in docker-compose.yml
--pelt-min-size 48
--bocpd-lambda 200.0
--hmm-states 5
```

2. **Re-run era detection** with better parameters:
```bash
# Increase min-size to create smaller eras
--pelt-min-size 288  # 1 hour minimum
--bocpd-lambda 50.0  # More sensitive to changes
```

3. **Validate era sizes**:
```sql
-- After re-running era detection
SELECT COUNT(*), 
       AVG(EXTRACT(EPOCH FROM (end_time - start_time))/3600) as avg_hours
FROM era_labels_level_a;
```

## Expected Results

With the shared memory fix:
- The CUDA launch error should disappear
- Features should compute successfully
- But processing will still be slow due to 100K rows per era

With proper era detection:
- Eras should be hours/days, not years
- Much faster processing
- No need for SQL LIMIT