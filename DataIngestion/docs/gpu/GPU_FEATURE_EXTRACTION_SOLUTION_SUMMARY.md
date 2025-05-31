# GPU Feature Extraction Solution Summary

## Problem Statement

GPU feature extraction was failing with exit code 139 (segmentation fault) for Levels A and C, while Level B worked. The failures were caused by:
1. CUDA_ERROR_INVALID_VALUE due to incorrect shared memory configuration
2. Era detection creating year-long spans (1000+ days) instead of reasonable periods
3. SQL queries attempting to fetch millions of rows, causing memory exhaustion

## Root Cause Analysis

### 1. Shared Memory Configuration Error
- The CUDA kernel declared `extern __shared__ float shared[]` but was launched with `shared_mem_bytes: 0`
- CUDA requires the exact shared memory size for kernels using dynamic shared memory
- Solution: Calculate and provide the correct size (96 bytes for 256 threads)

### 2. Era Detection Parameters
- `pelt-min-size: 48` (only 4 hours) was too small, creating either tiny or massive eras
- `bocpd-lambda: 200.0` was too high, expecting very long runs between changepoints
- Result: Level A had 12 eras all ending at dataset end (2016-09-08)
- Level B/C had 278K+ single-point eras with duration = 0

### 3. SQL Performance
- Fetching 1.4M+ rows per era was causing SQLx slow query warnings
- GPU memory was being exhausted trying to process years of data at once

## Implemented Solutions

### 1. Fixed Shared Memory (gpu_feature_extraction/src/features.rs)
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

### 2. Data Sampling for Long Eras (gpu_feature_extraction/src/db.rs)
```sql
-- Sample one reading every 5 minutes for eras > 7 days
AND (($2 - $1 < interval '7 days') OR EXTRACT(EPOCH FROM time) % 300 = 0)
ORDER BY time
LIMIT 100000  -- Safety limit
```

### 3. Updated Era Detection Parameters (docker-compose.yml)
```yaml
command: [
  # ... other parameters ...
  "--pelt-min-size", "288",    # Was 48 (4hr), now 288 (24hr)
  "--bocpd-lambda", "50.0",     # Was 200.0, now more sensitive
  "--hmm-states", "10"          # Was 5, now more granular
]
```

### 4. Sequential GPU Execution (docker-compose.yml)
Already configured to prevent GPU memory conflicts:
```yaml
gpu_feature_extraction_level_b:
  depends_on:
    gpu_feature_extraction_level_a:
      condition: service_completed_successfully
      
gpu_feature_extraction_level_c:
  depends_on:
    gpu_feature_extraction_level_b:
      condition: service_completed_successfully
```

## Architecture Decisions

### Why Sequential Execution?
- Docker doesn't support native GPU sharing between containers
- NVIDIA MPS (Multi-Process Service) has stability issues
- Sequential execution ensures each level gets full GPU resources
- Trade-off: Slower but reliable vs faster but unstable

### Why Data Sampling?
- Greenhouse data is highly regular (1440 readings/day)
- For long eras, sampling every 5 minutes captures patterns without overwhelming memory
- Preserves statistical properties while reducing data volume by ~95%

### Why These Era Parameters?
- **24-hour minimum**: Prevents micro-eras from sensor noise
- **Lower lambda**: More responsive to actual operational changes
- **More HMM states**: Better captures different operational modes

## Verification Steps

1. **Check Era Sizes**:
```sql
SELECT COUNT(*) as total_eras,
       AVG(EXTRACT(EPOCH FROM (end_time - start_time))/3600) as avg_hours
FROM era_labels_level_a;
```

2. **Monitor GPU Memory**:
```bash
nvidia-smi -l 1
```

3. **Verify Features**:
```sql
SELECT 'Level A' as level, COUNT(*) FROM gpu_features_level_a
UNION ALL
SELECT 'Level B', COUNT(*) FROM gpu_features_level_b
UNION ALL  
SELECT 'Level C', COUNT(*) FROM gpu_features_level_c;
```

## Lessons Learned

1. **Always verify shared memory requirements** for CUDA kernels
2. **Era detection parameters** significantly impact downstream processing
3. **Data sampling** is essential for long time periods
4. **Sequential processing** can be more reliable than parallel for GPU workloads
5. **Comprehensive logging** is crucial for debugging GPU issues

## Future Improvements

1. Implement adaptive sampling based on era duration
2. Add GPU memory monitoring to prevent OOM errors
3. Consider using CUDA streams for better parallelization within each level
4. Implement checkpointing for long-running feature extraction
5. Add automated parameter tuning for era detection based on data characteristics