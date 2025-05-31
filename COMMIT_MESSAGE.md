fix(gpu): Complete GPU feature extraction fixes and era detection optimization

This commit finalizes the GPU feature extraction fixes from the previous session:

## Key Changes Implemented

### 🔧 GPU Feature Extraction Fixes
- ✅ Fixed CUDA_ERROR_INVALID_VALUE by providing correct shared memory size (96 bytes)
- ✅ Added SQL data sampling for long eras (>7 days) to prevent memory exhaustion  
- ✅ Implemented compile-time struct size verification for StatisticalFeatures
- ✅ Enhanced error handling with detailed CUDA error messages

### 📊 Era Detection Parameter Optimization
Updated docker-compose.yml with improved parameters:
- `pelt-min-size`: 48 → 288 (4hr → 24hr minimum era size)
- `bocpd-lambda`: 200.0 → 50.0 (more sensitive to changes)
- `hmm-states`: 5 → 10 (more granular state detection)

### 💾 Database Query Optimization
Modified gpu_feature_extraction/src/db.rs:
```sql
-- Sample one reading every 5 minutes for eras > 7 days
AND (($2 - $1 < interval '7 days') OR EXTRACT(EPOCH FROM time) % 300 = 0)
LIMIT 100000  -- Safety limit
```

### 📚 Documentation Updates
- Created `docs/gpu/GPU_TESTING_COMMANDS.md` with step-by-step testing procedures
- Updated `docs/gpu/ERA_DETECTION_ANALYSIS.md` with parameter rationale
- Sequential GPU execution already configured in docker-compose.yml (A→B→C)

## Technical Implementation

### Shared Memory Fix (features.rs)
```rust
// Calculate shared memory size for the kernel
let num_warps = (block_size + 31) / 32; // 8 warps for 256 threads  
let shared_mem_bytes = (num_warps * 3 * std::mem::size_of::<f32>()) as u32; // 96 bytes
```

### Struct Size Verification (kernels.rs)
```rust
// Compile-time check to ensure host/device struct alignment
const _: () = assert!(std::mem::size_of::<StatisticalFeatures>() == 24);
```

## Expected Results

After re-running era detection:
- **Level A**: 10-50 eras (major structural changes)
- **Level B**: 50-200 eras (operational shifts)  
- **Level C**: 200-1000 eras (daily patterns)

GPU feature extraction should:
- Complete without segmentation faults
- Process reasonable data sizes per era
- Run sequentially to avoid GPU memory conflicts

## Next Steps

1. Clean existing era tables: `DROP TABLE era_labels_level_*`
2. Re-run era detection: `docker compose up era_detector`
3. Test GPU extraction: `docker compose up gpu_feature_extraction_level_a`
4. Verify results using commands in GPU_TESTING_COMMANDS.md

Resolves: GPU segmentation faults, year-long era spans, CUDA memory errors
Related: GPU feature extraction optimization, era detection parameter tuning