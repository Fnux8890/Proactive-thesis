# Comprehensive GPU Feature Extraction Fix

## Issues Summary
1. **GPU Memory Conflicts**: 3 containers using same GPU simultaneously → segmentation fault
2. **Level C Scale**: 850,710 eras is excessive (likely includes many tiny/invalid eras)
3. **Slow Queries**: Fetching 1.4M+ rows without pagination

## Step-by-Step Fix

### Step 1: Test Sequential Execution (Quick Fix)

```bash
# Make the script executable
chmod +x run_gpu_feature_extraction_sequential.sh

# Run services one at a time
./run_gpu_feature_extraction_sequential.sh
```

### Step 2: Filter Level C Eras (Reduce 850K to manageable number)

```bash
# Use the override file to increase min-era-rows
docker compose -f docker-compose.yml -f docker-compose.gpu-fix.yml up gpu_feature_extraction_level_c
```

### Step 3: Check How Many GPUs You Have

```bash
nvidia-smi --query-gpu=name,index --format=csv
```

If you have multiple GPUs, we can assign different ones to each service.

### Step 4: Query Optimization (Code Change)

The slow queries need pagination. Here's a patch for `db.rs`:

```rust
// In fetch_era_data, add LIMIT clause
let query = r#"
    SELECT 
        time as timestamp,
        air_temp_c,
        relative_humidity_percent,
        -- other columns...
    FROM sensor_data_merged
    WHERE time >= $1 AND time < $2
    ORDER BY time
    LIMIT 100000  -- Add limit to prevent huge queries
"#;
```

### Step 5: Environment Variable Tuning

Create `.env` file:
```env
# Reduce batch sizes
GPU_BATCH_SIZE=100

# Increase minimum era rows to filter
MIN_ERA_ROWS_A=1000
MIN_ERA_ROWS_B=500  
MIN_ERA_ROWS_C=5000

# Set specific GPU devices if you have multiple
CUDA_VISIBLE_DEVICES_A=0
CUDA_VISIBLE_DEVICES_B=0
CUDA_VISIBLE_DEVICES_C=0
```

## Recommended Approach

1. **First**: Check actual era distribution
```sql
-- Run this query to understand Level C
SELECT 
    COUNT(*) as era_count,
    MIN(rows) as min_rows,
    MAX(rows) as max_rows,
    AVG(rows) as avg_rows,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY rows) as median_rows
FROM era_labels_level_c;
```

2. **Then**: Set appropriate MIN_ERA_ROWS based on results

3. **Finally**: Run services sequentially or with proper GPU isolation

## Long-term Solutions

1. **Implement Streaming**: Process eras in chunks instead of loading all data
2. **Add Pagination**: Limit query results and process in batches
3. **GPU Scheduling**: Use Kubernetes or Docker Swarm for proper GPU scheduling
4. **Data Validation**: Many of those 850K eras might be invalid/duplicate