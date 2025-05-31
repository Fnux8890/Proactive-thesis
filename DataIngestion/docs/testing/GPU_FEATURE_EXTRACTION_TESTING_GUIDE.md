# GPU Feature Extraction Testing & Integration Guide

## Overview

This guide explains how to test and integrate the GPU-accelerated feature extraction service into the data pipeline.

## Prerequisites

1. **NVIDIA GPU** with CUDA Compute Capability ≥ 7.0
2. **NVIDIA Docker Runtime** installed
3. **CUDA drivers** installed on host
4. **Database** with preprocessed data and era detection results

## 1. Standalone Testing

### A. Build the Service

```bash
cd DataIngestion
docker compose build gpu_feature_extraction
```

### B. Test GPU Availability

```bash
# Check if Docker can see the GPU
docker run --rm --gpus all nvidia/cuda:12.4.1-base nvidia-smi

# Should show your GPU information
```

### C. Run Benchmark Mode

```bash
# Test with benchmark mode (uses sample data)
docker compose run --rm gpu_feature_extraction --benchmark

# Expected output:
# - GPU device initialization
# - Benchmark results for different batch sizes
# - Performance metrics (features/sec)
```

### D. Test with Limited Data

```bash
# Process only 10 eras for testing
docker compose run --rm gpu_feature_extraction \
  --era-level A \
  --max-eras 10 \
  --features-table feature_data_gpu_test

# Check results in database
docker compose exec db psql -U postgres -d postgres \
  -c "SELECT COUNT(*) FROM feature_data_gpu_test;"
```

## 2. Integration Testing

### A. Create Test Pipeline Script

Create `test_gpu_pipeline.sh`:

```bash
#!/bin/bash
# Test GPU feature extraction in pipeline

echo "=== GPU Feature Extraction Test Pipeline ==="
echo

# 1. Ensure database is ready
echo "1. Checking database..."
docker compose up -d db
sleep 5

# 2. Run preprocessing (if needed)
echo "2. Running preprocessing..."
docker compose run --rm preprocessing \
  python preprocess.py --start-date 2014-01-01 --end-date 2014-01-07

# 3. Run era detection
echo "3. Running era detection..."
docker compose run --rm era_detector

# 4. Run GPU feature extraction
echo "4. Running GPU feature extraction..."
time docker compose run --rm gpu_feature_extraction \
  --era-level A \
  --features-table feature_data_gpu

# 5. Compare with CPU version (optional)
echo "5. Running CPU feature extraction for comparison..."
time docker compose run --rm feature_extraction

# 6. Verify results
echo "6. Verifying results..."
docker compose exec db psql -U postgres -d postgres -c "
SELECT 
    'GPU Features' as source,
    COUNT(*) as count,
    COUNT(DISTINCT era_id) as unique_eras
FROM feature_data_gpu
UNION ALL
SELECT 
    'CPU Features' as source,
    COUNT(*) as count,
    COUNT(DISTINCT era_id) as unique_eras
FROM feature_data;
"

echo "Test complete!"
```

### B. Performance Comparison Script

Create `benchmark_gpu_vs_cpu.sh`:

```bash
#!/bin/bash
# Compare GPU vs CPU feature extraction performance

echo "=== GPU vs CPU Performance Comparison ==="
echo

# Ensure clean state
docker compose exec db psql -U postgres -d postgres -c "
DROP TABLE IF EXISTS feature_data_gpu_bench;
DROP TABLE IF EXISTS feature_data_cpu_bench;
"

# GPU benchmark
echo "Running GPU feature extraction..."
GPU_START=$(date +%s)
docker compose run --rm gpu_feature_extraction \
  --era-level B \
  --features-table feature_data_gpu_bench
GPU_END=$(date +%s)
GPU_TIME=$((GPU_END - GPU_START))

# CPU benchmark
echo "Running CPU feature extraction..."
CPU_START=$(date +%s)
docker compose run --rm feature_extraction
CPU_END=$(date +%s)
CPU_TIME=$((CPU_END - CPU_START))

# Results
echo
echo "Results:"
echo "GPU Time: ${GPU_TIME} seconds"
echo "CPU Time: ${CPU_TIME} seconds"
echo "Speedup: $(($CPU_TIME / $GPU_TIME))x"
```

## 3. Integration Options

### Option 1: Replace CPU Feature Extraction

Modify `docker-compose.yml`:

```yaml
# Comment out or remove the CPU feature extraction
# feature_extraction:
#   ...

# Use GPU version as default
feature_extraction:
  extends:
    service: gpu_feature_extraction
  profiles: []  # Always run
```

### Option 2: Parallel Services with Profiles

```yaml
# In docker-compose.yml
feature_extraction:
  profiles: ["cpu"]
  # ... existing config

gpu_feature_extraction:
  profiles: ["gpu"]
  # ... existing config
```

Then run with:
```bash
# CPU pipeline
docker compose --profile cpu up

# GPU pipeline
docker compose --profile gpu up
```

### Option 3: Conditional Selection Script

Create `run_pipeline.sh`:

```bash
#!/bin/bash
# Smart pipeline runner that uses GPU if available

if docker run --rm --gpus all nvidia/cuda:12.4.1-base nvidia-smi &>/dev/null; then
    echo "GPU detected - using GPU feature extraction"
    FEATURE_SERVICE="gpu_feature_extraction"
else
    echo "No GPU detected - using CPU feature extraction"
    FEATURE_SERVICE="feature_extraction"
fi

# Run pipeline stages
docker compose run --rm rust_pipeline
docker compose run --rm preprocessing
docker compose run --rm era_detector
docker compose run --rm $FEATURE_SERVICE
docker compose run --rm model_builder
```

## 4. Validation Testing

### A. Feature Consistency Check

```sql
-- Compare feature statistics between GPU and CPU
WITH gpu_stats AS (
    SELECT 
        era_id,
        jsonb_object_keys(features) as feature_name,
        (features->>jsonb_object_keys(features))::float as value
    FROM feature_data_gpu
),
cpu_stats AS (
    SELECT 
        era_id,
        jsonb_object_keys(features) as feature_name,
        (features->>jsonb_object_keys(features))::float as value
    FROM feature_data
)
SELECT 
    g.feature_name,
    AVG(g.value) as gpu_avg,
    AVG(c.value) as cpu_avg,
    ABS(AVG(g.value) - AVG(c.value)) as diff
FROM gpu_stats g
JOIN cpu_stats c ON g.era_id = c.era_id AND g.feature_name = c.feature_name
GROUP BY g.feature_name
ORDER BY diff DESC
LIMIT 20;
```

### B. Memory Usage Test

```bash
# Monitor GPU memory usage during execution
docker compose run --rm -d gpu_feature_extraction --era-level A
watch -n 1 nvidia-smi
```

## 5. Production Integration

### A. Environment Variables

Create `.env.gpu`:
```env
# GPU Feature Extraction Settings
CUDA_VISIBLE_DEVICES=0
GPU_BATCH_SIZE=2000
GPU_FEATURE_TABLE=feature_data
USE_GPU_FEATURES=true
```

### B. Docker Compose Override

Create `docker-compose.gpu.yml`:
```yaml
version: '3.8'

services:
  # Override feature extraction to use GPU
  feature_extraction:
    extends:
      file: docker-compose.yml
      service: gpu_feature_extraction
    profiles: []  # Always run

  # Model builder uses GPU features
  model_builder:
    environment:
      FEATURE_TABLE: ${GPU_FEATURE_TABLE:-feature_data}
```

Run with:
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

## 6. Monitoring & Debugging

### A. Enable Debug Logging

```bash
RUST_LOG=debug docker compose run --rm gpu_feature_extraction
```

### B. Profile Kernel Performance

```bash
# Run with NVIDIA Nsight Systems
docker run --rm --gpus all \
  -v $(pwd):/workspace \
  nvcr.io/nvidia/nsight-systems:2023.4.1 \
  profile --stats=true \
  /workspace/gpu_feature_extraction/target/release/gpu_feature_extraction
```

### C. Common Issues

1. **Out of Memory**
   ```bash
   # Reduce batch size
   GPU_BATCH_SIZE=500 docker compose run --rm gpu_feature_extraction
   ```

2. **CUDA Version Mismatch**
   ```bash
   # Check CUDA version
   nvidia-smi
   # Ensure Dockerfile uses matching CUDA base image
   ```

3. **Database Connection**
   ```bash
   # Test database connection
   docker compose run --rm gpu_feature_extraction \
     psql $DATABASE_URL -c "SELECT 1"
   ```

## 7. CI/CD Integration

### GitHub Actions Example

```yaml
name: Test GPU Pipeline

on: [push, pull_request]

jobs:
  test-gpu:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v3
      
      - name: Build GPU Feature Extraction
        run: docker compose build gpu_feature_extraction
        
      - name: Run GPU Tests
        run: |
          docker compose run --rm gpu_feature_extraction --benchmark
          ./test_gpu_pipeline.sh
```

## Summary

The GPU feature extraction can be integrated as:
1. **Drop-in replacement** for CPU feature extraction
2. **Parallel option** selected via profiles
3. **Automatic selection** based on hardware availability

For production, recommend Option 3 (automatic selection) for flexibility.