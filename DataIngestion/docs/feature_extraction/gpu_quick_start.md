# GPU Feature Extraction Quick Start

## 1. Test GPU Availability
```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base nvidia-smi
```

## 2. Build the Service
```bash
docker compose build gpu_feature_extraction
```

## 3. Quick Test (10 eras)
```bash
./test_gpu_pipeline.sh
```

## 4. Run Benchmark
```bash
# Built-in benchmark
docker compose run --rm gpu_feature_extraction --benchmark

# Compare with CPU
./benchmark_gpu_vs_cpu.sh
```

## 5. Full Pipeline with GPU
```bash
# Run complete pipeline for January 2014
./run_pipeline_with_gpu.sh 2014-01-01 2014-01-31 B
```

## 6. Use GPU by Default
```bash
# Use the GPU override compose file
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

## 7. Manual Testing

### Test specific era level:
```bash
docker compose run --rm gpu_feature_extraction \
  --era-level A \
  --features-table feature_data_gpu
```

### Process limited data:
```bash
docker compose run --rm gpu_feature_extraction \
  --era-level B \
  --max-eras 50 \
  --batch-size 2000
```

### Debug mode:
```bash
RUST_LOG=debug docker compose run --rm gpu_feature_extraction \
  --era-level C \
  --max-eras 5
```

## Common Issues

### Out of Memory
```bash
# Reduce batch size
GPU_BATCH_SIZE=500 docker compose run --rm gpu_feature_extraction
```

### Check GPU Usage
```bash
# In another terminal while running
watch -n 1 nvidia-smi
```

### Database Connection
```bash
# Test connection
docker compose run --rm gpu_feature_extraction --help
```

## Performance Tips

1. **Batch Size**: Larger = faster, but uses more memory
   - GTX 1660: 500-1000
   - RTX 3090: 2000-5000
   - A100: 5000-10000

2. **Era Level**: 
   - Level A: Fewer, larger eras (fastest)
   - Level B: Medium
   - Level C: Many small eras (slowest)

3. **Expected Speedup**:
   - 20-50x faster than CPU
   - Depends on GPU and feature complexity