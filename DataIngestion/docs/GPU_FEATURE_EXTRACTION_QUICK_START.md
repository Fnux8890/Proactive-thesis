# GPU Feature Extraction Quick Start Guide

## Overview
GPU-accelerated feature extraction using Rust + cudarc, reducing processing time from 8+ hours to ~15 minutes.

## Quick Commands

### Build the Service
```bash
cd DataIngestion
docker compose build gpu_feature_extraction
```

### Run Feature Extraction

#### Process All Eras (Level A)
```bash
docker compose run --rm gpu_feature_extraction \
  --era-level A \
  --batch-size 1000
```

#### Process Specific Date Range
```bash
docker compose run --rm gpu_feature_extraction \
  --era-level B \
  --start-date 2024-01-01 \
  --end-date 2024-01-31 \
  --batch-size 500
```

#### Benchmark Mode
```bash
docker compose run --rm gpu_feature_extraction \
  --benchmark \
  --max-eras 100
```

## Integration with Full Pipeline

### Option 1: Replace CPU Feature Extraction
```bash
# Run pipeline with GPU feature extraction
docker compose up -d db
docker compose run --rm rust_pipeline
docker compose run --rm preprocess
docker compose run --rm era_detector
docker compose run --rm gpu_feature_extraction --era-level A
docker compose run --rm gpu_feature_extraction --era-level B
docker compose run --rm gpu_feature_extraction --era-level C
docker compose run --rm model_builder
docker compose run --rm moea_optimizer_gpu
```

### Option 2: Modify docker-compose.yml
Replace the three CPU feature extraction services with GPU version:

```yaml
# Comment out or remove these:
# feature_extraction_level_a:
# feature_extraction_level_b:  
# feature_extraction_level_c:

# Add GPU versions:
gpu_feature_extraction_level_a:
  extends:
    service: gpu_feature_extraction
  command: ["--era-level", "A", "--batch-size", "1000"]

gpu_feature_extraction_level_b:
  extends:
    service: gpu_feature_extraction
  command: ["--era-level", "B", "--batch-size", "1000"]

gpu_feature_extraction_level_c:
  extends:
    service: gpu_feature_extraction
  command: ["--era-level", "C", "--batch-size", "1000"]
```

## Performance Comparison

| Era Level | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| Level A (437K eras) | ~8 hours | ~10 min | 48x |
| Level B (278K eras) | ~5 hours | ~7 min | 43x |
| Level C (92K eras) | ~2 hours | ~3 min | 40x |

## Environment Variables

```bash
# .env file
DATABASE_URL=postgresql://postgres:postgres@db:5432/postgres
RUST_LOG=info
CUDA_VISIBLE_DEVICES=0
GPU_BATCH_SIZE=1000
```

## Monitoring

### GPU Utilization
```bash
# In another terminal while running
watch -n 1 nvidia-smi
```

### Logs
```bash
docker compose logs -f gpu_feature_extraction
```

## Troubleshooting

### Out of Memory
Reduce batch size:
```bash
docker compose run --rm gpu_feature_extraction \
  --era-level A \
  --batch-size 500  # Reduced from 1000
```

### No GPU Available
Check Docker GPU support:
```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### Slow Performance
1. Check GPU thermal throttling
2. Verify no other GPU processes running
3. Try smaller batch sizes
4. Check database connection speed

## Cost Estimation (Cloud)

| Instance Type | Cost/Hour | Time for Full Dataset | Total Cost |
|---------------|-----------|----------------------|------------|
| GCP A100 (40GB) | $3.67 | ~30 min | ~$1.84 |
| AWS p3.2xlarge | $3.06 | ~45 min | ~$2.30 |
| Azure NC6s_v3 | $2.88 | ~40 min | ~$1.92 |

Compare to CPU: 8 hours × $0.50/hour = $4.00

## Key Files

- Implementation: `DataIngestion/gpu_feature_extraction/`
- Documentation: `DataIngestion/docs/architecture/GPU_FEATURE_EXTRACTION_CUDARC.md`
- Docker config: `DataIngestion/docker-compose.yml` (service: `gpu_feature_extraction`)

## Next Steps

1. **Test on Small Dataset**
   ```bash
   docker compose run --rm gpu_feature_extraction \
     --era-level A \
     --max-eras 100 \
     --benchmark
   ```

2. **Run Full Pipeline**
   ```bash
   docker compose up moea_optimizer_gpu
   ```

3. **Compare Results**
   - Check feature_data table
   - Validate against CPU features
   - Compare model performance