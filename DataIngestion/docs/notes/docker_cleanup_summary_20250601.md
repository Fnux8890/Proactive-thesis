# Docker Configuration Cleanup Summary

## Current Situation
- Multiple Docker files created during development causing confusion
- Only 535,072 rows in database (2013-12-01 to 2016-09-08)
- Need to use Rust 1.87 (latest)
- Python code passes ruff checks ✅

## Files to DELETE (not needed):
```bash
# In gpu_feature_extraction/
rm Dockerfile.hybrid      # Old hybrid attempt
rm Dockerfile.python-gpu  # Python-only attempt
rm docker-compose.hybrid.yml  # Old hybrid compose

# In DataIngestion/
rm docker-compose.yml     # Has missing era_detector service
rm docker-compose.prod.yml  # Production variant not needed yet
```

## Files to KEEP and USE:

### 1. **For Enhanced Pipeline (RECOMMENDED)**
- `gpu_feature_extraction/Dockerfile.enhanced` - Main Dockerfile with Rust 1.87 + Python GPU
- `docker-compose.enhanced.yml` - Clean pipeline configuration
- Command: `docker compose -f docker-compose.enhanced.yml up --build`

### 2. **For Basic Testing**
- `gpu_feature_extraction/Dockerfile` - CPU-only Rust (for debugging)
- `docker-compose.sparse.yml` - Original sparse pipeline

## Enhanced Pipeline Services:

1. **db** - TimescaleDB (required)
2. **rust_pipeline** - Data ingestion 
3. **enhanced_sparse_pipeline** - Hybrid Rust+Python with GPU (Stages 2-4)
4. **model_builder** - GPU-accelerated model training
5. **moea_optimizer** - Multi-objective optimization with GPU

## Data Issue:
- Current: 535,072 rows (seems incomplete)
- Available: Multiple JSON/CSV files in ../Data/aarslev/
- Action needed: May need to re-run rust_pipeline with all data sources

## To Run Enhanced Pipeline:

```bash
cd DataIngestion

# 1. Clean build and start
docker compose -f docker-compose.enhanced.yml down -v
docker compose -f docker-compose.enhanced.yml build --no-cache
docker compose -f docker-compose.enhanced.yml up

# 2. Or run specific stages
docker compose -f docker-compose.enhanced.yml up db rust_pipeline
docker compose -f docker-compose.enhanced.yml up enhanced_sparse_pipeline
```

## Key Environment Variables:
```bash
# Date range (use full data range)
START_DATE=2013-12-01
END_DATE=2016-09-08

# GPU settings
USE_GPU=true
CUDA_VISIBLE_DEVICES=0

# Feature flags
ENABLE_WEATHER_FEATURES=true
ENABLE_ENERGY_FEATURES=true
ENABLE_GROWTH_FEATURES=true
```

## Next Steps:
1. Build with Dockerfile.enhanced
2. Run full pipeline with docker-compose.enhanced.yml
3. Verify feature extraction works with sparse data
4. Check if we need to ingest more data files