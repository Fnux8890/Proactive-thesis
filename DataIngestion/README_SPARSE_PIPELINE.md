# Sparse Pipeline Usage Guide

## Quick Start

The sparse pipeline is now properly configured with environment variables and dependency chains. You can test the entire pipeline by just running the final service!

### 1. Setup Environment

```bash
# Copy the sparse environment configuration
cp .env.sparse .env

# Or use your existing .env and add sparse settings
# Key variables to set:
# SPARSE_START_DATE=2014-01-01
# SPARSE_END_DATE=2014-12-31
# DISABLE_GPU=false (or true if no GPU)
```

### 2. Test Full Pipeline

```bash
# This will automatically run all dependencies:
# db → rust_pipeline → sparse_pipeline → model_builder → moea_optimizer_gpu
docker compose -f docker-compose.sparse.yml up --build moea_optimizer_gpu

# Or for CPU version:
docker compose -f docker-compose.sparse.yml up --build moea_optimizer_cpu
```

### 3. Test Individual Stages

```bash
# Just data ingestion
docker compose -f docker-compose.sparse.yml up rust_pipeline

# Through feature extraction
docker compose -f docker-compose.sparse.yml up sparse_pipeline

# Through model building
docker compose -f docker-compose.sparse.yml up model_builder
```

## Environment Variables

All configuration is now in `.env` or `.env.sparse`:

### Date Range Control
```bash
SPARSE_START_DATE=2014-01-01  # Start date for processing
SPARSE_END_DATE=2014-12-31    # End date for processing
```

### GPU Control
```bash
DISABLE_GPU=false             # Set to true for CPU-only mode
CUDA_VISIBLE_DEVICES=0        # Which GPU to use
```

### Performance Tuning
```bash
SPARSE_BATCH_SIZE=24          # Window size in hours
SPARSE_MIN_HOURLY_COVERAGE=0.1  # Minimum 10% data coverage
SPARSE_MAX_INTERPOLATION_GAP=2  # Max gap filling in hours
```

## Dependency Chain

The services automatically run in order due to `depends_on` conditions:

```
db (PostgreSQL + TimescaleDB)
  ↓
rust_pipeline (Data Ingestion)
  ↓
sparse_pipeline (Feature Extraction)
  ↓
model_builder (Train Models)
  ↓
moea_optimizer_gpu/cpu (Optimization)
```

## Monitoring Progress

```bash
# Watch logs for all services
docker compose -f docker-compose.sparse.yml logs -f

# Check specific service
docker compose -f docker-compose.sparse.yml logs sparse_pipeline

# See what's running
docker compose -f docker-compose.sparse.yml ps
```

## Troubleshooting

### No GPU Available
```bash
# Set in .env
DISABLE_GPU=true

# Then use CPU optimizer
docker compose -f docker-compose.sparse.yml up moea_optimizer_cpu
```

### Out of Memory
```bash
# Reduce batch size in .env
SPARSE_BATCH_SIZE=12
GPU_BATCH_SIZE=250
```

### Change Date Range
```bash
# Edit .env
SPARSE_START_DATE=2014-06-01
SPARSE_END_DATE=2014-07-01

# Rebuild to pick up changes
docker compose -f docker-compose.sparse.yml up --build sparse_pipeline
```

### Clear Cache and Restart
```bash
# Stop everything
docker compose -f docker-compose.sparse.yml down

# Clear database
docker volume rm dataingestion_postgres-data

# Clear checkpoints
rm -rf gpu_feature_extraction/checkpoints/*

# Start fresh
docker compose -f docker-compose.sparse.yml up --build moea_optimizer_gpu
```

## Expected Runtime

With default settings processing all of 2014:
- rust_pipeline: ~30 seconds
- sparse_pipeline: ~2-3 minutes  
- model_builder: ~5-10 minutes
- moea_optimizer_gpu: ~10-20 minutes

Total: ~20-35 minutes for full pipeline

## Validation

Check that each stage completed:

```bash
# Data ingested
docker exec dataingestion-db-1 psql -U postgres -c "SELECT COUNT(*) FROM sensor_data;"

# Features extracted  
ls -la gpu_feature_extraction/checkpoints/

# Models trained
ls -la model_builder/models/

# MOEA results
ls -la moea_optimizer/results/
```