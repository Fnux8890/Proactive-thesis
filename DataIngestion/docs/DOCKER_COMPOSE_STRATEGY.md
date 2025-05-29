# Docker Compose Configuration Strategy

## Current Setup Issues

You're absolutely right - having separate "cloud" and "parallel" configurations doesn't make sense. The cloud instance (A2 with 4 GPUs and 48 vCPUs) is PERFECT for parallel processing!

## Proposed Simplification

### 1. **docker-compose.yml** (Base)
- Core service definitions
- Basic resource allocations
- Network and volume definitions

### 2. **docker-compose.override.yml** (Local Development)
- CPU-only overrides
- Minimal feature sets
- Debug logging
- Small batch sizes
- Code hot-reloading

### 3. **docker-compose.production.yml** (Cloud Production)
- Combines current cloud.yml + parallel features
- Full parallel processing (4 GPU + 6-8 CPU workers)
- Comprehensive feature extraction
- Production monitoring stack
- Optimized resource allocation

## Why This Makes More Sense

### Current Confusion:
- `cloud.yml` - Has monitoring but no parallel workers
- `parallel-feature.yml` - Has parallel workers but no monitoring
- Both are meant for the same A2 instance!

### Better Approach:
```bash
# Local development (auto-uses override)
docker compose up

# Cloud production (everything you need)
docker compose -f docker-compose.yml -f docker-compose.production.yml up
```

## Resource Utilization on A2 Instance

### Hardware Available:
- 4× NVIDIA A100 GPUs (40GB each)
- 48 vCPUs
- 340 GiB RAM

### Optimal Allocation:

#### GPU Usage:
- **Feature Extraction**: 4 GPUs (one per worker)
- **Model Training**: Can share GPUs after feature extraction
- **MOEA Optimization**: Can share GPUs after model training

#### CPU Usage:
- **Database**: 8 vCPUs
- **Infrastructure** (Redis, PgBouncer): 4 vCPUs
- **Data Ingestion**: 8-16 vCPUs
- **Preprocessing**: 8 vCPUs
- **Era Detection**: 8-16 vCPUs
- **CPU Feature Workers**: 12 vCPUs (6 workers × 2 cores)
- **Monitoring**: 4 vCPUs

#### Memory Usage:
- **GPU Workers**: 80GB each (total 320GB)
- **Database**: 32GB
- **Other Services**: ~100GB
- Total: ~450GB (oversubscribed but Docker handles this)

## Benefits of Unified Configuration

1. **Simplicity**: One production config instead of two
2. **Clarity**: Clear distinction between dev and prod
3. **Efficiency**: All resources utilized properly
4. **Monitoring**: Production always has monitoring
5. **Parallel by Default**: Cloud means parallel!

## Migration Path

1. Rename `docker-compose.cloud.yml` → `docker-compose.production.yml`
2. Merge parallel worker definitions from `parallel-feature.yml`
3. Delete `docker-compose.parallel-feature.yml`
4. Update documentation

## Example Production Run

```bash
# On Google Cloud A2 instance
git pull
docker compose -f docker-compose.yml -f docker-compose.production.yml up

# This gives you:
# ✓ Parallel data ingestion (16 threads)
# ✓ Parallel preprocessing (8 workers)  
# ✓ Parallel era detection (16 threads)
# ✓ Parallel feature extraction (4 GPU + 6 CPU workers)
# ✓ GPU model training
# ✓ GPU MOEA optimization
# ✓ Full monitoring stack
# ✓ All in one command!
```

## Summary

The cloud instance IS your parallel processing environment. Having separate configurations for "cloud" and "parallel" is redundant and confusing. A unified production configuration makes much more sense!