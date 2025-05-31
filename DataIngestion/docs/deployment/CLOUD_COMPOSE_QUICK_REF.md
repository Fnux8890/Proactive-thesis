# 🚀 Cloud Docker Compose Quick Reference

## Which Files Are Used in Cloud?

**Answer:** `docker-compose.yml` + `docker-compose.prod.yml`

```bash
# This is what runs in the cloud:
docker compose -f docker-compose.yml -f docker-compose.prod.yml up
```

## Test Cloud Config Locally

```bash
# 1. Quick validation
docker compose -f docker-compose.yml -f docker-compose.prod.yml config --quiet

# 2. See all services
docker compose -f docker-compose.yml -f docker-compose.prod.yml ps --services

# 3. Build everything
docker compose -f docker-compose.yml -f docker-compose.prod.yml build

# 4. Run automated test
./test_cloud_compose.sh
```

## Key Differences in Cloud (prod.yml)

| Component | Local | Cloud (prod.yml) |
|-----------|-------|------------------|
| **Database** | Local PostgreSQL container | Cloud SQL (db service disabled) |
| **GPU** | Optional | Required (4× A100s allocated) |
| **Monitoring** | None | Prometheus + Grafana + DCGM |
| **Resources** | Unlimited | CPU/Memory limits enforced |
| **Environment** | `.env` file | Cloud SQL via `${DB_HOST}` |

## Service GPU Allocation (Cloud)

- **feature_extraction**: 2× A100 GPUs
- **model_builder**: 2× A100 GPUs  
- **moea_optimizer_gpu**: 1× A100 GPU
- **dcgm-exporter**: All GPUs (monitoring)

## Test Commands

```bash
# Test specific service from cloud config
docker compose -f docker-compose.yml -f docker-compose.prod.yml build model_builder

# Check what image will be used
docker compose -f docker-compose.yml -f docker-compose.prod.yml config | grep -A 2 "model_builder:"

# See resource limits
docker compose -f docker-compose.yml -f docker-compose.prod.yml config | grep -B 5 -A 5 "limits:"
```

## Common Issues

1. **"db service not found"**
   - Normal! Cloud uses Cloud SQL, not local db
   - For local testing, use test override

2. **"GPU not available"**
   - Set `USE_GPU=false` for local testing
   - Cloud instances have GPUs

3. **"Cannot connect to database"**
   - Cloud expects `${DB_HOST}` environment variable
   - Use `.env.cloud-test` for local testing

## Files NOT Used in Cloud

These are for other purposes:
- ❌ `docker-compose.override.yml` (local dev only)
- ❌ `docker-compose.cloud.yml` (old/alternative)
- ❌ `docker-compose.parallel-feature.yml` (not used)
- ❌ `feature_extraction/docker-compose.*.yml` (subdirectory)

## Remember

**Always test with:**
```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml
```

This ensures you're testing what actually runs in the cloud!