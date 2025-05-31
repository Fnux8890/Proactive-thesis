# Docker Compose Files Guide

## Overview of All Docker Compose Files

### 🚀 **FOR CLOUD DEPLOYMENT**
The cloud deployment uses these files:
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml
```

- **`docker-compose.yml`** - Base configuration with all services
- **`docker-compose.prod.yml`** - Production overrides for cloud (GPU settings, Cloud SQL, resource limits)

### 📁 All Docker Compose Files Explained

1. **`docker-compose.yml`** (BASE)
   - Main file with all service definitions
   - Used as base for all environments
   - Contains default configurations

2. **`docker-compose.prod.yml`** ✅ (CLOUD PRODUCTION)
   - Production overrides for cloud deployment
   - Configures Cloud SQL instead of local DB
   - Sets GPU allocations (2 A100s for feature extraction, 2 for model builder)
   - Includes monitoring stack (Prometheus, Grafana)
   - **THIS IS WHAT TERRAFORM USES**

3. **`docker-compose.override.yml`**
   - Local development overrides
   - Automatically loaded when you run `docker compose up`
   - For local testing only

4. **`docker-compose.cloud.yml`**
   - Alternative cloud configuration (not used by Terraform)
   - Possibly older version

5. **`docker-compose.parallel-feature.yml`**
   - Parallel feature extraction configuration
   - Not used in current deployment

6. **`docker-compose.production.yml`**
   - Created by copying docker-compose.prod.yml during deployment
   - Used by systemd service

## How Docker Compose -f Works

```bash
# Syntax
docker-compose -f <base-file> -f <override-file> <command>

# The override file merges with and overrides the base file
# Later files override earlier files
```

### Examples:

```bash
# Local development (uses docker-compose.yml + docker-compose.override.yml automatically)
docker compose up

# Production cloud (what Terraform uses)
docker compose -f docker-compose.yml -f docker-compose.prod.yml up

# Test a specific combination
docker compose -f docker-compose.yml -f docker-compose.prod.yml config  # Shows merged config
```

## Testing the Cloud Configuration Locally

### 1. View the merged configuration:
```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml config > merged-cloud-config.yml
```

### 2. Test building cloud services:
```bash
# Build all services with cloud config
docker compose -f docker-compose.yml -f docker-compose.prod.yml build

# Build specific service
docker compose -f docker-compose.yml -f docker-compose.prod.yml build model_builder
```

### 3. Test running with cloud config (without Cloud SQL):
```bash
# Start only services that don't need Cloud SQL
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d redis prometheus grafana

# Test a pipeline service
docker compose -f docker-compose.yml -f docker-compose.prod.yml run --rm rust_pipeline cargo --version
```

## Key Differences in Production (docker-compose.prod.yml)

1. **Database**
   - Overrides local `db` service to use Cloud SQL
   - Sets `deploy.replicas: 0` to disable local PostgreSQL

2. **Environment Variables**
   - Uses `${DB_HOST}` and `${DB_PASSWORD}` for Cloud SQL
   - Sets `USE_GPU=true` for GPU services
   - Configures batch sizes and parallelism

3. **Resource Limits**
   - CPU and memory limits for each service
   - GPU device reservations

4. **Monitoring**
   - Adds Prometheus, Grafana, and DCGM GPU exporter
   - Configures dashboards and data retention

## Quick Test Commands

```bash
# See what services are defined in cloud config
docker compose -f docker-compose.yml -f docker-compose.prod.yml ps --services

# Validate the configuration
docker compose -f docker-compose.yml -f docker-compose.prod.yml config --quiet && echo "✅ Config is valid"

# Test build without running
docker compose -f docker-compose.yml -f docker-compose.prod.yml build --dry-run

# Check images that would be used
docker compose -f docker-compose.yml -f docker-compose.prod.yml config | grep "image:"
```

## Testing Cloud Pipeline Locally

Since the cloud config expects Cloud SQL, you need to either:

### Option 1: Use local DB for testing
```bash
# Create a test override file
cat > docker-compose.test-cloud.yml << 'EOF'
services:
  db:
    deploy:
      replicas: 1  # Re-enable local DB
    environment:
      POSTGRES_DB: greenhouse
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
EOF

# Test with local DB
docker compose -f docker-compose.yml -f docker-compose.prod.yml -f docker-compose.test-cloud.yml up -d db
```

### Option 2: Set environment variables
```bash
# Create test environment
cat > .env.cloud-test << 'EOF'
DB_HOST=db
DB_PASSWORD=postgres
DB_NAME=greenhouse
DB_USER=postgres
DATABASE_URL=postgresql://postgres:postgres@db:5432/greenhouse
EOF

# Run with test environment
docker compose --env-file .env.cloud-test -f docker-compose.yml -f docker-compose.prod.yml up
```

## Summary

**For cloud deployment testing, always use:**
```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml <command>
```

This ensures you're testing the exact configuration that will run in the cloud!