# 🧪 Testing Guide - Validate Before Cloud Deployment

## Overview

This guide helps you test all services locally before spending money on cloud resources. Follow these steps to ensure everything works correctly.

## Quick Start (5 minutes)

```bash
cd DataIngestion
./preflight_check.sh
```

This runs all essential checks. If it passes, you're ready to deploy!

## Detailed Testing Options

### 1. Configuration Validation (30 seconds)

```bash
./validate_services.sh
```

**What it checks:**
- ✅ All required files exist
- ✅ Dockerfiles are present
- ✅ Requirements.txt files exist
- ✅ Docker Compose syntax is valid
- ✅ No unusually large files

**Common issues:**
- Missing requirements.txt files
- Invalid YAML in docker-compose files
- Missing Dockerfiles

### 2. Minimal Pipeline Test (3-5 minutes)

```bash
./test_minimal_pipeline.sh
```

**What it does:**
- Tests each service with minimal data (January 15, 2014 only)
- Verifies all components can start and connect
- Checks basic functionality without full processing

**Perfect for:**
- Quick validation after code changes
- Testing without GPU
- Verifying database connectivity

### 3. Full Service Test (10-20 minutes)

```bash
./test_all_services.sh
```

**What it does:**
- Builds every service from scratch
- Tests GPU functionality (if available)
- Verifies all dependencies are installed
- Checks database extensions (TimescaleDB)

**When to use:**
- Before first deployment
- After major dependency updates
- When adding new services

## Testing Individual Services

### Test a specific service:

```bash
# Test just the Rust pipeline
docker compose build rust_pipeline
docker compose run --rm rust_pipeline cargo --version

# Test preprocessing
docker compose build preprocessing
docker compose run --rm preprocessing python -c "import pandas; print('OK')"

# Test GPU services (requires NVIDIA GPU)
docker compose build model_builder
docker compose run --rm --gpus all model_builder python -c "import torch; print(torch.cuda.is_available())"
```

### Common Service Issues & Fixes

#### 1. Rust Pipeline
```bash
# Error: "failed to compile"
# Fix: Update Rust dependencies
docker compose run --rm rust_pipeline cargo update

# Error: "connection refused"
# Fix: Check DATABASE_URL environment variable
```

#### 2. Feature Extraction
```bash
# Error: "ImportError: No module named tsfresh"
# Fix: Check requirements.txt has tsfresh>=0.20.0

# Error: "Out of memory"
# Fix: Reduce BATCH_SIZE environment variable
```

#### 3. Model Builder
```bash
# Error: "CUDA not found"
# Fix: Set USE_GPU=false for CPU-only testing

# Error: "lightgbm not installed"
# Fix: Add lightgbm to requirements.txt
```

#### 4. MOEA Optimizer
```bash
# Error: "No module named pymoo"
# Fix: Check requirements.txt has pymoo>=0.6.0

# Error: "Config file not found"
# Fix: Ensure config files are in moea_optimizer/config/
```

## Testing Without GPU

If you don't have a GPU locally:

```bash
# Set environment variable
export USE_GPU=false

# Run tests
./test_minimal_pipeline.sh

# Or test specific GPU service in CPU mode
docker compose run --rm -e USE_GPU=false model_builder python test_gpu.py
```

## Testing Database Connectivity

```bash
# Start database
docker compose up -d db

# Wait for it to be ready
docker compose exec db pg_isready -U postgres

# Test connection
docker compose exec db psql -U postgres -c "SELECT version();"

# Test TimescaleDB
docker compose exec db psql -U postgres -d greenhouse -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"
```

## Pre-Deployment Checklist

Before running `terraform apply`:

- [ ] Run `./preflight_check.sh` - all tests pass
- [ ] Check Docker Desktop has enough resources allocated (8GB+ RAM)
- [ ] Verify data exists in `../Data/` directory
- [ ] Ensure you have Google Cloud SDK installed and authenticated
- [ ] Review costs in `terraform/parallel-feature/DEPLOYMENT_STEPS.md`

## Debugging Failed Tests

### 1. Check logs
```bash
# Find the latest test log
ls -lt *.log | head -5

# View specific service logs
docker compose logs <service-name>
```

### 2. Run service interactively
```bash
# Start a shell in the service container
docker compose run --rm <service-name> /bin/bash

# Then test commands manually
python -c "import pandas; print(pandas.__version__)"
```

### 3. Check environment variables
```bash
# List all environment variables
docker compose run --rm <service-name> env | sort
```

## Cloud-Specific Testing

### Test cloud connectivity (without deploying):
```bash
# Test Google Cloud access
gcloud auth list
gcloud config list

# Test you can create resources
gcloud compute regions list

# Test Cloud Storage access
gsutil ls
```

### Estimate costs:
```bash
# Production (non-preemptible): ~$2-4/hour
# Development (preemptible): ~$0.40-0.80/hour
# Storage: ~$0.02/GB/month
```

## Getting Help

If tests fail:

1. Check the specific error message in the log files
2. Verify all dependencies in requirements.txt/Cargo.toml
3. Ensure Docker has enough resources allocated
4. Try testing services individually
5. Check the service's Dockerfile for build issues

## Summary

**Always run tests before deployment!** It's much cheaper to find issues locally than in the cloud.

```bash
# Recommended test sequence:
./preflight_check.sh          # 5 minutes
./test_minimal_pipeline.sh    # 3 minutes (if preflight fails)
./test_all_services.sh        # 15 minutes (for thorough testing)
```

Once all tests pass, you're ready for cloud deployment! 🚀