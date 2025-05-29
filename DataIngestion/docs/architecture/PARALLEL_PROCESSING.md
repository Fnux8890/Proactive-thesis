# Parallel Processing Architecture

This document consolidates all parallel processing information for the DataIngestion pipeline.

## Overview

The pipeline supports three types of parallelization:

1. **Data Ingestion Parallelization** (Rust)
   - See: [Rust Pipeline Operations](../../rust_pipeline/RUST_PIPELINE_OPERATIONS_GUIDE.md)
   - Concurrent file processing
   - Parallel CSV/JSON parsing
   - Batch database insertions

2. **Feature Extraction Parallelization** (Python/GPU)
   - See: [Parallel Feature Extraction](../../feature_extraction/parallel/README.md)
   - GPU+CPU hybrid processing
   - Smart work distribution
   - Redis-based task queue

3. **Cloud Deployment Architecture**
   - See: [Deployment Guide](../deployment/PARALLEL_DEPLOYMENT_GUIDE.md)
   - Google Cloud A2 instance optimization
   - 4 GPUs + 48 vCPUs utilization
   - Docker Compose configurations

## Quick Reference

### Local Development (CPU Parallel)
```bash
docker compose up  # Uses override.yml
```

### Cloud Production (GPU+CPU Parallel)
```bash
docker compose -f docker-compose.yml -f docker-compose.cloud.yml up
```

### Parallel Feature Extraction Only
```bash
docker compose -f docker-compose.yml -f docker-compose.parallel-feature.yml up
```

## Architecture Details

For specific implementation details, see:
- [GPU Feature Extraction](GPU_FEATURE_EXTRACTION.md)
- [Pipeline Flow](PIPELINE_FLOW.md)
- [Data Source Integration](DATA_SOURCE_INTEGRATION.md)
