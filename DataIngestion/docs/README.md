# DataIngestion Documentation

## Directory Structure

### `/architecture`
System design and architectural documentation
- `DATA_SOURCE_INTEGRATION.md` - Data source specifications
- `GPU_FEATURE_EXTRACTION.md` - GPU acceleration approach
- `PARALLEL_PROCESSING.md` - Parallel processing overview
- `PIPELINE_FLOW.md` - Complete pipeline architecture
- `/parallel` - Detailed parallel implementation docs

### `/database`
Database-related documentation and fixes
- `THREAD_SAFETY_IMPROVEMENTS.md` - Connection pooling and thread safety
- `ERA_DETECTION_HYBRID_SUPPORT.md` - Hybrid table storage support
- `ERA_DETECTION_TIMESTAMP_FIX.md` - Timestamp parsing fixes
- Migration guides and optimization reports

### `/deployment`
Deployment and infrastructure documentation
- `PARALLEL_DEPLOYMENT_GUIDE.md` - Google Cloud deployment guide

### `/migrations`
Historical migration and refactoring documentation
- `EPIC1B_REFACTORING_SUMMARY.md` - Major refactoring effort summary
- Storage migration guides

### `/operations`
Operational guides and fixes
- `ERA_DETECTION_OPERATIONS_GUIDE.md` - Era detection operations
- `ERA_DETECTION_IMPROVEMENTS.md` - Recent improvements
- `PREPROCESSING_OPERATIONS_GUIDE.md` - Preprocessing operations
- Bug fixes and operational notes

## Quick Links

- [Pipeline Overview](architecture/PIPELINE_FLOW.md)
- [Local Development](../README.md#local-development-setup)
- [Cloud Deployment](deployment/PARALLEL_DEPLOYMENT_GUIDE.md)
- [GPU Features](architecture/GPU_FEATURE_EXTRACTION.md)
