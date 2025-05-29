# Documentation Structure

## Main Documentation (`/docs`)

### Architecture (`/docs/architecture`)
- `DATA_SOURCE_INTEGRATION.md` - Data sources and integration points
- `GPU_FEATURE_EXTRACTION.md` - How GPU acceleration works with tsfresh
- `PARALLEL_PROCESSING.md` - Overview of all parallel processing
- `PIPELINE_FLOW.md` - Complete pipeline architecture
- `/parallel/` - Detailed parallel implementation docs

### Database (`/docs/database`)
- `db_utils_migration_guide.md` - Database utilities migration
- `db_utils_optimization_report.md` - Performance optimizations
- `preprocessing_storage_analysis.md` - Storage strategy analysis
- `ERA_DETECTION_HYBRID_SUPPORT.md` - Hybrid table support
- `ERA_DETECTION_TIMESTAMP_FIX.md` - Timestamp parsing fixes

### Operations (`/docs/operations`)
- `ERA_DETECTION_OPERATIONS_GUIDE.md` - Era detection operations
- `ERA_DETECTION_IMPROVEMENTS.md` - Recent improvements
- `ERA_DETECTOR_FINAL_FIXES.md` - Final bug fixes
- `ERA_DETECTOR_JSONB_FIX.md` - JSONB storage fix
- `PREPROCESSING_OPERATIONS_GUIDE.md` - Preprocessing operations
- `OBSERVABILITY_GUIDE.md` - Monitoring and observability
- `OPTIMAL_SIGNAL_SELECTION.md` - Signal selection strategy
- `TESTING_GUIDE.md` - Testing procedures

### Migrations (`/docs/migrations`)
- `EPIC1B_REFACTORING_SUMMARY.md` - Major refactoring history
- `COMPLETE_HYBRID_MIGRATION.md` - Complete migration guide
- `MIGRATION_TO_HYBRID.md` - Hybrid storage migration
- `STORAGE_ALTERNATIVES.md` - Storage options analysis

### Deployment (`/docs/deployment`)
- `PARALLEL_DEPLOYMENT_GUIDE.md` - Google Cloud deployment

## Component Documentation

Each component maintains its own README:
- `feature_extraction/README.md` - Main overview
- `feature_extraction/parallel/README.md` - Parallel processing
- `feature_extraction/feature/README.md` - CPU features
- `feature_extraction/pre_process/README.md` - Preprocessing
- `feature_extraction/benchmarks/README.md` - Benchmarks
- `feature_extraction/tests/README.md` - Tests

The preprocessing reports remain in:
`feature_extraction/pre_process/report_for_preprocess/`
