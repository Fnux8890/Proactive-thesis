# Feature Extraction Folder Structure

## Overview
The feature extraction pipeline has been reorganized for better maintainability and navigation.

## Directory Tree

```
feature_extraction/
│
├── benchmarks/                 # Performance benchmarking
│   ├── src/                   # Benchmark source code
│   │   └── benchmark_db_utils.py
│   ├── docker/                # Docker configurations
│   │   └── benchmark.dockerfile
│   ├── results/               # Benchmark results (JSON)
│   ├── benchmark_requirements.txt
│   └── README.md
│
├── docs/                      # All documentation
│   ├── operations/            # How-to guides
│   │   ├── OBSERVABILITY_GUIDE.md
│   │   ├── TESTING_GUIDE.md
│   │   ├── OPTIMAL_SIGNAL_SELECTION.md
│   │   ├── PREPROCESSING_OPERATIONS_GUIDE.md
│   │   └── ERA_DETECTION_OPERATIONS_GUIDE.md
│   ├── database/              # Database documentation
│   │   ├── db_utils_optimization_report.md
│   │   ├── db_utils_migration_guide.md
│   │   ├── preprocessing_storage_analysis.md
│   │   ├── THREAD_SAFETY_IMPROVEMENTS.md
│   │   ├── HYBRID_SUPPORT.md
│   │   └── TIMESTAMP_PARSING_FIX.md
│   ├── migrations/            # Migration guides
│   │   ├── COMPLETE_HYBRID_MIGRATION.md
│   │   ├── COMPLETE_MIGRATION_SCRIPT.sh
│   │   ├── STORAGE_ALTERNATIVES.md
│   │   └── MIGRATION_TO_HYBRID.md
│   └── README.md              # Documentation index
│
├── db/                        # Database utilities
│   ├── __init__.py
│   ├── connection.py          # Connection pooling
│   ├── chunked_query.py       # Chunked data retrieval
│   └── metrics.py             # Performance metrics
│
├── features/                  # Feature adapters
│   ├── __init__.py
│   └── adapters.py           # Type-safe adapters
│
├── tests/                     # Test suite
│   ├── README.md
│   ├── test_connection_*.py   # Connection tests
│   ├── test_adapters_*.py     # Adapter tests
│   └── test_*.py             # Other tests
│
├── pre_process/               # Pre-processing stage
├── era_detection/             # Python era detection
├── era_detection_rust/        # Rust era detection
├── feature/                   # Feature extraction
├── feature-gpu/               # GPU acceleration
├── examples/                  # Usage examples
│
├── README.md                  # Main documentation
├── EPIC1B_REFACTORING_SUMMARY.md
├── Makefile                   # Build automation
└── docker-compose.test.yaml   # Test configuration
```

## Key Benefits of New Structure

1. **Clear Separation of Concerns**
   - Source code, tests, benchmarks, and docs are clearly separated
   - Each component has its own directory

2. **Easy Navigation**
   - All documentation in `docs/` with subcategories
   - All benchmarks in `benchmarks/`
   - Consistent naming conventions

3. **Better Maintainability**
   - Related files are grouped together
   - Clear hierarchy for different types of content
   - Easy to find and update documentation

4. **Improved Developer Experience**
   - Quick access to guides and documentation
   - Benchmarks isolated from production code
   - Examples readily available

## Quick Navigation

- **Need to run tests?** → Check `tests/` and `docs/operations/TESTING_GUIDE.md`
- **Database optimization?** → See `docs/database/`
- **Migration help?** → Look in `docs/migrations/`
- **Performance testing?** → Go to `benchmarks/`
- **Usage examples?** → Check `examples/`