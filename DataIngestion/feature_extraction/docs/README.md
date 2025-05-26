# Feature Extraction Documentation

This directory contains comprehensive documentation for the feature extraction pipeline.

## Documentation Structure

```
docs/
├── operations/             # Operational guides and procedures
│   ├── OBSERVABILITY_GUIDE.md
│   ├── TESTING_GUIDE.md
│   └── OPTIMAL_SIGNAL_SELECTION.md
├── database/              # Database optimization and utilities
│   ├── db_utils_optimization_report.md
│   ├── db_utils_migration_guide.md
│   └── preprocessing_storage_analysis.md
├── migrations/            # Migration guides and scripts
│   ├── COMPLETE_HYBRID_MIGRATION.md
│   └── COMPLETE_MIGRATION_SCRIPT.sh
├── testing/              # Test documentation
│   └── (test-related docs)
└── README.md
```

## Quick Links

### Getting Started
- [Testing Guide](operations/TESTING_GUIDE.md) - How to run and write tests
- [Observability Guide](operations/OBSERVABILITY_GUIDE.md) - Monitoring and logging

### Database
- [DB Utils Migration Guide](database/db_utils_migration_guide.md) - Migrating to optimized database utilities
- [Storage Analysis](database/preprocessing_storage_analysis.md) - Storage options comparison
- [Optimization Report](database/db_utils_optimization_report.md) - Performance improvements

### Migrations
- [Hybrid Migration](migrations/COMPLETE_HYBRID_MIGRATION.md) - Complete guide to hybrid storage
- [Migration Script](migrations/COMPLETE_MIGRATION_SCRIPT.sh) - Automated migration script

### Operations
- [Signal Selection](operations/OPTIMAL_SIGNAL_SELECTION.md) - Choosing optimal signals for processing

## Component Documentation

### Pre-processing
See [pre_process/report_for_preprocess/](../pre_process/report_for_preprocess/) for detailed pre-processing reports.

### Era Detection
See [era_detection_rust/](../era_detection_rust/) for Rust-based era detection documentation.

### Feature Extraction
See [feature/](../feature/) for Python feature extraction documentation.

## Best Practices

1. **Documentation Updates**: Keep documentation synchronized with code changes
2. **Migration Planning**: Review migration guides before system updates
3. **Testing**: Follow testing guide for consistent test coverage
4. **Monitoring**: Implement observability recommendations for production