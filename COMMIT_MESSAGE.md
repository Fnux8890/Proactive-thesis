feat: Major refactoring of feature extraction pipeline (Epic 1B)

## Summary
Comprehensive refactoring of the feature extraction pipeline to improve maintainability, 
performance, and documentation. Fixed critical issues with JSONB parsing in era_detector 
and reorganized the entire project structure.

## Key Changes

### 🔧 Era Detector Fixes
- Fixed JSONB parsing in era_detector to handle hybrid table format
- Added OptimalSignals struct for prioritizing high-quality signals
- Implemented missing quantize_signal_f64 function for HMM processing
- Updated Rust to 1.87.0 and fixed all compilation warnings
- Fixed function calls from perform_hmm to viterbi_path_from_observations
- Added comprehensive error handling and logging

### 📁 Project Reorganization
- Created organized folder structure:
  - `benchmarks/` - Performance testing with src/, docker/, and results/
  - `docs/` - All documentation categorized by operations, database, migrations
  - `db/` - New database utilities with connection pooling
  - `features/` - Type-safe feature adapters
  - `tests/` - Comprehensive test suite
  - `examples/` - Usage examples

### 📚 Documentation
- Created comprehensive operations guides:
  - PREPROCESSING_OPERATIONS_GUIDE.md with sequence diagrams
  - ERA_DETECTION_OPERATIONS_GUIDE.md with algorithm details
  - RUST_PIPELINE_OPERATIONS_GUIDE.md with C# migration guide
- Added migration documentation:
  - Complete hybrid storage migration guide
  - Database optimization reports
  - Storage alternatives analysis
- Created folder structure documentation and Epic 1B summary

### 🚀 Performance Improvements
- Implemented thread-safe connection pooling
- Added chunked query utilities for large datasets
- Created optimized db_utils with retry logic
- Added performance metrics collection

### ✅ Testing
- Added comprehensive test suite for database connections
- Implemented thread safety tests
- Created type adapter tests
- Added performance benchmarks

### 🐛 Bug Fixes
- Fixed preprocessed_features table detection in preprocess.py
- Resolved JSONB vs hybrid schema mismatch
- Fixed parallel processing error handling
- Corrected signal selection logic

## Technical Details

### Database Changes
- Updated to use hybrid storage (native columns + JSONB)
- Optimized connection pooling with deadpool
- Added proper error handling for all DB operations

### Code Quality
- Fixed all Rust compilation warnings
- Improved error messages and logging
- Added proper type safety throughout
- Cleaned up unused imports and code

### Migration Support
- Created detailed C# migration guides for each component
- Documented all algorithms for implementation in other languages
- Provided sequence diagrams for better understanding

## Files Changed
- Modified: era_detection_rust/src/main.rs (JSONB parsing, signal optimization)
- Modified: pre_process/preprocess.py (table detection fix)
- Added: Multiple operations guides and migration documentation
- Added: Comprehensive test suite
- Added: Benchmarking infrastructure
- Reorganized: Entire folder structure for better maintainability

## Impact
This refactoring provides a solid foundation for the feature extraction pipeline with:
- Better error handling and reliability
- Improved performance through optimizations
- Clear documentation for maintenance and migration
- Organized structure for easier navigation
- Comprehensive testing for quality assurance

Resolves: JSONB parsing issues, performance bottlenecks, documentation gaps
Related: Epic 1B requirements for pipeline refactoring