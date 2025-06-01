# DataIngestion Documentation

Welcome to the comprehensive documentation for the Data Ingestion and Optimization Pipeline. This documentation covers architecture, deployment, operations, and development guides.

## 📚 Documentation Index

### 🏗️ Architecture
System design, analysis, and architectural decisions
- **Core Architecture**
  - [Pipeline Flow](architecture/PIPELINE_FLOW.md) - Complete pipeline architecture
  - [Data Source Integration](architecture/DATA_SOURCE_INTEGRATION.md) - Data source specifications
  - [GPU Feature Extraction](architecture/GPU_FEATURE_EXTRACTION.md) - GPU acceleration approach
  - [Parallel Processing](architecture/PARALLEL_PROCESSING.md) - Parallel processing overview
  - [Configuration Simplification](architecture/CONFIGURATION_SIMPLIFICATION.md) - Configuration improvements
  
- **Analysis Reports**
  - [Model Builder Code Analysis](architecture/analysis/model_builder_code_analysis.md)
  - [Model Builder Data Integration](architecture/analysis/model_builder_data_integration.md)
  - [Rust-Python Usability](architecture/analysis/rust_python_usability.md)

- **Implementations**
  - [GAN Synthetic Data](architecture/implementations/gan_synthetic_data.md) - Synthetic data generation
  - [Multi-Level Feature Extraction](MULTI_LEVEL_FEATURE_EXTRACTION.md) - Hierarchical feature extraction
  - [Sparse Pipeline Implementation](SPARSE_PIPELINE_IMPLEMENTATION.md) - 4-stage sparse data processing

### 💾 Database
Database schemas, optimizations, and storage strategies
- [DB Utils Optimization Report](database/db_utils_optimization_report.md) - Performance optimizations
- [Preprocessing Storage Analysis](database/preprocessing_storage_analysis.md)
- [Era Detection Hybrid Support](database/ERA_DETECTION_HYBRID_SUPPORT.md)
- [Era Detection Timestamp Fix](database/ERA_DETECTION_TIMESTAMP_FIX.md)
- [GAN Data Description](database/gan_data_description.md)

### 🚀 Deployment
Deployment guides and infrastructure setup
- **Docker & Compose**
  - [Docker Compose Guide](deployment/DOCKER_COMPOSE_GUIDE.md) - Complete Docker usage guide
  - [Cloud Compose Quick Reference](deployment/CLOUD_COMPOSE_QUICK_REF.md)
  - [Cloud Production Deployment](CLOUD_PRODUCTION_DEPLOYMENT.md)
  - [Parallel Deployment Guide](deployment/PARALLEL_DEPLOYMENT_GUIDE.md)

- **Terraform**
  - [Terraform Update Summary](deployment/terraform/update_summary.md)

### 🔧 Operations
Operational procedures, monitoring, and maintenance
- **Core Operations**
  - [Era Detection Operations](operations/ERA_DETECTION_OPERATIONS_GUIDE.md)
  - [Preprocessing Operations](operations/PREPROCESSING_OPERATIONS_GUIDE.md)
  - [Rust Pipeline Operations](operations/rust_pipeline_operations.md)
  - [Docker Services Status](operations/DOCKER_SERVICES_STATUS.md)

- **Fixes & Solutions**
  - [Model Builder Fix](operations/fixes/MODEL_BUILDER_FIX.md)
  - [Rust Warnings Fixed](operations/fixes/rust_warnings_fixed.md)
  - [Era Detector Final Fixes](operations/ERA_DETECTOR_FINAL_FIXES.md)
  - [Era Detector JSONB Fix](operations/ERA_DETECTOR_JSONB_FIX.md)

- **Implementations**
  - [Rust Null Value Logging](operations/implementations/rust_null_value_logging.md)

- **Cleanup Operations**
  - [Cleanup Summary](operations/cleanup/CLEANUP_SUMMARY.md)
  - [Workspace Clean](operations/cleanup/WORKSPACE_CLEAN.md)
  - [Feature Extraction Cleanup](operations/cleanup/feature_extraction_cleanup.md)

### 🧪 Testing
Testing procedures, guides, and results
- [Testing Guide](testing/TESTING_GUIDE.md) - Comprehensive testing procedures
- [Docker Services Test Results](testing/results/docker_services_test_results.md)

### 🔄 Migrations
Migration guides and historical changes
- [EPIC1B Refactoring Summary](migrations/EPIC1B_REFACTORING_SUMMARY.md)
- [Complete Hybrid Migration](migrations/COMPLETE_HYBRID_MIGRATION.md)
- [Migration to Hybrid](migrations/MIGRATION_TO_HYBRID.md)
- [Storage Alternatives](migrations/STORAGE_ALTERNATIVES.md)

### ✨ Features
Feature documentation and specifications
- [Multi-Level Feature Extraction](MULTI_LEVEL_FEATURE_EXTRACTION.md)
- [Parallel Feature Extraction Guide](PARALLEL_FEATURE_EXTRACTION_GUIDE.md)
- [Lightgroup Feature](features/lightgroup_feature.md)
- [Optimal Signal Selection](operations/OPTIMAL_SIGNAL_SELECTION.md)

### 📖 Additional Resources
- [Documentation Structure](DOCUMENTATION_STRUCTURE.md)
- [Documentation Reorganization Plan](DOCUMENTATION_REORGANIZATION.md)
- [Folder Structure](FOLDER_STRUCTURE.md)
- [MOEA Performance Comparison](MOEA_PERFORMANCE_COMPARISON.md)

## 🚦 Quick Start Guides

### For Developers
1. [Pipeline Overview](architecture/PIPELINE_FLOW.md) - Understand the system
2. [Docker Compose Guide](deployment/DOCKER_COMPOSE_GUIDE.md) - Local development
3. [Testing Guide](testing/TESTING_GUIDE.md) - Run tests

### For DevOps
1. [Cloud Production Deployment](CLOUD_PRODUCTION_DEPLOYMENT.md) - Deploy to cloud
2. [Docker Services Status](operations/DOCKER_SERVICES_STATUS.md) - Monitor services
3. [Parallel Deployment Guide](deployment/PARALLEL_DEPLOYMENT_GUIDE.md) - Scale deployment

### For Data Scientists
1. [Multi-Level Feature Extraction](MULTI_LEVEL_FEATURE_EXTRACTION.md) - Feature engineering
2. [GAN Synthetic Data](architecture/implementations/gan_synthetic_data.md) - Data generation
3. [MOEA Performance Comparison](MOEA_PERFORMANCE_COMPARISON.md) - Optimization results

## 📋 Documentation Standards

When adding new documentation:
1. Use descriptive filenames in UPPER_SNAKE_CASE
2. Include a clear title and overview section
3. Add to the appropriate category folder
4. Update this index with a link and description
5. Use markdown formatting consistently
6. Include code examples where relevant
7. Add diagrams for complex concepts

## 🔍 Finding Documentation

- **By Component**: Check component folders (feature_extraction/, model_builder/, etc.)
- **By Topic**: Use the categories above
- **By Search**: Use `grep -r "search term" docs/` from the DataIngestion directory

## 📝 Recent Updates

- Added [Multi-Level Feature Extraction](MULTI_LEVEL_FEATURE_EXTRACTION.md) documentation
- Reorganized documentation structure for better discoverability
- Added comprehensive testing and deployment guides
- Updated architecture documentation with latest improvements

## 📂 Guide to Documentation Subfolders

This section provides a brief overview of the main subfolders within `DataIngestion/docs/` to help you navigate the documentation:

- **`architecture/`**: Contains documents related to the system's design, architectural decisions, component breakdowns, and analysis reports. This includes high-level pipeline flows, specific component designs (e.g., GPU feature extraction), and diagrams.
  - `architecture/analysis/`: Detailed analysis of specific components or approaches.
  - `architecture/implementations/`: Documentation on specific implemented architectural patterns or significant features.

- **`database/`**: Focuses on database schemas, data model details, storage strategies, database-specific optimizations, and migration information related to database structure.

- **`deployment/`**: Includes guides and information for deploying the DataIngestion pipeline. This covers Docker, Docker Compose configurations, cloud deployment strategies (e.g., using Terraform), and environment setup.
  - `deployment/terraform/`: Specific documentation for Terraform-based infrastructure management.

- **`experiments/`**: (Assuming based on common practice, verify actual content) Likely contains records, results, and analyses of various experiments conducted during development, such as performance tuning, algorithm comparisons, or feature validation.

- **`feature_extraction/`**: Holds detailed documentation specific to the feature extraction processes, methodologies, and individual feature implementations. This is a deep dive into how features are engineered.

- **`features/`**: General documentation about specific features developed or integrated into the system, perhaps at a higher level than `feature_extraction/` or covering cross-cutting feature concerns.

- **`gpu/`**: Contains documentation specifically related to GPU acceleration, including setup, CUDA kernel details, RAPIDS/PyTorch usage in the pipeline, performance benchmarks, and troubleshooting for GPU components.

- **`migrations/`**: Documents the process and history of data schema migrations, significant code refactoring efforts that impact data or interfaces, and guides for upgrading between different versions of the pipeline or its components.

- **`notes/`**: A collection of miscellaneous notes, historical decisions, logs of specific fixes, developer thoughts, or meeting summaries that provide context but may not fit into formal documentation categories.

- **`operations/`**: Provides operational guides, how-to documents for running and maintaining the pipeline, monitoring procedures, troubleshooting common issues, and guides for specific operational tasks (e.g., running era detection, preprocessing).
  - `operations/cleanup/`: Guides for cleaning up data, logs, or intermediate artifacts.
  - `operations/fixes/`: Documentation for specific bug fixes and solutions implemented.
  - `operations/implementations/`: Notes on specific operational implementations or tools.

- **`pipelines/`**: Contains user guides and detailed explanations for specific end-to-end pipelines that can be run, such as the sparse pipeline or enhanced sparse pipeline. This includes prerequisites, setup, execution steps, and expected outputs.

- **`testing/`**: Includes testing strategies, guides for running tests, descriptions of test suites, test case details, and summaries of test results.
  - `testing/results/`: Stores outputs or summaries from test runs.

- **`tutorials/`**: (Assuming based on common practice, verify actual content) Likely provides step-by-step tutorials for common tasks, onboarding new developers, or using specific parts of the system.

Key standalone files in `docs/` often provide overarching summaries, indexes (like this `README.md`), or specific high-level plans (e.g., `ENHANCED_SPARSE_PIPELINE_README.md`, `PIPELINE_OVERVIEW.md`).
