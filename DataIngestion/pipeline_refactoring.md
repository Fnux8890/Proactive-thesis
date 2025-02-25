# Data Ingestion Pipeline Refactoring Plan

## Current Architecture Analysis

After reviewing the existing codebase and architecture, we've identified several areas that need improvement to achieve a more streamlined and maintainable data pipeline:

### Issues Identified

1. **Disconnected Data Flow**: The current pipeline architecture doesn't clearly enforce a sequential flow from ingestion through to feature extraction, with TimescaleDB as the central source of truth.

2. **Multiple Data Sources**: Feature extraction appears to be pulling from both raw files and database, creating duplication and potential inconsistencies.

3. **Unclear Boundaries**: The responsibilities between Elixir ingestion service and Python processing components have overlapping concerns.

4. **Airflow DAGs Structure**: Current DAGs don't properly reflect the logical sequence of operations and dependencies.

5. **Missing Database Migrations**: No clear database schema evolution strategy, which is essential for maintaining data integrity across pipeline changes.

## Correct Pipeline Architecture

The pipeline should follow this strict sequential flow:

```
Raw Data Sources → Ingestion → Validation → Transformation → Persistence (TimescaleDB) → Feature Extraction → Analysis/MOGA
```

### Key Principles

1. **Single Source of Truth**: TimescaleDB should be the canonical data store that all downstream processes read from.

2. **Clear Component Boundaries**:
   - Elixir Ingestion Service: File detection, validation, triggering
   - Python Processing Service: Data cleaning, transformation, persistence
   - Feature Extraction Service: Works exclusively from persisted data in TimescaleDB

3. **Explicit Data Flow**: Each step in the pipeline should have clear inputs and outputs with no circular dependencies.

4. **Metadata Tracking**: All data transformations should be tracked in the metadata catalog for lineage and audit purposes.

## Required Changes

### 1. Database Schema Restructuring

We need to establish a clear database schema with the following components:

1. **Staging Area**:

   ```
   staging.raw_aarslev
   staging.raw_knudjepsen
   ```

   - Contains raw data with minimal transformations
   - Includes metadata columns (_ingested_at, _source_file, _status)

2. **Timeseries Data**:

   ```
   timeseries.aarslev_data
   timeseries.knudjepsen_data
   ```

   - Cleaned, transformed, and normalized data
   - Hypertables with time partitioning
   - Includes metadata columns (_processed_at, _feature_extracted)

3. **Feature Storage**:

   ```
   features.extracted_features
   features.feature_importance
   features.feature_metadata
   ```

   - Stores extracted features linked to timeseries data
   - Maintains feature importance metrics
   - Tracks feature metadata (source, algorithm, parameters)

### 2. Component Refactoring

#### Elixir Ingestion Service

1. **File Watcher** (already implemented):
   - Monitor directories for new files
   - Validate file formats and basic structure
   - Register files for ingestion

2. **Ingestion API** (already implemented):
   - Endpoints for status monitoring
   - Metadata querying
   - Ingestion triggering

3. **Pipeline Interface**:
   - Clear separation from processing logic
   - Focus on orchestration, not transformation

#### Python Processing Service

1. **Data Loading**:
   - Read files detected by Elixir service
   - Parse into common format
   - Load into staging tables

2. **Data Transformation**:
   - Clean and normalize data
   - Apply domain-specific transformations
   - Generate derived measurements

3. **Data Persistence**:
   - Write to TimescaleDB timeseries tables
   - Update metadata about processing status

#### Feature Extraction Service

1. **Feature Discovery**:
   - Read exclusively from TimescaleDB
   - Apply time-series feature extraction algorithms
   - Calculate statistical properties

2. **Feature Selection**:
   - Evaluate feature importance
   - Select relevant features for MOGA
   - Store feature metadata

### 3. Airflow DAGs Restructuring

We need to restructure the Airflow DAGs to reflect the correct flow:

1. **Data Ingestion DAG**:
   - Check for new files (via Elixir API)
   - Trigger loading into staging
   - Verify data was loaded correctly
   - Clear separation from downstream processes

2. **Data Processing DAG**:
   - Process data from staging to timeseries tables
   - Apply transformations
   - Update processing status
   - Verify processing completed successfully

3. **Feature Extraction DAG**:
   - Query for unprocessed timeseries data
   - Apply feature extraction algorithms
   - Store features and metadata
   - Update feature extraction status

4. **Quality Monitoring DAG**:
   - Monitor data quality metrics
   - Track pipeline performance
   - Generate alerts for anomalies
   - Update feedback loop for continuous improvement

### 4. Metadata Catalog Enhancement

1. **Schema Registry**:
   - Track schema versions
   - Document table relationships
   - Log schema changes

2. **Data Lineage**:
   - Track data flow from source to features
   - Document transformations applied
   - Enable audit trail for regulatory compliance

3. **Quality Metrics**:
   - Track data quality over time
   - Monitor feature stability
   - Provide feedback to ingestion process

## Action Items

1. **Database Migrations**:
   - Create migration scripts for staging schema
   - Create migration scripts for timeseries schema
   - Create migration scripts for features schema

2. **Component Development**:
   - Update Elixir ingestion service to focus on orchestration
   - Refine Python processing services for clear boundaries
   - Ensure feature extraction reads only from TimescaleDB

3. **Pipeline Orchestration**:
   - Update Airflow DAGs to reflect correct sequence
   - Implement proper dependencies between tasks
   - Add validation steps between pipeline stages

4. **Monitoring and Feedback**:
   - Implement comprehensive logging
   - Create dashboards for pipeline monitoring
   - Establish feedback mechanisms for continuous improvement

## Implementation Timeline

### Week 1: Database Schema and Migrations

- Define complete database schema
- Create migration scripts
- Test data persistence patterns

### Week 2: Component Refactoring

- Update Elixir service
- Refine Python processing
- Ensure clear interfaces

### Week 3: Pipeline Orchestration

- Restructure Airflow DAGs
- Implement proper task dependencies
- Test end-to-end pipeline

### Week 4: Monitoring and Documentation

- Create monitoring dashboards
- Complete pipeline documentation
- System testing and optimization

## Conclusion

By implementing these changes, we will create a more robust, maintainable, and logical data pipeline that properly leverages TimescaleDB as the central source of truth. This will provide a solid foundation for feature extraction and subsequent MOGA optimization processes.

The refactored architecture will ensure clear separation of concerns between components, explicit data flow, and comprehensive metadata tracking – all essential for a production-grade data pipeline.
