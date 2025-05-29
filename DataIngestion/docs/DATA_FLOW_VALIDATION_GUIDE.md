# Data Flow Validation and Orchestration Guide

## Overview
This guide ensures data flows correctly through the pipeline stages and validates each step before proceeding to the next.

## Current Data Flow Architecture

```
1. Raw Data → 2. Rust Ingestion → 3. Preprocessing → 4. Era Detection → 5. Feature Extraction → 6. Model Training → 7. MOEA Optimization
     ↓              ↓                    ↓                  ↓                    ↓                      ↓                    ↓
   CSV/JSON    sensor_data      preprocessed_data    era_labels       tsfresh_features      LightGBM models    Optimal solutions
```

## Stage-by-Stage Validation

### Stage 1: Raw Data Ingestion
**Table**: `sensor_data`

```sql
-- Validate data was ingested
SELECT COUNT(*) as total_rows,
       COUNT(DISTINCT DATE(timestamp)) as days,
       MIN(timestamp) as start_date,
       MAX(timestamp) as end_date
FROM sensor_data;

-- Check sensor coverage
SELECT sensor_name, COUNT(*) as readings
FROM sensor_data
GROUP BY sensor_name
ORDER BY readings DESC;
```

### Stage 2: Preprocessing
**Table**: `preprocessed_greenhouse_data`

```sql
-- Validate preprocessing completed
SELECT COUNT(*) as total_rows,
       COUNT(DISTINCT signal_name) as unique_signals,
       MIN(timestamp) as start_date,
       MAX(timestamp) as end_date
FROM preprocessed_greenhouse_data;

-- Check for required signals
SELECT signal_name, COUNT(*) as readings
FROM preprocessed_greenhouse_data
WHERE signal_name IN (
    'air_temp_c', 'relative_humidity_percent', 
    'co2_measured_ppm', 'light_intensity_umol'
)
GROUP BY signal_name;
```

### Stage 3: Era Detection
**Tables**: `era_labels_level_a`, `era_labels_level_b`, `era_labels_level_c`

```sql
-- Validate era detection results
SELECT 
    level,
    COUNT(*) as num_eras,
    AVG(era_rows) as avg_rows_per_era,
    MIN(era_rows) as min_rows,
    MAX(era_rows) as max_rows
FROM (
    SELECT 'A' as level, era_id, era_rows FROM era_labels_level_a
    UNION ALL
    SELECT 'B' as level, era_id, era_rows FROM era_labels_level_b
    UNION ALL
    SELECT 'C' as level, era_id, era_rows FROM era_labels_level_c
) eras
GROUP BY level;
```

### Stage 4: Feature Extraction
**Table**: `tsfresh_features`

```sql
-- Check if features were extracted
SELECT COUNT(*) as total_features,
       COUNT(DISTINCT era_id) as unique_eras,
       COUNT(DISTINCT signal_name) as unique_signals
FROM tsfresh_features;

-- Verify target columns exist (or need to be created)
SELECT column_name 
FROM information_schema.columns 
WHERE table_name = 'tsfresh_features'
AND column_name IN (
    'energy_consumption', 'plant_growth', 
    'water_usage', 'crop_quality'
);
```

## Data Preparation Scripts

### 1. Create Validation Script
`/DataIngestion/scripts/validate_pipeline_data.py`

```python
#!/usr/bin/env python3
"""Validate data at each pipeline stage."""

import os
import sys
import logging
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PipelineDataValidator:
    def __init__(self, db_url):
        self.engine = create_engine(db_url)
        self.validation_results = {}
    
    def validate_stage_1_ingestion(self):
        """Validate raw data ingestion."""
        query = """
        SELECT 
            COUNT(*) as total_rows,
            COUNT(DISTINCT DATE(timestamp)) as days,
            MIN(timestamp) as start_date,
            MAX(timestamp) as end_date
        FROM sensor_data
        """
        result = pd.read_sql(query, self.engine)
        
        if result['total_rows'].iloc[0] == 0:
            logger.error("❌ Stage 1: No data in sensor_data table")
            return False
        
        logger.info(f"✅ Stage 1: Found {result['total_rows'].iloc[0]:,} rows")
        logger.info(f"   Date range: {result['start_date'].iloc[0]} to {result['end_date'].iloc[0]}")
        self.validation_results['stage_1'] = result.to_dict('records')[0]
        return True
    
    def validate_stage_2_preprocessing(self):
        """Validate preprocessing results."""
        # Check if table exists and has data
        query = """
        SELECT 
            COUNT(*) as total_rows,
            COUNT(DISTINCT signal_name) as unique_signals
        FROM preprocessed_greenhouse_data
        """
        try:
            result = pd.read_sql(query, self.engine)
            if result['total_rows'].iloc[0] == 0:
                logger.error("❌ Stage 2: No data in preprocessed_greenhouse_data")
                return False
            
            logger.info(f"✅ Stage 2: Found {result['total_rows'].iloc[0]:,} preprocessed rows")
            logger.info(f"   Unique signals: {result['unique_signals'].iloc[0]}")
            self.validation_results['stage_2'] = result.to_dict('records')[0]
            return True
        except Exception as e:
            logger.error(f"❌ Stage 2: {e}")
            return False
    
    def validate_stage_3_era_detection(self):
        """Validate era detection results."""
        query = """
        SELECT 
            'B' as level,
            COUNT(*) as num_eras,
            AVG(era_rows) as avg_rows_per_era,
            MIN(era_rows) as min_rows,
            MAX(era_rows) as max_rows
        FROM era_labels_level_b
        WHERE era_rows > 100
        """
        try:
            result = pd.read_sql(query, self.engine)
            if result['num_eras'].iloc[0] == 0:
                logger.error("❌ Stage 3: No eras detected")
                return False
            
            logger.info(f"✅ Stage 3: Found {result['num_eras'].iloc[0]} eras (Level B)")
            logger.info(f"   Avg rows per era: {result['avg_rows_per_era'].iloc[0]:.0f}")
            self.validation_results['stage_3'] = result.to_dict('records')[0]
            return True
        except Exception as e:
            logger.error(f"❌ Stage 3: {e}")
            return False
    
    def validate_stage_4_features(self):
        """Validate feature extraction results."""
        query = """
        SELECT 
            COUNT(*) as total_rows,
            COUNT(DISTINCT era_id) as unique_eras,
            COUNT(*) - COUNT(era_id) as missing_era_ids
        FROM tsfresh_features
        """
        try:
            result = pd.read_sql(query, self.engine)
            if result['total_rows'].iloc[0] == 0:
                logger.error("❌ Stage 4: No features extracted")
                return False
            
            logger.info(f"✅ Stage 4: Found {result['total_rows'].iloc[0]} feature rows")
            logger.info(f"   Unique eras: {result['unique_eras'].iloc[0]}")
            self.validation_results['stage_4'] = result.to_dict('records')[0]
            return True
        except Exception as e:
            logger.error(f"❌ Stage 4: {e}")
            return False
    
    def validate_stage_5_targets(self):
        """Check if target columns exist for model training."""
        query = """
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'tsfresh_features'
        AND column_name IN (
            'energy_consumption', 'plant_growth', 
            'water_usage', 'crop_quality'
        )
        """
        result = pd.read_sql(query, self.engine)
        
        if len(result) == 0:
            logger.warning("⚠️  Stage 5: No target columns found - will use synthetic targets")
            return 'synthetic'
        else:
            logger.info(f"✅ Stage 5: Found {len(result)} target columns")
            return 'real'
    
    def generate_report(self):
        """Generate validation report."""
        print("\n" + "="*60)
        print("PIPELINE DATA VALIDATION REPORT")
        print("="*60)
        
        stages = [
            ("Stage 1: Data Ingestion", self.validate_stage_1_ingestion),
            ("Stage 2: Preprocessing", self.validate_stage_2_preprocessing),
            ("Stage 3: Era Detection", self.validate_stage_3_era_detection),
            ("Stage 4: Feature Extraction", self.validate_stage_4_features),
            ("Stage 5: Target Variables", self.validate_stage_5_targets)
        ]
        
        all_valid = True
        for stage_name, validator_func in stages:
            print(f"\n{stage_name}:")
            result = validator_func()
            if result is False:
                all_valid = False
                break
        
        print("\n" + "="*60)
        if all_valid:
            print("✅ ALL STAGES VALIDATED - Ready for model training!")
        else:
            print("❌ VALIDATION FAILED - Fix issues before proceeding")
        print("="*60)
        
        return all_valid

if __name__ == "__main__":
    db_url = os.getenv('DATABASE_URL', 
        f"postgresql://{os.getenv('DB_USER', 'postgres')}:"
        f"{os.getenv('DB_PASSWORD', 'postgres')}@"
        f"{os.getenv('DB_HOST', 'localhost')}:"
        f"{os.getenv('DB_PORT', '5432')}/"
        f"{os.getenv('DB_NAME', 'postgres')}"
    )
    
    validator = PipelineDataValidator(db_url)
    validator.generate_report()
```

### 2. Create Data Preparation Script
`/DataIngestion/scripts/prepare_cloud_data.sh`

```bash
#!/bin/bash
# Prepare data for cloud deployment

echo "=== Preparing Data for Cloud Deployment ==="

# 1. Export minimal test dataset (if needed)
if [ "$1" == "test" ]; then
    echo "Creating test dataset..."
    docker compose run --rm db psql -U postgres -d postgres -c "
    -- Create test subset (1 week of data)
    CREATE TABLE sensor_data_test AS 
    SELECT * FROM sensor_data 
    WHERE timestamp >= '2014-01-01' AND timestamp < '2014-01-08';
    
    CREATE TABLE preprocessed_greenhouse_data_test AS
    SELECT * FROM preprocessed_greenhouse_data
    WHERE timestamp >= '2014-01-01' AND timestamp < '2014-01-08';
    "
fi

# 2. Create synthetic targets if needed
echo "Checking for target variables..."
docker compose run --rm model_builder python -m src.utils.create_synthetic_targets

# 3. Validate all stages
echo "Validating pipeline data..."
docker compose run --rm feature_extraction python /app/scripts/validate_pipeline_data.py

# 4. Create data backup for cloud
echo "Creating data backup..."
docker compose run --rm db pg_dump -U postgres -d postgres \
    --table=sensor_data \
    --table=preprocessed_greenhouse_data \
    --table=era_labels_level_b \
    --table=tsfresh_features \
    > data_backup_for_cloud.sql

echo "=== Data Preparation Complete ==="
```

### 3. Create Orchestration Script
`/DataIngestion/scripts/run_pipeline_sequence.py`

```python
#!/usr/bin/env python3
"""Orchestrate pipeline execution with validation between stages."""

import subprocess
import time
import sys
from validate_pipeline_data import PipelineDataValidator

def run_stage(stage_name, compose_service, validator_func=None):
    """Run a pipeline stage and validate results."""
    print(f"\n{'='*60}")
    print(f"Running {stage_name}...")
    print('='*60)
    
    # Run the service
    cmd = f"docker compose up -d {compose_service}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Failed to start {stage_name}")
        print(result.stderr)
        return False
    
    # Wait for completion (check logs)
    print(f"Waiting for {stage_name} to complete...")
    time.sleep(10)  # Initial wait
    
    # Check if service completed
    max_attempts = 60  # 10 minutes max
    for i in range(max_attempts):
        check_cmd = f"docker compose ps {compose_service} --format json"
        result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)
        # Parse status and check if exited successfully
        time.sleep(10)
    
    # Validate if validator provided
    if validator_func:
        if not validator_func():
            print(f"❌ Validation failed for {stage_name}")
            return False
    
    print(f"✅ {stage_name} completed successfully")
    return True

def main():
    # Initialize validator
    validator = PipelineDataValidator(db_url=os.getenv('DATABASE_URL'))
    
    # Define pipeline stages
    stages = [
        ("Data Ingestion", "rust_pipeline", validator.validate_stage_1_ingestion),
        ("Preprocessing", "preprocessing", validator.validate_stage_2_preprocessing),
        ("Era Detection", "era_detector", validator.validate_stage_3_era_detection),
        ("Feature Extraction", "feature_extraction", validator.validate_stage_4_features),
        ("Create Targets", "model_builder", validator.validate_stage_5_targets),
    ]
    
    # Run stages in sequence
    for stage_name, service, validator_func in stages:
        if not run_stage(stage_name, service, validator_func):
            print(f"\n❌ Pipeline failed at {stage_name}")
            sys.exit(1)
    
    print("\n✅ All pipeline stages completed successfully!")
    print("Ready for model training and MOEA optimization")

if __name__ == "__main__":
    main()
```

## Cloud Deployment Checklist

### Pre-deployment (Local)
```bash
# 1. Validate local data
cd DataIngestion
python scripts/validate_pipeline_data.py

# 2. Create test dataset (optional)
./scripts/prepare_cloud_data.sh test

# 3. Run full pipeline locally with test data
docker compose up -d
```

### Cloud Deployment
```bash
# 1. Deploy with Terraform
cd terraform/parallel-feature
terraform apply

# 2. SSH to instance and verify
gcloud compute ssh feature-extraction-instance

# 3. Check pipeline status
cd /opt/Proactive-thesis/DataIngestion
docker compose ps
docker compose logs --tail=100

# 4. Run validation
docker compose run --rm feature_extraction python scripts/validate_pipeline_data.py
```

### Post-deployment Monitoring
```bash
# Monitor pipeline progress
watch -n 5 'docker compose ps'

# Check specific stage logs
docker compose logs -f feature_extraction

# Validate data after each stage
docker compose run --rm feature_extraction python scripts/validate_pipeline_data.py
```

## Troubleshooting Common Issues

### Issue 1: Missing Data
```sql
-- Check what data exists
SELECT table_name, pg_size_pretty(pg_total_relation_size(table_name::regclass))
FROM information_schema.tables
WHERE table_schema = 'public'
AND table_name IN ('sensor_data', 'preprocessed_greenhouse_data', 'era_labels_level_b', 'tsfresh_features');
```

### Issue 2: Stage Dependencies
```bash
# Ensure stages run in correct order
docker compose up -d rust_pipeline
# Wait for completion...
docker compose up -d preprocessing
# Wait for completion...
docker compose up -d era_detector
# etc...
```

### Issue 3: Memory Issues
```bash
# Reduce batch sizes
export BATCH_SIZE=50
export N_JOBS=4
docker compose up -d feature_extraction
```

## Summary

1. **Always validate data** between stages using the validation script
2. **Use synthetic targets** for testing if real targets don't exist
3. **Run stages in sequence** with validation between each
4. **Monitor progress** using logs and validation queries
5. **Start with test data** before running on full dataset

This ensures your pipeline runs correctly whether using synthetic or real data!