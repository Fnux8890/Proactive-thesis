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
        try:
            result = pd.read_sql(text(query), self.engine)
            
            if result['total_rows'].iloc[0] == 0:
                logger.error("❌ Stage 1: No data in sensor_data table")
                return False
            
            logger.info(f"✅ Stage 1: Found {result['total_rows'].iloc[0]:,} rows")
            logger.info(f"   Date range: {result['start_date'].iloc[0]} to {result['end_date'].iloc[0]}")
            self.validation_results['stage_1'] = result.to_dict('records')[0]
            return True
        except Exception as e:
            logger.error(f"❌ Stage 1: {e}")
            return False
    
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
            result = pd.read_sql(text(query), self.engine)
            if result['total_rows'].iloc[0] == 0:
                logger.error("❌ Stage 2: No data in preprocessed_greenhouse_data")
                return False
            
            logger.info(f"✅ Stage 2: Found {result['total_rows'].iloc[0]:,} preprocessed rows")
            logger.info(f"   Unique signals: {result['unique_signals'].iloc[0]}")
            
            # Check for key signals
            signal_query = """
            SELECT signal_name, COUNT(*) as count
            FROM preprocessed_greenhouse_data
            WHERE signal_name IN ('air_temp_c', 'relative_humidity_percent', 'co2_measured_ppm', 'light_intensity_umol')
            GROUP BY signal_name
            """
            signals = pd.read_sql(text(signal_query), self.engine)
            logger.info(f"   Key signals found: {len(signals)}/4")
            
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
            result = pd.read_sql(text(query), self.engine)
            if result['num_eras'].iloc[0] == 0:
                logger.error("❌ Stage 3: No eras detected")
                return False
            
            logger.info(f"✅ Stage 3: Found {result['num_eras'].iloc[0]} eras (Level B)")
            logger.info(f"   Avg rows per era: {result['avg_rows_per_era'].iloc[0]:.0f}")
            logger.info(f"   Row range: {result['min_rows'].iloc[0]} - {result['max_rows'].iloc[0]}")
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
            result = pd.read_sql(text(query), self.engine)
            if result['total_rows'].iloc[0] == 0:
                logger.error("❌ Stage 4: No features extracted")
                return False
            
            logger.info(f"✅ Stage 4: Found {result['total_rows'].iloc[0]} feature rows")
            logger.info(f"   Unique eras: {result['unique_eras'].iloc[0]}")
            
            # Check number of features
            feature_count_query = """
            SELECT COUNT(*) as num_features
            FROM information_schema.columns
            WHERE table_name = 'tsfresh_features'
            AND column_name NOT IN ('era_id', 'signal_name', 'level', 'stage', 'start_time', 'end_time', 'era_rows')
            """
            feature_count = pd.read_sql(text(feature_count_query), self.engine)
            logger.info(f"   Number of features: {feature_count['num_features'].iloc[0]}")
            
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
            'water_usage', 'crop_quality',
            'production_time', 'climate_stability'
        )
        """
        try:
            result = pd.read_sql(text(query), self.engine)
            
            if len(result) == 0:
                logger.warning("⚠️  Stage 5: No target columns found - will need to create synthetic targets")
                return 'synthetic'
            else:
                logger.info(f"✅ Stage 5: Found {len(result)} target columns: {', '.join(result['column_name'].tolist())}")
                return 'real'
        except Exception as e:
            logger.error(f"❌ Stage 5: {e}")
            return False
    
    def check_table_sizes(self):
        """Check the size of each table."""
        query = """
        SELECT 
            table_name,
            pg_size_pretty(pg_total_relation_size(table_name::regclass)) as size
        FROM information_schema.tables
        WHERE table_schema = 'public'
        AND table_name IN (
            'sensor_data', 
            'preprocessed_greenhouse_data', 
            'era_labels_level_a',
            'era_labels_level_b',
            'era_labels_level_c',
            'tsfresh_features'
        )
        ORDER BY pg_total_relation_size(table_name::regclass) DESC
        """
        try:
            result = pd.read_sql(text(query), self.engine)
            logger.info("\n📊 Table Sizes:")
            for _, row in result.iterrows():
                logger.info(f"   {row['table_name']}: {row['size']}")
        except Exception as e:
            logger.warning(f"Could not check table sizes: {e}")
    
    def generate_report(self):
        """Generate validation report."""
        print("\n" + "="*60)
        print("PIPELINE DATA VALIDATION REPORT")
        print("="*60)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Database: {self.engine.url.database}")
        print("="*60)
        
        stages = [
            ("Stage 1: Data Ingestion", self.validate_stage_1_ingestion),
            ("Stage 2: Preprocessing", self.validate_stage_2_preprocessing),
            ("Stage 3: Era Detection", self.validate_stage_3_era_detection),
            ("Stage 4: Feature Extraction", self.validate_stage_4_features),
            ("Stage 5: Target Variables", self.validate_stage_5_targets)
        ]
        
        all_valid = True
        target_type = None
        
        for stage_name, validator_func in stages:
            print(f"\n{stage_name}:")
            result = validator_func()
            if result is False:
                all_valid = False
                print(f"\n⚠️  Pipeline stopped at {stage_name}")
                print("   Fix this stage before proceeding to the next")
                break
            elif stage_name == "Stage 5: Target Variables":
                target_type = result
        
        # Check table sizes
        self.check_table_sizes()
        
        print("\n" + "="*60)
        print("SUMMARY:")
        print("="*60)
        
        if all_valid:
            print("✅ ALL STAGES VALIDATED - Ready for next steps!")
            print("\nNext steps:")
            if target_type == 'synthetic':
                print("1. Run: docker compose run --rm model_builder python -m src.utils.create_synthetic_targets")
                print("2. Then run model training")
            else:
                print("1. Ready for model training")
                print("2. Run: docker compose up model_builder")
        else:
            print("❌ VALIDATION FAILED - Fix issues before proceeding")
            print("\nDebug commands:")
            print("- Check logs: docker compose logs <service_name>")
            print("- Check data: docker compose run --rm db psql -U postgres -d postgres")
        
        print("="*60)
        
        return all_valid

def main():
    # Build database URL
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        db_url = (
            f"postgresql://{os.getenv('DB_USER', 'postgres')}:"
            f"{os.getenv('DB_PASSWORD', 'postgres')}@"
            f"{os.getenv('DB_HOST', 'localhost')}:"
            f"{os.getenv('DB_PORT', '5432')}/"
            f"{os.getenv('DB_NAME', 'postgres')}"
        )
    
    print(f"Connecting to database: {db_url.split('@')[1]}")  # Hide password
    
    try:
        validator = PipelineDataValidator(db_url)
        validator.generate_report()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("1. Database is running: docker compose up -d db")
        print("2. Environment variables are set correctly")
        print("3. You're running from the DataIngestion directory")
        sys.exit(1)

if __name__ == "__main__":
    main()