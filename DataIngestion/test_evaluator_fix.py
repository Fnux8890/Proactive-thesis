#!/usr/bin/env python3
"""
Test script to validate the evaluator fix.
This script tests database connectivity and schema compatibility.
"""

import os
import sys
import pandas as pd
import psycopg2
from pathlib import Path

# Add the evaluation module to path
sys.path.append('/mnt/c/Users/fhj88/Documents/Github/Proactive-thesis/DataIngestion')

def test_database_connection():
    """Test database connection and schema"""
    
    database_url = os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/postgres')
    
    try:
        conn = psycopg2.connect(database_url)
        cursor = conn.cursor()
        
        # Test which tables exist
        cursor.execute("""
        SELECT table_name FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name LIKE '%features%'
        ORDER BY table_name
        """)
        
        tables = cursor.fetchall()
        print("Available feature tables:")
        for table in tables:
            print(f"  - {table[0]}")
        
        # Test enhanced_sparse_features_full if it exists
        cursor.execute("""
        SELECT table_name FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name = 'enhanced_sparse_features_full'
        """)
        
        enhanced_table_exists = bool(cursor.fetchone())
        print(f"\nenhanced_sparse_features_full exists: {enhanced_table_exists}")
        
        # Test sensor_data table
        cursor.execute("""
        SELECT COUNT(*) FROM sensor_data 
        WHERE time BETWEEN '2014-01-01' AND '2014-01-02'
        """)
        sensor_count = cursor.fetchone()[0]
        print(f"Sample sensor data available (Jan 1-2, 2014): {sensor_count} records")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False

def test_evaluator_data_loading():
    """Test the evaluator data loading methods"""
    
    try:
        from evaluation.evaluate_full_experiment import FullExperimentEvaluator
        
        database_url = os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/postgres')
        evaluator = FullExperimentEvaluator('/tmp/experiment_test', database_url)
        
        # Test loading historical data
        print("\nTesting evaluator data loading...")
        historical_data = evaluator.get_historical_data("2014-01-01", "2014-01-02")
        
        print(f"Loaded historical data: {len(historical_data)} records")
        if len(historical_data) > 0:
            print(f"Columns: {list(historical_data.columns)}")
            print(f"Sample data:")
            print(historical_data.head(2))
        
        return True
        
    except Exception as e:
        print(f"Evaluator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    
    print("=== Testing Evaluator Fix ===")
    
    # Test 1: Database connection
    print("\n1. Testing database connection...")
    db_ok = test_database_connection()
    
    if not db_ok:
        print("❌ Database connection failed")
        return 1
    
    # Test 2: Evaluator data loading
    print("\n2. Testing evaluator data loading...")
    eval_ok = test_evaluator_data_loading()
    
    if not eval_ok:
        print("❌ Evaluator data loading failed")
        return 1
    
    print("\n✅ All tests passed! Evaluator fix is working.")
    return 0

if __name__ == "__main__":
    sys.exit(main())