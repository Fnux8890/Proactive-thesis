#!/usr/bin/env python3
"""Script to check what columns exist in the feature tables."""

import psycopg2
import pandas as pd
from tabulate import tabulate

# Database connection parameters
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "postgres",
    "user": "postgres",
    "password": "postgres"
}

def check_table_columns(cursor, table_name):
    """Check columns in a table."""
    print(f"\n{'='*60}")
    print(f"Checking table: {table_name}")
    print('='*60)
    
    # Check if table exists
    cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = %s
        );
    """, (table_name,))
    
    exists = cursor.fetchone()[0]
    if not exists:
        print(f"Table {table_name} does NOT exist!")
        return
    
    # Get column information
    cursor.execute("""
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        ORDER BY ordinal_position;
    """, (table_name,))
    
    columns = cursor.fetchall()
    if columns:
        df = pd.DataFrame(columns, columns=['Column Name', 'Data Type', 'Nullable'])
        print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
        
        # Check for specific target columns
        column_names = [col[0] for col in columns]
        target_columns = ['energy_consumption', 'plant_growth', 'light_efficiency', 
                         'temperature_stability', 'humidity_control']
        
        print(f"\nTarget columns check:")
        for target in target_columns:
            if target in column_names:
                print(f"  ✓ {target} - FOUND")
            else:
                print(f"  ✗ {target} - MISSING")
                
        # Check row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        count = cursor.fetchone()[0]
        print(f"\nRow count: {count}")
    else:
        print("No columns found!")

def check_all_tables():
    """Check all relevant tables."""
    try:
        # Connect to database
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Tables to check
        tables = [
            'tsfresh_features_level_a',
            'tsfresh_features_level_b', 
            'tsfresh_features_level_c',
            'feature_targets',  # Check if this exists
            'synthetic_targets',  # Check if this exists
            'feature_data'  # Check if this exists
        ]
        
        for table in tables:
            check_table_columns(cursor, table)
            
        # Also check what tables exist with 'feature' or 'target' in name
        print(f"\n{'='*60}")
        print("All tables containing 'feature' or 'target':")
        print('='*60)
        
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND (table_name LIKE '%feature%' OR table_name LIKE '%target%')
            ORDER BY table_name;
        """)
        
        tables_found = cursor.fetchall()
        if tables_found:
            for table in tables_found:
                print(f"  - {table[0]}")
        else:
            print("  No tables found with 'feature' or 'target' in name")
            
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_all_tables()