#!/usr/bin/env python3
"""
Enhanced Pipeline Runner
Temporary workaround to run the enhanced pipeline using Python directly
"""
import os
import sys
import argparse
import logging
from datetime import datetime
import psycopg2
import pandas as pd
import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Enhanced Sparse Pipeline')
    parser.add_argument('--database-url', required=True, help='PostgreSQL connection string')
    parser.add_argument('--start-date', default='2013-12-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2016-09-08', help='End date (YYYY-MM-DD)')
    parser.add_argument('--features-table', default='enhanced_sparse_features', help='Output table name')
    args = parser.parse_args()
    
    logger.info("Starting Enhanced Sparse Pipeline (Python workaround)")
    logger.info(f"Period: {args.start_date} to {args.end_date}")
    logger.info(f"Output table: {args.features_table}")
    
    # Connect to database
    conn = psycopg2.connect(args.database_url)
    cur = conn.cursor()
    
    # Create output table if not exists
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {args.features_table} (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP WITH TIME ZONE,
        feature_name VARCHAR(255),
        feature_value DOUBLE PRECISION,
        era_id INTEGER,
        resolution VARCHAR(50),
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );
    """
    cur.execute(create_table_sql)
    conn.commit()
    
    # Load data
    logger.info("Loading sensor data...")
    query = """
    SELECT 
        time,
        air_temp_c,
        co2_measured_ppm,
        relative_humidity_percent,
        radiation_w_m2,
        light_intensity_umol,
        heating_setpoint_c,
        vpd_hpa
    FROM sensor_data
    WHERE time BETWEEN %s AND %s
    ORDER BY time
    -- Process all data, no limit
    """
    
    df = pd.read_sql_query(query, conn, params=(args.start_date, args.end_date))
    logger.info(f"Loaded {len(df)} rows")
    
    # Simple preprocessing
    logger.info("Preprocessing data...")
    df = df.fillna(method='ffill', limit=3)
    
    # Simple changepoint detection (every month)
    logger.info("Detecting changepoints...")
    df['month'] = pd.to_datetime(df['time']).dt.to_period('M')
    changepoints = df.groupby('month').first()['time'].tolist()
    logger.info(f"Found {len(changepoints)} monthly segments")
    
    # Enhanced feature extraction (simulating Rust + Python GPU processing)
    logger.info("Extracting enhanced features...")
    features = []
    
    # Process in windows for sparse data handling
    window_size = 24  # hours
    stride = 6  # hours
    
    for i in range(0, len(df) - window_size, stride):
        window = df.iloc[i:i+window_size]
        window_time = window['time'].iloc[0]
        
        # Basic statistics (would be done in Rust)
        for col in ['air_temp_c', 'co2_measured_ppm', 'relative_humidity_percent', 
                   'radiation_w_m2', 'light_intensity_umol', 'heating_setpoint_c', 'vpd_hpa']:
            if col in window.columns:
                col_data = window[col].dropna()
                if len(col_data) > 0:
                    # Sparse-aware features
                    coverage = len(col_data) / len(window)
                    features.extend([
                        (window_time, f'{col}_mean', col_data.mean(), 1, 'window_24h'),
                        (window_time, f'{col}_std', col_data.std(), 1, 'window_24h'),
                        (window_time, f'{col}_min', col_data.min(), 1, 'window_24h'),
                        (window_time, f'{col}_max', col_data.max(), 1, 'window_24h'),
                        (window_time, f'{col}_coverage', coverage, 1, 'window_24h'),
                        (window_time, f'{col}_skew', col_data.skew(), 1, 'window_24h'),
                        (window_time, f'{col}_kurtosis', col_data.kurtosis(), 1, 'window_24h'),
                    ])
                    
                    # Percentiles
                    for p in [25, 50, 75]:
                        features.append(
                            (window_time, f'{col}_p{p}', col_data.quantile(p/100), 1, 'window_24h')
                        )
        
        # Cross-sensor features (would be GPU accelerated in Python)
        temp_data = window['air_temp_c'].dropna()
        humidity_data = window['relative_humidity_percent'].dropna()
        
        if len(temp_data) > 5 and len(humidity_data) > 5:
            # Correlation features
            if len(temp_data) == len(humidity_data):
                corr = np.corrcoef(temp_data, humidity_data)[0, 1]
                features.append((window_time, 'temp_humidity_correlation', corr, 1, 'window_24h'))
            
            # VPD calculation
            vpd_calc = 0.611 * np.exp(17.27 * temp_data / (temp_data + 237.3)) * (1 - humidity_data/100)
            features.extend([
                (window_time, 'vpd_calculated_mean', vpd_calc.mean(), 1, 'window_24h'),
                (window_time, 'vpd_calculated_std', vpd_calc.std(), 1, 'window_24h'),
            ])
        
        # Time-based features
        hour = pd.to_datetime(window_time).hour
        features.extend([
            (window_time, 'hour_sin', np.sin(2 * np.pi * hour / 24), 1, 'window_24h'),
            (window_time, 'hour_cos', np.cos(2 * np.pi * hour / 24), 1, 'window_24h'),
            (window_time, 'is_day', 1 if 6 <= hour <= 18 else 0, 1, 'window_24h'),
        ])
    
    # Insert features
    logger.info(f"Inserting {len(features)} features...")
    insert_sql = f"""
    INSERT INTO {args.features_table} (timestamp, feature_name, feature_value, era_id, resolution)
    VALUES (%s, %s, %s, %s, %s)
    """
    cur.executemany(insert_sql, features)
    conn.commit()
    
    cur.close()
    conn.close()
    
    logger.info("Enhanced pipeline completed successfully!")
    return 0

if __name__ == '__main__':
    sys.exit(main())