"""Create test features table with synthetic data for model builder testing."""

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config.moea_objectives import OBJECTIVES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_test_features():
    """Create a test features table with synthetic tsfresh features and targets."""
    # Database connection
    db_url = (
        f"postgresql://{os.getenv('DB_USER', 'postgres')}:"
        f"{os.getenv('DB_PASSWORD', 'postgres')}@"
        f"{os.getenv('DB_HOST', 'localhost')}:"
        f"{os.getenv('DB_PORT', '5432')}/"
        f"{os.getenv('DB_NAME', 'postgres')}"
    )
    
    engine = create_engine(db_url)
    
    # Create synthetic data
    n_eras = 100  # More eras for better training
    n_features = 50  # Typical number of tsfresh features
    
    logger.info(f"Creating synthetic test data with {n_eras} eras and {n_features} features...")
    
    # Create base dataframe with metadata
    base_data = {
        'era_id': [f'test_era_{i}' for i in range(n_eras)],
        'signal_name': np.random.choice(['dli_sum', 'co2_measured_ppm', 'air_temp_c', 'radiation_w_m2'], n_eras),
        'level': 'B',
        'stage': np.random.choice(['PELT', 'BOCPD', 'HMM'], n_eras),
        'start_time': [datetime(2014, 1, 1) + timedelta(days=i*7) for i in range(n_eras)],
        'end_time': [datetime(2014, 1, 1) + timedelta(days=i*7+6) for i in range(n_eras)],
        'era_rows': np.random.randint(100, 1000, n_eras)
    }
    
    df = pd.DataFrame(base_data)
    
    # Add synthetic tsfresh features
    np.random.seed(42)
    for i in range(n_features):
        feature_name = f'air_temp_c__mean_{i}' if i < 10 else f'co2_measured_ppm__variance_{i-10}' if i < 20 else f'dli_sum__skewness_{i-20}'
        # Create correlated features based on signal type
        base_values = np.random.randn(n_eras)
        if 'temp' in feature_name:
            base_values = 20 + 5 * base_values  # Temperature range
        elif 'co2' in feature_name:
            base_values = 600 + 100 * base_values  # CO2 range
        elif 'dli' in feature_name:
            base_values = 15 + 5 * base_values  # DLI range
        
        df[feature_name] = base_values
    
    # Add synthetic targets based on combinations of features
    feature_cols = [col for col in df.columns if '__' in col]  # tsfresh features have '__' pattern
    
    for objective_name, objective in OBJECTIVES.items():
        logger.info(f"Creating synthetic target: {objective_name}")
        
        # Select random features to create target
        n_features_for_target = min(10, len(feature_cols))
        selected_features = np.random.choice(feature_cols, n_features_for_target, replace=False)
        weights = np.random.randn(n_features_for_target)
        weights = weights / np.sum(np.abs(weights))
        
        # Calculate target
        target_values = np.zeros(n_eras)
        for i, feat in enumerate(selected_features):
            target_values += weights[i] * df[feat].values
        
        # Add noise and scale appropriately
        noise = np.random.normal(0, 0.1 * np.std(target_values), n_eras)
        target_values += noise
        
        # Apply objective-specific scaling
        if objective_name == "energy_consumption":
            target_values = 150 + 50 * np.abs(target_values)  # kWh
        elif objective_name == "plant_growth":
            target_values = 8 + 3 * np.abs(target_values)  # g/day
        elif objective_name == "water_usage":
            target_values = 25 + 15 * np.abs(target_values)  # L
        elif objective_name == "crop_quality":
            target_values = 0.6 + 0.3 * (1 / (1 + np.exp(-target_values)))  # 0-1 index
        elif objective_name == "production_time":
            target_values = 70 + 30 * np.abs(target_values)  # days
        elif objective_name == "climate_stability":
            target_values = 0.2 + 0.8 * np.abs(target_values)  # variance
        
        df[objective_name] = target_values
    
    # Create the table (replace if exists)
    logger.info("Creating tsfresh_features_test table...")
    df.to_sql('tsfresh_features_test', engine, if_exists='replace', index=False)
    
    # Also update the main table for immediate testing
    logger.info("Updating main tsfresh_features table...")
    df.to_sql('tsfresh_features', engine, if_exists='replace', index=False)
    
    # Verify
    for table_name in ['tsfresh_features', 'tsfresh_features_test']:
        verify_query = f"SELECT COUNT(*) as row_count, COUNT(*) as col_count FROM {table_name}"
        result = pd.read_sql(text(verify_query), engine)
        logger.info(f"{table_name}: {result.iloc[0]['row_count']} rows")
        
        # Check columns
        col_query = f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}'"
        columns_df = pd.read_sql(text(col_query), engine)
        logger.info(f"  - Total columns: {len(columns_df)}")
        logger.info(f"  - Target columns: {[col for col in columns_df['column_name'] if col in OBJECTIVES]}")
    
    logger.info("Test features created successfully!")


if __name__ == "__main__":
    create_test_features()