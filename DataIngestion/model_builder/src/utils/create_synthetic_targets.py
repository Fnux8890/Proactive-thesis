"""Create synthetic target columns for testing the model builder.

This script adds synthetic target columns to the features table for testing purposes.
In a real scenario, these would be calculated from actual sensor data.
"""

import logging
import os
import sys
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


def create_synthetic_targets(table_name="tsfresh_features"):
    """Add synthetic target columns to the features table.
    
    Args:
        table_name: Name of the table to add targets to
    """
    # Database connection
    db_url = (
        f"postgresql://{os.getenv('DB_USER', 'postgres')}:"
        f"{os.getenv('DB_PASSWORD', 'postgres')}@"
        f"{os.getenv('DB_HOST', 'localhost')}:"
        f"{os.getenv('DB_PORT', '5432')}/"
        f"{os.getenv('DB_NAME', 'postgres')}"
    )
    
    engine = create_engine(db_url)
    
    # Load existing features
    query = f"SELECT * FROM {table_name}"
    logger.info(f"Loading existing features from {table_name}...")
    features_df = pd.read_sql(text(query), engine)
    logger.info(f"Loaded {len(features_df)} rows with {len(features_df.columns)} columns")
    
    # Get numeric columns (exclude metadata columns)
    numeric_cols = [col for col in features_df.columns 
                   if col not in ['era_id', 'signal_name', 'level', 'stage', 'start_time', 'end_time', 'era_rows']]
    
    if not numeric_cols:
        logger.error("No numeric feature columns found!")
        return
    
    logger.info(f"Found {len(numeric_cols)} numeric feature columns")
    
    # Create synthetic targets based on weighted combinations of features
    np.random.seed(42)  # For reproducibility
    
    for objective_name, objective in OBJECTIVES.items():
        logger.info(f"Creating synthetic target: {objective_name}")
        
        # Create synthetic target as weighted sum of random features + noise
        n_features_to_use = min(5, len(numeric_cols))
        selected_features = np.random.choice(numeric_cols, n_features_to_use, replace=False)
        weights = np.random.randn(n_features_to_use)
        
        # Normalize weights
        weights = weights / np.sum(np.abs(weights))
        
        # Calculate synthetic target
        target_values = np.zeros(len(features_df))
        for i, feat in enumerate(selected_features):
            # Handle any NaN values
            feat_values = features_df[feat].fillna(0).values
            target_values += weights[i] * feat_values
        
        # Add some noise
        noise = np.random.normal(0, 0.1 * np.std(target_values), len(target_values))
        target_values += noise
        
        # Scale based on objective type
        if objective.type.value == "minimize":
            # For minimize objectives, ensure positive values
            target_values = np.abs(target_values) + 1
        else:  # maximize
            # For maximize objectives, scale to reasonable range
            target_values = 10 + 5 * target_values
        
        # Add specific scaling for each objective
        if objective_name == "energy_consumption":
            target_values = 100 + 50 * np.abs(target_values)  # kWh range
        elif objective_name == "plant_growth":
            target_values = 5 + 2 * np.abs(target_values)  # g/day range
        elif objective_name == "water_usage":
            target_values = 20 + 10 * np.abs(target_values)  # L range
        elif objective_name == "crop_quality":
            target_values = 0.5 + 0.3 * (1 / (1 + np.exp(-target_values)))  # 0-1 quality index
        elif objective_name == "production_time":
            target_values = 60 + 20 * np.abs(target_values)  # days range
        elif objective_name == "climate_stability":
            target_values = 0.1 + 0.5 * np.abs(target_values)  # variance range
        
        # Add to dataframe
        features_df[objective_name] = target_values
        logger.info(f"  - Added {objective_name}: mean={np.mean(target_values):.2f}, std={np.std(target_values):.2f}")
    
    # Save updated features back to database
    logger.info(f"Saving updated features with synthetic targets to {table_name}...")
    features_df.to_sql(table_name, engine, if_exists='replace', index=False)
    logger.info(f"Successfully added synthetic targets to {table_name}")
    
    # Verify
    verify_query = f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}'"
    columns_df = pd.read_sql(text(verify_query), engine)
    logger.info(f"Updated {table_name} now has columns: {list(columns_df['column_name'])}")


def create_synthetic_targets_for_all_levels():
    """Create synthetic targets for all feature level tables."""
    tables = [
        'tsfresh_features_level_a',
        'tsfresh_features_level_b', 
        'tsfresh_features_level_c'
    ]
    
    for table in tables:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {table}")
            logger.info(f"{'='*60}")
            create_synthetic_targets(table)
        except Exception as e:
            logger.error(f"Failed to process {table}: {e}")
            continue


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # If table name provided as argument
        table_name = sys.argv[1]
        create_synthetic_targets(table_name)
    else:
        # Default: create for all level tables
        create_synthetic_targets_for_all_levels()