"""Create synthetic target columns for all multi-level feature tables.

This script adds synthetic target columns to all three feature table levels
(tsfresh_features_level_a, tsfresh_features_level_b, tsfresh_features_level_c)
for testing the multi-level model builder.
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


def add_targets_to_table(engine, table_name: str, seed_offset: int = 0):
    """Add synthetic target columns to a specific feature table.
    
    Args:
        engine: SQLAlchemy engine
        table_name: Name of the feature table
        seed_offset: Offset for random seed to ensure different targets per level
    """
    # Check if table exists
    check_query = f"""
    SELECT EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_name = '{table_name}'
    )
    """
    exists = pd.read_sql(text(check_query), engine).iloc[0, 0]
    
    if not exists:
        logger.warning(f"Table {table_name} does not exist, skipping...")
        return
    
    # Load existing features
    query = f"SELECT * FROM {table_name}"
    logger.info(f"Loading features from {table_name}...")
    features_df = pd.read_sql(text(query), engine)
    logger.info(f"Loaded {len(features_df)} rows with {len(features_df.columns)} columns")
    
    # Get numeric columns (exclude metadata columns)
    metadata_cols = ['era_id', 'index', 'signal_name', 'level', 'stage', 
                     'start_time', 'end_time', 'era_rows']
    numeric_cols = [col for col in features_df.columns 
                   if col not in metadata_cols and pd.api.types.is_numeric_dtype(features_df[col])]
    
    if not numeric_cols:
        logger.error(f"No numeric feature columns found in {table_name}!")
        return
    
    logger.info(f"Found {len(numeric_cols)} numeric feature columns")
    
    # Create synthetic targets based on weighted combinations of features
    np.random.seed(42 + seed_offset)  # Different seed per level for variety
    
    for objective_name, objective in OBJECTIVES.items():
        logger.info(f"Creating synthetic target: {objective_name}")
        
        # Create synthetic target as weighted sum of random features + noise
        n_features_to_use = min(10, len(numeric_cols))
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
        
        # Scale based on objective type and level
        # Different levels should have slightly different scales to test aggregation
        level_scale = 1.0
        if 'level_a' in table_name:
            level_scale = 1.2  # Level A has broader, more aggregated values
        elif 'level_b' in table_name:
            level_scale = 1.1  # Level B is intermediate
        # Level C uses base scale (1.0)
        
        if objective.type.value == "minimize":
            # For minimize objectives, ensure positive values
            target_values = np.abs(target_values) + 1
        else:  # maximize
            # For maximize objectives, scale to reasonable range
            target_values = 10 + 5 * target_values
        
        # Add specific scaling for each objective
        if objective_name == "energy_consumption":
            target_values = (100 + 50 * np.abs(target_values)) * level_scale  # kWh range
        elif objective_name == "plant_growth":
            target_values = (5 + 2 * np.abs(target_values)) * level_scale  # g/day range
        elif objective_name == "water_usage":
            target_values = (20 + 10 * np.abs(target_values)) * level_scale  # L range
        elif objective_name == "crop_quality":
            target_values = 0.5 + 0.3 * (1 / (1 + np.exp(-target_values)))  # 0-1 quality index
        elif objective_name == "production_time":
            target_values = (60 + 20 * np.abs(target_values)) * level_scale  # days range
        elif objective_name == "climate_stability":
            target_values = (0.1 + 0.5 * np.abs(target_values)) * level_scale  # variance range
        
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
    target_cols = [col for col in columns_df['column_name'] if col in OBJECTIVES]
    logger.info(f"Verified target columns in {table_name}: {target_cols}")


def create_multi_level_targets():
    """Add synthetic targets to all three feature table levels."""
    # Database connection
    db_url = (
        f"postgresql://{os.getenv('DB_USER', 'postgres')}:"
        f"{os.getenv('DB_PASSWORD', 'postgres')}@"
        f"{os.getenv('DB_HOST', 'localhost')}:"
        f"{os.getenv('DB_PORT', '5432')}/"
        f"{os.getenv('DB_NAME', 'postgres')}"
    )
    
    engine = create_engine(db_url)
    
    # Process all three levels
    levels = [
        ('tsfresh_features_level_a', 0),
        ('tsfresh_features_level_b', 10),
        ('tsfresh_features_level_c', 20),
    ]
    
    logger.info("=" * 60)
    logger.info("Creating synthetic targets for multi-level features")
    logger.info("=" * 60)
    
    for table_name, seed_offset in levels:
        logger.info(f"\nProcessing {table_name}...")
        add_targets_to_table(engine, table_name, seed_offset)
    
    # Also create/update the single-level table if it exists
    logger.info("\nProcessing single-level table (tsfresh_features)...")
    add_targets_to_table(engine, 'tsfresh_features', 30)
    
    logger.info("\n" + "=" * 60)
    logger.info("Multi-level target creation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    create_multi_level_targets()