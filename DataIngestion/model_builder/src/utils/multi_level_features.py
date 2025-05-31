"""
Multi-level feature integration for model training.
Combines features from Level A (PELT), Level B (BOCPD), and Level C (HMM).
"""

import pandas as pd
from sqlalchemy import create_engine
import os
from typing import List, Dict, Tuple


def load_multi_level_features(
    engine, 
    feature_tables: List[str] = None,
    target_column: str = 'energy_consumption'
) -> pd.DataFrame:
    """
    Load and combine features from multiple era detection levels.
    
    Args:
        engine: SQLAlchemy engine
        feature_tables: List of feature table names
        target_column: Target variable to predict
        
    Returns:
        Combined feature DataFrame with hierarchical relationships
    """
    if feature_tables is None:
        feature_tables = [
            'tsfresh_features_level_a',
            'tsfresh_features_level_b', 
            'tsfresh_features_level_c'
        ]
    
    # Load features from each level
    level_features = {}
    for table in feature_tables:
        query = f"""
        SELECT * FROM {table}
        WHERE {target_column} IS NOT NULL
        """
        df = pd.read_sql(query, engine)
        level = table.split('_')[-1]  # Extract 'a', 'b', or 'c'
        level_features[level] = df
        print(f"Loaded {len(df)} rows from {table}")
    
    # Combine features based on time overlaps
    combined_df = combine_hierarchical_features(level_features)
    
    return combined_df


def combine_hierarchical_features(level_features: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine features from different era levels using hierarchical relationships.
    
    Strategy:
    1. Level C (smallest) as base - most training samples
    2. Add aggregated Level B features for each Level C era
    3. Add aggregated Level A features for broader context
    """
    # Start with Level C (most granular)
    base_df = level_features['c'].copy()
    base_df['base_level'] = 'C'
    
    # For each Level C era, find overlapping Level B era
    if 'b' in level_features:
        level_b_features = []
        for idx, row in base_df.iterrows():
            # Find Level B era that contains this Level C era
            overlapping_b = level_features['b'][
                (level_features['b']['start_time'] <= row['start_time']) &
                (level_features['b']['end_time'] >= row['end_time'])
            ]
            
            if len(overlapping_b) > 0:
                # Add Level B features with prefix
                b_features = overlapping_b.iloc[0].to_dict()
                b_features = {f'level_b_{k}': v for k, v in b_features.items() 
                             if k not in ['era_id', 'start_time', 'end_time']}
                level_b_features.append(b_features)
            else:
                level_b_features.append({})
        
        # Add Level B features to base
        b_df = pd.DataFrame(level_b_features)
        base_df = pd.concat([base_df, b_df], axis=1)
    
    # Similarly for Level A (most macro)
    if 'a' in level_features:
        level_a_features = []
        for idx, row in base_df.iterrows():
            overlapping_a = level_features['a'][
                (level_features['a']['start_time'] <= row['start_time']) &
                (level_features['a']['end_time'] >= row['end_time'])
            ]
            
            if len(overlapping_a) > 0:
                a_features = overlapping_a.iloc[0].to_dict()
                a_features = {f'level_a_{k}': v for k, v in a_features.items()
                             if k not in ['era_id', 'start_time', 'end_time']}
                level_a_features.append(a_features)
            else:
                level_a_features.append({})
        
        a_df = pd.DataFrame(level_a_features)
        base_df = pd.concat([base_df, a_df], axis=1)
    
    # Create cross-level features
    base_df = create_cross_level_features(base_df)
    
    print(f"Combined features shape: {base_df.shape}")
    print(f"Features include: {len([c for c in base_df.columns if 'level_a_' in c])} from Level A")
    print(f"                  {len([c for c in base_df.columns if 'level_b_' in c])} from Level B")
    print(f"                  {len([c for c in base_df.columns if c.startswith('cross_')])} cross-level")
    
    return base_df


def create_cross_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features that capture relationships between era levels.
    """
    # Ratio features - compare same metric across levels
    feature_cols = [c for c in df.columns if '__' in c and 'level_' not in c]
    
    for col in feature_cols:
        # Check if same feature exists at other levels
        if f'level_b_{col}' in df.columns:
            df[f'cross_ratio_b_{col}'] = df[col] / (df[f'level_b_{col}'] + 1e-8)
            
        if f'level_a_{col}' in df.columns:
            df[f'cross_ratio_a_{col}'] = df[col] / (df[f'level_a_{col}'] + 1e-8)
            
        # Difference features
        if f'level_b_{col}' in df.columns and f'level_a_{col}' in df.columns:
            df[f'cross_hier_diff_{col}'] = df[f'level_b_{col}'] - df[f'level_a_{col}']
    
    # Era duration ratios (if available)
    if 'era_duration_hours' in df.columns and 'level_b_era_duration_hours' in df.columns:
        df['cross_duration_ratio_c_to_b'] = df['era_duration_hours'] / (df['level_b_era_duration_hours'] + 1e-8)
    
    return df


# Usage in model training:
if __name__ == "__main__":
    # Example of how model builder would use this
    db_url = os.getenv('DATABASE_URL')
    engine = create_engine(db_url)
    
    # Load multi-level features
    features_df = load_multi_level_features(
        engine,
        feature_tables=['tsfresh_features_level_a', 'tsfresh_features_level_b', 'tsfresh_features_level_c'],
        target_column='energy_consumption'
    )
    
    # Now train model with hierarchical features
    print(f"Training with {len(features_df)} samples and {len(features_df.columns)} features")
    print("Feature categories:")
    print(f"- Base Level C: {len([c for c in features_df.columns if 'level_' not in c and 'cross_' not in c])}")
    print(f"- From Level B: {len([c for c in features_df.columns if 'level_b_' in c])}")
    print(f"- From Level A: {len([c for c in features_df.columns if 'level_a_' in c])}")
    print(f"- Cross-level: {len([c for c in features_df.columns if 'cross_' in c])}")