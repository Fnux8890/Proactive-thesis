"""Multi-level feature data loader for model training.

This module extends the PostgreSQLDataLoader to support loading features
from multiple era detection levels (A, B, C) and combining them hierarchically.
"""

import logging

import numpy as np
import pandas as pd
from sqlalchemy import text

from ..training.train_lightgbm_surrogate import DataConfig, PostgreSQLDataLoader

logger = logging.getLogger(__name__)


class MultiLevelDataLoader(PostgreSQLDataLoader):
    """Extended data loader that supports multi-level feature integration."""

    def __init__(self, connection_string: str, config: DataConfig):
        super().__init__(connection_string, config)
        self._multi_level_features: pd.DataFrame | None = None

    def load_multi_level_features(
        self,
        feature_tables: list[str] | None = None,
        combine_method: str = "hierarchical"
    ) -> pd.DataFrame:
        """Load and combine features from multiple era levels.

        Args:
            feature_tables: List of feature table names (defaults to all three levels)
            combine_method: How to combine features ("hierarchical", "concat", "aggregate")

        Returns:
            Combined feature DataFrame
        """
        if self._multi_level_features is not None:
            return self._multi_level_features

        if feature_tables is None:
            feature_tables = [
                'tsfresh_features_level_a',
                'tsfresh_features_level_b',
                'tsfresh_features_level_c'
            ]

        logger.info(f"Loading features from {len(feature_tables)} tables")

        # Load features from each level
        level_features = {}
        for table in feature_tables:
            try:
                # First check which columns exist in the table
                col_check_query = f"""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = '{table}'
                AND column_name IN ('era_id', 'index')
                """
                existing_cols = pd.read_sql(text(col_check_query), self.engine)
                has_era_id = 'era_id' in existing_cols['column_name'].values
                has_index = 'index' in existing_cols['column_name'].values

                if has_era_id:
                    # Table has era_id column, use it directly
                    query = f"""
                    SELECT * FROM {table}
                    ORDER BY era_id
                    """
                elif has_index:
                    # Table has index column but not era_id, rename index to era_id
                    logger.info(f"Table {table} has 'index' column instead of 'era_id', renaming it")
                    query = f"""
                    SELECT
                        "index" AS era_id,
                        *
                    FROM {table}
                    ORDER BY "index"
                    """
                else:
                    raise ValueError(f"Table {table} has neither 'era_id' nor 'index' column")

                df = pd.read_sql(text(query), self.engine)

                # Remove duplicate columns if we renamed index to era_id
                if has_index and not has_era_id and 'index' in df.columns:
                    df = df.drop(columns=['index'])

                level = table.split('_')[-1]  # Extract 'a', 'b', or 'c'
                level_features[level] = df
                logger.info(f"Loaded {len(df)} rows from {table}")
            except Exception as e:
                logger.warning(f"Could not load {table}: {e}")

        if not level_features:
            raise ValueError("No feature tables could be loaded")

        # Combine features based on method
        if combine_method == "hierarchical":
            self._multi_level_features = self._combine_hierarchical(level_features)
        elif combine_method == "concat":
            self._multi_level_features = self._combine_concat(level_features)
        elif combine_method == "aggregate":
            self._multi_level_features = self._combine_aggregate(level_features)
        else:
            raise ValueError(f"Unknown combine method: {combine_method}")

        logger.info(f"Combined features shape: {self._multi_level_features.shape}")
        return self._multi_level_features

    def _combine_hierarchical(self, level_features: dict) -> pd.DataFrame:
        """Combine features using hierarchical relationships.

        Strategy:
        1. Use Level C (smallest) as base for most training samples
        2. Add aggregated Level B features for each Level C era
        3. Add aggregated Level A features for broader context
        """
        # Start with Level C (most granular)
        if 'c' not in level_features:
            # Fallback to Level B if C not available
            if 'b' in level_features:
                base_df = level_features['b'].copy()
                base_level = 'B'
            else:
                base_df = level_features['a'].copy()
                base_level = 'A'
        else:
            base_df = level_features['c'].copy()
            base_level = 'C'

        base_df['base_level'] = base_level
        logger.info(f"Using Level {base_level} as base with {len(base_df)} samples")

        # For each base era, find overlapping parent eras
        if base_level == 'C' and 'b' in level_features:
            base_df = self._add_parent_features(base_df, level_features['b'], 'b')

        if base_level in ['B', 'C'] and 'a' in level_features:
            base_df = self._add_parent_features(base_df, level_features['a'], 'a')

        # Create cross-level features
        base_df = self._create_cross_level_features(base_df)

        return base_df

    def _add_parent_features(
        self,
        base_df: pd.DataFrame,
        parent_df: pd.DataFrame,
        parent_level: str
    ) -> pd.DataFrame:
        """Add features from parent era level."""
        logger.info(f"Adding Level {parent_level.upper()} features to base")

        # Columns to skip when adding parent features
        skip_cols = {'era_id', 'signal_name', 'level', 'stage', 'start_time',
                     'end_time', 'era_rows', 'base_level'}

        # For each row in base, find overlapping parent era
        parent_features = []
        for _idx, row in base_df.iterrows():
            # Find parent era that contains this era
            # Assuming start_time and end_time columns exist
            if 'start_time' in row and 'end_time' in row:
                overlapping = parent_df[
                    (parent_df['start_time'] <= row['start_time']) &
                    (parent_df['end_time'] >= row['end_time'])
                ]

                if len(overlapping) > 0:
                    # Use the first overlapping parent (there should typically be only one)
                    parent_row = overlapping.iloc[0]
                    # Add parent features with prefix
                    parent_feat = {}
                    for col in parent_row.index:
                        if col not in skip_cols:
                            parent_feat[f'level_{parent_level}_{col}'] = parent_row[col]
                    parent_features.append(parent_feat)
                else:
                    # No overlapping parent found
                    parent_features.append({})
            else:
                parent_features.append({})

        # Convert to DataFrame and concatenate
        parent_df_features = pd.DataFrame(parent_features)
        if not parent_df_features.empty:
            base_df = pd.concat([base_df.reset_index(drop=True),
                                parent_df_features.reset_index(drop=True)], axis=1)

        return base_df

    def _create_cross_level_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features that capture relationships between era levels."""
        logger.info("Creating cross-level features")

        # Find numeric columns at base level
        base_numeric_cols = []
        for col in df.columns:
            if ('level_' not in col and
                col not in ['era_id', 'signal_name', 'level', 'stage',
                           'start_time', 'end_time', 'era_rows', 'base_level'] and
                pd.api.types.is_numeric_dtype(df[col])):
                base_numeric_cols.append(col)

        # Create ratio and difference features
        new_features = {}
        for col in base_numeric_cols:
            # Check if same feature exists at other levels
            if f'level_b_{col}' in df.columns:
                new_features[f'cross_ratio_b_{col}'] = df[col] / (df[f'level_b_{col}'] + 1e-8)

            if f'level_a_{col}' in df.columns:
                new_features[f'cross_ratio_a_{col}'] = df[col] / (df[f'level_a_{col}'] + 1e-8)

            # Hierarchical differences
            if f'level_b_{col}' in df.columns and f'level_a_{col}' in df.columns:
                new_features[f'cross_hier_diff_{col}'] = (
                    df[f'level_b_{col}'] - df[f'level_a_{col}']
                )

        # Add duration ratios if available
        if 'era_duration_hours' in df.columns and 'level_b_era_duration_hours' in df.columns:
            new_features['cross_duration_ratio_c_to_b'] = (
                df['era_duration_hours'] / (df['level_b_era_duration_hours'] + 1e-8)
            )

        # Add new features to dataframe using concat for better performance
        if new_features:
            new_features_df = pd.DataFrame(new_features, index=df.index)
            df = pd.concat([df, new_features_df], axis=1)

        logger.info(f"Added {len(new_features)} cross-level features")
        return df

    def _combine_concat(self, level_features: dict) -> pd.DataFrame:
        """Simple concatenation of all levels with level identifier."""
        all_dfs = []
        for level, df in level_features.items():
            df_copy = df.copy()
            df_copy['source_level'] = level.upper()
            all_dfs.append(df_copy)

        return pd.concat(all_dfs, ignore_index=True)

    def _combine_aggregate(self, level_features: dict) -> pd.DataFrame:
        """Aggregate features across levels for same time periods."""
        # This is a simplified implementation
        # In practice, you'd want more sophisticated time-based aggregation
        return self._combine_concat(level_features)

    def prepare_training_data(
        self,
        target: str,
        use_phenotypes: bool = False
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for training with multi-level support."""

        # Always use multi-level features for this loader
        use_multi_level = True

        if use_multi_level:
            # Check if multi-level tables are specified
            feature_tables = None
            if hasattr(self.config, 'feature_tables'):
                feature_tables = self.config.feature_tables

            features_df = self.load_multi_level_features(feature_tables)
        else:
            # Fall back to single-level loading
            features_df = self.load_features()

        # Check if target column exists
        target_col = self.config.target_columns.get(target)
        if not target_col or target_col not in features_df.columns:
            # Try to infer target from feature names
            possible_targets = [col for col in features_df.columns if target in col.lower()]
            if possible_targets:
                target_col = possible_targets[0]
                logger.warning(f"Target '{target}' not found, using inferred column: {target_col}")
            else:
                raise ValueError(f"Target column for '{target}' not found in features")

        # Separate features and target
        exclude_cols = {
            'era_id', 'signal_name', 'level', 'stage', 'start_time',
            'end_time', 'era_rows', 'base_level', 'source_level', target_col
        }

        feature_cols = [col for col in features_df.columns if col not in exclude_cols]

        X = features_df[feature_cols].copy()
        y = features_df[target_col].copy()

        # Add phenotype features if requested
        if use_phenotypes and self.config.use_phenotypes:
            phenotypes_df = self.load_phenotypes()
            logger.info("Adding phenotype features to training data")
            # Add aggregated phenotype features
            for col in phenotypes_df.select_dtypes(include=[np.number]).columns:
                X[f"phenotype_{col}_mean"] = phenotypes_df[col].mean()
                X[f"phenotype_{col}_std"] = phenotypes_df[col].std()

        # Remove any remaining NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]

        logger.info(f"Prepared {len(X)} samples with {len(X.columns)} features")

        # Log feature breakdown if multi-level
        if use_multi_level:
            base_features = len([c for c in X.columns if 'level_' not in c and 'cross_' not in c])
            level_b_features = len([c for c in X.columns if 'level_b_' in c])
            level_a_features = len([c for c in X.columns if 'level_a_' in c])
            cross_features = len([c for c in X.columns if 'cross_' in c])

            logger.info("Feature breakdown:")
            logger.info(f"  - Base features: {base_features}")
            logger.info(f"  - Level B features: {level_b_features}")
            logger.info(f"  - Level A features: {level_a_features}")
            logger.info(f"  - Cross-level features: {cross_features}")

        return X, y
