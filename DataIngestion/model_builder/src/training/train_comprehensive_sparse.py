#!/usr/bin/env python3
"""
Train LightGBM models using comprehensive enhanced sparse features.

This script loads the full feature set from the enhanced sparse pipeline
and trains multi-objective models for energy efficiency and plant growth.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import json
from datetime import datetime

# Machine learning imports (using LightGBM for efficiency with sparse features)
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("WARNING: LightGBM not available, falling back to basic model")

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveFeatureLoader:
    """Load and process comprehensive features from enhanced sparse pipeline."""
    
    def __init__(self, database_url, features_table='enhanced_sparse_features'):
        self.database_url = database_url
        self.features_table = features_table
        self.engine = create_engine(database_url)
    
    def load_all_features(self):
        """Load all features from JSONB-based enhanced sparse features table."""
        logger.info(f"Loading comprehensive features from {self.features_table}")
        
        # Load the JSONB data from enhanced sparse features table
        main_query = f"""
        SELECT 
            era_id,
            resolution,
            computed_at,
            sensor_features,
            extended_stats,
            weather_features,
            energy_features,
            growth_features,
            temporal_features,
            optimization_metrics
        FROM {self.features_table}
        ORDER BY computed_at, era_id
        """
        
        df = pd.read_sql_query(main_query, self.engine)
        logger.info(f"Loaded {len(df)} feature records")
        
        # Extract features from JSONB columns
        logger.info("Extracting features from JSONB columns...")
        feature_matrix_rows = []
        
        for idx, row in df.iterrows():
            feature_row = {
                'era_id': row['era_id'],
                'computed_at': row['computed_at'],
                'resolution': row['resolution']
            }
            
            # Extract features from each JSONB column
            for col in ['sensor_features', 'extended_stats', 'weather_features', 
                       'energy_features', 'growth_features', 'temporal_features', 
                       'optimization_metrics']:
                if row[col] is not None:
                    if isinstance(row[col], str):
                        import json
                        features = json.loads(row[col])
                    else:
                        features = row[col]
                    
                    # Add prefix to avoid column name conflicts
                    for feature_name, feature_value in features.items():
                        prefixed_name = f"{col}_{feature_name}"
                        feature_row[prefixed_name] = feature_value
            
            feature_matrix_rows.append(feature_row)
        
        feature_matrix = pd.DataFrame(feature_matrix_rows)
        
        # Get all feature column names (excluding metadata columns)
        feature_names = [col for col in feature_matrix.columns 
                        if col not in ['era_id', 'computed_at', 'resolution']]
        
        # Convert feature columns to numeric, coercing errors to NaN
        logger.info("Converting feature columns to numeric types...")
        for col in feature_names:
            feature_matrix[col] = pd.to_numeric(feature_matrix[col], errors='coerce')
        
        logger.info(f"Created feature matrix: {feature_matrix.shape}")
        logger.info(f"Feature columns: {feature_names[:10]}...")  # Show first 10
        logger.info(f"Total feature types: {len(feature_names)}")
        
        return feature_matrix, feature_names
    
    def analyze_feature_quality(self, feature_matrix):
        """Analyze the quality and completeness of features."""
        logger.info("Analyzing feature quality...")
        
        feature_cols = [col for col in feature_matrix.columns if col not in ['era_id', 'computed_at', 'resolution']]
        
        quality_stats = {}
        for col in feature_cols:
            non_null_count = feature_matrix[col].notna().sum()
            total_count = len(feature_matrix)
            completeness = non_null_count / total_count
            
            if non_null_count > 0:
                mean_val = feature_matrix[col].mean()
                std_val = feature_matrix[col].std()
                quality_stats[col] = {
                    'completeness': completeness,
                    'mean': mean_val,
                    'std': std_val,
                    'non_null_count': non_null_count
                }
        
        # Sort by completeness
        sorted_features = sorted(quality_stats.items(), key=lambda x: x[1]['completeness'], reverse=True)
        
        logger.info(f"Top 10 most complete features:")
        for feature, stats in sorted_features[:10]:
            logger.info(f"  {feature}: {stats['completeness']:.3f} complete, mean={stats['mean']:.3f}")
        
        return quality_stats

class MultiObjectiveTargetGenerator:
    """Generate realistic multi-objective targets for greenhouse optimization."""
    
    def __init__(self, feature_matrix):
        self.feature_matrix = feature_matrix
        
    def create_energy_consumption_target(self):
        """Create energy consumption target based on greenhouse operations."""
        logger.info("Creating energy consumption targets...")
        
        # Base energy from heating/cooling
        heating_energy = np.zeros(len(self.feature_matrix))
        cooling_energy = np.zeros(len(self.feature_matrix))
        lighting_energy = np.zeros(len(self.feature_matrix))
        
        # Heating energy (when temp < setpoint)
        if 'air_temp_c_mean' in self.feature_matrix.columns and 'heating_setpoint_c_mean' in self.feature_matrix.columns:
            temp_diff = (self.feature_matrix['heating_setpoint_c_mean'].fillna(20) - 
                        self.feature_matrix['air_temp_c_mean'].fillna(18))
            heating_energy = np.maximum(0, temp_diff * 2.0)  # 2 kW per degree
        
        # Lighting energy
        if 'light_intensity_umol_mean' in self.feature_matrix.columns:
            lighting_energy = self.feature_matrix['light_intensity_umol_mean'].fillna(0) * 0.01
        
        # Ventilation energy (CO2 control)
        ventilation_energy = np.zeros(len(self.feature_matrix))
        if 'co2_measured_ppm_mean' in self.feature_matrix.columns:
            co2_excess = np.maximum(0, self.feature_matrix['co2_measured_ppm_mean'].fillna(400) - 800)
            ventilation_energy = co2_excess * 0.005
        
        # Total energy with some noise
        total_energy = (heating_energy + cooling_energy + lighting_energy + ventilation_energy + 
                       np.random.normal(0, 0.5, len(self.feature_matrix)))
        
        return np.maximum(0, total_energy)  # Ensure non-negative
    
    def create_plant_growth_target(self):
        """Create plant growth target based on environmental conditions."""
        logger.info("Creating plant growth targets...")
        
        # Base growth from optimal conditions
        growth_rate = np.ones(len(self.feature_matrix))
        
        # Temperature effect (optimal around 22°C)
        if 'air_temp_c_mean' in self.feature_matrix.columns:
            temp = self.feature_matrix['air_temp_c_mean'].fillna(20)
            temp_factor = 1.0 - np.abs(temp - 22) * 0.05  # 5% reduction per degree from optimal
            growth_rate *= np.maximum(0.3, temp_factor)
        
        # Light effect
        if 'light_intensity_umol_mean' in self.feature_matrix.columns:
            light = self.feature_matrix['light_intensity_umol_mean'].fillna(100)
            light_factor = np.minimum(1.0, light / 200.0)  # Saturates at 200 μmol
            growth_rate *= light_factor
        
        # Humidity effect (optimal around 60%)
        if 'relative_humidity_percent_mean' in self.feature_matrix.columns:
            humidity = self.feature_matrix['relative_humidity_percent_mean'].fillna(60)
            humidity_factor = 1.0 - np.abs(humidity - 60) * 0.01  # 1% reduction per % from optimal
            growth_rate *= np.maximum(0.5, humidity_factor)
        
        # VPD effect (optimal around 0.8-1.2 kPa)
        if 'vpd_hpa_mean' in self.feature_matrix.columns:
            vpd = self.feature_matrix['vpd_hpa_mean'].fillna(0.8)
            vpd_optimal = np.where((vpd >= 0.8) & (vpd <= 1.2), 1.0, 
                                 np.maximum(0.6, 1.0 - np.abs(vpd - 1.0) * 0.2))
            growth_rate *= vpd_optimal
        
        # Add seasonal variation and noise
        growth_with_noise = growth_rate * (1 + np.random.normal(0, 0.1, len(self.feature_matrix)))
        
        return np.maximum(0, growth_with_noise)

class EnhancedModelTrainer:
    """Train LightGBM models for multi-objective greenhouse optimization."""
    
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
    
    def prepare_features(self, feature_matrix, min_completeness=0.1):
        """Prepare feature matrix for training."""
        logger.info("Preparing features for training...")
        
        # Remove metadata columns
        feature_cols = [col for col in feature_matrix.columns if col not in ['era_id', 'computed_at', 'resolution']]
        X = feature_matrix[feature_cols].copy()
        
        # Remove features with too much missing data
        completeness = X.notna().sum() / len(X)
        good_features = completeness[completeness >= min_completeness].index.tolist()
        X = X[good_features]
        
        logger.info(f"Using {len(good_features)} features with ≥{min_completeness*100:.0f}% completeness")
        
        # Fill remaining missing values with median
        X = X.fillna(X.median())
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        logger.info(f"Final feature matrix shape: {X.shape}")
        return X, good_features
    
    def train_lightgbm_model(self, X, y, model_name, objective='regression'):
        """Train a LightGBM model."""
        logger.info(f"Training LightGBM model for {model_name}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # LightGBM parameters optimized for sparse data
        params = {
            'objective': objective,
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        if self.use_gpu and LIGHTGBM_AVAILABLE:
            params['device_type'] = 'gpu'
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=100,
            callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(period=20)]
        )
        
        # Evaluate
        y_pred = model.predict(X_test, num_iteration=model.best_iteration)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"{model_name} - RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        # Store model and metrics
        self.models[model_name] = model
        self.feature_importance[model_name] = dict(zip(X.columns, model.feature_importance()))
        
        # Save model
        model_path = f"/models/{model_name}_lightgbm.txt"
        model.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
        
        return model, rmse, r2
    
    def train_fallback_model(self, X, y, model_name):
        """Train a simple fallback model if LightGBM is not available."""
        logger.info(f"Training fallback model for {model_name}")
        
        from sklearn.ensemble import RandomForestRegressor
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"{model_name} - RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        # Store model and scaler
        self.models[model_name] = model
        self.scalers[model_name] = scaler
        
        # Save model
        model_path = f"/models/{model_name}_rf.joblib"
        joblib.dump({'model': model, 'scaler': scaler}, model_path)
        logger.info(f"Model saved to {model_path}")
        
        return model, rmse, r2

def main():
    """Main training function."""
    logger.info("Starting comprehensive enhanced sparse feature model training")
    
    # Configuration
    database_url = os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/postgres')
    features_table = os.environ.get('FEATURE_TABLES', 'enhanced_sparse_features')
    use_gpu = os.environ.get('USE_GPU', 'false').lower() == 'true'
    
    logger.info(f"Configuration:")
    logger.info(f"  Database: {database_url}")
    logger.info(f"  Features table: {features_table}")
    logger.info(f"  Use GPU: {use_gpu}")
    logger.info(f"  LightGBM available: {LIGHTGBM_AVAILABLE}")
    
    # Load features
    loader = ComprehensiveFeatureLoader(database_url, features_table)
    
    try:
        feature_matrix, feature_names = loader.load_all_features()
    except Exception as e:
        logger.error(f"Failed to load features: {e}")
        return 1
    
    if len(feature_matrix) == 0:
        logger.error("No features found in database")
        return 1
    
    # Analyze feature quality
    quality_stats = loader.analyze_feature_quality(feature_matrix)
    
    # Create targets
    target_generator = MultiObjectiveTargetGenerator(feature_matrix)
    energy_targets = target_generator.create_energy_consumption_target()
    growth_targets = target_generator.create_plant_growth_target()
    
    logger.info(f"Created targets - Energy: {len(energy_targets)}, Growth: {len(growth_targets)}")
    
    # Prepare features
    trainer = EnhancedModelTrainer(use_gpu=use_gpu)
    X, good_features = trainer.prepare_features(feature_matrix, min_completeness=0.1)
    
    # Train models
    results = {}
    
    try:
        if LIGHTGBM_AVAILABLE:
            # Train LightGBM models
            energy_model, energy_rmse, energy_r2 = trainer.train_lightgbm_model(
                X, energy_targets, 'energy_consumption'
            )
            growth_model, growth_rmse, growth_r2 = trainer.train_lightgbm_model(
                X, growth_targets, 'plant_growth'
            )
        else:
            # Train fallback models
            energy_model, energy_rmse, energy_r2 = trainer.train_fallback_model(
                X, energy_targets, 'energy_consumption'
            )
            growth_model, growth_rmse, growth_r2 = trainer.train_fallback_model(
                X, growth_targets, 'plant_growth'
            )
        
        results = {
            'energy_consumption': {'rmse': energy_rmse, 'r2': energy_r2},
            'plant_growth': {'rmse': growth_rmse, 'r2': growth_r2},
            'num_features': len(good_features),
            'training_samples': len(X)
        }
        
        # Save training summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'features_table': features_table,
            'model_performance': results,
            'top_features': {
                name: list(trainer.feature_importance.get(name, {}).keys())[:10] 
                for name in ['energy_consumption', 'plant_growth']
            },
            'feature_count': len(good_features),
            'sample_count': len(X)
        }
        
        with open('/models/training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("Training completed successfully!")
        logger.info(f"Results: {results}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())