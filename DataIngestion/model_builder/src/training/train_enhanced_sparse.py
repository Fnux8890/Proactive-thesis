#!/usr/bin/env python
"""Train model using enhanced sparse features.

Simple training script that works with the enhanced_sparse_features table.
"""
import os
import sys
import logging
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_enhanced_features(database_url, features_table='enhanced_sparse_features'):
    """Load features from enhanced sparse features table."""
    logger.info(f"Loading features from {features_table}")
    
    # Create SQLAlchemy engine
    engine = create_engine(database_url)
    
    # Query to pivot features into columns
    query = f"""
    SELECT 
        timestamp,
        MAX(CASE WHEN feature_name = 'air_temp_c_mean' THEN feature_value END) as air_temp_c_mean,
        MAX(CASE WHEN feature_name = 'air_temp_c_std' THEN feature_value END) as air_temp_c_std,
        MAX(CASE WHEN feature_name = 'co2_measured_ppm_mean' THEN feature_value END) as co2_mean,
        MAX(CASE WHEN feature_name = 'co2_measured_ppm_std' THEN feature_value END) as co2_std,
        MAX(CASE WHEN feature_name = 'relative_humidity_percent_mean' THEN feature_value END) as humidity_mean,
        MAX(CASE WHEN feature_name = 'relative_humidity_percent_std' THEN feature_value END) as humidity_std
    FROM {features_table}
    GROUP BY timestamp
    ORDER BY timestamp
    """
    
    df = pd.read_sql_query(query, engine)
    logger.info(f"Loaded {len(df)} feature vectors")
    return df

def create_synthetic_targets(df):
    """Create synthetic targets for demonstration."""
    # Energy consumption (simplified)
    energy = (
        df['air_temp_c_mean'].fillna(20) * 0.5 +  # Heating/cooling
        df['co2_mean'].fillna(400) * 0.001 +      # Ventilation
        np.random.normal(0, 0.1, len(df))         # Noise
    )
    
    # Plant growth (simplified)
    growth = (
        np.clip(df['air_temp_c_mean'].fillna(20) - 15, 0, 10) * 0.1 +
        np.clip(df['humidity_mean'].fillna(60) - 40, 0, 30) * 0.01 +
        np.random.normal(0, 0.05, len(df))
    )
    
    return energy, growth

class SimpleNN(nn.Module):
    """Simple neural network for demonstration."""
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

def train_model(features, targets, model_name='energy'):
    """Train a simple model."""
    logger.info(f"Training {model_name} model")
    
    # Prepare data
    X = features.fillna(0).values
    y = targets.values.reshape(-1, 1)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.FloatTensor(y)
    
    # Create model
    model = SimpleNN(X.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    epochs = 50
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    # Save model
    model_path = f"/models/{model_name}_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'input_dim': X.shape[1]
    }, model_path)
    logger.info(f"Model saved to {model_path}")
    
    return model, scaler

def main():
    """Main training function."""
    logger.info("Starting enhanced sparse feature model training")
    
    # Get database URL
    database_url = os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@db:5432/postgres')
    features_table = os.environ.get('FEATURE_TABLES', 'enhanced_sparse_features')
    
    logger.info(f"Using features table: {features_table}")
    
    # Load features
    df = load_enhanced_features(database_url, features_table)
    
    if len(df) == 0:
        logger.error("No features found in database")
        return 1
    
    # Extract feature columns
    feature_cols = ['air_temp_c_mean', 'air_temp_c_std', 'co2_mean', 'co2_std', 'humidity_mean', 'humidity_std']
    features = df[feature_cols]
    
    # Create synthetic targets
    energy_targets, growth_targets = create_synthetic_targets(df)
    
    # Train models
    energy_model, energy_scaler = train_model(features, energy_targets, 'energy_consumption')
    growth_model, growth_scaler = train_model(features, growth_targets, 'plant_growth')
    
    logger.info("Training completed successfully!")
    return 0

if __name__ == '__main__':
    sys.exit(main())