"""
CPU-based feature extraction worker
Handles small to medium eras using tsfresh
"""

import os
import logging
from typing import Dict, Any
import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

from worker_base import FeatureWorkerBase, FeatureExtractionConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CPUFeatureWorker(FeatureWorkerBase):
    def __init__(self, worker_id: str, **kwargs):
        super().__init__(worker_id, "CPU", **kwargs)
        
        # Get number of jobs from environment
        self.n_jobs = int(os.environ.get('TSFRESH_N_JOBS', '4'))
        logger.info(f"CPU worker {worker_id} using {self.n_jobs} jobs for tsfresh")
    
    def transform_to_long_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform wide format data to long format for tsfresh"""
        # Select columns for melting
        id_vars = ['timestamp', 'compartment_id']
        value_vars = [col for col in FeatureExtractionConfig.FEATURE_COLUMNS 
                     if col in df.columns]
        
        # Melt to long format
        df_long = df.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name='variable',
            value_name='value'
        )
        
        # Add ID column for tsfresh
        df_long['id'] = df_long['compartment_id']
        
        # Remove NaN values
        df_long = df_long.dropna(subset=['value'])
        
        return df_long
    
    def process_era(self, task_data: Dict) -> Dict[str, Any]:
        """Process a single era using CPU-based tsfresh"""
        try:
            # Fetch era data
            df = self.fetch_era_data(
                task_data['era_id'],
                task_data['compartment_id'],
                task_data['start_time'],
                task_data['end_time']
            )
            
            if df.empty:
                return {
                    'status': 'skipped',
                    'features_extracted': 0,
                    'reason': 'No data found'
                }
            
            logger.info(f"Processing {len(df)} rows for era {task_data['era_id']}")
            
            # Fill missing values
            for col in FeatureExtractionConfig.FEATURE_COLUMNS:
                if col in df.columns:
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
            
            # Transform to long format
            df_long = self.transform_to_long_format(df)
            
            # Get feature parameters based on environment
            feature_params = FeatureExtractionConfig.get_feature_parameters()
            feature_set = FeatureExtractionConfig.get_feature_set()
            
            # Configure tsfresh settings
            extraction_settings = FeatureExtractionConfig.TSFRESH_SETTINGS.copy()
            extraction_settings['n_jobs'] = self.n_jobs
            
            # Extract features
            logger.info(f"Extracting features with tsfresh ({feature_set} set, n_jobs={self.n_jobs})")
            features = extract_features(
                df_long,
                column_id='id',
                column_sort='timestamp',
                column_kind='variable',
                column_value='value',
                default_fc_parameters=feature_params,
                **extraction_settings
            )
            
            # Impute missing values
            features = impute(features)
            
            # Basic feature selection - remove features with low variance
            feature_variance = features.var()
            selected_features = features[feature_variance[feature_variance > 0.01].index]
            
            logger.info(f"Selected {len(selected_features.columns)} features "
                       f"from {len(features.columns)} total")
            
            # Transform to long format for storage
            features_long = selected_features.reset_index().melt(
                id_vars=['index'],
                var_name='variable',
                value_name='value'
            )
            
            # Save features
            success = self.save_features(
                features_long,
                task_data['era_id'],
                task_data['compartment_id']
            )
            
            return {
                'status': 'success' if success else 'error',
                'features_extracted': len(features_long),
                'error': None if success else 'Failed to save features'
            }
            
        except Exception as e:
            logger.error(f"Error processing era: {str(e)}")
            return {
                'status': 'error',
                'features_extracted': 0,
                'error': str(e)
            }


if __name__ == "__main__":
    worker_id = os.environ.get('WORKER_ID', 'cpu-0')
    worker = CPUFeatureWorker(worker_id)
    worker.run()