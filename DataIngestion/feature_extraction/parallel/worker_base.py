"""
Base worker class for parallel feature extraction
Handles Redis queue, database connections, and common functionality
"""

"""Base worker class for parallel feature extraction.

Handles Redis queue, database connections, and common functionality.
"""

import json
import logging
import os
import time
import traceback
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import psycopg2
import redis
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureWorkerBase(ABC):
    """Abstract base class for feature extraction workers.
    
    Provides common functionality for GPU and CPU workers including:
    - Redis task queue management
    - Database connection pooling
    - Era data fetching and feature storage
    - Performance metrics tracking
    """
    
    def __init__(
        self,
        worker_id: str,
        worker_type: str,
        redis_url: str = "redis://localhost:6379",
        db_url: str = None,
        batch_size: int = 10000,
    ):
        self.worker_id = worker_id
        self.worker_type = worker_type
        self.batch_size = batch_size
        
        # Redis connection
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        
        # Database URL
        self.db_url = db_url or os.environ.get("DATABASE_URL")
        
        # Queue names
        self.task_queue = f"feature:{worker_type.lower()}:tasks"
        self.results_queue = "feature:results"
        
        # Performance metrics
        self.tasks_processed = 0
        self.total_processing_time = 0
        
        logger.info(f"Initialized {worker_type} worker: {worker_id}")
    
    def get_db_connection(self) -> psycopg2.extensions.connection:
        """Create database connection with connection pooling.
        
        Returns:
            PostgreSQL connection with RealDictCursor
        """
        return psycopg2.connect(self.db_url, cursor_factory=RealDictCursor)
    
    def fetch_era_data(
        self,
        era_id: int,
        compartment_id: int,
        start_time: str,
        end_time: str,
    ) -> pd.DataFrame:
        """Fetch data for a specific era from preprocessed table.
        
        Args:
            era_id: Era identifier
            compartment_id: Greenhouse compartment ID
            start_time: Era start time (ISO format)
            end_time: Era end time (ISO format)
            
        Returns:
            DataFrame with timestamp, compartment_id, and feature columns
        """
        query = """
        SELECT 
            timestamp,
            compartment_id,
            features
        FROM preprocessed_features
        WHERE compartment_id = %s
          AND timestamp >= %s
          AND timestamp < %s
        ORDER BY timestamp
        """
        
        with self.get_db_connection() as conn:
            df = pd.read_sql_query(
                query,
                conn,
                params=(compartment_id, start_time, end_time),
                parse_dates=["timestamp"],
            )
        
        if df.empty:
            logger.warning(f"No data found for era {era_id}, compartment {compartment_id}")
            return df
        
        # Expand JSONB features column
        features_df = pd.json_normalize(df["features"])
        df = pd.concat(
            [df[["timestamp", "compartment_id"]], features_df], axis=1
        )
        
        logger.info(f"Fetched {len(df)} rows for era {era_id}")
        return df
    
    def save_features(
        self, features_df: pd.DataFrame, era_id: int, compartment_id: int
    ) -> bool:
        """Save extracted features to database.
        
        Args:
            features_df: DataFrame with 'variable' and 'value' columns
            era_id: Era identifier
            compartment_id: Greenhouse compartment ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare data for insertion
            records = []
            for _, row in features_df.iterrows():
                record = {
                    "era_id": era_id,
                    "compartment_id": compartment_id,
                    "feature_name": row["variable"],
                    "feature_value": float(row["value"]),
                    "created_at": datetime.now(),
                }
                records.append(record)
            
            # Batch insert
            with self.get_db_connection() as conn:
                with conn.cursor() as cur:
                    # Create temporary table for bulk insert
                    cur.execute("""
                        CREATE TEMP TABLE temp_features (
                            era_id INTEGER,
                            compartment_id INTEGER,
                            feature_name TEXT,
                            feature_value FLOAT,
                            created_at TIMESTAMP
                        )
                    """)
                    
                    # Use COPY for efficient bulk insert
                    from io import StringIO
                    buffer = StringIO()
                    for r in records:
                        buffer.write(
                            f"{r['era_id']}\t{r['compartment_id']}\t"
                            f"{r['feature_name']}\t{r['feature_value']}\t"
                            f"{r['created_at']}\n"
                        )
                    buffer.seek(0)
                    
                    cur.copy_from(
                        buffer,
                        "temp_features",
                        columns=[
                            "era_id",
                            "compartment_id",
                            "feature_name",
                            "feature_value",
                            "created_at",
                        ],
                    )
                    
                    # Insert into main table
                    cur.execute("""
                        INSERT INTO tsfresh_features 
                        (era_id, compartment_id, feature_name, feature_value, created_at)
                        SELECT * FROM temp_features
                        ON CONFLICT (era_id, compartment_id, feature_name) 
                        DO UPDATE SET 
                            feature_value = EXCLUDED.feature_value,
                            created_at = EXCLUDED.created_at
                    """)
                    
                    conn.commit()
                    
            logger.info(f"Saved {len(records)} features for era {era_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save features: {str(e)}")
            return False
    
    @abstractmethod
    def process_era(self, task_data: Dict) -> Dict[str, Any]:
        """Process a single era - to be implemented by subclasses.
        
        Args:
            task_data: Task information including era_id, times, etc.
            
        Returns:
            Result dictionary with status and metrics
        """
        pass
    
    def run(self) -> None:
        """Run the main worker loop."""
        logger.info(f"Starting {self.worker_type} worker {self.worker_id}")
        
        while True:
            try:
                # Get task from queue (blocking with timeout)
                task_json = self.redis_client.brpop(self.task_queue, timeout=30)
                
                if task_json is None:
                    logger.info("No more tasks, worker shutting down")
                    break
                
                # Parse task
                _, task_str = task_json
                task = json.loads(task_str)
                
                # Process task
                start_time = time.time()
                logger.info(f"Processing era {task['era_id']} "
                          f"(compartment {task['compartment_id']})")
                
                result = self.process_era(task)

                # Update metrics
                processing_time = time.time() - start_time
                self.tasks_processed += 1
                self.total_processing_time += processing_time
                
                # Send result
                result_data = {
                    "worker_id": self.worker_id,
                    "era_id": task["era_id"],
                    "compartment_id": task["compartment_id"],
                    "status": result["status"],
                    "processing_time": processing_time,
                    "features_extracted": result.get("features_extracted", 0),
                    "error": result.get("error", None),
                }
                
                self.redis_client.lpush(
                    self.results_queue, json.dumps(result_data)
                )
                
                logger.info(f"Completed era {task['era_id']} in {processing_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Worker error: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Send error result
                error_result = {
                    "worker_id": self.worker_id,
                    "status": "error",
                    "error": str(e),
                }
                self.redis_client.lpush(
                    self.results_queue,
                    json.dumps(error_result)
                )
        
        # Log final statistics
        if self.tasks_processed > 0:
            avg_time = self.total_processing_time / self.tasks_processed
            logger.info(f"Worker {self.worker_id} completed {self.tasks_processed} tasks")
            logger.info(f"Average processing time: {avg_time:.2f}s per task")


class FeatureExtractionConfig:
    """Configuration for feature extraction.
    
    Based on analysis of the preprocessing pipeline, we extract features from:
    - Temperature sensors (air, pipe, flow)
    - Humidity sensors 
    - Light/radiation sensors
    - CO2 sensors
    - Ventilation position sensors
    - Binary sensors (lamp status, rain)
    - External weather data
    - Energy prices
    """
    
    # High-priority columns with efficient profile (more features)
    EFFICIENT_PROFILE_COLUMNS = [
        "air_temp_c",
        "relative_humidity_percent",
        "co2_measured_ppm",
        "light_intensity_umol",
        "radiation_w_m2",
        "dli_sum",
        "par_synth_umol",
    ]

    # All numeric columns to extract (minimal profile by default)
    FEATURE_COLUMNS = [
        # Temperature sensors
        "air_temp_c",
        "air_temp_middle_c",
        "outside_temp_c",
        "pipe_temp_1_c",
        "pipe_temp_2_c",
        "flow_temp_1_c",
        "flow_temp_2_c",
        # Humidity sensors
        "relative_humidity_percent",
        "relative_humidity_afd3_percent",
        "relative_humidity_afd4_percent",
        "humidity_deficit_g_m3",
        "humidity_deficit_afd3_g_m3",
        "humidity_deficit_afd4_g_m3",
        # Light/radiation sensors
        "light_intensity_umol",
        "light_intensity_lux",
        "radiation_w_m2",
        "outside_light_w_m2",
        "dli_sum",
        "par_synth_umol",
        # CO2 sensors
        "co2_measured_ppm",
        "co2_required_ppm",
        # Ventilation
        "vent_pos_1_percent",
        "vent_pos_2_percent",
        "vent_lee_afd3_percent",
        "vent_lee_afd4_percent",
        "vent_wind_afd3_percent",
        "vent_wind_afd4_percent",
        # Other controls
        "heating_setpoint_c",
        "curtain_1_percent",
        "curtain_2_percent",
        "curtain_3_percent",
        "curtain_4_percent",
        "window_1_percent",
        "window_2_percent",
        # Environmental
        "vpd_hpa",
        "rain_status",
        # Energy
        "spot_price_dkk_mwh",
    ]
    
    # tsfresh settings
    TSFRESH_SETTINGS = {
        "n_jobs": 4,
        "chunksize": 50,
        "show_warnings": False,
        "disable_progressbar": True,
    }
    
    # Feature selection settings
    FEATURE_SELECTION_THRESHOLD = 0.5
    FEATURE_SELECTION_METHOD = "fdr"  # False Discovery Rate
    
    @classmethod
    def get_feature_set(cls) -> str:
        """Get feature set from environment."""
        import os
        return os.environ.get("FEATURE_SET", "efficient").lower()
    
    @classmethod
    def get_feature_parameters(cls):
        """Get tsfresh parameters based on feature set."""
        from tsfresh.feature_extraction import (
            MinimalFCParameters,
            EfficientFCParameters,
            ComprehensiveFCParameters,
        )
        
        feature_set = cls.get_feature_set()
        
        if feature_set == "minimal":
            return MinimalFCParameters()
        elif feature_set == "comprehensive":
            return ComprehensiveFCParameters()
        else:  # efficient (default)
            return EfficientFCParameters()

    # GPU vs CPU distribution hints
    GPU_SUITABLE_COLUMNS = {
        # High-frequency continuous sensors
        "air_temp_c",
        "relative_humidity_percent",
        "light_intensity_umol",
        "radiation_w_m2",
        "co2_measured_ppm",
        # Sensors with simple stats
        "pipe_temp_1_c",
        "flow_temp_1_c",
        "outside_temp_c",
    }

    CPU_SUITABLE_COLUMNS = {
        # Binary/categorical sensors
        "rain_status",
        "co2_status",
        "co2_dosing_status",
        # Low-frequency data
        "spot_price_dkk_mwh",
        # Complex ventilation calculations
        "vent_pos_1_percent",
        "vent_pos_2_percent",
    }