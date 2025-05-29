"""Smart coordinator that distributes work based on sensor characteristics.

This coordinator analyzes the data characteristics of each era to make
intelligent decisions about GPU vs CPU processing.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import psycopg2
import redis
from psycopg2.extras import RealDictCursor

from worker_base import FeatureExtractionConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SmartFeatureCoordinator:
    """Enhanced coordinator with sensor-aware task distribution."""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        db_url: str = None,
        gpu_workers: int = 4,
        cpu_workers: int = 8,
        gpu_threshold: int = 500_000,  # Lowered threshold
    ):
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.db_url = db_url or os.environ.get("DATABASE_URL")
        self.gpu_workers = gpu_workers
        self.cpu_workers = cpu_workers
        self.gpu_threshold = gpu_threshold
        
        # Queue names
        self.gpu_queue = "feature:gpu:tasks"
        self.cpu_queue = "feature:cpu:tasks"
        self.results_queue = "feature:results"
        self.status_key = "feature:status"
        
        # Sensor characteristics for smarter distribution
        self.gpu_suitable_sensors = FeatureExtractionConfig.GPU_SUITABLE_COLUMNS
        self.cpu_suitable_sensors = FeatureExtractionConfig.CPU_SUITABLE_COLUMNS
        self.efficient_sensors = FeatureExtractionConfig.EFFICIENT_PROFILE_COLUMNS
        
    def get_db_connection(self):
        """Create database connection."""
        return psycopg2.connect(self.db_url, cursor_factory=RealDictCursor)
    
    def analyze_era_characteristics(self, era: Dict) -> Dict:
        """Analyze era to determine processing characteristics.
        
        Returns:
            Dictionary with analysis results including sensor types,
            data density, and recommended processing type.
        """
        query = """
        WITH sensor_summary AS (
            SELECT 
                COUNT(DISTINCT timestamp) as num_timestamps,
                jsonb_object_keys(features) as sensor,
                COUNT(*) as sensor_readings
            FROM preprocessed_features
            WHERE compartment_id = %s
              AND timestamp >= %s
              AND timestamp < %s
            GROUP BY sensor
        )
        SELECT 
            COUNT(DISTINCT sensor) as num_sensors,
            SUM(sensor_readings) as total_readings,
            MAX(num_timestamps) as unique_timestamps,
            json_agg(
                json_build_object(
                    'sensor', sensor,
                    'readings', sensor_readings
                )
            ) as sensor_details
        FROM sensor_summary
        """
        
        with self.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    (era["compartment_id"], era["start_time"], era["end_time"])
                )
                result = cur.fetchone()
        
        if not result or result["num_sensors"] is None:
            return {
                "num_sensors": 0,
                "total_readings": 0,
                "gpu_suitable_ratio": 0,
                "complexity_score": 0,
                "recommended": "skip"
            }
        
        # Analyze sensor types
        sensor_details = result["sensor_details"] or []
        gpu_suitable_count = 0
        cpu_suitable_count = 0
        efficient_count = 0
        
        for sensor_info in sensor_details:
            sensor_name = sensor_info["sensor"]
            if sensor_name in self.gpu_suitable_sensors:
                gpu_suitable_count += 1
            if sensor_name in self.cpu_suitable_sensors:
                cpu_suitable_count += 1
            if sensor_name in self.efficient_sensors:
                efficient_count += 1
        
        # Calculate metrics
        total_sensors = result["num_sensors"]
        gpu_suitable_ratio = gpu_suitable_count / total_sensors if total_sensors > 0 else 0
        efficient_ratio = efficient_count / total_sensors if total_sensors > 0 else 0
        
        # Complexity score (0-1) based on efficient sensors
        complexity_score = efficient_ratio
        
        # Determine recommendation
        if result["total_readings"] < 10000:
            recommended = "cpu"  # Too small for GPU overhead
        elif result["total_readings"] > self.gpu_threshold:
            recommended = "gpu"  # Large enough to benefit from GPU
        elif gpu_suitable_ratio > 0.6:
            recommended = "gpu"  # Many GPU-suitable sensors
        elif complexity_score > 0.5:
            recommended = "gpu"  # Complex features benefit from GPU preprocessing
        else:
            recommended = "cpu"
        
        return {
            "num_sensors": total_sensors,
            "total_readings": result["total_readings"],
            "unique_timestamps": result["unique_timestamps"],
            "gpu_suitable_ratio": gpu_suitable_ratio,
            "complexity_score": complexity_score,
            "recommended": recommended,
            "sensor_details": sensor_details
        }
    
    def distribute_eras_smart(self, eras: List[Dict]) -> Tuple[List[List[Dict]], List[List[Dict]]]:
        """Distribute eras using smart analysis of data characteristics."""
        logger.info("Analyzing era characteristics for smart distribution...")
        
        gpu_eras = []
        cpu_eras = []
        
        for era in eras:
            # Analyze each era
            analysis = self.analyze_era_characteristics(era)
            era["analysis"] = analysis
            
            # Skip empty eras
            if analysis["recommended"] == "skip":
                logger.info(f"Skipping empty era {era['era_id']}")
                continue
            
            # Assign based on analysis
            if analysis["recommended"] == "gpu":
                gpu_eras.append(era)
                logger.info(
                    f"Era {era['era_id']} -> GPU "
                    f"(readings: {analysis['total_readings']}, "
                    f"gpu_ratio: {analysis['gpu_suitable_ratio']:.2f}, "
                    f"complexity: {analysis['complexity_score']:.2f})"
                )
            else:
                cpu_eras.append(era)
                logger.info(
                    f"Era {era['era_id']} -> CPU "
                    f"(readings: {analysis['total_readings']}, "
                    f"gpu_ratio: {analysis['gpu_suitable_ratio']:.2f}, "
                    f"complexity: {analysis['complexity_score']:.2f})"
                )
        
        # Balance workloads
        gpu_queues = self._balance_workload(gpu_eras, self.gpu_workers, "total_readings")
        cpu_queues = self._balance_workload(cpu_eras, self.cpu_workers, "total_readings")
        
        return gpu_queues, cpu_queues
    
    def _balance_workload(self, eras: List[Dict], num_workers: int, 
                         weight_key: str = "estimated_rows") -> List[List[Dict]]:
        """Balance workload across workers using bin packing algorithm."""
        if not eras:
            return [[] for _ in range(num_workers)]
        
        # Sort by weight descending
        sorted_eras = sorted(
            eras, 
            key=lambda x: x.get("analysis", {}).get(weight_key, x.get(weight_key, 0)), 
            reverse=True
        )
        
        # Initialize queues and loads
        queues = [[] for _ in range(num_workers)]
        loads = [0] * num_workers
        
        # Assign eras to least loaded worker
        for era in sorted_eras:
            weight = era.get("analysis", {}).get(weight_key, era.get(weight_key, 0))
            min_idx = np.argmin(loads)
            queues[min_idx].append(era)
            loads[min_idx] += weight
        
        # Log distribution
        for i, load in enumerate(loads):
            logger.info(f"Worker {i} load: {load:,} {weight_key}")
        
        return queues
    
    def fetch_era_definitions(self, era_level: str = "B") -> List[Dict]:
        """Fetch all era definitions from database."""
        query = f"""
        SELECT 
            era_id,
            compartment_id,
            start_time,
            end_time,
            (EXTRACT(EPOCH FROM (end_time - start_time)) / 60) as duration_minutes
        FROM era_labels_level_{era_level}
        WHERE start_time >= '2014-01-01'
        ORDER BY start_time, compartment_id
        """
        
        with self.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                eras = cur.fetchall()
        
        logger.info(f"Fetched {len(eras)} era definitions")
        return eras
    
    def run(self, use_smart_distribution: bool = True):
        """Run the coordination with optional smart distribution."""
        logger.info(f"Starting coordinator (smart={use_smart_distribution})")
        
        # Fetch era definitions
        eras = self.fetch_era_definitions()
        
        # Distribute work
        if use_smart_distribution:
            gpu_queues, cpu_queues = self.distribute_eras_smart(eras)
        else:
            # Fall back to simple size-based distribution
            from coordinator import FeatureExtractionCoordinator
            simple_coord = FeatureExtractionCoordinator(
                redis_url=self.redis_client.connection_pool.connection_kwargs["url"],
                db_url=self.db_url,
                gpu_workers=self.gpu_workers,
                cpu_workers=self.cpu_workers,
                gpu_threshold=self.gpu_threshold
            )
            gpu_queues, cpu_queues = simple_coord.distribute_eras(eras)
        
        # Enqueue tasks (reuse from base coordinator)
        from coordinator import FeatureExtractionCoordinator
        coord = FeatureExtractionCoordinator(
            redis_url=self.redis_client.connection_pool.connection_kwargs["url"],
            db_url=self.db_url
        )
        coord.enqueue_tasks(gpu_queues, cpu_queues)
        coord.monitor_progress()


if __name__ == "__main__":
    # Use smart distribution by default
    use_smart = os.environ.get("USE_SMART_DISTRIBUTION", "true").lower() == "true"
    
    coordinator = SmartFeatureCoordinator()
    coordinator.run(use_smart_distribution=use_smart)