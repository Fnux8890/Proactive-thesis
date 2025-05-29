"""
Coordinator service for parallel feature extraction
Distributes work across GPU and CPU workers via Redis queue
"""

"""Coordinator service for parallel feature extraction.

Distributes work across GPU and CPU workers via Redis queue.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import psycopg2
import redis
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractionCoordinator:
    """Coordinates parallel feature extraction across GPU and CPU workers.
    
    Attributes:
        redis_client: Redis connection for task queue
        db_url: PostgreSQL connection string
        gpu_workers: Number of GPU workers
        cpu_workers: Number of CPU workers
        gpu_threshold: Row count threshold for GPU processing
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        db_url: str = None,
        gpu_workers: int = 4,
        cpu_workers: int = 8,
        gpu_threshold: int = 1_000_000,
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
        
    def get_db_connection(self):
        """Create database connection"""
        return psycopg2.connect(self.db_url, cursor_factory=RealDictCursor)
    
    def fetch_era_definitions(self, era_level: str = "B") -> List[Dict]:
        """Fetch all era definitions from database.
        
        Args:
            era_level: Era detection level (A, B, or C)
            
        Returns:
            List of era definitions with metadata
        """
        query = f"""
        SELECT 
            era_id,
            compartment_id,
            start_time,
            end_time,
            (EXTRACT(EPOCH FROM (end_time - start_time)) / 60) as duration_minutes,
            COUNT(*) OVER (PARTITION BY era_id, compartment_id) as estimated_rows
        FROM era_labels_level_{era_level}
        WHERE start_time >= '2014-01-01'
        ORDER BY start_time, compartment_id
        """
        
        with self.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                eras = cur.fetchall()
                
        # Estimate row count based on duration
        # Most sensors sample every 5 minutes, some every minute
        for era in eras:
            # Conservative estimate: 1 row per minute (some sensors are 5-min intervals)
            era["estimated_rows"] = int(era["duration_minutes"])
            
        logger.info(f"Fetched {len(eras)} era definitions")
        return eras
    
    def distribute_eras(
        self, eras: List[Dict]
    ) -> Tuple[List[List[Dict]], List[List[Dict]]]:
        """Distribute eras across GPU and CPU workers based on size and complexity.
        
        GPU workers handle:
        - Large eras (>1M rows)
        - High-frequency continuous sensors (temperature, humidity, light)
        - Simple statistical features that parallelize well
        
        CPU workers handle:
        - Small/medium eras
        - Binary/categorical sensors (lamp status)
        - Complex features (entropy, peak detection)
        
        Args:
            eras: List of era definitions
            
        Returns:
            Tuple of (gpu_queues, cpu_queues)
        """
        # Sort by estimated size
        sorted_eras = sorted(eras, key=lambda x: x['estimated_rows'], reverse=True)
        
        # Split into GPU and CPU tasks based on size and sensor characteristics
        gpu_eras = []
        cpu_eras = []
        
        for era in sorted_eras:
            # Large eras go to GPU for parallel processing
            if era["estimated_rows"] > self.gpu_threshold:
                gpu_eras.append(era)
            # Small eras or those with many binary sensors go to CPU
            else:
                cpu_eras.append(era)
        
        logger.info(f"GPU eras: {len(gpu_eras)}, CPU eras: {len(cpu_eras)}")
        
        # Balance GPU workload
        gpu_queues = [[] for _ in range(self.gpu_workers)]
        gpu_loads = [0] * self.gpu_workers
        
        for era in gpu_eras:
            # Assign to worker with least load
            min_idx = np.argmin(gpu_loads)
            gpu_queues[min_idx].append(era)
            gpu_loads[min_idx] += era['estimated_rows']
        
        # Balance CPU workload
        cpu_queues = [[] for _ in range(self.cpu_workers)]
        cpu_loads = [0] * self.cpu_workers
        
        for era in cpu_eras:
            min_idx = np.argmin(cpu_loads)
            cpu_queues[min_idx].append(era)
            cpu_loads[min_idx] += era['estimated_rows']
        
        # Log load distribution
        logger.info(f"GPU loads: {gpu_loads}")
        logger.info(f"CPU loads: {cpu_loads}")
        
        return gpu_queues, cpu_queues
    
    def enqueue_tasks(
        self, gpu_queues: List[List[Dict]], cpu_queues: List[List[Dict]]
    ) -> None:
        """Push tasks to Redis queues.
        
        Args:
            gpu_queues: Tasks for each GPU worker
            cpu_queues: Tasks for each CPU worker
        """
        # Clear existing queues
        self.redis_client.delete(self.gpu_queue, self.cpu_queue)
        
        # Enqueue GPU tasks
        for worker_id, tasks in enumerate(gpu_queues):
            for task in tasks:
                task_data = {
                    'worker_id': f'gpu-{worker_id}',
                    'era_id': task['era_id'],
                    'compartment_id': task['compartment_id'],
                    'start_time': task['start_time'].isoformat(),
                    'end_time': task['end_time'].isoformat(),
                    'estimated_rows': task['estimated_rows']
                }
                self.redis_client.lpush(self.gpu_queue, json.dumps(task_data))
        
        # Enqueue CPU tasks
        for worker_id, tasks in enumerate(cpu_queues):
            for task in tasks:
                task_data = {
                    'worker_id': f'cpu-{worker_id}',
                    'era_id': task['era_id'],
                    'compartment_id': task['compartment_id'],
                    'start_time': task['start_time'].isoformat(),
                    'end_time': task['end_time'].isoformat(),
                    'estimated_rows': task['estimated_rows']
                }
                self.redis_client.lpush(self.cpu_queue, json.dumps(task_data))
        
        # Set status
        total_tasks = sum(len(q) for q in gpu_queues) + sum(len(q) for q in cpu_queues)
        status = {
            'total_tasks': total_tasks,
            'gpu_tasks': sum(len(q) for q in gpu_queues),
            'cpu_tasks': sum(len(q) for q in cpu_queues),
            'completed': 0,
            'failed': 0,
            'start_time': datetime.now().isoformat()
        }
        self.redis_client.hset(self.status_key, mapping=status)
        
        logger.info(f"Enqueued {total_tasks} tasks")
    
    def monitor_progress(self) -> None:
        """Monitor task completion and aggregate results."""
        import time
        
        while True:
            # Get current status
            status = self.redis_client.hgetall(self.status_key)
            total = int(status.get('total_tasks', 0))
            completed = int(status.get('completed', 0))
            failed = int(status.get('failed', 0))
            
            if total > 0:
                progress = (completed + failed) / total * 100
                logger.info(
                    f"Progress: {progress:.1f}% "
                    f"({completed} completed, {failed} failed)"
                )
                
                if completed + failed >= total:
                    logger.info("All tasks completed!")
                    break
            
            # Check for results
            result = self.redis_client.rpop(self.results_queue)
            if result:
                result_data = json.loads(result)
                if result_data['status'] == 'success':
                    self.redis_client.hincrby(self.status_key, 'completed', 1)
                else:
                    self.redis_client.hincrby(self.status_key, 'failed', 1)
                    logger.error(f"Task failed: {result_data.get('error')}")
            
            time.sleep(5)
    
    def run(self) -> None:
        """Run the main coordination loop."""
        logger.info("Starting feature extraction coordinator")
        
        # Fetch era definitions
        eras = self.fetch_era_definitions()
        
        # Distribute work
        gpu_queues, cpu_queues = self.distribute_eras(eras)
        
        # Enqueue tasks
        self.enqueue_tasks(gpu_queues, cpu_queues)
        
        # Monitor progress
        self.monitor_progress()
        
        logger.info("Feature extraction completed")


if __name__ == "__main__":
    coordinator = FeatureExtractionCoordinator()
    coordinator.run()