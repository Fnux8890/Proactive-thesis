#!/usr/bin/env python3
"""
Demonstration of connection pool observability features.

This script shows how to:
1. Monitor pool metrics in real-time
2. Handle timeouts gracefully
3. Use custom metrics listeners
4. Tune pool configuration based on metrics
"""

import time
import threading
import logging
from concurrent.futures import ThreadPoolExecutor
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import (
    initialize_pool,
    get_connection,
    return_connection,
    configure_acquisition,
    get_pool_metrics,
    log_pool_summary,
    close_pool,
    get_metrics_collector
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def simulate_database_work(work_time: float = 0.5, worker_id: int = 0):
    """Simulate database operations."""
    try:
        logger.info(f"Worker {worker_id}: Requesting connection...")
        start_time = time.time()
        
        # Get connection with timeout
        conn = get_connection(timeout=2.0)
        wait_time = time.time() - start_time
        
        logger.info(f"Worker {worker_id}: Got connection after {wait_time:.3f}s")
        
        # Simulate work
        time.sleep(work_time)
        
        # Return connection
        return_connection(conn)
        logger.info(f"Worker {worker_id}: Returned connection")
        
        return True
        
    except TimeoutError as e:
        logger.error(f"Worker {worker_id}: Failed to get connection - {e}")
        return False
    except Exception as e:
        logger.error(f"Worker {worker_id}: Unexpected error - {e}")
        return False


def stress_test_pool(num_workers: int, work_time: float):
    """Run a stress test on the connection pool."""
    logger.info(f"\nStarting stress test with {num_workers} workers")
    
    success_count = 0
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = []
        for i in range(num_workers):
            future = executor.submit(simulate_database_work, work_time, i)
            futures.append(future)
            time.sleep(0.1)  # Stagger submissions slightly
        
        # Wait for completion
        for future in futures:
            if future.result():
                success_count += 1
    
    logger.info(f"Stress test complete: {success_count}/{num_workers} succeeded")
    return success_count


def adaptive_pool_tuning():
    """Demonstrate adaptive pool tuning based on metrics."""
    logger.info("\n=== Adaptive Pool Tuning Demo ===")
    
    # Start with small pool
    initialize_pool(minconn=2, maxconn=5)
    logger.info("Initial pool: min=2, max=5")
    
    # Monitor metrics during load
    def check_and_tune():
        metrics = get_pool_metrics()
        utilization = metrics['pool.utilization_percent']
        avg_wait = metrics['pool.wait_time.avg']
        
        logger.info(f"Current metrics: utilization={utilization:.1f}%, avg_wait={avg_wait:.3f}s")
        
        if utilization > 80 and avg_wait > 0.5:
            logger.warning("Pool under pressure, increasing size...")
            close_pool()
            initialize_pool(minconn=5, maxconn=10)
            logger.info("Resized pool: min=5, max=10")
            return True
        return False
    
    # Run load test
    stress_test_pool(num_workers=8, work_time=1.0)
    
    # Check if tuning needed
    if check_and_tune():
        logger.info("Re-running test with larger pool...")
        stress_test_pool(num_workers=8, work_time=1.0)
    
    # Final metrics
    log_pool_summary()


def real_time_monitoring():
    """Demonstrate real-time metrics monitoring."""
    logger.info("\n=== Real-Time Monitoring Demo ===")
    
    # Initialize pool
    close_pool()
    initialize_pool(minconn=3, maxconn=8)
    configure_acquisition(timeout=3.0, use_queue=True)
    
    # Add custom listener for high utilization alerts
    high_util_count = 0
    
    def utilization_monitor(metrics):
        nonlocal high_util_count
        if metrics['pool.utilization_percent'] > 75:
            high_util_count += 1
            if high_util_count >= 3:  # Alert after 3 consecutive high readings
                logger.warning(f"ALERT: Sustained high pool utilization: {metrics['pool.utilization_percent']:.1f}%")
                high_util_count = 0
    
    collector = get_metrics_collector()
    collector.add_listener(utilization_monitor)
    
    # Create sustained load
    logger.info("Creating sustained load...")
    active_connections = []
    
    try:
        # Gradually increase load
        for i in range(6):
            conn = get_connection()
            active_connections.append(conn)
            logger.info(f"Acquired connection {i+1}")
            
            # Log metrics
            metrics = get_pool_metrics()
            logger.info(
                f"Pool state: {metrics['pool.used_connections']}/{metrics['pool.max_size']} "
                f"({metrics['pool.utilization_percent']:.1f}%)"
            )
            time.sleep(0.5)
        
        # Hold connections to maintain high utilization
        logger.info("Holding connections to demonstrate sustained high utilization...")
        time.sleep(3)
        
    finally:
        # Clean up
        logger.info("Releasing connections...")
        for conn in active_connections:
            return_connection(conn)
        
        # Remove listener
        collector.remove_listener(utilization_monitor)
    
    # Final summary
    log_pool_summary()


def timeout_handling_demo():
    """Demonstrate timeout handling and metrics."""
    logger.info("\n=== Timeout Handling Demo ===")
    
    # Small pool to force timeouts
    close_pool()
    initialize_pool(minconn=1, maxconn=2)
    configure_acquisition(timeout=1.0, use_queue=True)
    
    # Hold all connections
    held_connections = []
    for i in range(2):
        conn = get_connection()
        held_connections.append(conn)
    
    logger.info("Pool exhausted, attempting more connections...")
    
    # Try to get more connections (will timeout)
    timeout_count = 0
    for i in range(3):
        try:
            logger.info(f"Attempt {i+1}: Requesting connection...")
            conn = get_connection()
            logger.info(f"Attempt {i+1}: Success!")
            return_connection(conn)
        except TimeoutError:
            logger.error(f"Attempt {i+1}: Timed out!")
            timeout_count += 1
    
    # Release held connections
    for conn in held_connections:
        return_connection(conn)
    
    # Show timeout metrics
    metrics = get_pool_metrics()
    logger.info(f"\nTimeout Statistics:")
    logger.info(f"- Total requests: {metrics['pool.total_requests']}")
    logger.info(f"- Successful: {metrics['pool.successful_acquisitions']}")
    logger.info(f"- Timeouts: {metrics['pool.timeouts']}")
    logger.info(f"- Success rate: {metrics['pool.success_rate']:.1f}%")


def main():
    """Run all demonstrations."""
    try:
        # Demo 1: Real-time monitoring
        real_time_monitoring()
        time.sleep(2)
        
        # Demo 2: Timeout handling
        timeout_handling_demo()
        time.sleep(2)
        
        # Demo 3: Adaptive tuning
        adaptive_pool_tuning()
        
        # Final cleanup
        logger.info("\n=== Final Metrics Summary ===")
        log_pool_summary()
        
        # Detailed metrics
        metrics = get_pool_metrics()
        logger.info("\nDetailed metrics:")
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.3f}")
            else:
                logger.info(f"  {key}: {value}")
    
    finally:
        close_pool()
        logger.info("\nDemo complete!")


if __name__ == "__main__":
    main()