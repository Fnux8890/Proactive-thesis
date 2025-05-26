"""
Connection pool metrics collection and reporting.

This module provides structured metrics for monitoring connection pool health,
including pool size, usage, wait times, and other key performance indicators.
"""

import time
import logging
import threading
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class PoolMetrics:
    """Connection pool metrics container."""
    # Pool configuration
    min_size: int = 0
    max_size: int = 0
    
    # Current state
    total_connections: int = 0
    used_connections: int = 0
    idle_connections: int = 0
    
    # Performance metrics
    total_requests: int = 0
    successful_acquisitions: int = 0
    failed_acquisitions: int = 0
    timeouts: int = 0
    
    # Wait time tracking (in seconds)
    total_wait_time: float = 0.0
    max_wait_time: float = 0.0
    avg_wait_time: float = 0.0
    
    # Recent wait times for rolling average
    recent_wait_times: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Timestamps
    last_updated: Optional[datetime] = None
    metrics_start_time: datetime = field(default_factory=datetime.now)
    
    def update_wait_time(self, wait_time: float) -> None:
        """Update wait time statistics."""
        self.recent_wait_times.append(wait_time)
        self.total_wait_time += wait_time
        self.max_wait_time = max(self.max_wait_time, wait_time)
        
        # Calculate rolling average
        if self.recent_wait_times:
            self.avg_wait_time = sum(self.recent_wait_times) / len(self.recent_wait_times)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for reporting."""
        uptime_seconds = (datetime.now() - self.metrics_start_time).total_seconds()
        
        return {
            # Pool configuration
            "pool.min_size": self.min_size,
            "pool.max_size": self.max_size,
            
            # Current state
            "pool.total_connections": self.total_connections,
            "pool.used_connections": self.used_connections,
            "pool.idle_connections": self.idle_connections,
            "pool.utilization_percent": (self.used_connections / self.max_size * 100) if self.max_size > 0 else 0,
            
            # Performance counters
            "pool.total_requests": self.total_requests,
            "pool.successful_acquisitions": self.successful_acquisitions,
            "pool.failed_acquisitions": self.failed_acquisitions,
            "pool.timeouts": self.timeouts,
            "pool.success_rate": (self.successful_acquisitions / self.total_requests * 100) if self.total_requests > 0 else 0,
            
            # Wait times
            "pool.wait_time.total": self.total_wait_time,
            "pool.wait_time.max": self.max_wait_time,
            "pool.wait_time.avg": self.avg_wait_time,
            "pool.wait_time.recent_avg": sum(self.recent_wait_times) / len(self.recent_wait_times) if self.recent_wait_times else 0,
            
            # Metadata
            "pool.uptime_seconds": uptime_seconds,
            "pool.last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }


class MetricsCollector:
    """Thread-safe metrics collector for connection pool monitoring."""
    
    def __init__(self):
        self._metrics = PoolMetrics()
        self._lock = threading.Lock()
        self._listeners: list[Callable[[Dict[str, Any]], None]] = []
    
    def set_pool_config(self, min_size: int, max_size: int) -> None:
        """Set pool configuration metrics."""
        with self._lock:
            self._metrics.min_size = min_size
            self._metrics.max_size = max_size
            # Assuming total_connections should reflect max_size initially or when config changes
            self._metrics.total_connections = max_size 
            self._metrics.last_updated = datetime.now()

    def increment_used_connections(self) -> None:
        """Atomically increment the count of used connections."""
        with self._lock:
            # Ensure used_connections does not exceed max_size
            self._metrics.used_connections = min(self._metrics.used_connections + 1, self._metrics.max_size)
            # idle_connections is max_size (total available) - used_connections
            self._metrics.idle_connections = self._metrics.max_size - self._metrics.used_connections
            self._metrics.last_updated = datetime.now()
            # Optionally, emit metrics here if needed on every change
            # self._emit_metrics()

    def decrement_used_connections(self) -> None:
        """Atomically decrement the count of used connections."""
        with self._lock:
            # Ensure used_connections does not go below 0
            self._metrics.used_connections = max(self._metrics.used_connections - 1, 0)
            # idle_connections is max_size (total available) - used_connections
            self._metrics.idle_connections = self._metrics.max_size - self._metrics.used_connections
            self._metrics.last_updated = datetime.now()
            # Optionally, emit metrics here if needed on every change
            # self._emit_metrics()

    def update_pool_state(self, total: int, used: int) -> None:
        """Update current pool state metrics."""
        with self._lock:
            self._metrics.total_connections = total
            self._metrics.used_connections = used
            self._metrics.idle_connections = total - used
            self._metrics.last_updated = datetime.now()
            
            # Emit metrics to listeners
            self._emit_metrics()
    
    def record_acquisition_attempt(self) -> 'AcquisitionTimer':
        """Record the start of a connection acquisition attempt."""
        with self._lock:
            self._metrics.total_requests += 1
        
        return AcquisitionTimer(self)
    
    def record_acquisition_success(self, wait_time: float) -> None:
        """Record successful connection acquisition."""
        with self._lock:
            self._metrics.successful_acquisitions += 1
            self._metrics.update_wait_time(wait_time)
            self._metrics.last_updated = datetime.now()
            
            # Log if wait time is concerning
            if wait_time > 1.0:
                logger.warning(f"Connection acquisition took {wait_time:.2f}s (threshold: 1.0s)")
    
    def record_acquisition_failure(self, reason: str = "unknown") -> None:
        """Record failed connection acquisition."""
        with self._lock:
            self._metrics.failed_acquisitions += 1
            self._metrics.last_updated = datetime.now()
            
        logger.error(f"Connection acquisition failed: {reason}")
    
    def record_timeout(self) -> None:
        """Record connection acquisition timeout."""
        with self._lock:
            self._metrics.timeouts += 1
            self._metrics.failed_acquisitions += 1
            self._metrics.last_updated = datetime.now()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics as dictionary."""
        with self._lock:
            return self._metrics.to_dict()
    
    def add_listener(self, listener: Callable[[Dict[str, Any]], None]) -> None:
        """Add a metrics listener that will be called on updates."""
        with self._lock:
            self._listeners.append(listener)
    
    def remove_listener(self, listener: Callable[[Dict[str, Any]], None]) -> None:
        """Remove a metrics listener."""
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)
    
    def _emit_metrics(self) -> None:
        """Emit metrics to all registered listeners."""
        metrics = self._metrics.to_dict()
        for listener in self._listeners:
            try:
                listener(metrics)
            except Exception as e:
                logger.error(f"Error in metrics listener: {e}")

    def reset(self) -> None:
        """Reset all metrics to their initial state, preserving pool configuration and start time."""
        with self._lock:
            # Preserve essential configuration that shouldn't be wiped by a reset
            # if it was already set.
            current_min_size = self._metrics.min_size
            current_max_size = self._metrics.max_size
            current_start_time = self._metrics.metrics_start_time
            
            self._metrics = PoolMetrics()
            
            # Restore preserved config
            self._metrics.min_size = current_min_size
            self._metrics.max_size = current_max_size
            self._metrics.metrics_start_time = current_start_time
            self._metrics.last_updated = datetime.now()
            logger.debug("MetricsCollector reset.")
    
    def log_summary(self) -> None:
        """Log a summary of current metrics."""
        metrics = self.get_metrics()
        
        logger.info(
            f"Pool metrics: "
            f"size={metrics['pool.used_connections']}/{metrics['pool.max_size']} "
            f"({metrics['pool.utilization_percent']:.1f}%), "
            f"requests={metrics['pool.total_requests']}, "
            f"success_rate={metrics['pool.success_rate']:.1f}%, "
            f"avg_wait={metrics['pool.wait_time.avg']:.3f}s"
        )


class AcquisitionTimer:
    """Context manager for timing connection acquisitions."""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
        self.start_time = time.time()
        self.success = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        wait_time = time.time() - self.start_time
        
        if exc_type is None and self.success:
            self.collector.record_acquisition_success(wait_time)
        elif exc_type is TimeoutError:
            self.collector.record_timeout()
        else:
            self.collector.record_acquisition_failure(
                str(exc_val) if exc_val else "unknown error"
            )
    
    def mark_success(self):
        """Mark this acquisition as successful."""
        self.success = True


# Global metrics collector instance
_metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return _metrics_collector


# Example metrics listener for logging
def log_metrics_listener(metrics: Dict[str, Any]) -> None:
    """Example listener that logs metrics periodically."""
    if metrics['pool.total_requests'] % 100 == 0:  # Log every 100 requests
        logger.info(
            f"Pool health: {metrics['pool.used_connections']}/{metrics['pool.max_size']} connections, "
            f"{metrics['pool.utilization_percent']:.1f}% utilization, "
            f"avg wait: {metrics['pool.wait_time.recent_avg']:.3f}s"
        )