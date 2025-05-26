"""
Database connection management for feature extraction.

This module provides thread-safe connection pooling and connection management utilities.
"""

import os
import logging
import threading
import time
from typing import Optional, Dict, Any
from contextlib import contextmanager
from queue import Queue, Empty, Full

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

from .metrics import get_metrics_collector, log_metrics_listener

logger = logging.getLogger(__name__)

# Global connection pool - now using ThreadedConnectionPool for thread safety
_connection_pool: Optional[pool.ThreadedConnectionPool] = None

# Thread lock for protecting global pool mutations
_pool_lock = threading.Lock()

# Thread-local storage for connections when no pool is used
_thread_local = threading.local()

# Configuration for connection acquisition
_acquisition_timeout: float = 30.0  # Default 30 seconds
_use_acquisition_queue: bool = True  # Enable queuing by default

# Queue for managing connection requests with back-pressure
_connection_queue: Optional[Queue] = None
_queue_lock = threading.Lock()


def configure_acquisition(timeout: float = 30.0, use_queue: bool = True) -> None:
    """
    Configure connection acquisition behavior.
    
    Args:
        timeout: Maximum time to wait for a connection (in seconds)
        use_queue: Whether to use a queue for managing connection requests
    """
    global _acquisition_timeout, _use_acquisition_queue
    _acquisition_timeout = timeout
    _use_acquisition_queue = use_queue
    logger.info(f"Configured acquisition: timeout={timeout}s, use_queue={use_queue}")


def _get_default_connection_params() -> Dict[str, Any]:
    """Get default database connection parameters from environment."""
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "database": os.getenv("DB_NAME", "postgres"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "postgres")
    }


def initialize_pool(
    minconn: int = 1,
    maxconn: int = 10,
    connection_params: Optional[Dict[str, Any]] = None
) -> None:
    """
    Initialize the global connection pool in a thread-safe manner.
    
    Args:
        minconn: Minimum number of connections
        maxconn: Maximum number of connections
        connection_params: Database connection parameters
    """
    global _connection_pool, _connection_queue
    
    with _pool_lock:
        # Check again inside the lock to prevent double initialization
        if _connection_pool is not None:
            logger.warning("Connection pool already initialized")
            return
        
        params = connection_params or _get_default_connection_params()
        
        try:
            # Use ThreadedConnectionPool for thread safety
            _connection_pool = pool.ThreadedConnectionPool(
                minconn,
                maxconn,
                **params
            )
            logger.info(f"Initialized thread-safe connection pool with {minconn}-{maxconn} connections")
            
            # Initialize metrics
            metrics = get_metrics_collector()
            metrics.set_pool_config(minconn, maxconn)
            metrics.add_listener(log_metrics_listener)
            
            # Initialize connection queue if enabled
            if _use_acquisition_queue:
                with _queue_lock:
                    _connection_queue = Queue(maxsize=maxconn * 2)  # Allow some queuing
                logger.info(f"Initialized connection request queue with size {maxconn * 2}")
                
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise


def get_connection(connection_params: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None):
    """
    Get a database connection in a thread-safe manner with optional timeout.
    
    If a connection pool is initialized, get from pool.
    Otherwise, create a new connection (using thread-local storage to avoid conflicts).
    
    Args:
        connection_params: Database connection parameters
        timeout: Maximum time to wait for connection (uses global default if None)
        
    Returns:
        Database connection object
        
    Raises:
        TimeoutError: If connection cannot be acquired within timeout
    """
    global _connection_pool, _connection_queue
    
    # Use provided timeout or global default
    acquisition_timeout = timeout if timeout is not None else _acquisition_timeout
    metrics = get_metrics_collector()
    
    # Try to get from pool first
    with _pool_lock: # Ensure thread-safe check of _connection_pool
        pool_is_initialized = _connection_pool is not None

    if pool_is_initialized:
        # Use provided timeout or global default
        current_acquisition_timeout = timeout if timeout is not None else _acquisition_timeout
        metrics = get_metrics_collector()

        with metrics.record_acquisition_attempt() as timer:
            start_time = time.time()
            try:
                # Directly attempt to get a connection from the pool.
                # psycopg2's ThreadedConnectionPool.getconn() will block if no connections are available
                # until one is returned to the pool or the pool is closed.
                # It does not directly support a timeout argument.
                # If a timeout is strictly necessary here, a more complex wrapper or a different pool might be needed.
                # For now, we rely on the pool's blocking behavior.
                conn = _connection_pool.getconn() # This blocks until a connection is available
                timer.mark_success()
                
                # Atomically increment used connections count
                metrics.increment_used_connections()
                
                logger.debug(f"Got connection from pool after {time.time() - start_time:.3f}s")
                return conn
            except pool.PoolError as e:
                # This might occur if the pool is closed while waiting, or other pool-specific errors.
                # Check if we've timed out based on our own timer if current_acquisition_timeout was intended to be used.
                # Note: getconn() itself doesn't timeout, so this check is more for consistency if a timeout wrapper was expected.
                if (time.time() - start_time) > current_acquisition_timeout:
                    logger.error(f"Connection pool acquisition timed out after {current_acquisition_timeout}s: {e}")
                    raise TimeoutError(f"Connection pool acquisition timed out after {current_acquisition_timeout}s") from e
                logger.error(f"Error getting connection from pool: {e}")
                raise # Re-raise the original pool error or a more generic one
            except Exception as e:
                # Catch any other unexpected errors during acquisition
                logger.error(f"Unexpected error getting connection from pool: {e}")
                # Check for timeout here as well, as a fallback
                if (time.time() - start_time) > current_acquisition_timeout:
                    raise TimeoutError(f"Connection pool acquisition timed out during unexpected error after {current_acquisition_timeout}s") from e
                raise

    # No pool exists, create new connection
    params = connection_params or _get_default_connection_params()
    
    try:
        # Store in thread-local to avoid cross-thread connection sharing
        if not hasattr(_thread_local, 'connection') or _thread_local.connection is None or _thread_local.connection.closed:
            _thread_local.connection = psycopg2.connect(**params)
            logger.debug(f"Created new database connection for thread {threading.current_thread().name}")
        return _thread_local.connection
    except Exception as e:
        logger.error(f"Failed to create database connection: {e}")
        raise


def return_connection(conn) -> None:
    """
    Return a connection to the pool or close it in a thread-safe manner.
    
    Args:
        conn: Connection to return
    """
    global _connection_pool
    
    if conn is None:
        return
    
    # Check if pool exists
    with _pool_lock:
        pool_exists = _connection_pool is not None
    
    # Return to pool if available
    if pool_exists:
        try:
            # ThreadedConnectionPool handles its own locking for putconn
            _connection_pool.putconn(conn)
            logger.debug("Returned connection to pool")
            
            # Atomically decrement used connections count
            metrics = get_metrics_collector()
            metrics.decrement_used_connections()
            return
        except Exception as e:
            logger.warning(f"Failed to return connection to pool: {e}")
    
    # Check if this is a thread-local connection
    if hasattr(_thread_local, 'connection') and _thread_local.connection == conn:
        # Don't close thread-local connections, they'll be reused
        logger.debug("Keeping thread-local connection open for reuse")
        return
    
    # Otherwise just close it
    try:
        conn.close()
        logger.debug("Closed database connection")
    except Exception as e:
        logger.warning(f"Failed to close connection: {e}")


@contextmanager
def connection_pool(
    minconn: int = 1,
    maxconn: int = 10,
    connection_params: Optional[Dict[str, Any]] = None
):
    """
    Context manager for connection pool lifecycle with proper restoration.
    
    This context manager ensures the previous pool is always restored,
    even if exceptions occur during pool usage.
    
    Example:
        >>> with connection_pool(maxconn=20):
        ...     # Use connections
        ...     pass
    """
    global _connection_pool
    
    # Save existing pool under lock
    with _pool_lock:
        old_pool = _connection_pool
        _connection_pool = None
    
    try:
        # Initialize new pool
        initialize_pool(minconn, maxconn, connection_params)
        yield _connection_pool
        
    except Exception as e:
        logger.error(f"Error during connection pool usage: {e}")
        # Re-raise with preserved stack trace
        raise e
        
    finally:
        # Always restore the previous pool state
        with _pool_lock:
            # Close current pool if it exists
            if _connection_pool is not None:
                try:
                    _connection_pool.closeall()
                    logger.info("Closed temporary connection pool")
                except Exception as e:
                    logger.error(f"Error closing temporary connection pool: {e}")
                    # Don't re-raise here to ensure restoration happens
            
            # Restore old pool
            _connection_pool = old_pool
            if old_pool is not None:
                logger.debug("Restored previous connection pool")


def close_pool() -> None:
    """Close the global connection pool in a thread-safe manner."""
    global _connection_pool
    
    with _pool_lock:
        if _connection_pool is not None:
            try:
                _connection_pool.closeall()
                _connection_pool = None
                logger.info("Closed global connection pool")
            except Exception as e:
                logger.error(f"Error closing connection pool: {e}")
                # Re-raise to preserve stack trace
                raise
        else:
            logger.debug("Connection pool already closed or not initialized")


def cleanup_thread_local_connection() -> None:
    """
    Clean up thread-local database connection.
    
    This should be called when a thread is done with database operations
    to free resources.
    """
    if hasattr(_thread_local, 'connection') and _thread_local.connection is not None:
        try:
            if not _thread_local.connection.closed:
                _thread_local.connection.close()
                logger.debug(f"Closed thread-local connection for thread {threading.current_thread().name}")
        except Exception as e:
            logger.warning(f"Error closing thread-local connection: {e}")
        finally:
            _thread_local.connection = None


def get_pool_metrics() -> Dict[str, Any]:
    """
    Get current connection pool metrics.
    
    Returns:
        Dictionary containing pool metrics including size, usage, wait times, etc.
    """
    return get_metrics_collector().get_metrics()


def log_pool_summary() -> None:
    """Log a summary of current pool metrics."""
    get_metrics_collector().log_summary()