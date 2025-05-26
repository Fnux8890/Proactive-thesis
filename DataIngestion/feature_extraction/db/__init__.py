"""
Database utilities module for feature extraction.
"""

from .chunked_query import chunked_query
from .connection import (
    get_connection, 
    return_connection,
    connection_pool,
    initialize_pool,
    close_pool,
    cleanup_thread_local_connection,
    configure_acquisition,
    get_pool_metrics,
    log_pool_summary
)
from .metrics import get_metrics_collector

__all__ = [
    "chunked_query", 
    "get_connection", 
    "return_connection",
    "connection_pool",
    "initialize_pool",
    "close_pool",
    "cleanup_thread_local_connection",
    "configure_acquisition",
    "get_pool_metrics",
    "log_pool_summary",
    "get_metrics_collector"
]