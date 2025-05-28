"""
Chunked query generator for efficient database operations.

This module provides a generator that yields DataFrame chunks from database
queries, properly managing connection lifecycle and memory usage.
"""

import logging
from typing import Iterator, Optional, Dict, Any
from contextlib import contextmanager

try:
    from ..backend import pd, DataFrame, USE_GPU
except ImportError:
    # Fallback for when running as script
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend import pd, DataFrame, USE_GPU
from .connection import get_connection

logger = logging.getLogger(__name__)


def chunked_query(
    query: str,
    params: Optional[Dict[str, Any]] = None,
    chunksize: int = 10000,
    connection_params: Optional[Dict[str, Any]] = None
) -> Iterator[DataFrame]:
    """
    Execute a database query and yield results in chunks.
    
    This generator owns its database connection for the duration of iteration,
    ensuring proper resource management.
    
    Args:
        query: SQL query to execute
        params: Query parameters (for parameterized queries)
        chunksize: Number of rows per chunk
        connection_params: Database connection parameters
        
    Yields:
        DataFrame chunks from the query results
        
    Example:
        >>> for chunk in chunked_query("SELECT * FROM sensor_data WHERE time > %s", 
        ...                           params={"time": "2024-01-01"}, 
        ...                           chunksize=5000):
        ...     process_chunk(chunk)
    """
    connection = None
    cursor = None
    
    try:
        # Get a new connection for this generator
        connection = get_connection(connection_params)
        
        # Log connection info
        logger.info(f"Executing chunked query with chunksize={chunksize}, backend={USE_GPU and 'GPU' or 'CPU'}")
        
        # Execute query with pandas read_sql
        # This automatically handles chunking and proper resource cleanup
        for chunk in pd.read_sql(
            query,
            connection,
            params=params,
            chunksize=chunksize
        ):
            # Yield the chunk
            yield chunk
            
            # Log progress
            logger.debug(f"Yielded chunk with {len(chunk)} rows")
            
    except Exception as e:
        logger.error(f"Error in chunked_query: {e}")
        raise
        
    finally:
        # Ensure connection is closed
        if connection is not None:
            try:
                connection.close()
                logger.debug("Closed database connection")
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")


@contextmanager
def chunked_query_context(
    query: str,
    params: Optional[Dict[str, Any]] = None,
    chunksize: int = 10000,
    connection_params: Optional[Dict[str, Any]] = None
) -> Iterator[Iterator[DataFrame]]:
    """
    Context manager version of chunked_query for explicit resource management.
    
    Example:
        >>> with chunked_query_context("SELECT * FROM data") as chunks:
        ...     for chunk in chunks:
        ...         process(chunk)
    """
    generator = chunked_query(query, params, chunksize, connection_params)
    try:
        yield generator
    finally:
        # Ensure generator is closed
        generator.close()