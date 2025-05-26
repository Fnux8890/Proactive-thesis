"""
Optimized database utilities for PostgreSQL connections.
This module provides efficient database connectivity with connection pooling,
performance monitoring, and optimized write operations.
"""

import os
import time
import logging
import weakref
import collections
from abc import ABC, abstractmethod
from contextlib import contextmanager, closing
from typing import Union, Optional, Any, Dict, Iterator

from sqlalchemy import create_engine, text, event, URL
from sqlalchemy.engine import Engine
from sqlalchemy.sql.expression import TextClause
from sqlalchemy.pool import NullPool, QueuePool
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd

# Try to import cuDF for type checking
try:
    import cudf
    HAS_CUDF = True
except ImportError:
    cudf = None
    HAS_CUDF = False

# Get logger for this module
logger = logging.getLogger(__name__)


class BaseDBConnector(ABC):
    """
    Abstract base class for database connectors.
    Implementations can either create their own connection or use dependency injection.
    """
    @abstractmethod
    def connect(self) -> Any:
        """Create and return a database connection or engine."""
        pass
    
    @abstractmethod
    def fetch_data_to_pandas(self, query: Union[str, TextClause], chunksize: Optional[int] = None, params: Optional[Dict[str, Any]] = None) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        """Fetch data from database using the provided query and return as pandas DataFrame or iterator of DataFrames."""
        pass

    @abstractmethod
    def write_dataframe(self, df: pd.DataFrame, table_name: str, **kwargs) -> None:
        """Write DataFrame to database table."""
        pass


class SQLAlchemyPostgresConnector(BaseDBConnector):
    """
    Optimized SQLAlchemy connector for PostgreSQL databases with connection pooling
    and performance monitoring.
    """
    _event_listeners_registered: weakref.WeakSet = weakref.WeakSet()  # Track registered listeners by engine ref
    
    def __init__(self, 
                 user: Optional[str] = None, 
                 password: Optional[str] = None, 
                 host: Optional[str] = None, 
                 port: Optional[str] = None, 
                 db_name: Optional[str] = None, 
                 engine: Optional[Engine] = None,
                 pool_size: int = 20,
                 max_overflow: int = 40,
                 pool_timeout: int = 30,
                 pool_recycle: int = 1800,
                 enable_performance_logging: bool = True):
        """
        Initialize the connector with optimized settings.
        
        Note: The connection is NOT established automatically. Call connect() explicitly
        or use the connector as a context manager for automatic connection management.
        
        Args:
            user: PostgreSQL username
            password: PostgreSQL password
            host: PostgreSQL host
            port: PostgreSQL port
            db_name: PostgreSQL database name
            engine: Existing SQLAlchemy engine (if provided, connection params are ignored)
            pool_size: Number of connections to maintain in pool
            max_overflow: Maximum overflow connections allowed
            pool_timeout: Timeout for getting connection from pool
            pool_recycle: Time (in seconds) to recycle connections
            enable_performance_logging: Whether to log performance metrics
        """
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.db_name = db_name
        self.engine = engine
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.enable_performance_logging = enable_performance_logging
        
        # Performance tracking with bounded memory usage
        self._operation_times: Dict[str, collections.deque] = {}
        # Aggregate tracking for chunked operations
        self._chunked_operation_stats: Dict[str, Dict[str, float]] = {}
        # Maximum number of operation times to keep per operation
        self._max_operation_history = 1000
        
        # Note: Removed automatic connection to avoid side effects during construction
        # Users must explicitly call connect() or use as context manager

    def _log_performance(self, operation: str, start_time: float, rows: Optional[int] = None):
        """Log performance metrics for database operations."""
        if not self.enable_performance_logging:
            return
            
        elapsed = time.time() - start_time
        
        # Check if this is a chunked operation
        if '[chunk_' in operation:
            # Extract base operation name (e.g., "fetch_data_to_pandas" from "fetch_data_to_pandas[chunk_1]")
            base_operation = operation.split('[')[0] + '_chunked'
            
            # Initialize aggregate stats if needed
            if base_operation not in self._chunked_operation_stats:
                self._chunked_operation_stats[base_operation] = {
                    'total_time': 0.0,
                    'total_rows': 0,
                    'chunk_count': 0
                }
            
            # Update aggregate stats
            self._chunked_operation_stats[base_operation]['total_time'] += elapsed
            self._chunked_operation_stats[base_operation]['total_rows'] += rows or 0
            self._chunked_operation_stats[base_operation]['chunk_count'] += 1
            
            # Log at DEBUG level for individual chunks
            if rows:
                rows_per_sec = rows / elapsed if elapsed > 0 else 0
                logger.debug(f"Chunk operation '{operation}' completed in {elapsed:.2f} seconds ({rows:,} rows, {rows_per_sec:,.0f} rows/sec)")
        else:
            # Track non-chunked operation times for analysis with bounded memory
            if operation not in self._operation_times:
                self._operation_times[operation] = collections.deque(maxlen=self._max_operation_history)
            self._operation_times[operation].append(elapsed)
            
            msg = f"Operation '{operation}' completed in {elapsed:.2f} seconds"
            if rows:
                rows_per_sec = rows / elapsed if elapsed > 0 else 0
                msg += f" ({rows:,} rows, {rows_per_sec:,.0f} rows/sec)"
            
            logger.info(msg)

    def connect(self) -> Any:
        """
        Create a new SQLAlchemy engine with optimized connection pooling.
        Returns the engine instance.
        """
        if self.engine is not None:
            return self.engine
            
        if not all([self.user, self.password, self.host, self.port, self.db_name]):
            raise ValueError("Cannot create connection: missing database connection parameters")
            
        start_time = time.time()
        
        try:
            # Use SQLAlchemy URL object to safely handle special characters
            db_url = URL.create(
                drivername="postgresql",
                username=self.user,
                password=self.password,
                host=self.host,
                port=int(self.port) if self.port else None,
                database=self.db_name
            )
            
            # Create engine with optimized pooling
            self.engine = create_engine(
                db_url,
                poolclass=QueuePool,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_timeout=self.pool_timeout,
                pool_recycle=self.pool_recycle,
                pool_pre_ping=True,  # Test connections before using
                echo=False,  # Set to True for SQL logging
                future=True  # Use SQLAlchemy 2.0 style
            )
            
            # Add event listener for connection pool monitoring (only once per engine)
            if self.engine not in self._event_listeners_registered:
                @event.listens_for(self.engine, "connect")
                def receive_connect(dbapi_connection, connection_record):
                    logger.debug(f"Pool connection established: {id(dbapi_connection)}")
                
                self._event_listeners_registered.add(self.engine)
                logger.debug(f"Registered event listener for engine {id(self.engine)}")
            
            # Test connection
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
                
            self._log_performance("connect", start_time)
            logger.info(f"Successfully connected to PostgreSQL database: {self.db_name}")
            logger.info(f"Connection pool configured: size={self.pool_size}, max_overflow={self.max_overflow}")
            
            return self.engine
            
        except Exception as e:
            logger.error(f"Error connecting to PostgreSQL: {e}")
            self.engine = None
            raise

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections with performance tracking."""
        if not self.engine:
            self.connect()
            
        conn = self.engine.connect()
        try:
            yield conn
        finally:
            conn.close()

    def fetch_data_to_pandas(self, query: Union[str, TextClause], chunksize: Optional[int] = None, params: Optional[Dict[str, Any]] = None) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        """
        Execute a query and return results as a pandas DataFrame with performance logging.
        
        Args:
            query: SQL query string or SQLAlchemy TextClause
            chunksize: If specified, return an iterator of DataFrames
            params: Optional parameters for SQL parameter binding
            
        Returns:
            pd.DataFrame or iterator of DataFrames if chunksize is specified
        """
        start_time = time.time()
        
        if not self.engine:
            self.connect()
            
        try:
            if chunksize:
                # Ensure engine is initialized
                if not self.engine:
                    self.connect()
                
                # Use self._get_connection() which is already a context manager
                # This makes fetch_data_to_pandas a generator function when chunksize is specified.
                with self._get_connection() as connection:
                    overall_iterator_start_time = time.time()
                    total_rows_processed = 0
                    processed_chunk_count = 0
                    # Initialize before the loop, to time the first chunk's query + processing
                    current_chunk_start_time = time.time() 
                    
                    db_iterator = pd.read_sql_query(
                        sql=query, 
                        con=connection, 
                        chunksize=chunksize, 
                        params=params
                    )

                    try:
                        for chunk in db_iterator:
                            processed_chunk_count += 1
                            num_rows_in_chunk = len(chunk)
                            total_rows_processed += num_rows_in_chunk
                            
                            self._log_performance(
                                f"fetch_data_to_pandas[chunk_{processed_chunk_count}]", 
                                current_chunk_start_time, 
                                rows=num_rows_in_chunk
                            )
                            yield chunk # Yield the chunk, making this a generator
                            # Reset timer for the next chunk's query + processing time
                            current_chunk_start_time = time.time() 

                        # Log success if the iterator was fully exhausted
                        logger.info(
                            f"Chunked read successful: {processed_chunk_count} chunks, "
                            f"{total_rows_processed} total rows, "
                            f"{time.time() - overall_iterator_start_time:.3f}s total for iterator."
                        )
                    except Exception as e:
                        logger.error(f"Error during chunked SQL read: {e}")
                        # Log details even if an error occurs mid-iteration
                        logger.info(
                            f"Chunked read failed after {processed_chunk_count} chunks, "
                            f"{total_rows_processed} total rows, "
                            f"{time.time() - overall_iterator_start_time:.3f}s total for iterator before error."
                        )
                        raise
                    # The 'with self._get_connection() as connection:' block ensures the connection
                    # is closed when the generator is exhausted, garbage collected, or if an error occurs.
            else:
                # For non-chunked reads, use context manager as before
                with self._get_connection() as connection:
                    df = pd.read_sql_query(sql=query, con=connection, params=params)
                    self._log_performance("fetch_data_to_pandas", start_time, rows=len(df))
                    return df
                    
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise

    def write_dataframe(self, 
                       df: pd.DataFrame, 
                       table_name: str,
                       if_exists: str = "append", 
                       index: bool = False,
                       index_label: Optional[str] = None,
                       chunksize: int = 5000,
                       method: str = 'multi',
                       schema: Optional[str] = None) -> None:
        """
        Write DataFrame to database table with optimized performance.
        
        Parameters
        ----------
        df: DataFrame to persist
        table_name: Destination table name
        if_exists: How to behave if table exists ('fail', 'replace', 'append')
        index: Whether to write DataFrame index as a column
        index_label: Column label for index column
        chunksize: Number of rows to write at a time
        method: Insert method - 'multi' for batch inserts (PostgreSQL)
        schema: Database schema to write to (optional)
        """
        # Check if df is either pandas or cuDF DataFrame
        is_valid_df = isinstance(df, pd.DataFrame)
        if HAS_CUDF and not is_valid_df:
            is_valid_df = isinstance(df, cudf.DataFrame)
            
        if not is_valid_df:
            raise TypeError("df must be a pandas or cuDF DataFrame")

        if df.empty:
            logger.warning(f"No data to write to table {table_name}")
            return

        if not self.engine:
            self.connect()

        start_time = time.time()
        rows_written = len(df)
        
        # Convert cuDF to pandas if necessary since SQLAlchemy doesn't support cuDF directly
        df_to_write = df
        if HAS_CUDF and isinstance(df, cudf.DataFrame):
            logger.info("Converting cuDF DataFrame to pandas for database write")
            df_to_write = df.to_pandas()

        try:
            # Use transaction for atomic writes
            with self.engine.begin() as connection:
                # Use optimized parameters for PostgreSQL
                df_to_write.to_sql(
                    name=table_name,
                    con=connection,
                    if_exists=if_exists,
                    index=index,
                    index_label=index_label,
                    chunksize=chunksize,
                    method=method,  # 'multi' for efficient batch inserts
                    schema=schema
                )
            
            self._log_performance(f"write_dataframe[{table_name}]", start_time, rows=rows_written)
            
        except SQLAlchemyError as exc:
            logger.error(f"Database error writing DataFrame to table {table_name}: {exc}")
            raise

    def execute_query(self, query: Union[str, TextClause], params: Optional[dict] = None) -> None:
        """
        Execute a query without returning results (e.g., DDL, INSERT, UPDATE).
        
        WARNING: For raw string queries without parameters, ensure the query is 
        from a trusted source to prevent SQL injection. Always prefer parameterized
        queries when handling user input.
        
        Args:
            query: SQL query to execute (TextClause or parameterized string)
            params: Optional parameters for parameterized queries
        """
        start_time = time.time()
        
        if not self.engine:
            self.connect()
            
        try:
            # Use explicit transaction context
            with self.engine.begin() as connection:
                if isinstance(query, str):
                    if params:
                        # Safe parameterized query
                        query = text(query).bindparams(**params)
                    else:
                        # Log warning for raw string queries
                        logger.warning(
                            "Executing raw string query without parameters. "
                            "Ensure this query is from a trusted source."
                        )
                        query = text(query)
                    
                connection.execute(query)
                # Transaction automatically commits when exiting the context
                
            self._log_performance("execute_query", start_time)
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise

    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get performance statistics for all tracked operations.
        
        Returns:
            Dictionary with operation names as keys and stats as values
        """
        stats = {}
        
        # Add regular operation stats
        for operation, times in self._operation_times.items():
            if times:
                stats[operation] = {
                    'count': len(times),
                    'total_time': sum(times),
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times)
                }
        
        # Add chunked operation aggregate stats
        for operation, agg_stats in self._chunked_operation_stats.items():
            if agg_stats['chunk_count'] > 0:
                avg_time_per_chunk = agg_stats['total_time'] / agg_stats['chunk_count']
                avg_rows_per_chunk = agg_stats['total_rows'] / agg_stats['chunk_count'] if agg_stats['total_rows'] > 0 else 0
                rows_per_sec = agg_stats['total_rows'] / agg_stats['total_time'] if agg_stats['total_time'] > 0 else 0
                
                stats[operation] = {
                    'chunk_count': agg_stats['chunk_count'],
                    'total_time': agg_stats['total_time'],
                    'total_rows': agg_stats['total_rows'],
                    'avg_time_per_chunk': avg_time_per_chunk,
                    'avg_rows_per_chunk': avg_rows_per_chunk,
                    'overall_rows_per_sec': rows_per_sec
                }
        
        return stats

    def close(self) -> None:
        """
        Explicitly close the database connection and dispose of the engine.
        This should be called when the connector is no longer needed.
        """
        if hasattr(self, 'engine') and self.engine:
            self.engine.dispose()
            self.engine = None
            logger.info("Database connection closed and engine disposed")
    
    def dispose(self) -> None:
        """Alias for close() for backwards compatibility."""
        self.close()
    
    def __enter__(self):
        """
        Enter the runtime context for this database connector.
        Establishes connection if not already connected.
        """
        if not self.engine:
            self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the runtime context and close the database connection.
        """
        self.close()
        return False  # Don't suppress exceptions


# Factory function for creating connectors
def create_connector(connector_type: str = "sqlalchemy_postgres", **kwargs) -> BaseDBConnector:
    """
    Factory function to create a database connector of the specified type.
    
    Args:
        connector_type: Type of connector to create
        **kwargs: Parameters to pass to the connector's constructor
        
    Returns:
        BaseDBConnector: An instance of the specified connector type
    """
    if connector_type == "sqlalchemy_postgres":
        return SQLAlchemyPostgresConnector(**kwargs)
    else:
        raise ValueError(f"Unsupported connector type: {connector_type}")


# Example usage and testing
if __name__ == '__main__':
    import numpy as np  # Import numpy for demo code
    
    # Load environment variables
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "postgres")

    print("=== Optimized DB Connector Test ===")
    print(f"Connecting to: {DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

    try:
        # Create optimized connector
        connector = SQLAlchemyPostgresConnector(
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            db_name=DB_NAME,
            pool_size=20,
            max_overflow=40,
            enable_performance_logging=True
        )
        
        # Test query
        print("\n--- Testing fetch operation ---")
        query = "SELECT tablename FROM pg_tables WHERE schemaname = 'public' LIMIT 10"
        df = connector.fetch_data_to_pandas(query)
        print(f"Fetched {len(df)} rows")
        print(df.head())
        
        # Test write operation (if you want to test)
        # print("\n--- Testing write operation ---")
        # test_df = pd.DataFrame({
        #     'id': range(1000),
        #     'value': np.random.random(1000)
        # })
        # connector.write_dataframe(test_df, 'test_table', if_exists='replace')
        
        # Show performance stats
        print("\n--- Performance Statistics ---")
        stats = connector.get_performance_stats()
        for op, metrics in stats.items():
            print(f"\n{op}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.3f}")
                
    except Exception as e:
        print(f"Error during testing: {e}")