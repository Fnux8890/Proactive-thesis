#!/usr/bin/env python
"""
Benchmark script to compare current vs optimized db_utils performance.
Run this to measure actual performance gains.
"""

import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import re
from sqlalchemy import text, MetaData, Table
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_test_data(num_rows: int) -> pd.DataFrame:
    """Generate test data similar to preprocessed greenhouse data."""
    logger.info(f"Generating {num_rows:,} rows of test data...")
    
    base_time = datetime.now()
    return pd.DataFrame({
        'time': [base_time + timedelta(minutes=15*i) for i in range(num_rows)],
        'era_identifier': np.random.choice(['Era1', 'Era2', 'Era3'], num_rows),
        'temperature': np.random.normal(22, 2, num_rows),
        'humidity': np.random.normal(60, 10, num_rows),
        'co2_level': np.random.normal(400, 50, num_rows),
        'light_intensity': np.random.uniform(0, 1000, num_rows),
        'energy_consumption': np.random.exponential(5, num_rows),
        'feature_1': np.random.random(num_rows),
        'feature_2': np.random.random(num_rows),
        'feature_3': np.random.random(num_rows),
    })


def benchmark_connector(connector_class, connector_name: str, **connector_kwargs):
    """Benchmark a database connector implementation."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Benchmarking: {connector_name}")
    logger.info(f"{'='*60}")
    
    results = {}
    connector = None
    
    try:
        # Initialize connector
        start = time.time()
        connector = connector_class(**connector_kwargs)
        results['connection_time'] = time.time() - start
        logger.info(f"Connection time: {results['connection_time']:.3f} seconds")
        
        # Test read performance
        logger.info("\n--- Read Performance ---")
        queries = [
            ("simple_count", "SELECT COUNT(*) FROM pg_tables"),
            ("medium_query", "SELECT * FROM pg_tables LIMIT 100"),
            # Add more realistic queries if you have test tables
        ]
        
        for query_name, query in queries:
            start = time.time()
            
            # Check if connector has fetch_data_to_pandas method
            if hasattr(connector, 'fetch_data_to_pandas'):
                df = connector.fetch_data_to_pandas(query)
            else:
                # Fallback to pandas.read_sql with connector.engine
                if hasattr(connector, 'engine'):
                    # Read in chunks to avoid OOM errors for large results
                    chunk_list = []
                    logger.debug(f"Executing query in chunks: {query_name}")
                    for idx, chunk_df in enumerate(pd.read_sql(query, connector.engine, chunksize=50000)):
                        logger.debug(f"Read chunk {idx} with {len(chunk_df)} rows for query: {query_name}")
                        chunk_list.append(chunk_df)
                    
                    if chunk_list:
                        df = pd.concat(chunk_list, ignore_index=True)
                        logger.debug(f"Concatenated {len(chunk_list)} chunks into DataFrame with {len(df)} rows for query: {query_name}")
                    else:
                        df = pd.DataFrame() # Ensure df is a DataFrame even if query returns no data
                        logger.debug(f"Query {query_name} returned no data.")
                else:
                    logger.warning(f"Connector has no fetch_data_to_pandas or engine attribute, skipping query: {query_name}")
                    continue
                    
            elapsed = time.time() - start
            results[f'read_{query_name}'] = elapsed
            logger.info(f"{query_name}: {elapsed:.3f} seconds ({len(df)} rows)")
        
        # Test write performance
        logger.info("\n--- Write Performance ---")
        test_sizes = [1000, 10000, 50000]
        
        for size in test_sizes:
            df = generate_test_data(size)
            table_name = f'benchmark_test_{size}'
            
            start = time.time()
            try:
                # Always try to pass method and chunksize parameters
                connector.write_dataframe(
                    df, 
                    table_name, 
                    if_exists='replace',
                    method='multi',
                    chunksize=5000,
                    index=False
                )
            except TypeError as e:
                # Catch TypeError related to unsupported parameters
                error_msg = str(e).lower()
                if 'method' in error_msg or 'chunksize' in error_msg or 'index' in error_msg:
                    # Try with minimal parameters for maximum compatibility
                    try:
                        connector.write_dataframe(
                            df, 
                            table_name, 
                            if_exists='replace'
                        )
                    except TypeError as e2:
                        # If still failing with minimal params, it's a real issue
                        logger.error(f"Failed to write dataframe even with minimal parameters: {e2}")
                        raise
                    except Exception as e2:
                        # Any other exception should propagate immediately
                        logger.error(f"Database/network error during write: {e2}")
                        raise
                else:
                    # Re-raise if it's an unrelated TypeError
                    raise
            except Exception as e:
                # Any non-TypeError exception should propagate immediately
                logger.error(f"Unexpected error during dataframe write: {type(e).__name__}: {e}")
                raise
            elapsed = time.time() - start
            rows_per_sec = size / elapsed if elapsed > 0 else 0
            
            results[f'write_{size}_rows'] = elapsed
            results[f'write_{size}_rows_per_sec'] = rows_per_sec
            logger.info(f"{size} rows: {elapsed:.3f} seconds ({rows_per_sec:,.0f} rows/sec)")
            
            # Clean up test table
            try:
                if hasattr(connector, 'execute_query'):
                    connector.execute_query(f"DROP TABLE IF EXISTS {table_name}")
                else:
                    # Fallback for connectors with an engine but no direct execute_query for DROP
                    if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', table_name):
                        try:
                            metadata_obj = MetaData()
                            # As requested, using autoload_with=engine for the Table object
                            table_to_drop = Table(table_name, metadata_obj, autoload_with=connector.engine)
                            # The 'bind' parameter in drop connects it to the engine; checkfirst=True handles "IF EXISTS"
                            table_to_drop.drop(bind=connector.engine, checkfirst=True)
                            logger.debug(f"Table {table_name} dropped using SQLAlchemy Table.drop(autoload_with=engine).")
                        except Exception as e_drop:
                            logger.warning(f"SQLAlchemy Table.drop for '{table_name}' failed: {e_drop}. This might occur if the table does not exist and checkfirst=True was not fully effective, or due to other DB issues.")
                    else:
                        logger.warning(f"Invalid table name for cleanup, skipping DROP: {table_name}")
            except Exception as e:
                logger.warning(f"Failed to drop test table {table_name}: {e}")
        
        # Get performance stats if available
        if hasattr(connector, 'get_performance_stats'):
            logger.info("\n--- Performance Statistics ---")
            stats = connector.get_performance_stats()
            for op, metrics in stats.items():
                logger.info(f"{op}: avg={metrics['avg_time']:.3f}s, count={metrics['count']}")
        
    except Exception as e:
        logger.error(f"Error during benchmark: {e}")
        results['error'] = str(e)
    finally:
        # Ensure connector is properly disposed
        if connector:
            try:
                if hasattr(connector, 'dispose'):
                    connector.dispose()
                elif hasattr(connector, 'close'):
                    connector.close()
                elif hasattr(connector, 'engine'):
                    connector.engine.dispose()
                logger.info(f"Connector for {connector_name} properly disposed")
            except Exception as e:
                logger.warning(f"Error disposing connector: {e}")
    
    return results


def main():
    """Run benchmarks comparing current and optimized implementations."""
    
    # Database connection parameters
    db_params = {
        'user': os.getenv("DB_USER", "postgres"),
        'password': os.getenv("DB_PASSWORD", "postgres"),
        'host': os.getenv("DB_HOST", "localhost"),
        'port': os.getenv("DB_PORT", "5432"),
        'db_name': os.getenv("DB_NAME", "postgres")
    }
    
    logger.info("Starting DB Utils Benchmark")
    logger.info(f"Database: {db_params['user']}@{db_params['host']}:{db_params['port']}/{db_params['db_name']}")
    
    all_results = {}
    
    # Add parent directory of 'feature' package to sys.path
    # Assumes structure: DataIngestion/feature_extraction/benchmark_db_utils.py
    # And: DataIngestion/feature_extraction/feature/db_utils.py
    # So, 'DataIngestion/feature_extraction' (i.e., script_dir) needs to be in path for 'from feature import ...'
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    if current_script_dir not in sys.path: # Avoid duplicate entries
        sys.path.insert(0, current_script_dir)

    # Benchmark current implementation
    try:
        from feature.db_utils import SQLAlchemyPostgresConnector as CurrentConnector
        results = benchmark_connector(
            CurrentConnector,
            "Current Implementation",
            **db_params
        )
        all_results['current'] = results
    except ImportError:
        logger.warning("Could not import current implementation")
    
    # Benchmark optimized implementation
    try:
        from db_utils_optimized import SQLAlchemyPostgresConnector as OptimizedConnector
        results = benchmark_connector(
            OptimizedConnector,
            "Optimized Implementation",
            **db_params,
            pool_size=20,
            max_overflow=40,
            enable_performance_logging=True
        )
        all_results['optimized'] = results
    except ImportError:
        logger.warning("Could not import optimized implementation")
    
    # Compare results
    if len(all_results) == 2:
        logger.info(f"\n{'='*60}")
        logger.info("Performance Comparison")
        logger.info(f"{'='*60}")
        
        current = all_results['current']
        optimized = all_results['optimized']
        
        for key in current:
            if key in optimized and isinstance(current[key], (int, float)):
                # Skip if current value is zero to avoid division by zero
                if current[key] == 0:
                    if optimized[key] == 0:
                        logger.info(f"{key}: Both implementations took 0 time")
                    else:
                        logger.info(f"{key}: Current took 0 time, optimized took {optimized[key]:.3f}s")
                    continue
                
                improvement = (current[key] - optimized[key]) / current[key] * 100
                if improvement > 0:
                    logger.info(f"{key}: {improvement:.1f}% faster")
                else:
                    logger.info(f"{key}: {-improvement:.1f}% slower")
    
    return all_results


if __name__ == '__main__':
    results = main()
    
    # Save results to file
    import json
    import os
    
    # Create output directory if it doesn't exist
    output_dir = 'benchmark_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'benchmark_results_{timestamp}.json')
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nBenchmark complete. Results saved to {output_file}")