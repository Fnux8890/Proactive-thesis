import pandas as pd
import logging
from datetime import datetime
from typing import Optional, List, Any
from sqlalchemy.engine import Engine
from sqlalchemy import text

logger = logging.getLogger(__name__)


def retrieve_data(
    engine: Optional[Engine],
    start_time: Optional[datetime],
    end_time: Optional[datetime],
    columns: Optional[List[str]] = None,
    table_name: str = "public.sensor_data_merged"
) -> Optional[pd.DataFrame]:
    """Retrieves data from the specified table, optionally filtering by time.

    Connects to the database using the provided SQLAlchemy engine and fetches
    data based on the specified time window and columns.

    Args:
        engine: An active SQLAlchemy Engine object.
        start_time: The start of the time window (UTC, inclusive).
        end_time: The end of the time window (UTC, exclusive).
        columns: Optional list of column names. If None, retrieves all ('*').
                 'time' is always included if specific columns are requested.
        table_name: The name of the table to query (defaults to sensor_data_merged).

    Returns:
        Optional[pd.DataFrame]: DataFrame with data, indexed by 'time'.
            Returns empty DataFrame if no records match.
            Returns None on critical errors (e.g., invalid engine).
    """
    if engine is None:
        logger.error("Database engine is not valid.")
        return None

    selected_cols_str = "*"
    if columns:
        clean_columns = [col.strip() for col in columns if col and col.strip()]
        if "time" not in [c.lower() for c in clean_columns]:
            clean_columns.insert(0, "time")
        selected_cols_str = ", ".join([f'"{col}"' for col in clean_columns])
        logger.debug(f"Selecting specific columns: {selected_cols_str} from {table_name}")
    else:
        logger.debug(f"No specific columns requested, selecting all columns (*) from {table_name}.")

    query_base = f"SELECT {selected_cols_str} FROM {table_name}"
    where_clauses = []
    params = {}

    if start_time is not None and end_time is not None:
        if start_time >= end_time:
            logger.error(f"Start time ({start_time}) must be before end time ({end_time}). Cannot retrieve data.")
            return pd.DataFrame()
        where_clauses.append("\"time\" >= :start_time AND \"time\" < :end_time")
        params["start_time"] = start_time
        params["end_time"] = end_time
        logger.info(f"Retrieving data from {table_name} between {start_time} and {end_time}")
    elif start_time is not None or end_time is not None:
        logger.warning(f"Partial time window specified for {table_name}. Both start and end time required for filtering. Retrieving all data.")
        logger.info(f"Retrieving all data from {table_name}.")
    else:
        logger.info(f"Retrieving all data from {table_name}.")

    if where_clauses:
        query_final_str = f"{query_base} WHERE {' AND '.join(where_clauses)} ORDER BY \"time\" ASC;"
    else:
        query_final_str = f"{query_base} ORDER BY \"time\" ASC;"

    try:
        logger.info(f"Executing query: {query_final_str}")
        df = pd.read_sql_query(sql=text(query_final_str), con=engine, params=params)
        logger.info(f"Successfully fetched {len(df)} records from {table_name}.")

        if df.empty:
            return pd.DataFrame(columns=columns if columns else [])

        if 'time' in df.columns:
            try:
                df['time'] = pd.to_datetime(df['time'], errors='coerce')
                original_len = len(df)
                df = df.dropna(subset=['time'])
                if len(df) < original_len:
                    logger.warning(f"Dropped {original_len - len(df)} rows due to unparseable 'time' values.")
                df = df.set_index('time', drop=False)
                df = df.sort_index()
            except Exception as time_e:
                logger.error(f"Error converting 'time' column or setting index: {time_e}")
                return None
        else:
            logger.warning("'time' column not found in query results, cannot set DatetimeIndex.")
        
        return df

    except Exception as e:
        logger.exception(f"An unexpected error occurred during data retrieval from {table_name}: {e}")
        return None

# Example of how it might be called (for testing)
if __name__ == '__main__':
    from .db_connector import get_db_engine, close_db_connection

    logging.basicConfig(level=logging.DEBUG)
    logger.info(f"Testing retriever.py standalone...")

    print(f"Attempting to get engine via db_connector (relies on its config/env var logic)...")
    test_engine = None
    try:
        test_engine = get_db_engine()

        if test_engine:
            logger.info("Engine obtained. Testing data retrieval...")
            
            start_date_test = datetime(2014, 1, 1, 0, 0, 0)
            end_date_test = datetime(2014, 1, 2, 0, 0, 0)
            df_test1 = retrieve_data(test_engine, start_date_test, end_date_test)
            if df_test1 is not None:
                print(f"\nTest 1: Retrieved {len(df_test1)} rows from {start_date_test} to {end_date_test}.")
                print(df_test1.head())

        else:
            logger.error("Failed to obtain database engine for testing.")

    except Exception as e_main:
        logger.exception(f"Error in retriever.py __main__ block: {e_main}")
    finally:
        if test_engine:
            test_engine.dispose()
            logger.info("Test engine disposed.") 