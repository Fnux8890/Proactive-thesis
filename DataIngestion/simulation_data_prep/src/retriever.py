import pandas as pd
import logging
from datetime import datetime
from typing import Optional, Any
import psycopg
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)


def retrieve_data(
    conn: Optional[psycopg.Connection],
    start_time: Optional[datetime],
    end_time: Optional[datetime],
    columns: Optional[list[str]] = None,
) -> Optional[pd.DataFrame]:
    """Retrieves data from the sensor_data table, optionally filtering by time.

    Connects to the database using the provided psycopg connection and fetches
    data based on the specified time window and columns. If start_time and
    end_time are None, retrieves all data. Converts the results fetched as
    dictionaries into a pandas DataFrame.

    Args:
        conn: An active psycopg database connection object.
        start_time: The start of the time window (UTC, inclusive).
                    If None (along with end_time=None), no lower time bound is applied.
        end_time: The end of the time window (UTC, exclusive).
                  If None (along with start_time=None), no upper time bound is applied.
        columns: An optional list of column names to retrieve. If None or empty,
                 retrieves all columns ('SELECT *'). The 'time' column is always
                 added if specific columns are requested and it's not present.

    Returns:
        Optional[pd.DataFrame]: A pandas DataFrame containing the requested data,
            correctly indexed by 'time' (converted to datetime objects).
            Returns an empty DataFrame if no records match the criteria.
            Returns None if a database connection error occurs, the connection is
            invalid, or if start_time and end_time define an invalid range.

    Raises:
        Catches psycopg.Error and general Exceptions during query execution,
        logs them, and returns None.
    """
    if not conn or conn.closed:
        logger.error("Database connection is not valid or closed.")
        return None

    # Determine selected columns
    if not columns:
        select_cols = "*"
        logger.debug("No specific columns requested, selecting all columns (*).")
    else:
        # Ensure 'time' column is always included if specific columns are requested
        # Make check case-insensitive and handle potential extra whitespace
        clean_columns = [col.strip() for col in columns if col and col.strip()]
        if "time" not in [c.lower() for c in clean_columns]:
            logger.debug("'time' column not in requested list, adding it.")
            clean_columns.insert(0, "time")
        # Quote column names to handle potential spaces or special characters safely
        select_cols = ", ".join([f'"{col}"' for col in clean_columns])
        logger.debug(f"Selecting specific columns: {select_cols}")

    # Build the query dynamically
    query_base = f"SELECT {select_cols} FROM sensor_data"
    where_clauses = []
    params = []

    if start_time is not None and end_time is not None:
        if start_time >= end_time:
             logger.error(f"Start time ({start_time}) must be before end time ({end_time}). Cannot retrieve data.")
             return None # Return None for invalid range
        where_clauses.append("\"time\" >= %s AND \"time\" < %s") # Quote time column
        params.extend([start_time, end_time])
        logger.info(f"Retrieving data from {start_time} to {end_time}")
    elif start_time is not None or end_time is not None:
        # Allow filtering by only start OR end time?
        # For now, require both or neither as per previous logic.
        logger.warning("Both start_time and end_time must be provided for time window filtering. Retrieving all data.")
        logger.info("Retrieving all data from the table.")
    else:
        # Both start_time and end_time are None
        logger.info("Retrieving all data from the table.")

    # Construct the final query
    if where_clauses:
        query = f"{query_base} WHERE {' AND '.join(where_clauses)} ORDER BY \"time\" ASC;" # Quote time
    else:
        # No WHERE clauses, fetch all data ordered by time
        query = f"{query_base} ORDER BY \"time\" ASC;" # Quote time

    try:
        logger.info(f"Executing query: {query}") # Params logging might be too verbose/sensitive

        # Use a cursor with dict_row factory
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(query, params=params)
            results: list[dict[str, Any]] = cur.fetchall()
            logger.info(f"Successfully fetched {len(results)} records as dictionaries.")

            # Get column names from cursor description; needed esp. if results are empty
            col_names = [desc[0] for desc in cur.description] if cur.description else []

            if not results:
                # Return empty DataFrame with correct columns if no results
                logger.info("Query returned no records.")
                return pd.DataFrame([], columns=col_names)
            else:
                # Convert list of dictionaries to DataFrame
                df = pd.DataFrame(results, columns=col_names) # Ensure column order matches query
                # Ensure the 'time' column is datetime type and set as index
                if 'time' in df.columns:
                    try:
                         # Use errors='coerce' to handle potential parsing issues
                         df['time'] = pd.to_datetime(df['time'], errors='coerce')
                         # Drop rows where time could not be parsed
                         original_len = len(df)
                         df = df.dropna(subset=['time']) 
                         if len(df) < original_len:
                             logger.warning(f"Dropped {original_len - len(df)} rows due to unparseable 'time' values.")
                         df = df.set_index('time', drop=False) # Keep time column
                         df = df.sort_index() # Ensure index is sorted
                    except Exception as time_e:
                         logger.error(f"Error converting 'time' column to datetime or setting index: {time_e}")
                         # Decide on behavior: return None, or return df without time index?
                         # Returning None might be safer if time index is critical downstream.
                         return None
                else:
                    logger.warning("'time' column not found in query results, cannot set DatetimeIndex.")

                return df

    except psycopg.Error as db_e:
        logger.error(f"Database error during data retrieval: {db_e}")
        return None
    except Exception as e:
        logger.exception(f"An unexpected error occurred during data retrieval: {e}")
        return None 