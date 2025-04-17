import os
import psycopg
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_db_connection() -> Optional[psycopg.Connection]:
    """Establishes a connection to the PostgreSQL/TimescaleDB database.

    Reads connection parameters from environment variables:
        - DB_HOST (default: 'localhost')
        - DB_PORT (default: '5432')
        - DB_USER
        - DB_PASSWORD
        - DB_NAME

    Uses the psycopg3 library for the connection.

    Returns:
        Optional[psycopg.Connection]: A psycopg connection object if successful,
            otherwise None.

    Raises:
        Implicitly relies on environment variable access. Errors during connection
        (e.g., OperationalError) are logged, and None is returned.
    """
    try:
        host = os.getenv('DB_HOST', 'localhost')
        port = os.getenv('DB_PORT', '5432')
        user = os.getenv('DB_USER')
        password = os.getenv('DB_PASSWORD')
        dbname = os.getenv('DB_NAME')

        # Check if essential variables are set
        if not all([user, password, dbname]):
            missing_vars = []
            if not user: missing_vars.append("DB_USER")
            if not password: missing_vars.append("DB_PASSWORD")
            if not dbname: missing_vars.append("DB_NAME")
            logger.error(f"Database connection failed: Missing environment variables: {missing_vars}")
            return None

        conn_str = f"host={host} port={port} user={user} password={password} dbname={dbname}"
        logger.debug(f"Attempting DB connection with string: host={host} port={port} user={user} dbname={dbname} password=***")

        # psycopg.connect uses a connection string or keyword arguments
        conn = psycopg.connect(conn_str, autocommit=False) # Keep autocommit False generally
        logger.info("Database connection established successfully.")
        return conn
    except psycopg.OperationalError as e:
        # Log common operational errors like bad credentials, host not found, db doesn't exist
        logger.error(f"Database connection failed (OperationalError): {e}")
        return None
    except Exception as e:
        logger.exception(f"An unexpected error occurred during DB connection: {e}") # Use exception for stack trace
        return None


def close_db_connection(conn: Optional[psycopg.Connection]):
    """Closes the database connection if it is open.

    Args:
        conn: The psycopg connection object to close.
              Can be None or already closed, in which case the function does nothing.
    """
    try:
        if conn and not conn.closed:
            conn.close()
            logger.info("Database connection closed.")
        elif conn and conn.closed:
            logger.debug("Attempted to close an already closed connection.")
        # If conn is None, do nothing silently
    except Exception as e:
         logger.exception(f"An unexpected error occurred while closing DB connection: {e}") 