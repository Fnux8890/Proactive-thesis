import os
import psycopg
import logging
from typing import Optional, Dict, Any

# Assuming DBConnectionConfig is defined in your .config module
# from .config import DBConnectionConfig # This would be the ideal import

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_db_connection(db_config: Optional[Dict[str, Any]] = None) -> Optional[psycopg.Connection]:
    """Establishes a connection to the PostgreSQL/TimescaleDB database.

    Uses connection parameters from the provided db_config dictionary,
    which should ideally conform to DBConnectionConfig structure.
    Falls back to environment variables if db_config is not provided or incomplete.

    Args:
        db_config: A dictionary with keys like 'host', 'port', 'user', 'password', 'dbname'.

    Returns:
        Optional[psycopg.Connection]: A psycopg connection object if successful,
            otherwise None.

    Raises:
        Implicitly relies on environment variable access. Errors during connection
        (e.g., OperationalError) are logged, and None is returned.
    """
    try:
        if db_config and all(k in db_config for k in ['host', 'port', 'user', 'password', 'dbname']):
            host = str(db_config['host'])
            port = str(db_config['port']) # psycopg conn_str expects strings
            user = str(db_config['user'])
            password = str(db_config['password'])
            dbname = str(db_config['dbname'])
            logger.info("Using DB parameters from provided db_config.")
        else:
            logger.warning("db_config not provided or incomplete. Falling back to environment variables.")
            host = os.getenv('DB_HOST', 'localhost')
            port = os.getenv('DB_PORT', '5432')
            user = os.getenv('DB_USER')
            password = os.getenv('DB_PASSWORD')
            dbname = os.getenv('DB_NAME')

        # Check if essential variables are set
        if not all([user, password, dbname]):
            missing_vars = []
            if not user: missing_vars.append("DB_USER or config.user")
            if not password: missing_vars.append("DB_PASSWORD or config.password")
            if not dbname: missing_vars.append("DB_NAME or config.dbname")
            logger.error(f"Database connection failed: Missing connection parameters: {missing_vars}")
            return None

        conn_str = f"host={host} port={port} user={user} password={password} dbname={dbname}"
        logger.debug(f"Attempting DB connection with string: host={host} port={port} user={user} dbname={dbname} password=***")

        conn = psycopg.connect(conn_str, autocommit=False)
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

# Example of how it might be called (for testing, not part of the class typically)
if __name__ == '__main__':
    logger.info("Testing db_connector.py standalone...")
    
    # Scenario 1: Simulating config loaded from src.config.DataProcessingConfig
    print("\n--- Scenario 1: Using a mock config dictionary ---")
    mock_db_conf = {
        "host": os.getenv("DB_HOST_TEST", "localhost"), 
        "port": int(os.getenv("DB_PORT_TEST", 5432)), 
        "user": os.getenv("DB_USER_TEST", "postgres"), 
        "password": os.getenv("DB_PASSWORD_TEST", "postgres"), 
        "dbname": os.getenv("DB_NAME_TEST", "postgres") 
    }
    print(f"Mock config to be passed: { {k: mock_db_conf[k] if k != 'password' else '******' for k in mock_db_conf} }")
    conn1 = None
    try:
        conn1 = get_db_connection(db_config=mock_db_conf)
        if conn1:
            print("Connection 1 successful.")
    except Exception as e:
        print(f"Error with mock config: {e}")
    finally:
        if conn1:
            close_db_connection(conn1)

    print("\n--- Scenario 2: Relying on environment variables (ensure they are set for this test) ---")
    # Make sure DB_USER, DB_PASSWORD, DB_NAME are set in your environment for this to work
    print("If DB_USER, DB_PASSWORD, DB_NAME (and optionally DB_HOST, DB_PORT) are set, this should connect.")
    print(f"Reading: DB_HOST={os.getenv('DB_HOST')}, DB_PORT={os.getenv('DB_PORT')}, DB_USER={os.getenv('DB_USER')}, DB_NAME={os.getenv('DB_NAME')}, DB_PASSWORD=****")
    conn2 = None
    try:
        conn2 = get_db_connection() # No config passed, relies on os.getenv within the function
        if conn2:
            print("Connection 2 successful.")
    except Exception as e:
        print(f"Error with env var fallback: {e}")
    finally:
        if conn2:
            close_db_connection(conn2) 