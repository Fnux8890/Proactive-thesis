#!/usr/bin/env -S uv run --isolated
# /// script
# dependencies = ["pandas", "psycopg2-binary"]
# ///

import psycopg2
import pandas as pd
import os # Keep os for potential future use, though not strictly needed for hardcoded values

# Database parameters are now HARDCODED below.
# >>>>>>>>>>>> EDIT THE VALUES BELOW WITH YOUR ACTUAL DATABASE DETAILS <<<<<<<<<<<<
HARDCODED_DB_HOST = "localhost"  # e.g., "localhost" or IP address
HARDCODED_DB_PORT = 5432                   # e.g., 5432
HARDCODED_DB_USER = "postgres" # e.g., "postgres"
HARDCODED_DB_PASSWORD = "postgres" # Your specific password
HARDCODED_DB_NAME = "postgres" # e.g., "postgres"
# >>>>>>>>>>>> FINISH EDITING YOUR DATABASE DETAILS HERE <<<<<<<<<<<<

def get_db_connection():
    """Establishes a connection to the PostgreSQL database using hardcoded parameters."""
    db_params = {
        "host": HARDCODED_DB_HOST,
        "port": HARDCODED_DB_PORT,
        "user": HARDCODED_DB_USER,
        "password": HARDCODED_DB_PASSWORD,
        "dbname": HARDCODED_DB_NAME
    }
    
    print(f"Attempting connection with HARDCODED parameters: host='{db_params['host']}', port={db_params['port']}, user='{db_params['user']}', dbname='{db_params['dbname']}'")
    
    try:
        conn = psycopg2.connect(**db_params)
        # It's good practice to check the actual client encoding the connection settled on.
        # This can be influenced by server settings, environment variables like PGCLIENTENCODING, or driver defaults.
        try:
            print(f"Connection successful. Actual client encoding for this connection: {conn.encoding}")
        except Exception as enc_e:
            print(f"Connection successful. Could not determine client encoding directly: {enc_e}")
        
        # === Encoding settings remain available if needed ===
        # conn.set_client_encoding('LATIN1')
        # print(f"Client encoding MANUALLY SET TO: {conn.encoding}")
        
        return conn
    except psycopg2.OperationalError as e:
        print(f"Error connecting to the database: {e}")
        print("Please ensure database is running and connection details are correctly hardcoded in db_utils.py.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during connection: {e}")
        raise

def load_data_from_db(query: str, conn) -> pd.DataFrame:
    """Loads data from the database using the given query and psycopg2 connection."""
    if conn is None or conn.closed:
        print("Database connection is closed or invalid.")
        return pd.DataFrame()
        
    try:
        # For psycopg2, it's good to know what encoding the connection is using before pandas reads from it.
        # The previous print in get_db_connection covers this.
        print(f"Executing query (db_utils.py with psycopg2) - encoding: {conn.encoding if not conn.closed else 'N/A (closed)'}")
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        print(f"Error loading data from database: {e}")
        raise

if __name__ == '__main__':
    print("Attempting to connect to the database using HARDCODED values (db_utils.py test)...")
    connection = None 
    try:
        print(f"Using hardcoded parameters for test: host='{HARDCODED_DB_HOST}', etc.")
        
        connection = get_db_connection()
        if connection:
            print(f"Connection successful! (db_utils.py test) Client encoding: {connection.encoding}")
            
            test_query = "SELECT * FROM public.sensor_data_merged LIMIT 2;"
            print(f"Executing test query: {test_query}")
            df_test = load_data_from_db(test_query, connection)
            print("Test data loaded successfully (db_utils.py test):")
            print(df_test.head())
        else:
            print("Failed to establish database connection in test.")
            
    except Exception as e:
        print(f"An error occurred during the db_utils.py test: {e}")
    finally:
        if connection is not None and not connection.closed:
            connection.close()
            print("Connection closed (db_utils.py test).")
        elif connection is None:
            print("Connection was not established in test.")
        else: # Connection was closed before finally block
            print("Connection was already closed before final check in test.") 