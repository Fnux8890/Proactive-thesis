#!/usr/bin/env -S uv run --isolated
# /// script
# dependencies = ["pandas", "psycopg2-binary", "sqlalchemy"]
# ///

import psycopg2 # Still needed by SQLAlchemy for postgresql+psycopg2 dialect
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# Database parameters are now HARDCODED below.
# >>>>>>>>>>>> EDIT THE VALUES BELOW WITH YOUR ACTUAL DATABASE DETAILS <<<<<<<<<<<<
HARDCODED_DB_HOST = "localhost"  # e.g., "localhost" or IP address
HARDCODED_DB_PORT = 5432                   # e.g., 5432
HARDCODED_DB_USER = "postgres" # e.g., "postgres"
HARDCODED_DB_PASSWORD = "postgres" # Your specific password
HARDCODED_DB_NAME = "postgres" # e.g., "postgres"
# >>>>>>>>>>>> FINISH EDITING YOUR DATABASE DETAILS HERE <<<<<<<<<<<<

def get_db_engine():
    """Creates and returns a SQLAlchemy engine using hardcoded PostgreSQL parameters."""
    db_params = {
        "host": HARDCODED_DB_HOST,
        "port": HARDCODED_DB_PORT,
        "user": HARDCODED_DB_USER,
        "password": HARDCODED_DB_PASSWORD,
        "dbname": HARDCODED_DB_NAME
    }
    
    db_url = f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}"
    
    print(f"Attempting to create SQLAlchemy engine for URL: postgresql+psycopg2://{db_params['user']}@***:{db_params['port']}/{db_params['dbname']}")
    
    try:
        engine = create_engine(db_url)
        # Test the connection - typically, engine creation is lazy.
        # A simple way to test is to try to connect and execute a trivial query.
        with engine.connect() as connection:
            connection.execute(text("SELECT 1")) # sqlalchemy.text is used for literal SQL
        print("SQLAlchemy engine created and connection tested successfully.")
        return engine
    except SQLAlchemyError as e:
        print(f"Error creating SQLAlchemy engine or testing connection: {e}")
        print("Please ensure database is running and connection details are correctly hardcoded in db_utils.py.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during SQLAlchemy engine creation: {e}")
        raise

def load_data_from_db(query: str, engine) -> pd.DataFrame:
    """Loads data from the database using the given query and SQLAlchemy engine."""
    if engine is None:
        print("SQLAlchemy engine is None.")
        return pd.DataFrame()
        
    try:
        print(f"Executing query (db_utils.py with SQLAlchemy engine: {engine.url.drivername})")
        df = pd.read_sql_query(sql=text(query), con=engine)
        return df
    except SQLAlchemyError as e:
        print(f"SQLAlchemyError loading data from database: {e}")
        raise
    except Exception as e:
        print(f"Error loading data from database with SQLAlchemy: {e}")
        raise

if __name__ == '__main__':
    print("Attempting to connect to the database using SQLAlchemy (db_utils.py test)...")
    engine = None 
    try:
        print(f"Using hardcoded parameters for test: host='{HARDCODED_DB_HOST}', etc.")
        
        engine = get_db_engine()
        if engine:
            print(f"SQLAlchemy engine created successfully for test! Driver: {engine.url.drivername}")
            
            test_query = "SELECT * FROM public.sensor_data_merged LIMIT 2;"
            print(f"Executing test query: {test_query}")
            df_test = load_data_from_db(test_query, engine)
            print("Test data loaded successfully (db_utils.py with SQLAlchemy test):")
            print(df_test.head())
        else:
            print("Failed to create SQLAlchemy engine in test.")
            
    except Exception as e:
        print(f"An error occurred during the db_utils.py SQLAlchemy test: {e}")
    finally:
        if engine is not None:
            engine.dispose()
            print("SQLAlchemy engine disposed (db_utils.py test).")
        else:
            print("SQLAlchemy engine was not established in test.") 