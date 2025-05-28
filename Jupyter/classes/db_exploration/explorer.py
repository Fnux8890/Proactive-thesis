import pandas as pd
# import psycopg2
import psycopg # Import psycopg (v3)
import numpy as np # For describe include
from psycopg import errors # Import errors module
import time # Import time for sleep

class DatabaseExplorer:
    """Handles connection and exploration queries for database tables using psycopg (v3)."""

    def __init__(self, db_config):
        """Initializes the explorer and establishes a database connection."""
        self.db_config = db_config
        self.conn: psycopg.Connection | None = None # Type hint for psycopg connection
        self.last_merged_sample = pd.DataFrame() # Initialize attribute
        self.connect()

    def connect(self):
        """Establishes the database connection using psycopg (v3)."""
        if self.conn and not self.conn.closed:
            print("Connection already established.")
            return
        try:
            # Build connection string - preferred way for psycopg v3 often
            conn_str = (
                f"host='{self.db_config.host}' "
                f"port='{self.db_config.port}' "
                f"user='{self.db_config.user}' "
                f"password='{self.db_config.password}' " # Ensure password has no chars breaking the string itself
                f"dbname='{self.db_config.dbname}'"
            )
            # Debug print for the connection string (mask password)
            debug_conn_str = (
                f"host='{self.db_config.host}' "
                f"port='{self.db_config.port}' "
                f"user='{self.db_config.user}' "
                f"password='*****' "
                f"dbname='{self.db_config.dbname}'"
            )
            print(f"--- Debug: Attempting psycopg v3 connection with string: ---")
            print(debug_conn_str)
            print("------------------------------------------------------------")

            self.conn = psycopg.connect(conn_str, autocommit=False)

            print("Database connection established successfully using psycopg v3.")
        except UnicodeDecodeError as ude:
            # This error is less likely with conn string but keep for consistency
            print(f"Database connection failed due to an encoding error: {ude}")
            print("----> Check connection parameters (esp. password) for incompatible characters.")
            self.conn = None
        except psycopg.OperationalError as db_err:
            # Catch specific psycopg operational errors
            print(f"Database connection failed (psycopg OperationalError): {db_err}")
            print("----> Check database server status, network, credentials, dbname.")
            self.conn = None
        except Exception as e:
            print(f"An unexpected error occurred during database connection: {e}")
            self.conn = None

    def _run_query(self, query: str, params=None) -> pd.DataFrame | None:
        """
        Helper method to run a query.
        Returns a DataFrame on success, None on failure (especially if table not found).
        """
        if not self.conn or self.conn.closed:
            print("Query skipped: No active database connection.")
            return None
        try:
            # pandas read_sql_query works with psycopg v3 connections too
            # Handle potential warnings about DBAPI2 usage
            # with warnings.catch_warnings():
            #     warnings.filterwarnings("ignore", category=UserWarning, message='.*support*SQLAlchemy*.')
            df = pd.read_sql_query(query, self.conn, params=params)
            return df
        except errors.UndefinedTable as ut_err:
            # Catch specific error for missing table
            # Make this message less alarming as we might retry
            print(f"Query info: Relation (table) does not appear to exist yet. Details: {ut_err}")
            return None
        except Exception as e:
            print(f"Error running query '{query[:60]}...': {e}")
            return None # Return None for any other query error

    def explore_table(self, table_name: str):
        """Runs standard exploration queries for a given table, retrying if table not found initially."""
        print(f"\n--- Exploring Table: {table_name} ---")
        if not self.conn or self.conn.closed:
            print("Cannot explore table, no database connection.")
            return

        # --- Query to list all tables (runs only once) --- 
        if not hasattr(self, '_printed_public_tables'):
            print("Listing all tables found in schema 'public' by this connection...")
            list_tables_query = "SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname = 'public' ORDER BY tablename;"
            all_tables_df = self._run_query(list_tables_query)
            if all_tables_df is not None:
                if not all_tables_df.empty:
                    print(all_tables_df['tablename'].to_list()) # Print as a list
                else:
                    print("(No tables found in public schema)")
            else:
                print("(Error querying pg_catalog for table list)")
            print("---")
            self._printed_public_tables = True

        # --- Catalog check for the SPECIFIC table WITH RETRY --- 
        max_retries = 5
        retry_delay = 5 # seconds
        table_found_in_catalog = False

        for attempt in range(max_retries):
            print(f"Checking pg_catalog for table 'public.{table_name}' (Attempt {attempt + 1}/{max_retries})...")
            catalog_query = "SELECT 1 FROM pg_catalog.pg_tables WHERE schemaname = 'public' AND tablename = %s LIMIT 1;"
            catalog_df = self._run_query(catalog_query, params=(table_name,))

            if catalog_df is None:
                # Error during query itself
                print("Error querying pg_catalog.")
                # Decide if we should stop or retry? Let's retry for now.
            elif catalog_df.empty:
                print(f"Table 'public.{table_name}' NOT FOUND in pg_catalog.")
                # Table not found, wait before retrying (unless it's the last attempt)
            else:
                print(f"Table 'public.{table_name}' FOUND in pg_catalog.")
                table_found_in_catalog = True
                break # Exit loop if found

            # Wait before next retry, unless it was the last attempt
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                 print(f"Table 'public.{table_name}' not found in catalog after {max_retries} attempts.")
        print("---") # Separator after catalog check block
        # --- End catalog check --- 

        # Proceed only if table was found in catalog
        if not table_found_in_catalog:
             print(f"Skipping further exploration as table 'public.{table_name}' was not found in catalog.")
             # Clear any potentially stale sample data if this was the merged table
             if table_name == 'sensor_data_merged':
                 self.last_merged_sample = pd.DataFrame()
             return

        # --- Table exists, proceed with exploration --- 
        table_exists = True # We confirmed it exists in catalog
        query_count = f"SELECT COUNT(*) FROM public.{table_name};"
        count_df = self._run_query(query_count)
        print("Total rows:")
        if count_df is not None:
            print(count_df)
        else:
            print("(Query failed, see previous error)") # Should ideally not happen if catalog check passed
        print("\n---\n")

        # Schema
        query_schema = f"""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = '{table_name}';
        """
        schema_df = self._run_query(query_schema)
        print("Schema:")
        if schema_df is not None:
            print(schema_df)
        else:
            print("(Could not retrieve schema)")
        print("\n---\n")

        # Time range
        print("Time range:")
        try:
            query_time_range = f"SELECT MIN(time) AS min_time, MAX(time) AS max_time FROM public.{table_name} WHERE time IS NOT NULL;"
            time_range_df = self._run_query(query_time_range)
            if time_range_df is not None:
                if not time_range_df.empty and time_range_df.iloc[0]['min_time'] is not None:
                    print(time_range_df)
                else:
                    print("(Table might be empty or 'time' column has only NULLs)")
            else:
                print("(Query failed, see previous error)")
        except Exception as e:
            print(f"(Error checking time range: {e})")
        print("\n---\n")

        # Sample data
        print("Sample (latest 10 rows by time):")
        try:
            query_sample = f"SELECT * FROM public.{table_name} ORDER BY time DESC LIMIT 10;"
            sample_df = self._run_query(query_sample)
            if sample_df is not None:
                print(sample_df)
                if table_name == 'sensor_data_merged':
                    self.last_merged_sample = sample_df.copy()
            else:
                 print("(Query failed, see previous error)")
                 if table_name == 'sensor_data_merged':
                     self.last_merged_sample = pd.DataFrame()

        except Exception as e:
            print(f"(Error getting sample: {e})")
            if table_name == 'sensor_data_merged':
                self.last_merged_sample = pd.DataFrame()
        print("\n---")

    def check_merged_data_quality(self):
        """Runs specific data quality checks on the sensor_data_merged table."""
        table_name = 'sensor_data_merged'
        print(f"\n--- Quality Checks: {table_name} ---")

        # Determine if we need to load a sample
        sample_df_for_check = None
        if hasattr(self, 'last_merged_sample') and not self.last_merged_sample.empty:
            sample_df_for_check = self.last_merged_sample
            print("Performing checks on the previously loaded sample:")
        else:
            print("No pre-loaded sample found, attempting to load fresh sample for quality checks...")
            if not self.conn or self.conn.closed:
                print("Cannot perform checks, no database connection.")
                return
            # Try loading a sample specifically for checks
            query_sample_check = f"SELECT * FROM public.{table_name} ORDER BY time DESC LIMIT 100;"
            sample_df_for_check = self._run_query(query_sample_check)
            if sample_df_for_check is None:
                 print("Could not load sample data for quality checks (Table likely missing or query failed).")
                 return # Exit if loading failed
            elif sample_df_for_check.empty:
                 print("Loaded empty sample for quality checks.")
                 # Proceed with checks on empty dataframe if desired, or return
            else:
                print(f"Loaded sample of {len(sample_df_for_check)} rows for quality checks.")

        # --- Perform checks on sample_df_for_check --- 
        # Null counts
        print("\nNull counts in columns (from sample):")
        print(sample_df_for_check.isnull().sum())
        print("\n---\n")

        # Data types
        print("Data types of columns (from sample):")
        print(sample_df_for_check.dtypes)
        print("\n---\n")

        # Descriptive statistics
        print("Descriptive statistics for numeric columns (from sample):")
        try:
            # Only print if there are numeric columns
            numeric_df = sample_df_for_check.select_dtypes(include=np.number)
            if not numeric_df.empty:
                print(numeric_df.describe())
            else:
                print("(No numeric columns found in sample)")
        except Exception as e:
             print(f"Could not calculate descriptive statistics: {e}")
        print("\n---")

    def close_connection(self):
        """Closes the database connection if it is open."""
        if self.conn and not self.conn.closed:
            try:
                self.conn.close()
                print("Database connection closed.")
            except Exception as e:
                print(f"Error closing connection: {e}")
        else:
            print("Connection already closed or never established.")
        self.conn = None

    def __del__(self):
        """Ensure connection is closed when the object is destroyed."""
        self.close_connection() 