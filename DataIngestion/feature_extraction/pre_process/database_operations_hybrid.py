"""
Hybrid database operations for preprocessing - Core columns + JSONB
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
import logging
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import execute_values
import json

logger = logging.getLogger(__name__)


def _ensure_preprocessed_features_table_exists(cur, table_name: str):
    """
    Ensures the specified table exists with the correct hybrid schema and is a hypertable.
    If an old schema (JSONB-only 'features' column) is detected, it drops and recreates the table.
    Creates the table and converts it to a hypertable if it doesn't exist initially.
    If table exists with compatible schema but isn't a hypertable, it's converted.
    """
    # Check if table exists initially
    cur.execute("""
        SELECT EXISTS (
            SELECT FROM pg_tables
            WHERE schemaname = 'public' AND tablename = %s
        );
    """, (table_name,))
    table_exists_initially = cur.fetchone()[0]
    
    should_create_new_table = False

    if table_exists_initially:
        logger.debug(f"Table {table_name} exists. Checking schema...")
        # Get existing column names to check for old schema
        cur.execute("""
            SELECT array_agg(column_name::text)
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s;
        """, (table_name,))
        existing_columns_row = cur.fetchone()
        existing_columns = set(existing_columns_row[0]) if existing_columns_row and existing_columns_row[0] else set()
        
        # Heuristic for old schema: 'features' column exists AND 'extended_features' does not.
        is_old_schema = 'features' in existing_columns and 'extended_features' not in existing_columns
        
        if is_old_schema:
            logger.warning(f"Table {table_name} found with old schema (contains 'features' column, missing 'extended_features'). Dropping for recreation.")
            cur.execute(f"DROP TABLE public.{table_name};") # Use f-string for table name, ensure it's safe
            should_create_new_table = True
        else:
            logger.debug(f"Table {table_name} exists with a presumed compatible/new schema.")
            # Table exists and schema is not the old one, check if it's a hypertable
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1
                    FROM timescaledb_information.hypertables
                    WHERE hypertable_schema = 'public' AND hypertable_name = %s
                );
            """, (table_name,))
            is_hypertable = cur.fetchone()[0]
            if not is_hypertable:
                logger.info(f"Table {table_name} exists with compatible schema but is not a hypertable. Converting...")
                # For create_hypertable, table_name is passed as a parameter for psycopg2 to handle quoting
                cur.execute("SELECT create_hypertable(%s, 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '7 days');", (table_name,))
                logger.info(f"Table {table_name} converted to hypertable.")
            else:
                logger.debug(f"Table {table_name} is already a hypertable with a compatible schema.")
    else: # Table did not exist initially
        should_create_new_table = True

    if should_create_new_table:
        action_verb = "Recreating" if table_exists_initially else "Creating"
        logger.info(f"{action_verb} table {table_name} with new hybrid schema...")
        
        # Define the new hybrid schema (matches original successful creation logic)
        create_table_sql = f"""
        CREATE TABLE public.{table_name} (
            time TIMESTAMPTZ NOT NULL,
            era_identifier TEXT NOT NULL,
            air_temp_c REAL,
            relative_humidity_percent REAL,
            co2_measured_ppm REAL,
            light_intensity_umol REAL,
            radiation_w_m2 REAL,
            total_lamps_on REAL,
            dli_sum REAL,
            vpd_hpa REAL,
            heating_setpoint_c REAL,
            co2_status INTEGER,
            source_file TEXT,
            format_type TEXT,
            extended_features JSONB,
            PRIMARY KEY (time, era_identifier)
        );
        """
        cur.execute(create_table_sql) # f-string for table name in CREATE TABLE
        logger.info(f"Table {table_name} {action_verb.lower().replace('ing', 'ed')}.")

        # Convert to TimescaleDB hypertable
        # Pass table_name as parameter for psycopg2 to handle quoting
        create_hypertable_sql = "SELECT create_hypertable(%s, 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '7 days');"
        cur.execute(create_hypertable_sql, (table_name,))
        logger.info(f"Table {table_name} converted to hypertable.")

# Core columns that should be stored as native PostgreSQL columns
CORE_COLUMNS = {
    'air_temp_c', 'relative_humidity_percent', 'co2_measured_ppm',
    'light_intensity_umol', 'radiation_w_m2', 'total_lamps_on',
    'dli_sum', 'vpd_hpa', 'heating_setpoint_c', 'co2_status',
    'source_file', 'format_type'
}

def save_to_timescaledb_hybrid(
    df: pd.DataFrame,
    era_identifier: str,
    db_config: dict,
    table_name: str = 'preprocessed_features',
    batch_size: int = 10000
) -> None:
    """
    Save DataFrame to TimescaleDB using hybrid approach.
    Core columns are stored natively, extended features in JSONB.
    """
    if df.empty:
        logger.warning(f"Empty DataFrame for era {era_identifier}, skipping database save")
        return
    
    logger.info(f"Saving {len(df)} rows to {table_name} for era {era_identifier}")
    
    # Ensure time column exists
    if 'time' not in df.columns:
        if df.index.name == 'time' or isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            if df.columns[0] != 'time':
                df = df.rename(columns={df.columns[0]: 'time'})
        else:
            raise ValueError("No time column or datetime index found")
    
    # Separate core columns from extended features
    core_cols = [col for col in CORE_COLUMNS if col in df.columns]
    extended_cols = [col for col in df.columns if col not in CORE_COLUMNS and col != 'time']
    
    # Prepare data for insertion
    records = []
    for idx, row in df.iterrows():
        # Core column values
        record = [
            row['time'],
            era_identifier
        ]
        
        # Add core column values in order
        for col in ['air_temp_c', 'relative_humidity_percent', 'co2_measured_ppm',
                   'light_intensity_umol', 'radiation_w_m2', 'total_lamps_on',
                   'dli_sum', 'vpd_hpa', 'heating_setpoint_c', 'co2_status',
                   'source_file', 'format_type']:
            if col in df.columns:
                val = row[col]
                # Handle NaN/None values
                if pd.isna(val):
                    record.append(None)
                else:
                    record.append(val)
            else:
                record.append(None)
        
        # Extended features as JSONB
        extended_data = {}
        for col in extended_cols:
            val = row[col]
            if not pd.isna(val):
                # Convert numpy types to Python types
                if isinstance(val, (np.integer, np.floating)):
                    val = val.item()
                elif isinstance(val, np.bool_):
                    val = bool(val)
                extended_data[col] = val
        
        record.append(json.dumps(extended_data) if extended_data else None)
        records.append(record)
    
    # Insert in batches
    columns = ['time', 'era_identifier', 
               'air_temp_c', 'relative_humidity_percent', 'co2_measured_ppm',
               'light_intensity_umol', 'radiation_w_m2', 'total_lamps_on',
               'dli_sum', 'vpd_hpa', 'heating_setpoint_c', 'co2_status',
               'source_file', 'format_type', 'extended_features']
    
    insert_sql = f"""
        INSERT INTO {table_name} ({', '.join(columns)})
        VALUES %s
        ON CONFLICT (time, era_identifier) DO UPDATE SET
            air_temp_c = EXCLUDED.air_temp_c,
            relative_humidity_percent = EXCLUDED.relative_humidity_percent,
            co2_measured_ppm = EXCLUDED.co2_measured_ppm,
            light_intensity_umol = EXCLUDED.light_intensity_umol,
            radiation_w_m2 = EXCLUDED.radiation_w_m2,
            total_lamps_on = EXCLUDED.total_lamps_on,
            dli_sum = EXCLUDED.dli_sum,
            vpd_hpa = EXCLUDED.vpd_hpa,
            heating_setpoint_c = EXCLUDED.heating_setpoint_c,
            co2_status = EXCLUDED.co2_status,
            source_file = EXCLUDED.source_file,
            format_type = EXCLUDED.format_type,
            extended_features = EXCLUDED.extended_features
    """
    
    try:
        with get_db_connection(db_config) as conn:
            with conn.cursor() as cur:
                # Ensure table and hypertable exist before any operations
                _ensure_preprocessed_features_table_exists(cur, table_name)
                conn.commit() # Commit DDL changes from table creation/check

                # Insert in batches
                for i in range(0, len(records), batch_size):
                    batch = records[i:i + batch_size]
                    execute_values(cur, insert_sql, batch, template=None, page_size=100)
                    conn.commit()
                    logger.info(f"Inserted batch {i//batch_size + 1}/{(len(records)-1)//batch_size + 1}")
        
        logger.info(f"Successfully saved {len(df)} rows to {table_name}")
        
    except Exception as e:
        logger.error(f"Error saving to database: {str(e)}")
        raise


def migrate_jsonb_to_hybrid(
    db_config: dict,
    source_table: str = 'preprocessed_features',
    target_table: str = 'preprocessed_features',
    batch_size: int = 10000
) -> None:
    """
    Migrate data from JSONB-only table to hybrid table.
    """
    logger.info(f"Starting migration from {source_table} to {target_table}")
    
    migration_sql = f"""
        INSERT INTO {target_table} (
            time, era_identifier,
            air_temp_c, relative_humidity_percent, co2_measured_ppm,
            light_intensity_umol, radiation_w_m2, total_lamps_on,
            dli_sum, vpd_hpa, heating_setpoint_c, co2_status,
            source_file, format_type, extended_features
        )
        SELECT 
            time,
            era_identifier,
            (features->>'air_temp_c')::REAL,
            (features->>'relative_humidity_percent')::REAL,
            (features->>'co2_measured_ppm')::REAL,
            (features->>'light_intensity_umol')::REAL,
            (features->>'radiation_w_m2')::REAL,
            (features->>'total_lamps_on')::REAL,
            (features->>'dli_sum')::REAL,
            (features->>'vpd_hpa')::REAL,
            (features->>'heating_setpoint_c')::REAL,
            (features->>'co2_status')::INTEGER,
            features->>'source_file',
            features->>'format_type',
            features - ARRAY['air_temp_c', 'relative_humidity_percent', 'co2_measured_ppm',
                           'light_intensity_umol', 'radiation_w_m2', 'total_lamps_on',
                           'dli_sum', 'vpd_hpa', 'heating_setpoint_c', 'co2_status',
                           'source_file', 'format_type'] AS extended_features
        FROM {source_table}
        WHERE time >= %s AND time < %s
        ON CONFLICT (time, era_identifier) DO NOTHING
    """
    
    try:
        with get_db_connection(db_config) as conn:
            with conn.cursor() as cur:
                # Get time range
                cur.execute(f"SELECT MIN(time), MAX(time) FROM {source_table}")
                min_time, max_time = cur.fetchone()
                
                if not min_time or not max_time:
                    logger.warning("No data found in source table")
                    return
                
                # Migrate in time-based batches
                current_time = min_time
                while current_time < max_time:
                    next_time = current_time + pd.Timedelta(days=1)
                    
                    cur.execute(migration_sql, (current_time, next_time))
                    rows_migrated = cur.rowcount
                    conn.commit()
                    
                    logger.info(f"Migrated {rows_migrated} rows for period {current_time} to {next_time}")
                    current_time = next_time
        
        logger.info("Migration completed successfully")
        
    except Exception as e:
        logger.error(f"Error during migration: {str(e)}")
        raise


@contextmanager
def get_db_connection(db_config: dict):
    """Create a database connection context manager."""
    conn = None
    try:
        conn = psycopg2.connect(**db_config)
        yield conn
    finally:
        if conn:
            conn.close()