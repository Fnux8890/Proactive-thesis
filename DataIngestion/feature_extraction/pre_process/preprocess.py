import pandas as pd
import json
import os
import sys
from pathlib import Path # Added for robust path construction
from db_utils import SQLAlchemyPostgresConnector # Assuming db_utils.py is in the same directory
from processing_steps import OutlierHandler, ImputationHandler, DataSegmenter # Assuming processing_steps.py is in the same directory
from sklearn.preprocessing import MinMaxScaler # For Normalization/Scaling
import joblib # For saving scalers
from typing import Any, Tuple, Dict, List # Added Tuple, Dict, List
from sqlalchemy import text
from sqlalchemy import create_engine

# Define paths (these will correspond to paths inside the Docker container)
CONFIG_PATH = os.getenv('APP_CONFIG_PATH', '/app/config/data_processing_config.json')
OUTPUT_DATA_DIR = Path(os.getenv('APP_OUTPUT_DIR', '/app/data/output')) # Use Path object
SCALERS_DIR = OUTPUT_DATA_DIR / "scalers" # Directory to save scalers
SUMMARY_REPORT_FILENAME_TEMPLATE = "preprocessing_summary_report_{era_identifier}.txt"
OUTPUT_FILENAME_TEMPLATE = "{era_identifier}_processed_segment_{segment_num}.parquet"

# Database connection details (can be overridden by config or remain as env vars)
DB_USER_ENV = os.getenv("DB_USER", "postgres")
DB_PASSWORD_ENV = os.getenv("DB_PASSWORD", "postgres")
DB_HOST_ENV = os.getenv("DB_HOST", "db") 
DB_PORT_ENV = os.getenv("DB_PORT", "5432")
DB_NAME_ENV = os.getenv("DB_NAME", "postgres")
# DB_TABLE_ENV = os.getenv("DB_TABLE", "sensor_data_merged") # Table will come from era_config

def load_config(path: Path) -> Dict[str, Any]:
    print(f"Loading configuration from {path}...")
    try:
        with open(path, 'r') as f:
            config = json.load(f)
        print("Configuration loaded successfully.")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {path}. Exiting.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {path}. Exiting.")
        sys.exit(1)

def fetch_source_data(era_identifier: str, era_config: Dict[str, Any], global_config: Dict[str, Any], engine=None) -> pd.DataFrame:
    print(f"\n--- Fetching Source Data for Era: {era_identifier} from Database ---")
    
    db_table = era_config.get("db_table")
    start_date = era_config.get("start_date")
    end_date = era_config.get("end_date")
    time_col = global_config.get("common_settings", {}).get("time_col", "time") # Get time_col from common_settings

    if not all([db_table, start_date, end_date, time_col]):
        print(f"Error: Missing db_table, start_date, end_date, or common time_col in config for era '{era_identifier}'.")
        return pd.DataFrame()

    # Use DB connection details from global_config, allowing hardcoded or env var based
    db_conn_settings = global_config.get("database_connection", {})
    db_user = db_conn_settings.get("user", DB_USER_ENV)
    db_password = db_conn_settings.get("password", DB_PASSWORD_ENV)
    db_host = db_conn_settings.get("host", DB_HOST_ENV)
    db_port = db_conn_settings.get("port", DB_PORT_ENV)
    db_name = db_conn_settings.get("dbname", DB_NAME_ENV)

    db_connector = None
    try:
        # If we have an engine, use dependency injection
        if engine is not None:
            print(f"Using existing database engine for Era '{era_identifier}'")
            db_connector = SQLAlchemyPostgresConnector(engine=engine)
        else:
            # Otherwise create a new connector with credentials
            db_connector = SQLAlchemyPostgresConnector(
                user=db_user,
                password=db_password,
                host=db_host,
                port=db_port,
                db_name=db_name
            )
        # The connect() method is called in __init__ of SQLAlchemyPostgresConnector if needed
    except Exception as e:
        print(f"Failed to initialize database connector for Era '{era_identifier}': {e}. Exiting.")
        # No sys.exit here, allow main loop to handle or report for multiple eras
        return pd.DataFrame()

    query = f'''
        SELECT *
        FROM {db_table} 
        WHERE "{time_col}" >= :start_date AND "{time_col}" <= :end_date
        ORDER BY "{time_col}" ASC;
    '''
    params = {'start_date': start_date, 'end_date': end_date}
    
    print(f"Executing query for Era '{era_identifier}': from {start_date} to {end_date} on table {db_table}")
    try:
        from sqlalchemy import text as sql_text # Avoid conflict with other 'text' imports if any
        df = db_connector.fetch_data_to_pandas(sql_text(query).bindparams(**params))
        print(f"Data fetched successfully for Era '{era_identifier}'. Shape: {df.shape}")
        if df.empty:
            print(f"Warning: Fetched DataFrame is empty for Era '{era_identifier}'. Check table, query, and date ranges.")
        return df
    except Exception as e:
        print(f"Failed to fetch data from database for Era '{era_identifier}': {e}. Returning empty DataFrame.")
        return pd.DataFrame()
    # SQLAlchemy engine disposes connections automatically, no explicit close needed for engine scope


def sort_and_prepare_df(df: pd.DataFrame, config: Dict[str, Any], era_identifier: str) -> pd.DataFrame:
    if df.empty: # Added safety check
        return df

    time_col_name = config.get('common_settings',{}).get('time_col', 'time')
    if time_col_name not in df.columns:
        print(f"Error (Era: {era_identifier}): Time column '{time_col_name}' not found in fetched data.")
        return pd.DataFrame()
    
    df[time_col_name] = pd.to_datetime(df[time_col_name], errors='coerce', utc=True)
    df.dropna(subset=[time_col_name], inplace=True)
    df = df.sort_values(by=time_col_name).reset_index(drop=True)
    
    id_col_name = config.get('common_settings',{}).get('id_col', 'entity_id')
    # Use era_identifier for default_id_value to make it specific
    default_id_value = config.get('era_definitions',{}).get(era_identifier,{}).get('description', era_identifier) 
    if id_col_name not in df.columns:
        print(f"(Era: {era_identifier}) '{id_col_name}' not found. Adding default ID column with value: '{default_id_value}'.")
        df[id_col_name] = default_id_value

    # Explicit type conversion for boolean/status columns to 0/1 int
    era_specific_config = config.get('era_definitions', {}).get(era_identifier, {})
    boolean_cols_to_int = era_specific_config.get("boolean_columns_to_int", 
                                            config.get('common_settings',{}).get("boolean_columns_to_int", [])
                                           )
    if boolean_cols_to_int:
        print(f"(Era: {era_identifier}) Converting boolean columns to int (0/1): {boolean_cols_to_int}")
        for col in boolean_cols_to_int:
            if col in df.columns:
                # Robust conversion for boolean-like columns
                # Map common string representations of True/False and 0/1 to integers
                # Also handle cases where it might already be bool or numeric 0/1
                if df[col].dtype == 'object':
                    # Standardize to lowercase strings for mapping
                    # Fill NA before mapping to avoid errors, then decide how to handle original NAs if needed
                    # For now, NAs in object columns will become NA after mapping if not in map keys
                    df[col] = df[col].astype(str).str.lower().map({
                        'true': 1, 'false': 0,
                        '1': 1, '0': 0,
                        '1.0': 1, '0.0': 0,
                        'on': 1, 'off': 0,
                        'yes': 1, 'no': 0,
                        # Add other mappings as needed
                    }).astype(pd.Int64Dtype()) # Use nullable Int64Dtype
                    print(f"  Converted object column '{col}' to int (0/1) using mapping.")
                elif df[col].dtype == 'bool':
                    df[col] = df[col].astype(pd.Int64Dtype()) # Use nullable Int64Dtype
                    print(f"  Converted boolean column '{col}' to int.")
                elif pd.api.types.is_numeric_dtype(df[col]):
                    # For numeric types, check if they are effectively 0 or 1
                    # Handle potential NaNs before checking isin
                    is_01 = df[col].dropna().isin([0, 1]).all()
                    if is_01:
                        df[col] = df[col].astype(pd.Int64Dtype()) # Use nullable Int64Dtype
                        print(f"  Ensured numeric (0/1) column '{col}' is integer type.")
                    else:
                        # If numeric but not 0/1, it might be a status with other values.
                        # Log this but don't convert unless a specific mapping rule is applied.
                        print(f"  Warning: Numeric column '{col}' is not strictly 0/1 (contains other values or all NaN after dropna). Skipping direct 0/1 conversion. Dtype: {df[col].dtype}")
                else:
                    print(f"  Warning: Column '{col}' intended for bool->int conversion has unhandled dtype: {df[col].dtype}. Skipping conversion for this column.")
            else:
                print(f"  Warning: Column '{col}' for bool->int conversion not found in DataFrame.")
    return df

def save_data(df: pd.DataFrame, path: Path): # Ensure path is Path object
    print(f"Saving processed data to {path}...")
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == '.parquet':
            df.to_parquet(path, index=False)
        elif path.suffix == '.csv':
            df.to_csv(path, index=False)
        else:
            raise ValueError("Unsupported file format for saving. Please use .parquet or .csv")
        print(f"Data saved successfully to {path}")
    except Exception as e:
        print(f"Error saving data to {path}: {e}")

def generate_summary_report(report_items: list, output_dir: Path, filename: str):
    """Generates a simple text summary report."""
    report_path = output_dir / filename
    print(f"\nGenerating summary report at: {report_path}")
    try:
        with open(report_path, 'w') as f:
            for item_type, content in report_items:
                f.write(f"--- {item_type} ---\n")
                if isinstance(content, pd.DataFrame):
                    f.write(content.to_string() + "\n\n")
                elif isinstance(content, pd.Series):
                    f.write(content.to_string() + "\n\n")
                elif isinstance(content, dict):
                    for key, val in content.items():
                        f.write(f"{key}: {val}\n")
                    f.write("\n")
                else:
                    f.write(str(content) + "\n\n")
        print(f"Summary report saved successfully to {report_path}")
    except Exception as e:
        print(f"Error saving summary report: {e}")

def resample_data_for_era(df: pd.DataFrame, era_identifier: str, era_config: dict, common_config: dict) -> pd.DataFrame:
    """Resamples data to a consistent target frequency for the given era."""
    if df.empty:
        print(f"(Era: {era_identifier}) DataFrame is empty, skipping resampling.")
        return df

    time_col = common_config.get("time_col", "time")
    target_freq = era_config.get("target_frequency")

    if not target_freq:
        print(f"(Era: {era_identifier}) No target_frequency defined. Skipping resampling.")
        return df

    if time_col not in df.columns:
        print(f"(Era: {era_identifier}) Time column '{time_col}' not found. Cannot resample.")
        return df # Or raise error
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        print(f"(Era: {era_identifier}) Time column '{time_col}' is not datetime. Cannot resample.")
        return df

    print(f"\n--- Resampling Data for Era: {era_identifier} to frequency: {target_freq} ---")
    
    # Ensure time column is the index
    if df.index.name != time_col:
        print(f"Setting '{time_col}' as index for resampling...")
        df = df.set_index(time_col)
    elif not isinstance(df.index, pd.DatetimeIndex):
         print(f"Index is '{time_col}' but not DatetimeIndex. Converting...")
         df.index = pd.to_datetime(df.index, utc=True)

    try:
        df_resampled = df.resample(target_freq).asfreq()
        print(f"Resampling complete. Shape before: {df.shape}, Shape after: {df_resampled.shape}")
        if not df_resampled.empty:
            print(f"Data range after resampling: {df_resampled.index.min()} to {df_resampled.index.max()}")
        else:
            print("DataFrame is empty after resampling.")
        return df_resampled
    except Exception as e:
        print(f"Error during resampling for Era '{era_identifier}': {e}")
        return df.reset_index() # Return original with reset index on error

def scale_data_for_era(df: pd.DataFrame, era_identifier: str, era_config: Dict[str, Any], global_config: Dict[str, Any], fit_scalers: bool = True, existing_scalers: Dict[str, MinMaxScaler] = None) -> Tuple[pd.DataFrame, Dict[str, MinMaxScaler]]:
    """Scales numerical data for the given era and saves/loads scalers."""
    if df.empty:
        print(f"(Era: {era_identifier}) DataFrame is empty, skipping scaling.")
        return df, {}

    print(f"\n--- Scaling Data for Era: {era_identifier} (fit_scalers={fit_scalers}) ---")
    SCALERS_DIR.mkdir(parents=True, exist_ok=True)
    era_scalers_path = SCALERS_DIR / f"{era_identifier}_scalers.joblib"

    cols_to_scale = []
    # Get boolean columns that were converted to int (0/1) - these typically are not scaled further unless desired
    # Or, if scaling to [-1,1], they might be mapped from 0,1 to -1,1.
    # For now, we will exclude them from MinMaxScaling to [-1,1] if they are 0/1 integers.
    boolean_cols_converted = era_config.get("boolean_columns_to_int", 
                                          global_config.get('common_settings',{}).get("boolean_columns_to_int", []))

    defined_cols_in_rules = set()
    outlier_rules_ref = era_config.get('outlier_rules_ref', 'default_outlier_rules')
    imputation_rules_ref = era_config.get('imputation_rules_ref', 'default_imputation_rules')
    outlier_rules = global_config.get('preprocessing_rules', {}).get(outlier_rules_ref, [])
    imputation_rules = global_config.get('preprocessing_rules', {}).get(imputation_rules_ref, [])
    for rule in outlier_rules: defined_cols_in_rules.add(rule['column'])
    for rule in imputation_rules: defined_cols_in_rules.add(rule['column'])
    
    for col in df.columns:
        if col in boolean_cols_converted:
            # Check if these 0/1 int columns need specific mapping to [-1,1] for GANs
            # For now, assume they are fine as 0/1 and skip general MinMaxScaling
            print(f"  Skipping MinMaxScaling for column '{col}' (identified as boolean/status 0/1).")
            continue
        
        if df[col].dtype == 'float64' or df[col].dtype == 'float32':
            cols_to_scale.append(col)
        elif col in defined_cols_in_rules and pd.api.types.is_numeric_dtype(df[col]):
             if col not in cols_to_scale: cols_to_scale.append(col)
    
    print(f"Identified columns for MinMaxScaling to [-1,1]: {cols_to_scale}")

    df_scaled = df.copy()
    current_scalers = existing_scalers if existing_scalers is not None else {}

    if fit_scalers:
        for col in cols_to_scale:
            if col in df_scaled.columns and not df_scaled[col].isnull().all():
                # Ensure column is float for scaler, even if it was Int64 from boolean conversion
                data_to_scale = df_scaled[[col]].astype(float)
                if data_to_scale.dropna().empty: # Check if all values became NaN after astype(float) or were already all NaN
                    print(f"  Column '{col}' contains all NaN or non-convertible values. Skipping scaling.")
                    continue
                scaler = MinMaxScaler(feature_range=(-1, 1))
                df_scaled[col] = scaler.fit_transform(data_to_scale)
                current_scalers[col] = scaler
                print(f"  Fitted and transformed column: {col}")
        if current_scalers:
            joblib.dump(current_scalers, era_scalers_path)
            print(f"Saved fitted scalers for Era '{era_identifier}' to {era_scalers_path}")
    else: 
        if not current_scalers and era_scalers_path.exists():
            current_scalers = joblib.load(era_scalers_path)
            print(f"Loaded scalers for Era '{era_identifier}' from {era_scalers_path}")
        elif not current_scalers and not era_scalers_path.exists():
            print(f"Warning: fit_scalers is False and Scaler file not found at {era_scalers_path}. Cannot transform data.")
            return df_scaled, {}
        
        for col in cols_to_scale:
            if col in df_scaled.columns and col in current_scalers and not df_scaled[col].isnull().all():
                data_to_transform = df_scaled[[col]].astype(float)
                if data_to_transform.dropna().empty:
                    print(f"  Column '{col}' contains all NaN or non-convertible values after astype(float). Skipping transform.")
                    continue
                df_scaled[col] = current_scalers[col].transform(data_to_transform)
                print(f"  Transformed column '{col}' using provided/loaded scaler.")
            elif col in df_scaled.columns and not df_scaled[col].isnull().all():
                print(f"  Warning: No scaler provided/loaded for column '{col}'. Skipping scaling for it.")

    return df_scaled, current_scalers

def run_sql_script(engine, script_path: Path) -> bool:
    """Executes a SQL script file using a SQLAlchemy engine."""
    try:
        print(f"Executing SQL script: {script_path}")
        if not script_path.exists():
            print(f"Error: SQL script not found at {script_path}")
            return False
            
        with open(script_path, 'r') as f:
            sql = f.read()
            
        # Split on semicolons but respect those within quotes/comments
        import re
        statements = []
        current_statement = ""
        for line in sql.splitlines():
            # Skip empty lines and comments
            stripped_line = line.strip()
            if not stripped_line or stripped_line.startswith('--'):
                continue
                
            current_statement += line + " "
            if re.search(r';\s*$', line):  # Check for statement end
                statements.append(current_statement.strip())
                current_statement = ""
                
        # Add any remaining statement
        if current_statement.strip():
            statements.append(current_statement.strip())
            
        if not statements:
            print("No SQL statements found in script.")
            return False
            
        with engine.begin() as conn:
            for stmt in statements:
                if stmt and ';' in stmt:
                    # Remove trailing semicolon for SQLAlchemy
                    stmt = stmt.rstrip(';')
                    conn.execute(text(stmt))
                    
        print(f"SQL script executed successfully: {script_path}")
        return True
    except Exception as e:
        print(f"Error executing SQL script {script_path}: {e}")
        return False

def verify_table_exists(engine, table_name: str) -> bool:
    """Checks if a table exists in the connected PostgreSQL database."""
    try:
        query = f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name = '{table_name}'
            );
        """
        with engine.connect() as conn:
            result = conn.execute(text(query)).scalar()
            
        if result:
            print(f"Verified: Table '{table_name}' exists.")
        else:
            print(f"Warning: Table '{table_name}' does NOT exist!")
        return result
    except Exception as e:
        print(f"Error verifying table '{table_name}': {e}")
        return False

def save_to_timescaledb(df: pd.DataFrame, era_identifier: str, engine, time_col: str = 'time') -> bool:
    """
    Save processed data to the preprocessed_features TimescaleDB hypertable.
    
    Args:
        df: DataFrame with processed data
        era_identifier: Identifier for the era (e.g., 'Era1')
        engine: SQLAlchemy engine for database connection
        time_col: Name of the time column in the DataFrame
        
    Returns:
        bool: True if save was successful, False otherwise
    """
    if df.empty:
        print(f"DataFrame is empty, skipping save to TimescaleDB for era '{era_identifier}'")
        return False
        
    print(f"\n--- Saving Processed Data to TimescaleDB for Era: {era_identifier} ---")
    
    try:
        # Create a new DataFrame with the required structure for the preprocessed_features table
        result_df = pd.DataFrame()
        
        # Ensure time column is available 
        if time_col in df.columns:
            result_df['time'] = df[time_col]
        elif df.index.name == time_col:
            # If time is the index, reset it to make it a column
            result_df['time'] = df.index
        else:
            print(f"Error: Time column '{time_col}' not found in DataFrame or index.")
            return False
            
        # Add era_identifier column
        result_df['era_identifier'] = era_identifier
        
        # Convert all other columns to a JSONB-compatible dictionary
        features_list = []
        
        for _, row in df.iterrows():
            # Create features dict, excluding time if it's a column
            features = {}
            for col in df.columns:
                if col != time_col:  # Skip time column if it exists
                    # Handle NaN, infinity, etc.
                    val = row[col]
                    if pd.isna(val):
                        features[col] = None
                    elif isinstance(val, (int, float, bool, str)):
                        features[col] = val
                    else:
                        # Convert other types to string
                        features[col] = str(val)
            features_list.append(features)
            
        result_df['features'] = features_list
        
        # Insert to database
        print(f"Inserting {len(result_df)} rows into preprocessed_features table...")
        import json
        
        # Convert features column to valid JSON strings
        result_df['features'] = result_df['features'].apply(json.dumps)
        
        # Insert using to_sql
        result_df.to_sql('preprocessed_features', engine, if_exists='append', index=False, 
                         method='multi', chunksize=1000)
        
        print(f"Successfully saved {len(result_df)} rows to TimescaleDB for era '{era_identifier}'")
        return True
            
    except Exception as e:
        print(f"Error saving data to TimescaleDB for era '{era_identifier}': {e}")
        return False

if __name__ == "__main__":
    print("--- Starting Preprocessing Stage --- ")
    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SCALERS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        app_config = load_config(CONFIG_PATH)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Failed to load or parse config: {e}. Exiting.")
        sys.exit(1)

    # Get database connection details for creating the hypertable
    db_conn_settings = app_config.get("database_connection", {})
    db_user = db_conn_settings.get("user", DB_USER_ENV)
    db_password = db_conn_settings.get("password", DB_PASSWORD_ENV)
    db_host = db_conn_settings.get("host", DB_HOST_ENV)
    db_port = db_conn_settings.get("port", DB_PORT_ENV)
    db_name = db_conn_settings.get("dbname", DB_NAME_ENV)
    
    # Create SQLAlchemy engine for DB operations
    db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    engine = None
    try:
        # Add pooling parameters here
        engine = create_engine(
            db_url,
            pool_size=5,          # Number of connections to keep open in the pool
            max_overflow=10,      # Number of additional connections allowed beyond pool_size
            pool_timeout=30,      # Seconds to wait before timing out when getting a connection
            pool_recycle=1800     # Recycle connections after 30 minutes (1800 seconds)
        )
        print(f"Database engine with connection pooling created for: {db_url.replace(db_password, '******')}")
        
        # Run SQL script to create/ensure the preprocessed_features table exists
        sql_script_path = Path(__file__).parent / "create_preprocessed_hypertable.sql"
        if not run_sql_script(engine, sql_script_path):
            print("Failed to execute the SQL script to create the preprocessed_features table. Proceeding with caution.")
        
        # Verify the table exists
        if not verify_table_exists(engine, "preprocessed_features"):
            print("Warning: preprocessed_features table was not found! Data will only be saved as Parquet files.")
    except Exception as e:
        print(f"Error setting up database engine or executing SQL script: {e}")
        print("Proceeding with preprocessing, but data will only be saved as Parquet files.")

    eras_to_process_config = app_config.get('era_definitions', {})
    if not eras_to_process_config:
        print("No 'era_definitions' found in config. Exiting.")
        sys.exit(1)

    single_era_env = os.getenv("PROCESS_ERA_IDENTIFIER")
    eras_to_run_keys = []
    if single_era_env and single_era_env in eras_to_process_config:
        eras_to_run_keys = [single_era_env]
        print(f"Found PROCESS_ERA_IDENTIFIER: '{single_era_env}'. Processing only this era.")
    else:
        if single_era_env:
            print(f"Warning: PROCESS_ERA_IDENTIFIER '{single_era_env}' not found in config. Processing all defined eras.")
        eras_to_run_keys = list(eras_to_process_config.keys())
        print(f"Processing all defined eras: {eras_to_run_keys}")

    for era_id in eras_to_run_keys:
        era_conf_details = eras_to_process_config[era_id]
        print(f"\n===== PROCESSING ERA: {era_id} =====")
        current_era_summary_items = [] # Summary items specific to this era
        current_era_summary_items.append(("Era Identifier", era_id))
        current_era_summary_items.append(("Era Configuration", json.dumps(era_conf_details, indent=4)))
        
        # Pass the engine to fetch_source_data
        source_df = fetch_source_data(era_identifier=era_id, era_config=era_conf_details, global_config=app_config, engine=engine)
        current_era_summary_items.append((f"Era {era_id} - Initial Data Shape", source_df.shape))
        if source_df.empty:
            print(f"No data fetched for Era '{era_id}'. Skipping further processing for this era.")
            generate_summary_report(current_era_summary_items, OUTPUT_DATA_DIR, SUMMARY_REPORT_FILENAME_TEMPLATE.format(era_identifier=era_id))
            continue

        prepared_df = sort_and_prepare_df(source_df, app_config, era_identifier=era_id)
        current_era_summary_items.append((f"Era {era_id} - Shape After Sorting, ID & Type Prep", prepared_df.shape))
        if prepared_df.empty: 
            print(f"Data became empty after sorting/ID prep for Era '{era_id}'. Skipping.")
            generate_summary_report(current_era_summary_items, OUTPUT_DATA_DIR, SUMMARY_REPORT_FILENAME_TEMPLATE.format(era_identifier=era_id))
            continue

        # --- Resampling --- 
        df_resampled = resample_data_for_era(prepared_df, era_id, era_conf_details, app_config.get('common_settings',{}))
        current_era_summary_items.append((f"Era {era_id} - Shape After Resampling to {era_conf_details.get('target_frequency')}", df_resampled.shape))
        if df_resampled.empty:
            print(f"Data became empty after resampling for Era '{era_id}'. Skipping.")
            generate_summary_report(current_era_summary_items, OUTPUT_DATA_DIR, SUMMARY_REPORT_FILENAME_TEMPLATE.format(era_identifier=era_id))
            continue

        # Get specific preprocessing rules for the era
        era_outlier_rules_ref = era_conf_details.get('outlier_rules_ref', 'default_outlier_rules')
        outlier_rules = app_config.get('preprocessing_rules', {}).get(era_outlier_rules_ref, [])
        outlier_handler = OutlierHandler(outlier_rules)
        df_after_outliers = outlier_handler.clip_outliers(df_resampled) # Apply to resampled data
        current_era_summary_items.append((f"Era {era_id} - Shape After Outlier Clipping", df_after_outliers.shape))

        # Segmentation (operates on resampled, outlier-handled data)
        segmentation_config_base = app_config.get('segmentation', { 'min_gap_hours': 24 })
        segmentation_config_actual = segmentation_config_base.copy() # Avoid modifying global config
        if 'time_col' not in segmentation_config_actual:
             segmentation_config_actual['time_col'] = app_config.get('common_settings',{}).get('time_col', 'timestamp')
        # Pass era_conf_details if DataSegmenter needs more era-specific segmentation params
        segmenter = DataSegmenter(era_conf_details, common_config=app_config.get('common_settings',{}))
        data_segments = segmenter.segment_by_availability(df_after_outliers)
        current_era_summary_items.append((f"Era {era_id} - Number of Segments Found", len(data_segments)))

        if not data_segments:
            print(f"No data segments found for Era '{era_id}' after segmentation. Skipping saving.")
            generate_summary_report(current_era_summary_items, OUTPUT_DATA_DIR, SUMMARY_REPORT_FILENAME_TEMPLATE.format(era_identifier=era_id))
            continue

        processed_segment_paths_era = []
        for i, segment_df_original in enumerate(data_segments):
            segment_label = f"Era {era_id} - Segment {i+1}/{len(data_segments)}"
            current_era_summary_items.append((f"{segment_label} - Initial Shape", segment_df_original.shape))
            if segment_df_original.empty: continue

            # Imputation per segment
            era_imputation_rules_ref = era_conf_details.get('imputation_rules_ref', 'default_imputation_rules')
            imputation_rules = app_config.get('preprocessing_rules', {}).get(era_imputation_rules_ref, [])
            imputation_handler = ImputationHandler(imputation_rules)
            df_after_imputation = imputation_handler.impute_data(segment_df_original)
            current_era_summary_items.append((f"{segment_label} - Shape After Imputation", df_after_imputation.shape))
            nan_counts_segment = df_after_imputation.isnull().sum()
            current_era_summary_items.append((f"{segment_label} - NaN Counts Post-Imputation", nan_counts_segment[nan_counts_segment > 0].to_dict() or "No NaNs"))

            # Scaling per segment (or per Era before segmentation - chose per segment for now if scalers differ)
            # For GANs, typically you fit scalers on the whole training set of an Era, then transform segments.
            # Let's assume we fit scalers once per Era and apply to segments.
            # This requires a decision: fit scalers on df_after_outliers (pre-segmentation) for the whole Era, 
            # or fit on the first segment and reuse, or pass all segments to a combined scaling step.
            # For simplicity, let's scale the imputed segment directly here. 
            # If training a GAN, scalers should be fit on training data only.
            # This example scales each segment independently for demonstration of the function.
            # A more robust approach would be to scale the entire era data (df_after_outliers or df_after_imputation if imputation is done globally for an era first)
            # and then segment, or pass all segments to a function that scales them based on combined stats.
            # For now, this shows the function call per segment.
            df_scaled, _ = scale_data_for_era(df_after_imputation, era_id, era_conf_details, app_config, fit_scalers=True) # fit_scalers=True for each segment here is likely NOT for GAN training set
            current_era_summary_items.append((f"{segment_label} - Shape After Scaling", df_scaled.shape))
            
            output_filename = OUTPUT_FILENAME_TEMPLATE.format(era_identifier=era_id, segment_num=i+1)
            output_path = OUTPUT_DATA_DIR / output_filename 
            save_data(df_scaled, output_path) # Save scaled data
            processed_segment_paths_era.append(str(output_path))
            current_era_summary_items.append((f"{segment_label} - Saved To", str(output_path)))
            
            # Also save to TimescaleDB if available
            if 'engine' in locals() and verify_table_exists(engine, "preprocessed_features"):
                success = save_to_timescaledb(
                    df=df_scaled, 
                    era_identifier=era_id, 
                    engine=engine, 
                    time_col=app_config.get('common_settings',{}).get('time_col', 'time')
                )
                current_era_summary_items.append((f"{segment_label} - Saved to TimescaleDB", "Success" if success else "Failed"))

        if processed_segment_paths_era:
            print(f"\n--- Era '{era_id}' Preprocessing Completed Successfully. Processed segments saved: ---")
            for p in processed_segment_paths_era:
                print(f"- {p}")
        else:
            print(f"\n--- Era '{era_id}' Preprocessing Completed, but no segments were processed or saved. ---") 
        
        generate_summary_report(current_era_summary_items, OUTPUT_DATA_DIR, SUMMARY_REPORT_FILENAME_TEMPLATE.format(era_identifier=era_id))

    print("\n===== ALL ERAS PROCESSED (if multiple were specified) =====") 