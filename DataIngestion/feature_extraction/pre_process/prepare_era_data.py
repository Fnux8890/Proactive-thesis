import pandas as pd
import numpy as np
from pathlib import Path
import json
import psycopg2 # For PostgreSQL connection
import os # For reading environment variables
# from sklearn.preprocessing import MinMaxScaler # Import if/when needed for scaling
# import joblib # Import if/when needed for saving scalers

CONFIG_PATH = Path("/app/config/data_processing_config.json")
# Assuming raw data might be stored relative to the main feature_extraction directory or a defined path
# This needs to be configured in your JSON or passed to functions
# RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "raw_era_dumps" 

def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Loads the data processing configuration file."""
    print(f"Loading configuration from: {config_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def get_db_connection_details(config: dict) -> dict:
    """Fetches DB connection details from config (which points to env vars)."""
    conn_config = config.get("database_connection", {})
    details = {
        'user': "postgres",
        'password': "postgres",
        'host': "localhost",
        'port': "5432",
        'dbname': "postgres",
    }
    
    missing_details = [k for k, v in details.items() if v is None]
    if missing_details:
        raise ValueError(f"Missing database connection environment variables for: {missing_details}. Please set them.")
    return details

def load_era_data(era_identifier: str, config: dict) -> pd.DataFrame:
    """
    Loads raw data for a specific era by querying the database.
    """
    print(f"\n--- Loading Data for Era: {era_identifier} from Database ---")
    
    era_config = config.get("data_sources", {}).get(era_identifier)
    if not era_config:
        print(f"Error: No data source configuration found for era '{era_identifier}' in config.")
        return pd.DataFrame()

    db_table = era_config.get("db_table")
    start_date = era_config.get("start_date")
    end_date = era_config.get("end_date")
    time_column = config.get("common_settings", {}).get("time_column_name", "time")

    if not all([db_table, start_date, end_date, time_column]):
        print(f"Error: Missing db_table, start_date, end_date, or time_column in config for era '{era_identifier}'.")
        return pd.DataFrame()

    try:
        conn_details = get_db_connection_details(config)
        conn = psycopg2.connect(**conn_details)
        print(f"Successfully connected to database '{conn_details['dbname']}' on host '{conn_details['host']}'.")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return pd.DataFrame()

    # Construct the SQL query
    # Using placeholders for dates to prevent SQL injection, though less critical here as they come from config.
    # Ensure start_date and end_date from config are valid ISO 8601 strings for PostgreSQL.
    query = f"""
        SELECT *
        FROM {db_table}
        WHERE "{time_column}" >= %s AND "{time_column}" <= %s
        ORDER BY "{time_column}" ASC;
    """
    
    print(f"Executing query for Era '{era_identifier}': from {start_date} to {end_date} on table {db_table}")
    try:
        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        print(f"Successfully loaded {len(df)} rows from database for Era '{era_identifier}'.")
    except Exception as e:
        print(f"Error executing SQL query or reading into DataFrame: {e}")
        df = pd.DataFrame()
    finally:
        if 'conn' in locals() and conn:
            conn.close()
            print("Database connection closed.")
            
    return df

def initial_clean_era_data(df: pd.DataFrame, era_identifier: str, config: dict) -> pd.DataFrame:
    """
    Performs initial cleaning: time conversion, sorting, dropping dead columns.
    """
    if df.empty:
        return df
    print(f"\n--- Initial Cleaning for Era: {era_identifier} ---")
    
    time_column = config.get("common_settings", {}).get("time_column_name", "time")
    dead_columns_era = config.get("era_specific_settings", {}).get(era_identifier, {}).get("dead_columns_to_drop", [])

    if time_column not in df.columns:
        print(f"Error: Time column '{time_column}' not found in DataFrame.")
        # Potentially, try to infer or raise a more specific error
        return pd.DataFrame() # Or df, if we allow processing without time for some reason

    # 1. Time Conversion & Sorting
    print(f"Converting column '{time_column}' to datetime and setting to UTC...")
    df[time_column] = pd.to_datetime(df[time_column], errors='coerce', utc=True)
    df.dropna(subset=[time_column], inplace=True) # Drop rows where time conversion failed
    df.sort_values(by=time_column, inplace=True)
    print(f"Time column processed. Data range: {df[time_column].min()} to {df[time_column].max()}")

    # 2. Drop Dead/Excluded Columns for this Era
    if dead_columns_era:
        cols_to_drop = [col for col in dead_columns_era if col in df.columns]
        if cols_to_drop:
            print(f"Dropping specified dead columns for Era '{era_identifier}': {cols_to_drop}")
            df.drop(columns=cols_to_drop, inplace=True)
        else:
            print(f"No specified dead columns found in the DataFrame for Era '{era_identifier}'.")
    else:
        print(f"No dead columns specified for dropping in Era '{era_identifier}'.")
    
    print(f"Shape after initial cleaning: {df.shape}")
    return df

def resample_era_data(df: pd.DataFrame, era_identifier: str, config: dict) -> pd.DataFrame:
    """
    Resamples data to a consistent frequency and sets DatetimeIndex.
    """
    if df.empty:
        return df
    print(f"\n--- Resampling Data for Era: {era_identifier} ---")
    
    time_column = config.get("common_settings", {}).get("time_column_name", "time")
    target_freq = config.get("era_specific_settings", {}).get(era_identifier, {}).get("target_frequency", "1T") # Default to 1 minute
    # Aggregation methods might also be part of the config per column for downsampling
    # For now, this function focuses on setting the frequency; imputation handles NaNs.

    if time_column not in df.columns:
        print(f"Error during resampling: Time column '{time_column}' not found. Skipping resampling.")
        return df
    if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
        print(f"Error during resampling: Time column '{time_column}' is not datetime. Skipping resampling.")
        return df
    
    print(f"Setting '{time_column}' as index and resampling to frequency: {target_freq}...")
    df = df.set_index(time_column)
    
    # Determine original frequency to decide if upsampling or downsampling path is more appropriate
    # This is a heuristic. A more robust way would be to analyze median diff or rely on config.
    # For this example, we'll just resample. If it's upsampling, NaNs appear.
    # If it's downsampling without explicit agg, it might error or take first/last.
    # For robust downsampling, aggregation functions should be specified per column.
    
    # For now, we primarily prepare for upsampling by setting the frequency.
    # If actual downsampling with aggregation is needed, the logic here needs to be more complex,
    # iterating through columns and applying specified aggregations.
    try:
        # This call ensures the index has the target frequency. If upsampling, NaNs are introduced.
        # If the original data already matches target_freq and is regular, this might not change much
        # other than ensuring regularity if there were implied gaps.
        df_resampled = df.resample(target_freq).asfreq() # Use .asfreq() to just conform to new freq, NaN for new slots
        # To apply aggregations for downsampling (example, would need per-column logic):
        # aggs = {'columnA': 'mean', 'columnB': 'sum', 'status_col': 'last'}
        # df_resampled = df.resample(target_freq).agg(aggs) 
        print(f"Resampling complete. New shape: {df_resampled.shape}")
        print(f"Data range after resampling: {df_resampled.index.min()} to {df_resampled.index.max()}")
    except Exception as e:
        print(f"Error during resampling for Era '{era_identifier}' to freq '{target_freq}': {e}")
        return df # Return original df on error
        
    return df_resampled

def impute_era_data(df: pd.DataFrame, era_identifier: str, config: dict) -> pd.DataFrame:
    """
    Handles missing values (NaNs) after resampling.
    """
    if df.empty:
        return df
    print(f"\n--- Imputing Data for Era: {era_identifier} ---")
    
    imputation_config = config.get("era_specific_settings", {}).get(era_identifier, {}).get("imputation", {})
    default_method = imputation_config.get("default_method", "ffill") # e.g., ffill, linear
    default_limit = imputation_config.get("default_limit") # e.g., 5 (for 5 minutes if 1T freq)
    # Column-specific imputation strategies could be added to config here

    print(f"Applying imputation strategy (default: method='{default_method}', limit={default_limit})...")
    
    if default_method == "ffill":
        df.fillna(method='ffill', limit=default_limit, inplace=True)
    elif default_method == "bfill":
        df.fillna(method='bfill', limit=default_limit, inplace=True)
    elif default_method == "linear":
        df.interpolate(method='linear', limit_direction='forward', limit_area='inside', limit=default_limit, inplace=True)
    else:
        print(f"Warning: Unknown default imputation method '{default_method}'. No imputation applied by default.")

    # Example of a second pass for remaining NaNs, e.g., bfill for leading NaNs
    # if df.isnull().values.any():
    #     print("Applying secondary bfill for any remaining leading NaNs...")
    #     df.fillna(method='bfill', limit=default_limit, inplace=True)

    nan_counts = df.isnull().sum()
    print("NaN counts per column after imputation:")
    print(nan_counts[nan_counts > 0])
    print(f"Shape after imputation: {df.shape}")
    return df

# Placeholder for scaling function - to be developed
# def scale_era_data(df: pd.DataFrame, era_identifier: str, config: dict, scalers_path: Path) -> (pd.DataFrame, dict):
#     pass

def main():
    """Main pipeline for preprocessing data for a specific era."""
    try:
        config = load_config()
    except FileNotFoundError as e:
        print(e)
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from config file: {e}")
        return
    
    era_to_process = config.get("common_settings", {}).get("default_era_to_process_for_script", "Era1") 
    # eras_to_process = config.get("era_specific_settings", {}).keys() # To process all defined eras

    print(f"===== STARTING PREPROCESSING FOR ERA: {era_to_process} =====")
    
    try:
        df_raw = load_era_data(era_identifier=era_to_process, config=config)
    except ValueError as e: # Catch missing DB env vars
        print(e)
        return
        
    if df_raw.empty:
        print(f"Stopping preprocessing for Era '{era_to_process}' due to loading error or no data.")
        return

    # 2. Initial Cleaning
    df_cleaned = initial_clean_era_data(df=df_raw, era_identifier=era_to_process, config=config)
    if df_cleaned.empty:
        print(f"Stopping preprocessing for Era '{era_to_process}' after cleaning step.")
        return

    # 3. Resample to Consistent Frequency
    df_resampled = resample_era_data(df=df_cleaned, era_identifier=era_to_process, config=config)
    if df_resampled.empty:
        print(f"Stopping preprocessing for Era '{era_to_process}' after resampling step.")
        return

    # 4. Imputation
    df_imputed = impute_era_data(df=df_resampled, era_identifier=era_to_process, config=config)
    if df_imputed.empty:
        print(f"Stopping preprocessing for Era '{era_to_process}' after imputation step.")
        return

    # 5. Scaling (placeholder - to be implemented)
    # scalers_output_dir = Path(__file__).parent / "scalers" / era_to_process
    # scalers_output_dir.mkdir(parents=True, exist_ok=True)
    # df_scaled, fitted_scalers = scale_era_data(df=df_imputed, era_identifier=era_to_process, config=config, scalers_path=scalers_output_dir)
    # print("\n--- Data Scaling (Placeholder) ---")
    # print(f"Shape after scaling (if implemented): {df_scaled.shape if 'df_scaled' in locals() else df_imputed.shape}")

    # At this point, df_imputed (or df_scaled if implemented) is ready for windowing and feature engineering
    print(f"\n--- Preprocessing Complete for Era: {era_to_process} ---")
    if not df_imputed.empty:
        print("Final DataFrame sample (head):")
        print(df_imputed.head())
    else:
        print("Final DataFrame is empty.")
    print(f"Final DataFrame shape: {df_imputed.shape}")
    
    # Next steps would be windowing this df_imputed/df_scaled and then feature engineering per window.
    # Output of this script could be a Parquet/CSV file of the preprocessed data for this era.
    output_dir = Path(__file__).parent.parent / "data" / "preprocessed"
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_file_path = output_dir / f"{era_to_process}_preprocessed.parquet"
    try:
        df_imputed.to_parquet(preprocessed_file_path)
        print(f"Saved preprocessed data for Era '{era_to_process}' to: {preprocessed_file_path}")
    except Exception as e:
        print(f"Error saving preprocessed data for Era '{era_to_process}': {e}")

if __name__ == "__main__":
    main() 
