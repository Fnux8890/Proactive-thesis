from typing import Dict
import os
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine
from dotenv import load_dotenv


def load_all_data(root_path: str) -> Dict[str, pd.DataFrame]:
    """
    Load all CSV files from each subdirectory in root_path.
    Returns a dictionary mapping subfolder names to a concatenated DataFrame of their CSV files.
    """
    data = {}
    # Iterate over each item in the root directory
    for folder in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder)
        if os.path.isdir(folder_path):
            # Get all CSV files in this subfolder
            csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
            if not csv_files:
                print(f"No CSV files found in {folder_path}.")
                continue
            dfs = []
            for file in csv_files:
                file_path = os.path.join(folder_path, file)
                try:
                    df = pd.read_csv(file_path)
                    dfs.append(df)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                data[folder] = combined_df
    return data


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the loaded DataFrame by ensuring a timestamp column exists and is correctly formatted.
    Additional cleaning and wrangling steps can be added here.
    """
    if 'timestamp' not in df.columns:
        df['timestamp'] = datetime.now()
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Additional processing logic can be added below
    return df


def save_to_timescaledb(df: pd.DataFrame, connection_string: str, table_name: str) -> None:
    """
    Save the processed data to TimescaleDB into the specified table.
    """
    engine = create_engine(connection_string)
    df.to_sql(table_name, engine, if_exists='append', index=False)
    print(f"Data saved to table {table_name}.")


def main() -> None:
    """
    Main pipeline that loads data from subdirectories, processes each batch, and saves them to TimescaleDB.
    Each subfolder's data is saved into its own table based on the folder name.
    """
    # Load environment variables
    load_dotenv()
    
    input_path = os.getenv('DATA_INPUT_PATH', '/app/data')
    db_url = os.getenv('PROCESSING_DB_URL')
    
    if not db_url:
        raise ValueError("Database URL not found in environment variables")
    
    data_dict = load_all_data(input_path)
    
    if not data_dict:
        print(f"No data found in {input_path}.")
        return
    
    for source, df in data_dict.items():
        print(f"Processing data from {source}...")
        processed_df = process_data(df)
        table_name = f"processed_data_{source}"
        save_to_timescaledb(processed_df, db_url, table_name)
    
    print("All data processed and saved successfully!")


if __name__ == "__main__":
    main() 