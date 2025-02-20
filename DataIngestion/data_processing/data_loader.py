#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#     "pandas",
#     "sqlalchemy",
#     "python-dotenv",
#     "pathlib",
#     "logging",
#     "psycopg2"
# ]
# ///

from typing import Dict, List, Optional
import os
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import logging
from pathlib import Path


class DataLoader:
    def __init__(self, db_url: str):
        """Initialize the DataLoader with database connection."""
        self.engine = create_engine(db_url)
        self.setup_logging()
        
    def setup_logging(self) -> None:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_staging_schema(self) -> None:
        """Create staging schema if it doesn't exist."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("CREATE SCHEMA IF NOT EXISTS staging;"))
                conn.commit()
            self.logger.info("Staging schema created or verified.")
        except Exception as e:
            self.logger.exception(f"Failed to setup staging schema: {e}")
            
    def create_staging_table(self, table_name: str, df: pd.DataFrame) -> None:
        """Create a staging table with appropriate columns based on DataFrame structure."""
        staging_table = f"staging.raw_{table_name}"
        # Add metadata columns
        df['_loaded_at'] = datetime.now()
        df['_source_file'] = None  # Will be filled during load
        df['_status'] = 'new'
        try:
            # Create table if not exists (using an empty DataFrame)
            df.head(0).to_sql(
                staging_table.split('.')[1],
                self.engine,
                schema='staging',
                if_exists='replace',
                index=False
            )
            self.logger.info(f"Created staging table: {staging_table}")
        except Exception as e:
            self.logger.exception(f"Failed to create staging table {staging_table}: {e}")
            raise  # Reraise to let the calling method handle it
            
    def load_file_to_staging(
        self,
        file_path: Path,
        source_type: str,
        **kwargs
    ) -> Optional[str]:
        """
        Load a single file to staging area.
        Returns the staging table name if successful, None otherwise.
        """
        try:
            # Determine file type and read accordingly
            if file_path.suffix == '.csv':
                try:
                    # First try to infer the delimiter and number of columns
                    with open(file_path, 'r', encoding='utf-8') as f:
                        first_line = f.readline().strip()
                        for delimiter in [',', ';', '\t']:
                            if delimiter in first_line:
                                self.logger.info(f"Detected delimiter: {delimiter} for file {file_path}")
                                df = pd.read_csv(
                                    file_path,
                                    delimiter=delimiter,
                                    encoding='utf-8',
                                    on_bad_lines='warn',
                                    **kwargs
                                )
                                break
                        else:
                            self.logger.warning(f"Could not detect delimiter for {file_path}, using comma")
                            df = pd.read_csv(
                                file_path,
                                encoding='utf-8',
                                on_bad_lines='warn',
                                **kwargs
                            )
                except UnicodeDecodeError:
                    # If UTF-8 fails, try with different encodings
                    for encoding in ['latin1', 'iso-8859-1', 'cp1252']:
                        try:
                            df = pd.read_csv(
                                file_path,
                                encoding=encoding,
                                on_bad_lines='warn',
                                **kwargs
                            )
                            self.logger.info(f"Successfully read file with {encoding} encoding")
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        raise ValueError(f"Could not read file {file_path} with any known encoding")
            elif file_path.suffix == '.json':
                df = pd.read_json(file_path, **kwargs)
            elif file_path.suffix in ['.xls', '.xlsx']:
                df = pd.read_excel(file_path, **kwargs)
            else:
                self.logger.error(f"Unsupported file type: {file_path.suffix}")
                return None
            
            # Create staging table if needed
            staging_table = f"raw_{source_type}"
            self.create_staging_table(source_type, df)
            
            # Add metadata
            df['_loaded_at'] = datetime.now()
            df['_source_file'] = str(file_path)
            df['_status'] = 'new'
            
            # Load to staging
            df.to_sql(
                staging_table,
                self.engine,
                schema='staging',
                if_exists='append',
                index=False
            )
            
            self.logger.info(f"Successfully loaded {file_path} to {staging_table}")
            return staging_table
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {str(e)}")
            self.logger.exception("Detailed error information:")
            return None
            
    def load_directory_to_staging(
        self,
        directory: Path,
        source_type: str,
        file_pattern: str = "*.*",
        **kwargs
    ) -> List[str]:
        """
        Load all matching files in a directory to staging.
        Returns list of successfully loaded staging tables.
        """
        loaded_tables = []
        directory = Path(directory)
        for file_path in directory.glob(file_pattern):
            self.logger.info(f"Processing file: {file_path}")
            table_name = self.load_file_to_staging(file_path, source_type, **kwargs)
            if table_name:
                loaded_tables.append(table_name)
            else:
                self.logger.error(f"Failed to load file: {file_path}")
        return loaded_tables
        
    def get_staging_status(self, table_name: str) -> Dict:
        """Get status of records in a staging table."""
        query = f"""
        SELECT _status, COUNT(*) as count 
        FROM staging.{table_name}
        GROUP BY _status;
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                return dict(result.fetchall())
        except Exception as e:
            self.logger.exception(f"Error getting status for table {table_name}: {e}")
            return {}


def main():
    """Main function to demonstrate usage."""
    load_dotenv()
    
    db_url = os.getenv('PROCESSING_DB_URL')
    input_path = Path(os.getenv('DATA_INPUT_PATH', '/app/data'))
    
    if not db_url:
        raise ValueError("Database URL not found in environment variables")
    
    # Initialize loader
    loader = DataLoader(db_url)
    
    # Setup staging area
    loader.setup_staging_schema()
    
    # Load data from each subdirectory
    for folder in input_path.iterdir():
        if folder.is_dir():
            source_type = folder.name
            loader.logger.info(f"Loading data from directory: {folder}")
            tables = loader.load_directory_to_staging(
                folder,
                source_type,
                file_pattern="*.csv"  # Adjust pattern as needed
            )
            if tables:
                loader.logger.info(f"Loaded tables from {folder}: {tables}")
            else:
                loader.logger.error(f"No files loaded successfully from directory: {folder}")
            
    print("Data loading complete!")


if __name__ == "__main__":
    main()
