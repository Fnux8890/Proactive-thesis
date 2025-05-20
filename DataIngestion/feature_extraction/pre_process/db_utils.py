import os
from abc import ABC, abstractmethod
from sqlalchemy import create_engine, text
from sqlalchemy.sql.expression import TextClause
import pandas as pd
from typing import Union, Optional, Any
# import polars as pl # Uncomment if you plan to use Polars directly

class BaseDBConnector(ABC):
    """
    Abstract base class for database connectors.
    Implementations can either create their own connection or use dependency injection.
    """
    @abstractmethod
    def connect(self) -> Any:
        """Create and return a database connection or engine."""
        pass
    
    @abstractmethod
    def fetch_data_to_pandas(self, query: Union[str, TextClause]) -> pd.DataFrame:
        """Fetch data from database using the provided query and return as pandas DataFrame."""
        pass

    # @abstractmethod
    # def fetch_data_to_polars(self, query: str) -> pl.DataFrame: # Uncomment for Polars
    #     pass

class SQLAlchemyPostgresConnector(BaseDBConnector):
    """
    SQLAlchemy connector for PostgreSQL databases.
    Supports both creating its own engine or using dependency injection.
    """
    def __init__(self, user: Optional[str] = None, password: Optional[str] = None, 
                 host: Optional[str] = None, port: Optional[str] = None, 
                 db_name: Optional[str] = None, engine = None):
        """
        Initialize the connector either with connection parameters or an existing engine.
        
        Args:
            user: PostgreSQL username (not needed if engine is provided)
            password: PostgreSQL password (not needed if engine is provided)
            host: PostgreSQL host (not needed if engine is provided)
            port: PostgreSQL port (not needed if engine is provided)
            db_name: PostgreSQL database name (not needed if engine is provided)
            engine: Existing SQLAlchemy engine (if provided, connection params are ignored)
        """
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.db_name = db_name
        self.engine = engine
        
        # If engine not provided, try to connect if all credentials are given
        if self.engine is None and all([user, password, host, port, db_name]):
            self.connect()

    def connect(self) -> Any:
        """
        Create a new SQLAlchemy engine if one wasn't provided during initialization.
        Returns the engine instance.
        """
        if self.engine is not None:
            return self.engine
            
        if not all([self.user, self.password, self.host, self.port, self.db_name]):
            raise ValueError("Cannot create connection: missing database connection parameters")
            
        try:
            db_url = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db_name}"
            self.engine = create_engine(db_url)
            # Test connection
            with self.engine.connect() as connection:
                print(f"Successfully connected to PostgreSQL database: {self.db_name}")
            return self.engine
        except Exception as e:
            print(f"Error connecting to PostgreSQL: {e}")
            self.engine = None
            raise
            
    def fetch_data_to_pandas(self, query: Union[str, TextClause]) -> pd.DataFrame:
        """
        Execute a query and return results as a pandas DataFrame.
        If no engine exists, tries to create one if possible.
        """
        if not self.engine:
            try:
                self.connect()
            except:
                raise ConnectionError("Database engine not initialized and cannot be created with provided parameters.")
        try:
            with self.engine.connect() as connection:
                # Pass query directly, as it might be a pre-formed TextClause
                # with bound parameters from the calling function.
                df = pd.read_sql_query(sql=query, con=connection)
            return df
        except Exception as e:
            print(f"Error fetching data to Pandas DataFrame: {e}")
            raise

    # def fetch_data_to_polars(self, query: str) -> pl.DataFrame: # Uncomment for Polars
    #     if not self.engine:
    #         print("Database engine not initialized. Call connect() first.")
    #         raise ConnectionError("Database engine not initialized.")
    #     try:
    #         # Polars can read directly from a SQLAlchemy connection or a query result
    #         # For simplicity, using pandas as an intermediary for now if direct polars.read_sql is complex
    #         # Alternatively, use a library like connectorx: pl.read_sql(query, connection_uri)
    #         with self.engine.connect() as connection:
    #             # This is one way; polars.read_database might be more direct with a URI
    #             # For now, let's assume pandas conversion if direct polars is not straightforward
    #             # Polars 0.19+ has read_database
    #             # df_pd = pd.read_sql_query(sql=text(query), con=connection)
    #             # return pl.from_pandas(df_pd)

    #             # Using Polars read_sql (requires connection string)
    #             db_url = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db_name}"
    #             df_pl = pl.read_database(query=query, connection=db_url)

    #         return df_pl
    #     except Exception as e:
    #         print(f"Error fetching data to Polars DataFrame: {e}")
    #         raise

# Factory function to make creating connectors easier
def create_connector(connector_type: str = "sqlalchemy_postgres", **kwargs):
    """
    Factory function to create a database connector of the specified type.
    
    Args:
        connector_type: Type of connector to create (e.g., "sqlalchemy_postgres")
        **kwargs: Parameters to pass to the connector's constructor
        
    Returns:
        BaseDBConnector: An instance of the specified connector type
    """
    if connector_type == "sqlalchemy_postgres":
        return SQLAlchemyPostgresConnector(**kwargs)
    else:
        raise ValueError(f"Unsupported connector type: {connector_type}")

if __name__ == '__main__':
    # Example Usage 1: Connect with parameters
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
    DB_HOST = os.getenv("DB_HOST", "db") # 'db' if running in Docker network, 'localhost' otherwise
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "postgres")

    print(f"Attempting to connect with: User={DB_USER}, Host={DB_HOST}, Port={DB_PORT}, DB={DB_NAME}")

    try:
        # Method 1: Create connector with parameters 
        connector = SQLAlchemyPostgresConnector(
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            db_name=DB_NAME
        )
        
        # Method 2: Use factory function
        # connector = create_connector(
        #     connector_type="sqlalchemy_postgres",
        #     user=DB_USER,
        #     password=DB_PASSWORD,
        #     host=DB_HOST,
        #     port=DB_PORT,
        #     db_name=DB_NAME
        # )
        
        # Example query
        query_string = "SELECT COUNT(*) FROM information_schema.tables;"

        print(f"Fetching data with query: {query_string}")
        df_pandas = connector.fetch_data_to_pandas(query_string)
        print("\nPandas DataFrame:")
        print(df_pandas.head())
        
        # Method 3 (dependency injection): Create engine directly and inject it
        # engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
        # connector_with_engine = SQLAlchemyPostgresConnector(engine=engine)
        # df_pandas_2 = connector_with_engine.fetch_data_to_pandas(query_string)
        # print("\nPandas DataFrame (using injected engine):")
        # print(df_pandas_2.head())

    except Exception as e:
        print(f"An error occurred during the example usage: {e}") 
