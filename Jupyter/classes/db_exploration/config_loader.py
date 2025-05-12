import os
import sys
# import psycopg2 # Removed unused import causing errors with uv run
import psycopg # Keep import here for potential fallback/type hints

# Attempt to import necessary components from the main src directory
try:
    # Calculate path to the 'src' directory relative to this file
    # config_loader.py -> db_exploration -> classes -> jupyter -> src
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..', 'src'))

    if src_dir not in sys.path:
        sys.path.insert(0, src_dir) # Prepend to avoid conflicts if possible
        print(f"Added {src_dir} to sys.path")
    else:
        print(f"{src_dir} is already in sys.path")

    from config import DBConnectionConfig, DataProcessingConfig, load_data_processing_config
    print("Successfully imported config module from src.")
    CONFIG_LOADED = True
except ImportError as e:
    print(f"Error importing from src/config.py: {e}")
    print("Will need to use fallback manual DB connection details if DBConnectionConfig is used.")
    DBConnectionConfig = None
    DataProcessingConfig = None
    load_data_processing_config = None
    src_dir = None # Mark src_dir as invalid for config loading
    CONFIG_LOADED = False
    # Define fallback class if import failed and direct instantiation is needed later
    class FallbackDBConfig:
        host: str = 'localhost' # Default fallback
        port: int = 5432        # Default fallback
        user: str = 'postgres'  # Default fallback
        password: str = 'postgres' # TODO: Replace with your actual password if using fallback
        dbname: str = 'postgres'   # Default fallback

def get_database_config() -> object:
    """
    Loads database configuration, prioritizing data_processing_config.json,
    then Pydantic defaults (using environment variables), then a hardcoded fallback.

    Returns:
        An object (either DBConnectionConfig or FallbackDBConfig) with connection attributes.
    """
    db_config_obj = None

    if CONFIG_LOADED:
        # Start with Pydantic defaults (which should check env vars)
        try:
            db_config_obj = DBConnectionConfig()
            print("Initialized DB config with Pydantic defaults (checking env vars).")
        except Exception as e:
            print(f"Error initializing DBConnectionConfig with defaults: {e}. Will try JSON or fallback.")

        # Try loading from JSON file in the src directory
        if src_dir:
            config_file_path = os.path.join(src_dir, 'data_processing_config.json')
            print(f"Attempting to load specific config from: {config_file_path}")
            if os.path.exists(config_file_path):
                try:
                    data_proc_config_loaded = load_data_processing_config(config_file_path)
                    if data_proc_config_loaded and isinstance(data_proc_config_loaded, DataProcessingConfig):
                        db_config_obj = data_proc_config_loaded.db_connection # Override default with file content
                        print(f"Successfully loaded and validated DB config from {config_file_path}")
                    else:
                        print(f"Loaded config file {config_file_path}, but failed to validate or get DB settings.")
                except Exception as e:
                    print(f"Error loading or processing {config_file_path}: {e}.")
            else:
                print(f"Config file {config_file_path} not found.")
        else:
             print("src_dir not valid, cannot load config file.")

    # If db_config_obj is still None (e.g., initial import failed, or Pydantic default failed)
    if db_config_obj is None:
        print("Using hardcoded fallback DB connection details.")
        db_config_obj = FallbackDBConfig()

    # --- Override host for local execution --- 
    # If the loaded/default host is 'db' (likely from Docker config),
    # override it to 'localhost' for local script execution.
    # This avoids changing the shared JSON config file.
    original_host = getattr(db_config_obj, 'host', 'N/A')
    if original_host == 'db':
        print(f"Detected host='db', overriding to 'localhost' for local execution.")
        # Check if the object is mutable (Pydantic model or our Fallback class)
        if hasattr(db_config_obj, 'host'): 
             try:
                 setattr(db_config_obj, 'host', 'localhost')
             except AttributeError:
                 print("Warning: Could not override host attribute. Config object might be immutable.")
        else:
             print("Warning: Config object does not have a host attribute to override.")
    # --- End override ---

    print(f"Using DB Config -> Host: {getattr(db_config_obj, 'host', 'N/A')}, Port: {getattr(db_config_obj, 'port', 'N/A')}, DB: {getattr(db_config_obj, 'dbname', 'N/A')}, User: {getattr(db_config_obj, 'user', 'N/A')}")
    return db_config_obj

# Example usage (for testing this module directly)
if __name__ == '__main__':
    print("\nTesting config loader...")
    config_to_use = get_database_config()
    # You could add a connection test here if desired 