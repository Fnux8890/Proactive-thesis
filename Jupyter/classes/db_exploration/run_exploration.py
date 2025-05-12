from config_loader import get_database_config
from explorer import DatabaseExplorer

def main():
    """Loads config, connects to DB, runs exploration, and closes connection."""
    print("Starting database exploration script...")

    # 1. Load Configuration
    db_config = get_database_config()
    if not db_config:
        print("Failed to load database configuration. Exiting.")
        return

    # 2. Initialize Explorer (which connects automatically)
    explorer = DatabaseExplorer(db_config)

    # 3. Check if connection was successful
    if not explorer.conn or explorer.conn.closed:
        print("Database connection failed during initialization. Cannot proceed.")
        return

    # 4. Explore tables
    try:
        explorer.explore_table('sensor_data')
        explorer.explore_table('sensor_data_merged')

        # 5. Run quality checks
        explorer.check_merged_data_quality()

    except Exception as e:
        print(f"An error occurred during exploration: {e}")
    finally:
        # 6. Ensure connection is closed
        explorer.close_connection()

    print("\nDatabase exploration script finished.")

if __name__ == "__main__":
    main() 