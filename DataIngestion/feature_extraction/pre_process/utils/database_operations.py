from typing import Any

import pandas as pd

# Assuming db_utils.py is in the same directory or accessible in PYTHONPATH
from .db_utils import SQLAlchemyPostgresConnector
from sqlalchemy import text

# Database connection details (can be overridden by config or remain as env vars)
# These are defaults if not found in config in preprocess.py
DB_USER_ENV = "postgres"
DB_PASSWORD_ENV = "postgres"
DB_HOST_ENV = "db"
DB_PORT_ENV = "5432"
DB_NAME_ENV = "postgres"


def fetch_source_data(
    era_identifier: str, era_config: dict[str, Any], global_config: dict[str, Any], engine=None
) -> pd.DataFrame:
    print(f"\n--- Fetching Source Data for Era: {era_identifier} from Database ---")

    db_table = era_config.get("db_table")
    start_date = era_config.get("start_date")
    end_date = era_config.get("end_date")
    time_col = global_config.get("common_settings", {}).get(
        "time_col", "time"
    )  # Get time_col from common_settings

    if not all([db_table, start_date, end_date, time_col]):
        print(
            f"Error: Missing db_table, start_date, end_date, or common time_col in config for era '{era_identifier}'."
        )
        return pd.DataFrame()

    db_conn_settings = global_config.get("database_connection", {})
    db_user = db_conn_settings.get("user", DB_USER_ENV)
    db_password = db_conn_settings.get("password", DB_PASSWORD_ENV)
    db_host = db_conn_settings.get("host", DB_HOST_ENV)
    db_port = db_conn_settings.get("port", DB_PORT_ENV)
    db_name = db_conn_settings.get("dbname", DB_NAME_ENV)

    db_connector = None
    try:
        if engine is not None:
            print(f"Using existing database engine for Era '{era_identifier}'")
            db_connector = SQLAlchemyPostgresConnector(engine=engine)
        else:
            db_connector = SQLAlchemyPostgresConnector(
                user=db_user, password=db_password, host=db_host, port=db_port, db_name=db_name
            )
    except Exception as e:
        print(f"Failed to initialize database connector for Era '{era_identifier}': {e}. Exiting.")
        return pd.DataFrame()

    query = f'''
        SELECT *
        FROM {db_table}
        WHERE "{time_col}" >= :start_date AND "{time_col}" <= :end_date
        ORDER BY "{time_col}" ASC;
    '''
    params = {"start_date": start_date, "end_date": end_date}

    print(
        f"Executing query for Era '{era_identifier}': from {start_date} to {end_date} on table {db_table}"
    )
    try:
        from sqlalchemy import text as sql_text

        df = db_connector.fetch_data_to_pandas(sql_text(query).bindparams(**params))
        print(f"Data fetched successfully for Era '{era_identifier}'. Shape: {df.shape}")
        if df.empty:
            print(
                f"Warning: Fetched DataFrame is empty for Era '{era_identifier}'. Check table, query, and date ranges."
            )
        return df
    except Exception as e:
        print(
            f"Failed to fetch data from database for Era '{era_identifier}': {e}. Returning empty DataFrame."
        )
        return pd.DataFrame()


def run_sql_script(engine, script_path) -> bool:
    """Executes a SQL script file using a SQLAlchemy engine."""
    import re

    try:
        print(f"Executing SQL script: {script_path}")
        if not getattr(script_path, "exists", None):
            print(
                "Error: script_path must be a Path-like object with an .exists() method or a string"
            )
            return False
        if not script_path.exists():
            print(f"Error: SQL script not found at {script_path}")
            return False

        with open(script_path) as f:
            script_content = f.read()
        # Remove comments and split into individual statements (handle ; inside strings by regex)
        statements = [
            stmt.strip()
            for stmt in re.split(r";\s*(?=(?:[^']*'[^']*')*[^']*$)", script_content)
            if stmt.strip()
        ]
        with engine.begin() as conn:
            for stmt in statements:
                if stmt:
                    conn.execute(text(stmt))
        print(f"SQL script executed successfully: {script_path}")
        return True
    except Exception as e:
        print(f"Error executing SQL script {script_path}: {e}")
        return False
    """Executes a SQL script file using a SQLAlchemy engine."""
    import re

    try:
        print(f"Executing SQL script: {script_path}")
        if not script_path.exists():
            print(f"Error: SQL script not found at {script_path}")
            return False

        with open(script_path) as f:
            sql = f.read()

        statements = []
        current_statement = ""
        for line in sql.splitlines():
            stripped_line = line.strip()
            if not stripped_line or stripped_line.startswith("--"):
                continue

            current_statement += line + " "
            if re.search(r";\s*$", line):
                statements.append(current_statement.strip())
                current_statement = ""

        if current_statement.strip():
            statements.append(current_statement.strip())

        if not statements:
            print("No SQL statements found in script.")
            return False

        with engine.begin() as conn:
            for stmt in statements:
                if stmt and ";" in stmt:
                    stmt = stmt.rstrip(";")
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


def save_to_timescaledb(
    df: pd.DataFrame, era_identifier: str, engine, time_col: str = "time"
) -> bool:
    """
    Save processed data to the preprocessed_features TimescaleDB hypertable.
    """
    if df.empty:
        print(f"DataFrame is empty, skipping save to TimescaleDB for era '{era_identifier}'")
        return False

    print(f"\n--- Saving Processed Data to TimescaleDB for Era: {era_identifier} ---")

    try:
        result_df = pd.DataFrame()

        if time_col in df.columns:
            result_df["time"] = df[time_col]
        elif df.index.name == time_col:
            result_df["time"] = df.index
        else:
            print(f"Error: Time column '{time_col}' not found in DataFrame or index.")
            return False

        result_df["era_identifier"] = era_identifier

        features_list = []

        for _, row in df.iterrows():
            features = {}
            for col_df in df.columns:  # Renamed col to col_df to avoid clash with outer time_col
                if col_df != time_col:
                    val = row[col_df]
                    if pd.isna(val):
                        features[col_df] = None
                    elif isinstance(val, int | float | bool | str):
                        features[col_df] = val
                    else:
                        features[col_df] = str(val)
            features_list.append(features)

        result_df["features"] = features_list

        print(f"Inserting {len(result_df)} rows into preprocessed_features table...")
        import json  # Moved import here

        result_df["features"] = result_df["features"].apply(json.dumps)

        result_df.to_sql(
            "preprocessed_features",
            engine,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=1000,
        )

        print(f"Successfully saved {len(result_df)} rows to TimescaleDB for era '{era_identifier}'")
        return True

    except Exception as e:
        print(f"Error saving data to TimescaleDB for era '{era_identifier}': {e}")
        return False


def fetch_and_prepare_external_weather_for_era(
    era_start_date_str: str,
    era_end_date_str: str,
    target_frequency: str,
    time_col_name: str,
    engine,
) -> pd.DataFrame:
    """Fetches weather data from public.external_weather_aarhus, sets DatetimeIndex, and resamples."""
    print(f"  Fetching external weather data for range: {era_start_date_str} to {era_end_date_str}")

    query = """
    SELECT *
    FROM public.external_weather_aarhus
    WHERE time >= :start_date AND time <= :end_date
    ORDER BY time ASC;
    """
    params = {"start_date": era_start_date_str, "end_date": era_end_date_str}

    try:
        test_existence_query = text(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'external_weather_aarhus');"
        )
        with engine.connect() as conn_debug:
            table_actually_exists = conn_debug.execute(test_existence_query).scalar()
            print(
                f"DEBUG (fetch_weather in database_operations.py): Table 'external_weather_aarhus' exists according to this engine? {table_actually_exists}"
            )

        from sqlalchemy import text as sql_text

        weather_df = pd.read_sql_query(sql=sql_text(query).bindparams(**params), con=engine)
        print(f"  Successfully fetched {len(weather_df)} raw external weather records.")

        if weather_df.empty:
            return pd.DataFrame()

        if time_col_name not in weather_df.columns:
            print(f"  Error: Time column '{time_col_name}' not found in external weather data.")
            return pd.DataFrame()

        weather_df[time_col_name] = pd.to_datetime(weather_df[time_col_name], utc=True)
        weather_df = weather_df.set_index(time_col_name)

        if not weather_df.index.is_unique:
            print(
                "  Warning: Duplicate timestamps found in external weather data. Keeping first occurrence."
            )
            weather_df = weather_df[~weather_df.index.duplicated(keep="first")]

        print(f"  Resampling external weather data to target frequency: {target_frequency}")
        weather_df_resampled = weather_df.resample(target_frequency).ffill()
        print(f"  Resampled external weather data shape: {weather_df_resampled.shape}")
        return weather_df_resampled

    except Exception as e:
        print(f"  Error fetching or preparing external weather data: {e}")
        return pd.DataFrame()


def fetch_and_prepare_energy_prices_for_era(
    era_start_date_str: str,
    era_end_date_str: str,
    target_frequency: str,
    common_config: dict[str, Any],
    engine,
) -> pd.DataFrame:
    """Fetches energy prices from the database, filters by area, sets DatetimeIndex, and resamples."""

    energy_db_table = common_config.get("energy_db_table", "public.external_energy_prices_dk")
    energy_time_col = common_config.get("energy_time_col", "HourUTC")
    energy_price_area_col = common_config.get("energy_price_area_col", "PriceArea")
    energy_spot_price_col = common_config.get("energy_spot_price_col", "SpotPriceDKK")
    target_price_area = common_config.get("target_price_area", "DK1")
    output_spot_price_col_name = "spot_price_dkk_mwh"

    print(
        f"  Fetching external energy prices for range: {era_start_date_str} to {era_end_date_str} for area {target_price_area}"
    )

    query = f'''
    SELECT "{energy_time_col}" AS time, "{energy_spot_price_col}" AS {output_spot_price_col_name}
    FROM {energy_db_table}
    WHERE "{energy_time_col}" >= :start_date AND "{energy_time_col}" <= :end_date
    AND "{energy_price_area_col}" = :price_area
    ORDER BY "{energy_time_col}" ASC;
    '''
    params = {
        "start_date": era_start_date_str,
        "end_date": era_end_date_str,
        "price_area": target_price_area,
    }

    try:
        from sqlalchemy import text as sql_text

        energy_df = pd.read_sql_query(sql=sql_text(query).bindparams(**params), con=engine)
        print(
            f"  Successfully fetched {len(energy_df)} raw external energy price records for {target_price_area}."
        )

        if energy_df.empty:
            return pd.DataFrame()

        if "time" not in energy_df.columns:
            print("  Error: Standardized time column 'time' not found in fetched energy prices.")
            return pd.DataFrame()

        energy_df["time"] = pd.to_datetime(energy_df["time"], utc=True)
        energy_df = energy_df.set_index("time")

        if not energy_df.index.is_unique:
            print(
                f"  Warning: Duplicate timestamps found in external energy prices for {target_price_area}. Keeping first occurrence."
            )
            energy_df = energy_df[~energy_df.index.duplicated(keep="first")]

        print(f"  Resampling energy prices to target frequency: {target_frequency}")
        energy_df_resampled = energy_df.resample(target_frequency).ffill()
        print(f"  Resampled energy prices shape: {energy_df_resampled.shape}")
        return energy_df_resampled

    except Exception as e:
        print(f"  Error fetching or preparing external energy prices: {e}")
        return pd.DataFrame()
