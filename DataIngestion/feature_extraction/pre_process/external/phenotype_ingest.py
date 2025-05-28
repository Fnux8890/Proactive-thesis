#!/usr/bin/env -S uv run --isolated
# /// script
# dependencies = ["SQLAlchemy", "psycopg2-binary", "ijson", "pandas"] # pandas for db_utils if used fully
# ///

import json
import os
from collections.abc import Iterable, Iterator  # Added Iterable, Iterator
from pathlib import Path
from typing import Any

import ijson  # For streaming JSON parsing
from db_utils import (
    SQLAlchemyPostgresConnector,
)  # Assuming db_utils.py is in the same directory or PYTHONPATH
from sqlalchemy import exc as sqlalchemy_exc
from sqlalchemy import text

# --- Configuration ---
# Attempt to get the script's directory for relative paths
try:
    SCRIPT_DIR = Path(__file__).parent
except NameError:  # Fallback for interactive environments
    SCRIPT_DIR = Path.cwd()

PHENOTYPE_JSON_PATH = SCRIPT_DIR / "phenotype.json"
PHENOTYPE_TABLE_NAME = "literature_kalanchoe_phenotypes"
INSERT_BATCH_SIZE = 1000  # Number of records to insert in one DB transaction

# Database connection details (can be environment variables or hardcoded for the script)
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_HOST = os.getenv("DB_HOST", "db")  # Assuming 'db' is your TimescaleDB service name in Docker
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")


def create_literature_phenotypes_table(engine):
    """
    Creates the literature_kalanchoe_phenotypes table in TimescaleDB.
    - `entry_id` is the PRIMARY KEY. Its constraint is explicitly named (e.g., using the
      pattern '{table_name}_pkey', resulting in '{PHENOTYPE_TABLE_NAME}_pkey').
    - `uq_phenotype_entry` provides semantic uniqueness for a combination of other fields.

    Regarding INSERT ON CONFLICT:
    The `CREATE TABLE` statement defines constraints. To handle violations of these constraints
    (e.g., on `entry_id` (primary key) or `uq_phenotype_entry` (unique constraint))
    during `INSERT` operations without aborting a batch, the `INSERT` statement
    itself must include an `ON CONFLICT` clause.

    For example:
    - To handle primary key conflicts: `INSERT ... ON CONFLICT (entry_id) DO NOTHING;`
      (or `DO UPDATE SET ...`).
    - To handle `uq_phenotype_entry` conflicts:
      `INSERT ... ON CONFLICT ON CONSTRAINT uq_phenotype_entry DO NOTHING;` (or `DO UPDATE SET ...`).
    - To generally ignore rows causing any conflict: `INSERT ... ON CONFLICT DO NOTHING;`.

    A single `INSERT` statement's `ON CONFLICT` clause targets one specific conflict condition
    (e.g., a specific constraint or set of columns forming a unique index).
    It is not possible to list multiple, distinct constraints (like the primary key on `entry_id`
    and the `uq_phenotype_entry` constraint) as a single, combined conflict target within
    one `ON CONFLICT` specification in an `INSERT` statement. The choice of `ON CONFLICT`
    strategy within the `INSERT` statement is key to achieving the desired conflict resolution.
    """
    table_sql = f"""
    CREATE TABLE IF NOT EXISTS public.{PHENOTYPE_TABLE_NAME} (
        entry_id INTEGER NOT NULL,
        publication_source VARCHAR(255),
        species VARCHAR(255),
        cultivar_or_line VARCHAR(255),
        experiment_description TEXT,
        phenotype_name VARCHAR(255) NOT NULL,
        phenotype_value REAL,
        phenotype_unit VARCHAR(50),
        measurement_condition_notes TEXT,
        control_group_identifier VARCHAR(255),
        environment_temp_day_C REAL,
        environment_temp_night_C REAL,
        environment_photoperiod_h REAL,
        environment_DLI_mol_m2_d REAL,
        environment_light_intensity_umol_m2_s REAL,
        environment_light_quality_description TEXT,
        environment_CO2_ppm REAL,
        environment_RH_percent REAL,
        environment_stressor_type VARCHAR(100),
        environment_stressor_level VARCHAR(100),
        data_extraction_date TIMESTAMPTZ DEFAULT NOW(), -- DB default if not provided
        additional_notes TEXT,
        CONSTRAINT {PHENOTYPE_TABLE_NAME}_pkey PRIMARY KEY (entry_id),
        CONSTRAINT uq_phenotype_entry UNIQUE (
            publication_source, species, cultivar_or_line,
            experiment_description, phenotype_name, phenotype_value,
            measurement_condition_notes, environment_temp_day_C, environment_DLI_mol_m2_d, environment_photoperiod_h,
            environment_temp_night_C
        )
    );
    """
    try:
        with engine.begin() as connection:
            connection.execute(text(table_sql))
            connection.commit()
        print(f"Table '{PHENOTYPE_TABLE_NAME}' created successfully or already exists.")
    except Exception as e:
        print(f"Error creating table '{PHENOTYPE_TABLE_NAME}': {e}")
        raise


def convert_to_hypertable_optional(
    engine, table_name: str, time_column_name: str = "data_extraction_date"
):
    """
    Optionally converts the specified table to a TimescaleDB hypertable.
    """
    check_hyper_sql = f"""
    SELECT EXISTS (
        SELECT 1
        FROM timescaledb_information.hypertables
        WHERE hypertable_schema = 'public' AND hypertable_name = '{table_name}'
    );
    """
    try:
        with engine.connect() as connection:
            is_hypertable = connection.execute(text(check_hyper_sql)).scalar_one_or_none()

        if is_hypertable:
            print(f"Table '{table_name}' is already a hypertable.")
            return

        hypertable_sql = f"SELECT create_hypertable('public.{table_name}', '{time_column_name}', if_not_exists => TRUE, migrate_data => TRUE);"
        with engine.begin() as connection:
            connection.execute(text(hypertable_sql))
        print(
            f"Table '{table_name}' successfully converted/confirmed as a hypertable on column '{time_column_name}'."
        )
    except Exception as e:
        print(f"Warning: Could not convert '{table_name}' to hypertable (or check status): {e}")
        print(
            "This might be okay if TimescaleDB specific features are not strictly needed for this table, or if the extension is not enabled."
        )


def load_json_data_stream(json_path: Path) -> Iterator[dict[str, Any]]:
    """Loads data from a JSON file by streaming records one by one using ijson."""
    print(f"Streaming phenotype data from: {json_path}")
    if not json_path.exists():
        print(f"Error: Phenotype JSON file not found at {json_path}")
        return iter([])  # Return an empty iterator
    try:
        with open(json_path, encoding="utf-8") as f:  # Use text mode for json.load
            data = json.load(f)

        phenotype_records = data.get("phenotype")

        if not isinstance(phenotype_records, list):
            print(f"Error: Key 'phenotype' in {json_path} did not contain a list of records.")
            return iter([])

        print(
            f"Successfully loaded {len(phenotype_records)} records from {json_path} using json.load()."
        )
        # Yield records one by one to keep the generator interface for insert_phenotype_data_bulk
        yield from phenotype_records

    except ijson.JSONError as e:  # More specific error for ijson parsing issues
        raise RuntimeError(f"Malformed phenotype JSON at {json_path} during streaming: {e}") from e
    except json.JSONDecodeError as e:  # Catch standard json lib decode errors
        print(f"Error decoding JSON from {json_path} using json.load(): {e}")
        raise RuntimeError(f"Malformed phenotype JSON at {json_path}: {e}") from e
    except Exception as e:
        print(f"An unexpected error occurred while streaming {json_path}: {e}")
        # Depending on the error, you might want to reraise or handle differently
        return iter([])  # Return an empty iterator on other errors


def insert_phenotype_data_bulk(
    engine,
    data_stream: Iterable[dict[str, Any]],
    table_name: str,
    batch_size: int = INSERT_BATCH_SIZE,
):
    """
    Inserts phenotype data from a stream into the specified table using batched executemany,
    reusing a single database connection for all batches. Each batch is its own transaction.
    """
    total_processed_records_from_stream = 0
    total_skipped_due_to_missing_pk = 0
    total_batches_processed = 0
    total_rows_attempted_in_db = 0  # Renamed from total_rows_affected_in_db

    batch_params: list[dict[str, Any]] = []

    column_names = [
        "entry_id",
        "publication_source",
        "species",
        "cultivar_or_line",
        "experiment_description",
        "phenotype_name",
        "phenotype_value",
        "phenotype_unit",
        "measurement_condition_notes",
        "control_group_identifier",
        "environment_temp_day_C",
        "environment_temp_night_C",
        "environment_photoperiod_h",
        "environment_DLI_mol_m2_d",
        "environment_light_intensity_umol_m2_s",
        "environment_light_quality_description",
        "environment_CO2_ppm",
        "environment_RH_percent",
        "environment_stressor_type",
        "environment_stressor_level",
        "data_extraction_date",
        "additional_notes",
    ]

    stmt_template_final = f"""
    INSERT INTO public.{table_name} ({", ".join(column_names)})
    VALUES ({", ".join([f":{col}" for col in column_names])})
    ON CONFLICT DO NOTHING;
    """

    with engine.connect() as connection:  # A single connection for all batches
        for record in data_stream:
            total_processed_records_from_stream += 1
            if record.get("entry_id") is None:
                print(
                    f"Skipping record (stream index approx {total_processed_records_from_stream - 1}) due to missing 'entry_id'."
                )
                total_skipped_due_to_missing_pk += 1
                continue

            params = {}
            for col_name in column_names:
                params[col_name] = record.get(col_name)
                if params[col_name] is None and col_name in [
                    "phenotype_value",
                    "environment_temp_day_C",
                    "environment_temp_night_C",
                    "environment_photoperiod_h",
                    "environment_DLI_mol_m2_d",
                    "environment_light_intensity_umol_m2_s",
                    "environment_CO2_ppm",
                    "environment_RH_percent",
                ]:
                    params[col_name] = (
                        None  # Explicitly set to SQL NULL if missing and nullable numeric
                    )
            batch_params.append(params)

            if len(batch_params) >= batch_size:
                current_batch_rowcount = None
                try:
                    # Each batch is its own transaction using the single connection
                    with connection.begin():
                        connection.execute(text(stmt_template_final), batch_params)
                        # Update row counting to use len(batch_params)
                        total_rows_attempted_in_db += len(batch_params)
                        current_batch_rowcount = len(batch_params)  # Rows attempted in this batch
                    print(
                        f"Processed batch of {len(batch_params)}. DB rows attempted in batch: {current_batch_rowcount}"
                    )
                except sqlalchemy_exc.IntegrityError as ie:
                    print(
                        f"IntegrityError during a batch insert (first entry_id: {batch_params[0].get('entry_id', 'N/A')}). Batch skipped. Error: {ie}"
                    )
                except Exception as e:
                    print(
                        f"Error during a batch insert (first entry_id: {batch_params[0].get('entry_id', 'N/A')}): {e}. Batch skipped."
                    )
                finally:
                    total_batches_processed += 1
                    batch_params = []

        if batch_params:  # Process the last batch
            current_batch_rowcount = None
            try:
                # Final batch is also its own transaction
                with connection.begin():
                    connection.execute(text(stmt_template_final), batch_params)
                    # Update row counting for the final batch
                    total_rows_attempted_in_db += len(batch_params)
                    current_batch_rowcount = len(batch_params)  # Rows attempted in this batch
                print(
                    f"Processed final batch of {len(batch_params)}. DB rows attempted in batch: {current_batch_rowcount}"
                )
            except sqlalchemy_exc.IntegrityError as ie:
                print(
                    f"IntegrityError during final batch insert (first entry_id: {batch_params[0].get('entry_id', 'N/A')}). Batch skipped. Error: {ie}"
                )
            except Exception as e:
                print(
                    f"Error during final batch insert (first entry_id: {batch_params[0].get('entry_id', 'N/A')}): {e}. Batch skipped."
                )
            finally:
                total_batches_processed += 1

    print("Bulk data insertion attempt finished.")
    print(f"  Total records processed from stream: {total_processed_records_from_stream}")
    print(
        f"  Records skipped due to missing 'entry_id' before batching: {total_skipped_due_to_missing_pk}"
    )
    print(f"  Total batches processed: {total_batches_processed}")
    print(f"  Total DB rows attempted by INSERTs: {total_rows_attempted_in_db}")
    print(
        "  Note: Exact count of newly inserted vs. skipped-by-conflict rows is best verified by querying the DB."
    )


if __name__ == "__main__":
    print("--- Starting Phenotype Data Ingestion Script ---")

    db_connector = SQLAlchemyPostgresConnector(
        user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT, db_name=DB_NAME
    )

    if db_connector.engine:
        print("Database engine created successfully.")

        # 1. Create the table
        create_literature_phenotypes_table(db_connector.engine)

        # 2. Optionally convert to hypertable (if TimescaleDB is enabled and it makes sense for this table)
        # For a literature table, hypertable might be overkill unless 'data_extraction_date' is heavily used for time-series queries.
        # convert_to_hypertable_optional(db_connector.engine, PHENOTYPE_TABLE_NAME, "data_extraction_date")

        # 3. Load data from JSON file
        phenotype_data_stream = load_json_data_stream(PHENOTYPE_JSON_PATH)

        # 4. Insert data into the table
        if phenotype_data_stream:
            insert_phenotype_data_bulk(
                db_connector.engine, phenotype_data_stream, PHENOTYPE_TABLE_NAME
            )
        else:
            print("No phenotype data loaded from JSON stream, skipping database insertion.")

        # Clean up the engine created by this script instance
        db_connector.engine.dispose()
        print("Database engine disposed.")
    else:
        print(
            "Failed to create database engine. Please check connection details and database server."
        )
    print("--- Phenotype Data Ingestion Script Finished ---")
