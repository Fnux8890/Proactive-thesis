# Data Ingestion Pipeline Setup Guide

This guide details the setup and orchestration of a data ingestion pipeline using Docker Compose. In this pipeline, Apache Airflow is used to schedule and orchestrate tasks, while custom Docker images handle data cleaning and wrangling. Processed data is then stored in a time-series database such as **InfluxDB** or **TimescaleDB** for further analysis.

## Architecture Overview

The pipeline consists of the following components:

- **Apache Airflow:** Orchestrates and schedules ingestion tasks.
- **Data Processing Service:** A custom Docker container that handles data cleaning and wrangling.
- **Time-Series Database:** Stores processed data. We provide configurations for either InfluxDB or TimescaleDB.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed on your system.
- Docker Compose capability (using the `docker compose` command on Windows with Docker Desktop).
- (Optional) For Python package management within containers, use `uv`, e.g. `uv pip install package_name`.

## Setup Instructions

1. **Customize Configuration:**
   - Create a `.env` file if required to hold environment-specific configurations (e.g., database credentials, Airflow settings).
   - Update any configuration files as needed for your environment.

2. **Build and Start the Pipeline:**

   ```powershell
   docker compose up --build
   ```

   This command builds the Docker images and starts all defined services.

3. **Accessing Services:**
   - **Airflow Webserver:** Accessible at [http://localhost:8080](http://localhost:8080).
   - **Time-Series Database:** Depending on your configuration:
     - For **InfluxDB**, the default port is `8086`.
     - For **TimescaleDB** (typically based on PostgreSQL), the default port is `5432`.

## Folder Structure

The recommended directory layout for this project is:

DataIngestion/
├── airflow_dags/         # Contains the Airflow DAGs.
├── data_processing/      # Contains the data processing code and its Dockerfile.
├── .env                  # Environment variables for Docker Compose.
├── docker-compose.yml    # Docker Compose configuration file.
└── README.md             # This documentation file.

## Docker Compose and Service Configuration

Below is a sample snippet of a `docker-compose.yml` that outlines the pipeline structure:

```yaml
services:
  airflow:
    image: apache/airflow:2.3.0
    environment:
      - AIRFLOW__CORE__EXECUTOR=LocalExecutor
      - AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@db:5432/airflow
    ports:
      - "8080:8080"
    depends_on:
      - db

  data_processing:
    build: ./data_processing
    depends_on:
      - airflow
      - db
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/datadb

  db:
    image: timescale/timescaledb:latest-pg12
    environment:
      - POSTGRES_USER=airflow
      - POSTGRES_PASSWORD=airflow
      - POSTGRES_DB=airflow
    ports:
      - "5432:5432"
```

## Additional Notes

- **Airflow DAGs:** Place your DAG files in the directory mapped in your Airflow container.
- **Custom Dockerfile for Data Processing:** Ensure that your `./data_processing` directory includes a valid Dockerfile that sets up your data cleaning and wrangling scripts.
- **Persistence:** Consider adding volume mappings in your `docker-compose.yml` to persist data across container restarts.

## TimescaleDB Initialization and Migrations

The pipeline uses TimescaleDB with automated database migrations for schema management and initialization. Here's how it works:

### Migration Scripts

Migration scripts are located in the `database_migrations` directory and are executed automatically when the database container starts for the first time. The scripts are executed in numerical order (based on filename prefix). 

The migration structure:
- `01_create_schemas.sql` - Creates the base schemas and extensions
- `02_create_staging_tables.sql` - Creates tables for incoming raw data
- `03_create_timeseries_tables.sql` - Creates hypertables for time-series data
- `04_create_feature_tables.sql` - Creates tables for processed features

### How Migrations Work

1. Migration scripts are mounted to the container's `/docker-entrypoint-initdb.d` directory
2. PostgreSQL automatically executes all scripts in this directory during initialization
3. Each migration logs its execution in the `metadata.schema_migrations` table
4. Extensions like PostGIS and UUID-OSSP are automatically enabled

### Adding New Migrations

To add a new migration:
1. Create a SQL file with a sequential prefix (e.g., `05_...sql`)
2. Place it in the `database_migrations` directory
3. Restart the database container or run `docker-compose down -v && docker-compose up -d timescaledb` to test

## Troubleshooting

- Verify environment variables in your `.env` file.
- Check service logs using `docker compose logs <service_name>`.
- Ensure that no port conflicts exist on your host system.

## File Format Handling

The ingestion pipeline supports multiple file formats with robust detection and processing capabilities:

### CSV Files
- Automatic delimiter detection (supports comma, semicolon, tab, and pipe delimiters)
- Multiple encoding support (UTF-8 with fallback to Latin-1 and other encodings)
- Robust error handling for malformed CSV files

### JSON Files
- Support for different JSON structures:
  - JSON arrays of objects (standard format)
  - Single JSON objects
  - Nested JSON structures with automatic flattening
  - Special handling for common patterns like data/results/items arrays

### Excel Files
- Support for multiple sheets
- Automatic header detection and mapping

## Verification Mechanism

The Elixir ingestion service now includes a robust verification system:

1. **Verification Files**: For each processed file, a detailed verification report is generated in the `/verification` directory, including:
   - Timestamps and processing metadata
   - File format details (delimiter, encoding, structure)
   - Record counts and sample data
   - Success/failure status and any error messages

2. **Summary File**: A master summary file (`ingestion_summary.txt`) tracks all files processed with their status.

3. **External Access**: All verification files are mounted outside the Docker container for easy inspection.

To access verification files:
```bash
# View the summary file
cat ./verification/ingestion_summary.txt

# List all verification reports
ls -la ./verification
```

## Conclusion

This pipeline setup provides a robust starting point for ingesting, processing, and storing time-series data. Feel free to extend and modify the configuration as needed to fit your requirements.
