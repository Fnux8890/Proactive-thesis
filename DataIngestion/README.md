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

## Troubleshooting

- Verify environment variables in your `.env` file.
- Check service logs using `docker compose logs <service_name>`.
- Ensure that no port conflicts exist on your host system.

## Conclusion

This pipeline setup provides a robust starting point for ingesting, processing, and storing time-series data. Feel free to extend and modify the configuration as needed to fit your requirements.
