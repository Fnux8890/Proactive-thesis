# Start from the official MLflow image
FROM ghcr.io/mlflow/mlflow:v2.9.2

# Install the PostgreSQL database driver
# Using psycopg2-binary for easier installation
RUN pip install --no-cache-dir psycopg2-binary

# The original entrypoint/cmd should be inherited 