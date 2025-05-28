# Start from the official MLflow image
FROM ghcr.io/mlflow/mlflow:v2.9.2

# Install the PostgreSQL database driver
# Using psycopg2-binary for easier installation
RUN pip install --no-cache-dir psycopg2-binary

# Create artifact directory and set permissions for default user (usually UID 1000)
RUN mkdir -p /mlflow/artifacts && chown 1000:1000 /mlflow/artifacts

# The original entrypoint/cmd should be inherited 