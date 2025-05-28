# Start from the official Prefect image
FROM prefecthq/prefect:3-latest

# Add curl for healthcheck
RUN apt-get update && apt-get install -y curl --no-install-recommends && rm -rf /var/lib/apt/lists/*

# Install the PostgreSQL database drivers
# psycopg2-binary: Needed for SQLAlchemy to recognize the postgresql+psycopg2 scheme
# asyncpg: Required by Prefect Server's internal async engine
RUN pip install --no-cache-dir psycopg2-binary asyncpg

# The original entrypoint/cmd should be inherited 