# Start from the official Prefect image
FROM prefecthq/prefect:2-latest

# Install the PostgreSQL database drivers
# psycopg2-binary: Needed for SQLAlchemy to recognize the postgresql+psycopg2 scheme
# asyncpg: Required by Prefect Server's internal async engine
RUN pip install --no-cache-dir psycopg2-binary asyncpg

# The original entrypoint/cmd should be inherited 