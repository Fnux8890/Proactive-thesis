# Base Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install uv
RUN pip install uv

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies using uv
RUN uv pip install --system --no-cache -r requirements.txt
RUN apt-get update && apt-get install -y postgresql-client && rm -rf /var/lib/apt/lists/*

# Copy all application python files from the pre_process directory
COPY db_utils.py .
COPY processing_steps.py .
COPY preprocess.py .
COPY create_preprocessed_hypertable.sql /app/create_preprocessed_hypertable.sql

# Config will be mounted via volume in docker-compose

# CMD will be provided by docker-compose, but if run directly, it would be:
CMD ["sh", "-c", "psql postgresql://${DB_USER:-postgres}:${DB_PASSWORD:-postgres}@${DB_HOST:-db}:${DB_PORT:-5432}/${DB_NAME:-postgres} -f /app/create_preprocessed_hypertable.sql && uv run python preprocess.py"]
