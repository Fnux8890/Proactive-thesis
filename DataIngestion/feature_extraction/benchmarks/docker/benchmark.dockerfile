# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for PostgreSQL client and Python packages
# libpq-dev: PostgreSQL client library development files
# gcc: C compiler for building native extensions
# python3-dev: Python development headers for building C extensions
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY benchmark_requirements.txt .

# Upgrade pip to latest version for better package resolution and binary wheel support
RUN pip install --no-cache-dir --upgrade pip

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r benchmark_requirements.txt

# Copy the benchmark script and the db_utils modules into the container
COPY benchmark_db_utils.py .
COPY db_utils_optimized.py .
# Copy the entire feature directory to make it a Python package
# This is needed for imports like `from feature.db_utils import ...` in benchmark_db_utils.py
COPY ./feature ./feature

# Define environment variables for database connection (can be overridden in docker-compose)
ENV DB_USER=postgres
ENV DB_PASSWORD=postgres
ENV DB_HOST=db
ENV DB_PORT=5432
ENV DB_NAME=postgres
ENV PYTHONUNBUFFERED=1

# Command to run the benchmark script when the container launches
CMD ["python", "benchmark_db_utils.py"]
