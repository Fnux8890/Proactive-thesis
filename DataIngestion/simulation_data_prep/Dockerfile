# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install PostgreSQL client development libraries (needed for psycopg)
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y \
    libpq5 \
    libpq-dev \
    # Add build-essential potentially if C compilation is needed during pip install
    # build-essential \
    # Clean up apt cache to reduce image size
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt using uv
# Using --system allows uv to manage packages globally within the container
RUN uv pip install --system -r requirements.txt

# Copy the source code into the container
COPY src/ ./src/
# Copy plant configuration for the Flow and tests
COPY plant_config.json ./plant_config.json
# Copy Great Expectations config for validation
COPY great_expectations ./great_expectations
# Copy tests for pytest
COPY src/tests ./src/tests

# Command to run the application
CMD ["python", "src/main.py"] 