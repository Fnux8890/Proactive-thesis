# Base Python image for a small utility script
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install uv (lightweight package installer)
RUN pip install uv

# Copy requirements if any (pandas is the main one for auto_era_config_gen.py)
# Create a small requirements_auto_era_gen.txt if needed, or install directly.
# For this script, pandas and pyarrow (for parquet) are primary.
COPY requirements_auto_era_gen.txt .
RUN uv pip install --system --no-cache -r requirements_auto_era_gen.txt

# Copy the script itself
COPY auto_era_config_gen.py .

# CMD will be provided by docker-compose.yml
# Example if run directly (not used by compose usually):
# CMD ["uv", "run", "python", "auto_era_config_gen.py", "/path/to/input.parquet", "/path/to/output.json"] 