# Use a specific stable Python version
FROM python:3.11-slim

WORKDIR /app

# Install uv globally using pip
# Using --no-cache-dir is good practice for smaller layers
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir uv

# Copy the script (from ./data_analysis/config_analyser.py relative to build context)
COPY ./data_analysis/config_analyser.py ./analysis_scripts/config_analyser.py

# Copy the main configuration file (from ./data_processing_config.json relative to build context)
COPY ./data_processing_config.json ./input_config/data_processing_config.json

# The script uses #!/usr/bin/env -S uv run --isolated and has embedded dependencies.
# So, we can run it directly with "uv run <script_path>".
# No separate requirements.txt or pip install for script dependencies needed in Dockerfile.
CMD ["uv", "run", "./analysis_scripts/config_analyser.py"]



