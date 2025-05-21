# Dockerfile for GPU-accelerated feature extraction
# Located in DataIngestion/feature_extraction/feature/feature.dockerfile

# Use an official RAPIDS base image matching the desired version (25.04)
# This image includes cudf, cupy, dask-cuda, etc. installed via Conda.
# Following the tag scheme on NGC: RAPIDS_version-cudaCUDA_version-pyPYTHON_version
FROM nvcr.io/nvidia/rapidsai/base:25.04-cuda12.0-py3.10

WORKDIR /app

# Install uv using the pip from the base RAPIDS conda environment
RUN pip install --no-cache-dir uv

# Copy requirements file
COPY requirements.txt .

# Install remaining dependencies from requirements.txt using uv
# The --system flag tells uv to install into the current environment
RUN uv pip install --system --no-cache -r requirements.txt

ENV USE_GPU=true

# Copy essential Python source files required for the application.
# These typically include: __init__.py, config.py, db_utils.py, extract_features.py, feature_utils.py, etc.
# The wildcard ensures all .py files in the current directory context are copied.
COPY *.py .

# Run the script using the python from the main RAPIDS conda environment
CMD ["uv", "run", "extract_features.py" ]
