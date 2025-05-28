# Dockerfile for GPU-accelerated feature extraction
# Located in DataIngestion/feature_extraction/feature-gpu/feature_gpu.dockerfile

# Use an official RAPIDS base image matching the desired version (25.04)
# This image includes cudf, cupy, dask-cuda, etc. installed via Conda.
# Following the tag scheme on NGC: RAPIDS_version-cudaCUDA_version-pyPYTHON_version
FROM nvcr.io/nvidia/rapidsai/base:25.04-cuda12.0-py3.10

WORKDIR /app

# Add conda paths to PATH explicitly (important for finding executables)
ENV PATH=/opt/conda/bin:/opt/conda/condabin:${PATH}

# The base image uses conda. The PATH is set to include /opt/conda/bin.
# Try installing uv using pip (assuming it's on the PATH from the base conda env).
RUN echo "--- Installing uv using pip (from PATH) ---" && \
    pip install uv

# Copy requirements (which should now ONLY contain non-RAPIDS deps)
COPY requirements_gpu.txt .

# Try installing requirements using uv (assuming it's on the PATH after previous step).
RUN echo "--- Installing requirements using uv (from PATH) ---" && \
    uv pip install --system --no-cache -r requirements_gpu.txt

COPY db_utils.py .
COPY extract_features_gpu.py .

# Execute the python script directly using the python from the PATH
CMD ["python", "extract_features_gpu.py"]
