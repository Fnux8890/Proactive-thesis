# Dockerfile for GPU-accelerated feature extraction
# Located in DataIngestion/feature_extraction/feature-gpu/feature_gpu.dockerfile

# Use an official RAPIDS base image matching the desired version (25.04)
# This image includes cudf, cupy, dask-cuda, etc. installed via Conda.
# Following the tag scheme on NGC: RAPIDS_version-cudaCUDA_version-pyPYTHON_version
FROM nvcr.io/nvidia/rapidsai/base:25.04-cuda12.0-py3.10

WORKDIR /app

# The base image uses conda, and the 'rapids' env should have pip.
# Activate the environment and then install uv using pip.
RUN /bin/bash -c "source /opt/conda/bin/activate rapids && pip install uv"

# Copy requirements (which should now ONLY contain non-RAPIDS deps)
COPY requirements_gpu.txt .

# Activate the environment and then install remaining dependencies using uv
RUN /bin/bash -c "source /opt/conda/bin/activate rapids && uv pip install --system --no-cache -r requirements_gpu.txt"

COPY db_utils.py .
COPY extract_features_gpu.py .

CMD ["conda", "run", "--no-capture-output", "-n", "rapids", "python", "extract_features_gpu.py"]
