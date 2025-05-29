FROM rapidsai/base:25.04-cuda12.0-py3.10

WORKDIR /app

# Install additional dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy existing feature extraction code that we'll reuse
COPY feature/feature_utils.py /app/
COPY feature/db_utils.py /app/
COPY db_utils_optimized.py /app/

# Install Python dependencies
COPY parallel/requirements-gpu.txt /app/requirements-gpu.txt
RUN pip install --no-cache-dir -r requirements-gpu.txt

# Copy parallel worker code
COPY parallel/worker_base.py /app/
COPY parallel/gpu_worker.py /app/

# Create logs directory
RUN mkdir -p /app/logs

CMD ["python", "gpu_worker.py"]