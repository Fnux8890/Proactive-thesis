FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy existing feature extraction code
COPY feature/feature_utils.py /app/
COPY feature/db_utils.py /app/
COPY db_utils_optimized.py /app/

# Install Python dependencies
COPY parallel/requirements-cpu.txt /app/requirements-cpu.txt
RUN pip install --no-cache-dir -r requirements-cpu.txt

# Copy parallel worker code
COPY parallel/worker_base.py /app/
COPY parallel/cpu_worker.py /app/

# Create logs directory
RUN mkdir -p /app/logs

CMD ["python", "cpu_worker.py"]