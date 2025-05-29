FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY parallel/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY parallel/coordinator.py /app/
COPY parallel/worker_base.py /app/

# Create logs directory
RUN mkdir -p /app/logs

CMD ["python", "coordinator.py"]