FROM python:3.9-slim

# build tools for any wheels that need compiling
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch (skips the big CUDA chain)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.2.2

RUN pip install --no-cache-dir --upgrade uv

WORKDIR /app

COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

RUN apt-get update && \
    apt-get install -y --no-install-recommends postgresql-client && \
    rm -rf /var/lib/apt/lists/*

COPY db_utils.py .
COPY processing_steps.py .
COPY database_operations.py .
COPY data_preparation_utils.py .
COPY data_enrichment_utils.py .
COPY preprocess.py .
COPY fetch_external_weather.py .
COPY fetch_energy.py .
COPY phenotype_ingest.py .
COPY phenotype.json .
COPY phenotype.schema.json .
COPY create_preprocessed_hypertable.sql /app/create_preprocessed_hypertable.sql

CMD ["sh", "-c", "\
     psql postgresql://${DB_USER:-postgres}:${DB_PASSWORD:-postgres}@${DB_HOST:-db}:${DB_PORT:-5432}/${DB_NAME:-postgres} \
          -f /app/create_preprocessed_hypertable.sql && \
     python preprocess.py"]
