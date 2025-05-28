FROM python:3.11-slim

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

COPY core ./core
COPY utils ./utils
COPY external ./external
COPY preprocess.py .
COPY phenotype.json .
COPY phenotype.schema.json .
COPY create_preprocessed_hypertable.sql /app/create_preprocessed_hypertable.sql

CMD ["python", "preprocess.py"]
