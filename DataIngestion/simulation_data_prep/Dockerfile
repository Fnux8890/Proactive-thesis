# syntax=docker/dockerfile:1.4
FROM python:3.10-slim AS runtime

# -----------------------------
# System dependencies & tooling
# -----------------------------
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    libpq5 \
    libpq-dev && \
    pip install --no-cache-dir uv && \
    rm -rf /var/lib/apt/lists/*

# -----------------------------
# Python dependencies
# -----------------------------
WORKDIR /app
COPY requirements.txt ./
COPY pyproject.toml .

# Install Python dependencies using uv
RUN uv pip install --system --no-cache-dir \
    --extra-index-url https://pypi.nvidia.com \
    -r requirements.txt \
    black

# Install the project in editable mode
RUN uv pip install --system --no-cache-dir -e .

# -----------------------------
# Project sources
# -----------------------------
COPY src/ ./src/
COPY plant_config.json ./plant_config.json
COPY great_expectations ./great_expectations
COPY src/tests ./src/tests
COPY dao/ ./dao/
COPY transforms/ ./transforms/
# COPY loading/ ./loading/

# -----------------------------
# Runtime env variables
# -----------------------------
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# -----------------------------
# Entrypoint
# -----------------------------
CMD ["python", "src/main.py"]