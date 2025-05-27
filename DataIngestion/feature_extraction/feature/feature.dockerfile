# syntax=docker/dockerfile:1.4

# Stage 1: Dependencies only (rebuild only when requirements.txt changes)
FROM nvcr.io/nvidia/rapidsai/base:25.04-cuda12.0-py3.10 AS dependencies

WORKDIR /app

# Install system dependencies and Python tools
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install uv ruff black mypy

# Copy requirements files
COPY requirements.txt .
COPY ../pyproject.toml .

# Install Python dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

# Stage 2: Code validation
FROM dependencies AS validator

WORKDIR /app

# Copy all source code for validation
COPY . .
COPY ../features ./features
COPY ../db ./db
COPY ../backend ./backend

# Run validation checks
RUN echo "Running code quality checks..." && \
    black --check . && \
    ruff check . && \
    echo "✓ Code quality checks passed"

# Stage 3: Final runtime image
FROM dependencies AS runtime

WORKDIR /app

# Environment variables
ENV USE_GPU=true
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Copy validated code from validator stage
COPY --from=validator /app .

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs

# Add validation script to PATH
RUN chmod +x validate_pipeline.py && \
    ln -s /app/validate_pipeline.py /usr/local/bin/validate-pipeline

# Default command with pre-validation
CMD ["sh", "-c", "validate-pipeline && python extract_features.py"]