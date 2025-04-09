# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first
COPY requirements.txt .

# Install Python dependencies using uv (mounted temporarily)
# Use --mount for buildkit cache and temporary uv binary
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=from=ghcr.io/astral-sh/uv:latest,source=/uv,target=/usr/local/bin/uv \
    uv pip install --system --no-cache -r requirements.txt

# Copy the rest of the application code
COPY . .

# Run the pipeline module by default (processing all discovered files)
CMD ["python", "-m", "pipeline_pkg.pipeline"] 