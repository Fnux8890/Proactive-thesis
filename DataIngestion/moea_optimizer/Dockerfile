FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libboost-dev \
    libboost-system-dev \
    libboost-filesystem-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    numpy==1.26.* \
    pandas==2.* \
    scipy==1.* \
    matplotlib==3.* \
    seaborn==0.13.* \
    joblib==1.* \
    lightgbm \
    pymoo==0.6.* \
    tomli \
    sqlalchemy==2.* \
    psycopg2-binary \
    plotly==5.* \
    mlflow-skinny \
    tabulate

# Create working directory
WORKDIR /app

# Copy source code
COPY src/ ./src/
COPY config/ ./config/

# Create models directory (will be mounted at runtime via docker-compose)
RUN mkdir -p ./models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Default command - can be overridden
CMD ["python", "-m", "src.run_optimization"]