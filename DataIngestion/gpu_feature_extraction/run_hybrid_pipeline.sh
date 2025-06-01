#!/bin/bash
# Run the hybrid GPU feature extraction pipeline

set -e

echo "=== Hybrid GPU Feature Extraction Pipeline ==="
echo

# Default values
START_DATE="${START_DATE:-2014-01-01}"
END_DATE="${END_DATE:-2014-12-31}"
DATABASE_URL="${DATABASE_URL:-postgresql://postgres:postgres@localhost:5432/postgres}"
USE_DOCKER="${USE_DOCKER:-true}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --start-date)
            START_DATE="$2"
            shift 2
            ;;
        --end-date)
            END_DATE="$2"
            shift 2
            ;;
        --database-url)
            DATABASE_URL="$2"
            shift 2
            ;;
        --no-docker)
            USE_DOCKER="false"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD] [--database-url URL] [--no-docker]"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Start Date: $START_DATE"
echo "  End Date: $END_DATE"
echo "  Database URL: $DATABASE_URL"
echo "  Use Docker: $USE_DOCKER"
echo

if [ "$USE_DOCKER" = "true" ]; then
    echo "=== Building Docker Images ==="
    
    # Build Python GPU image
    echo "Building Python GPU service..."
    docker build -f Dockerfile.python-gpu -t gpu-feature-python:latest . || {
        echo "Failed to build Python GPU image"
        exit 1
    }
    
    # Build Rust image
    echo "Building Rust pipeline..."
    docker build -f Dockerfile -t gpu-feature-rust:latest . || {
        echo "Failed to build Rust image"
        exit 1
    }
    
    echo
    echo "=== Running Hybrid Pipeline with Docker Compose ==="
    
    # Create temporary docker-compose override
    cat > docker-compose.hybrid.override.yml << EOF
version: '3.8'

services:
  gpu-feature-rust-hybrid:
    environment:
      - DATABASE_URL=$DATABASE_URL
    command: ["--hybrid-mode", "--start-date", "$START_DATE", "--end-date", "$END_DATE"]
EOF
    
    # Run with docker-compose
    docker-compose -f docker-compose.hybrid.yml -f docker-compose.hybrid.override.yml up --abort-on-container-exit
    
    # Cleanup
    rm -f docker-compose.hybrid.override.yml
    
else
    echo "=== Running Hybrid Pipeline Locally ==="
    
    # Check if Python script exists
    if [ ! -f "minimal_gpu_features.py" ]; then
        echo "Error: minimal_gpu_features.py not found"
        exit 1
    fi
    
    # Check Python dependencies
    echo "Checking Python dependencies..."
    python3 -c "import cudf, cupy" 2>/dev/null || {
        echo "Warning: GPU Python dependencies not found. Install with:"
        echo "  pip install cudf-cu11 cupy-cuda11x"
        echo "Continuing with CPU fallback..."
    }
    
    # Build Rust binary
    echo "Building Rust binary..."
    cargo build --release
    
    # Run the hybrid pipeline
    echo "Starting hybrid pipeline..."
    DATABASE_URL="$DATABASE_URL" \
    RUST_LOG=info \
    cargo run --release -- \
        --hybrid-mode \
        --start-date "$START_DATE" \
        --end-date "$END_DATE"
fi

echo
echo "=== Pipeline Complete ==="