#!/bin/bash
# Simulate cloud deployment of parallel feature extraction
# This script tests the setup with cloud-like parameters

echo "=== Simulating Cloud Deployment of Feature Extraction ==="
echo "This simulates running on a cloud instance with:"
echo "- 16-32 CPU cores"
echo "- GPU support (if available)"
echo "- Large batch sizes"
echo "- Comprehensive feature set"
echo ""

# Set cloud-optimized parameters
export BATCH_SIZE=500
export N_JOBS=-1  # Use all available cores
export USE_GPU=false  # Set to true if GPU available
export FEATURE_SET=comprehensive
export MIN_ERA_ROWS=200

# Show configuration
echo "Configuration:"
echo "- BATCH_SIZE: $BATCH_SIZE"
echo "- N_JOBS: $N_JOBS (all available cores)"
echo "- USE_GPU: $USE_GPU"
echo "- FEATURE_SET: $FEATURE_SET"
echo "- MIN_ERA_ROWS: $MIN_ERA_ROWS"
echo ""

# Check if we should use the cloud compose file
if [ -f "docker-compose.cloud.yml" ]; then
    echo "Using cloud-optimized docker-compose configuration..."
    COMPOSE_CMD="docker compose -f docker-compose.feature.yml -f docker-compose.cloud.yml"
else
    echo "Using standard docker-compose configuration..."
    COMPOSE_CMD="docker compose -f docker-compose.feature.yml"
fi

# Build the image
echo ""
echo "Building feature extraction image..."
$COMPOSE_CMD build feature_extraction

# Run the test
echo ""
echo "Running parallel setup test with cloud parameters..."
docker run --rm \
    -e BATCH_SIZE=$BATCH_SIZE \
    -e N_JOBS=$N_JOBS \
    -e USE_GPU=$USE_GPU \
    -e FEATURE_SET=$FEATURE_SET \
    -e MIN_ERA_ROWS=$MIN_ERA_ROWS \
    -v $(pwd)/test_parallel_setup.py:/app/test_parallel_setup.py \
    feature_extraction-feature_extraction:latest \
    python /app/test_parallel_setup.py

echo ""
echo "=== Cloud Deployment Commands ==="
echo "To deploy with these settings:"
echo ""
echo "1. For standard cloud deployment:"
echo "   $COMPOSE_CMD up -d"
echo ""
echo "2. For production cloud with monitoring:"
echo "   $COMPOSE_CMD --profile monitoring up -d"
echo ""
echo "3. To scale horizontally (future):"
echo "   $COMPOSE_CMD up -d --scale feature_extraction=4"
echo ""
echo "=== Estimated Performance ==="
echo "With cloud resources (32 cores + GPU):"
echo "- Expected throughput: 500-1000 rows/second"
echo "- Processing 1M rows: ~20-30 minutes"
echo "- Processing 10M rows: ~3-5 hours"