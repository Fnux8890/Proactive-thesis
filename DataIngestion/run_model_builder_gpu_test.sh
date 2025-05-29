#!/bin/bash
# Script to test GPU availability in model_builder container

echo "Testing GPU availability in model_builder container..."
echo "=============================================="

# First, test GPU in the container
echo "Running GPU test..."
docker compose run --rm model_builder python -m src.test_gpu

echo ""
echo "=============================================="
echo "If GPU is detected, you can run the model builder with:"
echo "docker compose up model_builder"
echo ""
echo "Or for a specific model type:"
echo "docker compose --profile lightgbm up model_builder_lightgbm"