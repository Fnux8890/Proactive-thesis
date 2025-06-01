#!/bin/bash

# Simple test for GPU Hybrid Pipeline
# Uses existing database from main docker-compose

set -e

echo "================================"
echo "GPU Hybrid Pipeline Simple Test"
echo "================================"

# Parameters
START_DATE="${1:-2014-01-01}"
END_DATE="${2:-2014-01-07}"  # Default to 1 week for quick test

echo "Test Period: $START_DATE to $END_DATE"

# Step 1: Build the containers
echo -e "\nBuilding containers..."
cd gpu_feature_extraction
docker compose -f docker-compose.hybrid.yml build

# Step 2: Run with external database
echo -e "\nRunning hybrid pipeline..."
docker compose -f docker-compose.hybrid.yml run --rm \
    -e DATABASE_URL=postgresql://postgres:postgres@host.docker.internal:5432/postgres \
    gpu-feature-rust-hybrid \
    --hybrid-mode \
    --start-date "$START_DATE" \
    --end-date "$END_DATE"

echo -e "\nTest complete!"