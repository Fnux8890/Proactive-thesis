#!/bin/bash

# Test GPU Hybrid Pipeline
# This script tests the Rust + Python GPU hybrid feature extraction pipeline

set -e

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}GPU Hybrid Pipeline Test${NC}"
echo -e "${BLUE}================================${NC}"

# Default parameters
START_DATE="${START_DATE:-2014-01-01}"
END_DATE="${END_DATE:-2014-01-31}"

echo -e "${GREEN}Test Period:${NC} $START_DATE to $END_DATE"

# Step 1: Ensure main database is running
echo -e "\n${YELLOW}Step 1: Checking database...${NC}"
if docker compose ps db | grep -q "running"; then
    echo -e "${GREEN}✓ Main database is running${NC}"
else
    echo -e "${YELLOW}Starting main database...${NC}"
    docker compose up -d db
    echo "Waiting for database to be ready..."
    sleep 15
fi

# Check if we have data
echo -e "\n${YELLOW}Checking for sensor data...${NC}"
RECORD_COUNT=$(docker compose exec -T db psql -U postgres -d postgres -t -c \
    "SELECT COUNT(*) FROM sensor_data WHERE time >= '$START_DATE' AND time < '$END_DATE'::date + interval '1 day'" 2>/dev/null || echo "0")

if [ "$RECORD_COUNT" -eq "0" ] || [ -z "$RECORD_COUNT" ]; then
    echo -e "${RED}No sensor data found for the test period!${NC}"
    echo "Please run the Rust ingestion pipeline first:"
    echo "  docker compose up rust-ingestion"
    exit 1
fi

echo -e "${GREEN}✓ Found $RECORD_COUNT sensor records${NC}"

# Step 2: Build the hybrid pipeline images
echo -e "\n${YELLOW}Step 2: Building Docker images...${NC}"
cd gpu_feature_extraction

# Build Python GPU image
echo "Building Python GPU feature extraction image..."
docker build -f Dockerfile.python-gpu -t gpu-feature-python:latest . || {
    echo -e "${RED}Failed to build Python GPU image${NC}"
    exit 1
}

# Build Rust hybrid pipeline image
echo "Building Rust hybrid pipeline image..."
docker build -f Dockerfile -t gpu-feature-rust-hybrid:latest . || {
    echo -e "${RED}Failed to build Rust hybrid image${NC}"
    exit 1
}

echo -e "${GREEN}✓ Images built successfully${NC}"

# Step 3: Run the hybrid pipeline
echo -e "\n${YELLOW}Step 3: Running hybrid pipeline...${NC}"

# Create docker-compose override to use main network
cat > docker-compose.hybrid.override.yml << EOF
version: '3.8'

services:
  gpu-feature-rust-hybrid:
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@db:5432/postgres
    networks:
      - pipeline-net
    external_links:
      - db
    command: ["--hybrid-mode", "--start-date", "$START_DATE", "--end-date", "$END_DATE"]

  gpu-feature-python:
    networks:
      - pipeline-net

networks:
  pipeline-net:
    external: true
    name: dataingestion_pipeline-net
EOF

# Run the hybrid pipeline
echo "Starting GPU feature extraction..."
docker compose -f docker-compose.hybrid.yml -f docker-compose.hybrid.override.yml up --abort-on-container-exit || {
    echo -e "${RED}Pipeline execution failed!${NC}"
    docker compose -f docker-compose.hybrid.yml -f docker-compose.hybrid.override.yml logs
    exit 1
}

# Step 4: Verify results
echo -e "\n${YELLOW}Step 4: Verifying results...${NC}"

# Check if features were extracted
FEATURE_COUNT=$(docker compose exec -T db psql -U postgres -d postgres -t -c \
    "SELECT COUNT(DISTINCT era_id) FROM gpu_features WHERE computed_at >= NOW() - INTERVAL '1 hour'" 2>/dev/null || echo "0")

if [ "$FEATURE_COUNT" -gt "0" ]; then
    echo -e "${GREEN}✓ Successfully extracted features for $FEATURE_COUNT eras${NC}"
    
    # Show sample features
    echo -e "\n${BLUE}Sample extracted features:${NC}"
    docker compose exec -T db psql -U postgres -d postgres -c \
        "SELECT era_id, jsonb_object_keys(features) as feature_name 
         FROM gpu_features 
         WHERE computed_at >= NOW() - INTERVAL '1 hour' 
         LIMIT 20"
else
    echo -e "${RED}No features found in database!${NC}"
fi

# Cleanup
echo -e "\n${YELLOW}Cleaning up...${NC}"
rm -f docker-compose.hybrid.override.yml
docker compose -f docker-compose.hybrid.yml down

echo -e "\n${GREEN}Test complete!${NC}"

# Return to main directory
cd ..