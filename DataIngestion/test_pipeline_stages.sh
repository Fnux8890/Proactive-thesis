#!/bin/bash
# Test Pipeline Stages Individually
# This validates each stage works before running the full pipeline

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== TESTING PIPELINE STAGES INDIVIDUALLY ===${NC}"

# Stage 1: Test Database
echo -e "${YELLOW}Testing Stage 1: Database...${NC}"
docker compose -f docker-compose.full-comparison.yml up -d db
sleep 10

if docker compose -f docker-compose.full-comparison.yml ps db | grep -q "healthy\|Up"; then
    echo -e "${GREEN}✓ Database is running${NC}"
else
    echo -e "${RED}✗ Database failed to start${NC}"
    exit 1
fi

# Stage 2: Test Data Ingestion
echo -e "${YELLOW}Testing Stage 2: Data Ingestion...${NC}"
docker compose -f docker-compose.full-comparison.yml run --rm rust_pipeline

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Data ingestion works${NC}"
else
    echo -e "${RED}✗ Data ingestion failed${NC}"
    exit 1
fi

# Test database has data
echo -e "${YELLOW}Checking database has data...${NC}"
RECORD_COUNT=$(docker compose -f docker-compose.full-comparison.yml exec -T db psql -U postgres -d postgres -t -c "SELECT COUNT(*) FROM sensor_data;" | tr -d ' ')

if [ "$RECORD_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✓ Database contains $RECORD_COUNT records${NC}"
else
    echo -e "${RED}✗ No data in database${NC}"
    exit 1
fi

# Stage 3: Test Enhanced Pipeline Build
echo -e "${YELLOW}Testing Stage 3: Enhanced Pipeline Build...${NC}"
docker compose -f docker-compose.full-comparison.yml build enhanced_sparse_pipeline

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Enhanced pipeline builds successfully${NC}"
else
    echo -e "${RED}✗ Enhanced pipeline build failed${NC}"
    exit 1
fi

# Test enhanced pipeline help
echo -e "${YELLOW}Testing enhanced pipeline command arguments...${NC}"
docker compose -f docker-compose.full-comparison.yml run --rm enhanced_sparse_pipeline --help

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Enhanced pipeline binary works${NC}"
else
    echo -e "${RED}✗ Enhanced pipeline binary failed${NC}"
    exit 1
fi

# Stage 4: Test Model Builder Build
echo -e "${YELLOW}Testing Stage 4: Model Builder Build...${NC}"
docker compose -f docker-compose.full-comparison.yml build model_builder

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Model builder builds successfully${NC}"
else
    echo -e "${RED}✗ Model builder build failed${NC}"
    exit 1
fi

# Stage 5: Test MOEA Optimizer Builds
echo -e "${YELLOW}Testing Stage 5: MOEA Optimizer Builds...${NC}"
docker compose -f docker-compose.full-comparison.yml build moea_optimizer_cpu
docker compose -f docker-compose.full-comparison.yml build moea_optimizer_gpu

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ MOEA optimizers build successfully${NC}"
else
    echo -e "${RED}✗ MOEA optimizer build failed${NC}"
    exit 1
fi

# Stage 6: Test Evaluator Build
echo -e "${YELLOW}Testing Stage 6: Results Evaluator Build...${NC}"
docker compose -f docker-compose.full-comparison.yml build results_evaluator

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Results evaluator builds successfully${NC}"
else
    echo -e "${RED}✗ Results evaluator build failed${NC}"
    exit 1
fi

# Cleanup
echo -e "${YELLOW}Cleaning up...${NC}"
docker compose -f docker-compose.full-comparison.yml down

echo -e "${GREEN}=== ALL PIPELINE STAGES VALIDATED ===${NC}"
echo -e "${GREEN}✓ Database: Working${NC}"
echo -e "${GREEN}✓ Data Ingestion: Working with $RECORD_COUNT records${NC}"
echo -e "${GREEN}✓ Enhanced Pipeline: Builds and runs${NC}"
echo -e "${GREEN}✓ Model Builder: Builds${NC}"
echo -e "${GREEN}✓ MOEA Optimizers: Build${NC}"
echo -e "${GREEN}✓ Results Evaluator: Builds${NC}"
echo ""
echo -e "${BLUE}Pipeline is ready! You can now run:${NC}"
echo -e "${BLUE}  ./run_full_pipeline_experiment.sh${NC}"