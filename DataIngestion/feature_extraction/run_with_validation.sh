#!/bin/bash
# run_with_validation.sh - Run feature extraction with pre-flight checks

set -e

echo "🚀 Feature Extraction Pipeline - Pre-flight Checks"
echo "================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $2"
    else
        echo -e "${RED}✗${NC} $2"
        return 1
    fi
}

# Change to script directory
cd "$(dirname "$0")"

echo -e "\n${YELLOW}1. Checking Python code quality...${NC}"
make format
make lint
LINT_STATUS=$?
print_status $LINT_STATUS "Code quality checks"

echo -e "\n${YELLOW}2. Running validation script...${NC}"
python validate_pipeline.py
VALIDATE_STATUS=$?
print_status $VALIDATE_STATUS "Pipeline validation"

echo -e "\n${YELLOW}3. Checking Docker environment...${NC}"
docker --version > /dev/null 2>&1
DOCKER_STATUS=$?
print_status $DOCKER_STATUS "Docker available"

docker compose version > /dev/null 2>&1
COMPOSE_STATUS=$?
print_status $COMPOSE_STATUS "Docker Compose available"

echo -e "\n${YELLOW}4. Checking GPU availability...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    GPU_STATUS=0
else
    GPU_STATUS=1
    echo "No GPU detected - will run in CPU mode"
fi
print_status $GPU_STATUS "GPU check" || true

echo -e "\n${YELLOW}5. Environment configuration:${NC}"
echo "  USE_GPU=${USE_GPU:-true}"
echo "  FEATURE_SET=${FEATURE_SET:-efficient}"
echo "  BATCH_SIZE=${BATCH_SIZE:-10000}"
echo "  FEATURES_TABLE=${FEATURES_TABLE:-feature_data}"

# Check if all validations passed
if [ $LINT_STATUS -ne 0 ] || [ $VALIDATE_STATUS -ne 0 ] || [ $DOCKER_STATUS -ne 0 ] || [ $COMPOSE_STATUS -ne 0 ]; then
    echo -e "\n${RED}❌ Pre-flight checks failed!${NC}"
    echo "Please fix the issues above before running the pipeline."
    exit 1
fi

echo -e "\n${GREEN}✓ All pre-flight checks passed!${NC}"

# Ask for confirmation
echo -e "\n${YELLOW}Ready to run feature extraction pipeline.${NC}"
read -p "Continue? [Y/n] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
    echo "Aborted."
    exit 0
fi

# Run with docker-compose
echo -e "\n${YELLOW}Starting feature extraction with Docker Compose...${NC}"
cd ../..
docker compose up -d db
docker compose up --build feature_extractor

echo -e "\n${GREEN}✓ Feature extraction pipeline completed!${NC}"