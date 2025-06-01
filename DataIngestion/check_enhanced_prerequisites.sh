#!/bin/bash

# Check Prerequisites for Enhanced Sparse Pipeline
echo "🔍 Checking Prerequisites for Enhanced Sparse Pipeline"
echo "====================================================="

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check counter
TOTAL_CHECKS=0
PASSED_CHECKS=0

check_item() {
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    if [ "$1" = "true" ]; then
        echo -e "${GREEN}✅ $2${NC}"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
    else
        echo -e "${RED}❌ $2${NC}"
        echo "   Fix: $3"
    fi
}

echo ""
echo "1. Docker & GPU Environment"
echo "---------------------------"

# Check Docker
if command -v docker &> /dev/null; then
    check_item "true" "Docker is installed"
else
    check_item "false" "Docker is installed" "Install Docker: https://docs.docker.com/get-docker/"
fi

# Check Docker Compose
if command -v docker compose &> /dev/null || command -v docker-compose &> /dev/null; then
    check_item "true" "Docker Compose is available"
else
    check_item "false" "Docker Compose is available" "Install Docker Compose"
fi

# Check NVIDIA Docker Runtime
if docker info 2>/dev/null | grep -q nvidia; then
    check_item "true" "NVIDIA Docker runtime is configured"
else
    check_item "false" "NVIDIA Docker runtime is configured" "Install nvidia-docker2 package"
fi

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        check_item "true" "GPU detected: $GPU_NAME"
    else
        check_item "false" "GPU is accessible" "Check NVIDIA driver installation"
    fi
else
    check_item "false" "nvidia-smi command available" "Install NVIDIA drivers"
fi

echo ""
echo "2. Required Files & Configuration"
echo "---------------------------------"

# Check enhanced environment file
if [ -f ".env.enhanced" ]; then
    check_item "true" ".env.enhanced configuration exists"
else
    check_item "false" ".env.enhanced configuration exists" "File should have been created"
fi

# Check if .env is configured
if [ -f ".env" ]; then
    if grep -q "ENHANCED_MODE=true" .env 2>/dev/null; then
        check_item "true" ".env configured for enhanced mode"
    else
        check_item "false" ".env configured for enhanced mode" "Copy .env.enhanced to .env"
    fi
else
    check_item "false" ".env file exists" "cp .env.enhanced .env"
fi

# Check phenotype data
if [ -f "feature_extraction/pre_process/phenotype.json" ]; then
    if grep -q "Kalanchoe blossfeldiana" feature_extraction/pre_process/phenotype.json; then
        check_item "true" "Phenotype data for Kalanchoe blossfeldiana exists"
    else
        check_item "false" "Kalanchoe phenotype data found" "Check phenotype.json content"
    fi
else
    check_item "false" "phenotype.json exists" "Missing plant phenotype data"
fi

# Check Docker Compose files
if [ -f "docker-compose.sparse.yml" ]; then
    check_item "true" "docker-compose.sparse.yml exists"
else
    check_item "false" "docker-compose.sparse.yml exists" "Missing sparse pipeline compose file"
fi

# Check GPU feature extraction directory
if [ -d "gpu_feature_extraction" ]; then
    if [ -f "gpu_feature_extraction/Dockerfile" ]; then
        check_item "true" "GPU feature extraction Dockerfile exists"
    else
        check_item "false" "GPU feature extraction Dockerfile exists" "Missing Dockerfile"
    fi
else
    check_item "false" "gpu_feature_extraction directory exists" "Missing GPU implementation"
fi

echo ""
echo "3. Database Requirements"
echo "------------------------"

# Check if we can connect to database (if running)
if docker compose ps 2>/dev/null | grep -q "db.*running"; then
    check_item "true" "Database service is running"
    
    # Check if external tables SQL exists
    if [ -f "feature_extraction/pre_process/create_external_data_tables.sql" ]; then
        check_item "true" "External data tables SQL script exists"
    else
        check_item "false" "External data tables SQL script exists" "Create external tables schema"
    fi
else
    echo -e "${YELLOW}⚠️  Database not running (normal if not started yet)${NC}"
fi

echo ""
echo "4. External Data Sources"
echo "------------------------"

# Check Open-Meteo API (no key required)
check_item "true" "Open-Meteo weather API (no key required)"

# Check if energy price fetcher exists
if [ -f "feature_extraction/pre_process/external/fetch_energy.py" ]; then
    check_item "true" "Energy price fetcher script exists"
else
    check_item "false" "Energy price fetcher exists" "Missing external/fetch_energy.py"
fi

echo ""
echo "5. Required Data"
echo "----------------"

# Check if sensor data exists
if [ -d "../Data" ]; then
    CSV_COUNT=$(find ../Data -name "*.csv" 2>/dev/null | wc -l)
    if [ $CSV_COUNT -gt 0 ]; then
        check_item "true" "Sensor data files found: $CSV_COUNT CSV files"
    else
        check_item "false" "CSV sensor data found" "Add sensor data to ../Data directory"
    fi
else
    check_item "false" "../Data directory exists" "Create ../Data and add sensor CSV files"
fi

echo ""
echo "================================"
echo "Summary: $PASSED_CHECKS/$TOTAL_CHECKS checks passed"
echo ""

if [ $PASSED_CHECKS -eq $TOTAL_CHECKS ]; then
    echo -e "${GREEN}✅ All prerequisites met! Ready to run enhanced pipeline.${NC}"
    echo ""
    echo "Quick start commands:"
    echo "1. Start database: docker compose -f docker-compose.sparse.yml up -d db"
    echo "2. Load external tables: docker compose -f docker-compose.sparse.yml exec db psql -U postgres -f /docker-entrypoint-initdb.d/create_external_data_tables.sql"
    echo "3. Run pipeline: docker compose -f docker-compose.sparse.yml run --rm sparse_pipeline --enhanced-mode --start-date 2014-01-01 --end-date 2014-07-01"
else
    echo -e "${RED}❌ Some prerequisites missing. Please address the issues above.${NC}"
    echo ""
    echo "Critical items to fix:"
    if [ ! -f ".env" ] || ! grep -q "ENHANCED_MODE=true" .env 2>/dev/null; then
        echo "- Copy configuration: cp .env.enhanced .env"
    fi
    if [ ! -f "feature_extraction/pre_process/create_external_data_tables.sql" ]; then
        echo "- External tables SQL was just created, add to docker-compose volume mount"
    fi
fi

echo ""
echo "Optional: Fetch external data before running pipeline:"
echo "- Weather: cd feature_extraction/pre_process && python external/fetch_external_weather.py"
echo "- Energy: cd feature_extraction/pre_process && python external/fetch_energy.py"