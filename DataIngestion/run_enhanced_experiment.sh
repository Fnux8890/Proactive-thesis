#!/bin/bash
# Enhanced Pipeline Experiment Runner
# Runs the complete enhanced sparse pipeline with configurable parameters

set -e

# Default configuration
START_DATE="${START_DATE:-2013-12-01}"
END_DATE="${END_DATE:-2014-08-27}"  # Era 1 only for faster testing
BATCH_SIZE="${BATCH_SIZE:-24}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-enhanced_sparse_experiment_$(date +%Y%m%d_%H%M%S)}"
USE_GPU="${USE_GPU:-true}"
SKIP_INGESTION="${SKIP_INGESTION:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Enhanced Sparse Pipeline Experiment ===${NC}"
echo -e "${BLUE}Experiment: ${EXPERIMENT_NAME}${NC}"
echo -e "${BLUE}Date range: ${START_DATE} to ${END_DATE}${NC}"
echo -e "${BLUE}GPU enabled: ${USE_GPU}${NC}"
echo ""

# Function to check if service is ready
wait_for_service() {
    local service=$1
    local max_attempts=30
    local attempt=1
    
    echo -e "${YELLOW}Waiting for ${service} to be ready...${NC}"
    
    while [ $attempt -le $max_attempts ]; do
        if docker compose -f docker-compose.enhanced.yml ps $service | grep -q "healthy\|running"; then
            echo -e "${GREEN}✓ ${service} is ready${NC}"
            return 0
        fi
        
        echo -e "${YELLOW}  Attempt ${attempt}/${max_attempts}...${NC}"
        sleep 10
        attempt=$((attempt + 1))
    done
    
    echo -e "${RED}✗ ${service} failed to start${NC}"
    return 1
}

# Function to check service logs for errors
check_service_logs() {
    local service=$1
    echo -e "${BLUE}Checking ${service} logs...${NC}"
    
    # Get last 20 lines and check for common error patterns
    if docker compose -f docker-compose.enhanced.yml logs --tail=20 $service | grep -E "(ERROR|FATAL|panic|failed|error)"; then
        echo -e "${YELLOW}⚠ Found potential issues in ${service} logs${NC}"
    else
        echo -e "${GREEN}✓ ${service} logs look clean${NC}"
    fi
}

# Create experiment directory
EXPERIMENT_DIR="experiments/results/${EXPERIMENT_NAME}"
mkdir -p $EXPERIMENT_DIR

# Set environment variables for Docker Compose
export START_DATE
export END_DATE
export BATCH_SIZE
export USE_GPU
export FEATURES_TABLE="enhanced_sparse_features_${EXPERIMENT_NAME}"

echo -e "${BLUE}Starting experiment with enhanced pipeline...${NC}"

# Stage 1: Database
echo -e "${YELLOW}Stage 1: Starting database...${NC}"
docker compose -f docker-compose.enhanced.yml up -d db
wait_for_service db

# Stage 2: Data ingestion (skip if requested)
if [ "$SKIP_INGESTION" = "false" ]; then
    echo -e "${YELLOW}Stage 2: Running data ingestion...${NC}"
    docker compose -f docker-compose.enhanced.yml up rust_pipeline
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Data ingestion completed${NC}"
    else
        echo -e "${RED}✗ Data ingestion failed${NC}"
        check_service_logs rust_pipeline
        exit 1
    fi
else
    echo -e "${YELLOW}Stage 2: Skipping data ingestion (SKIP_INGESTION=true)${NC}"
fi

# Stage 3: Enhanced sparse pipeline
echo -e "${YELLOW}Stage 3: Running enhanced sparse feature extraction...${NC}"
docker compose -f docker-compose.enhanced.yml up enhanced_sparse_pipeline

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Enhanced feature extraction completed${NC}"
else
    echo -e "${RED}✗ Enhanced feature extraction failed${NC}"
    check_service_logs enhanced_sparse_pipeline
    exit 1
fi

# Stage 4: Model building
echo -e "${YELLOW}Stage 4: Training enhanced models...${NC}"
docker compose -f docker-compose.enhanced.yml up model_builder

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Model training completed${NC}"
else
    echo -e "${RED}✗ Model training failed${NC}"
    check_service_logs model_builder
    exit 1
fi

# Stage 5: MOEA optimization
echo -e "${YELLOW}Stage 5: Running MOEA optimization...${NC}"
docker compose -f docker-compose.enhanced.yml up moea_optimizer

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ MOEA optimization completed${NC}"
else
    echo -e "${RED}✗ MOEA optimization failed${NC}"
    check_service_logs moea_optimizer
    exit 1
fi

# Collect results
echo -e "${YELLOW}Collecting experiment results...${NC}"

# Copy feature extraction checkpoints
if [ -d "./gpu_feature_extraction/checkpoints" ]; then
    cp -r ./gpu_feature_extraction/checkpoints "$EXPERIMENT_DIR/"
    echo -e "${GREEN}✓ Copied feature extraction checkpoints${NC}"
fi

# Copy model artifacts
if [ -d "./model_builder/models" ]; then
    cp -r ./model_builder/models "$EXPERIMENT_DIR/"
    echo -e "${GREEN}✓ Copied trained models${NC}"
fi

# Copy MOEA results
if [ -d "./moea_optimizer/results" ]; then
    cp -r ./moea_optimizer/results "$EXPERIMENT_DIR/"
    echo -e "${GREEN}✓ Copied MOEA results${NC}"
fi

# Generate experiment summary
cat > "$EXPERIMENT_DIR/experiment_summary.json" << EOF
{
  "experiment_name": "$EXPERIMENT_NAME",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "configuration": {
    "start_date": "$START_DATE",
    "end_date": "$END_DATE",
    "batch_size": $BATCH_SIZE,
    "use_gpu": $USE_GPU,
    "features_table": "$FEATURES_TABLE"
  },
  "pipeline_stages": [
    "data_ingestion",
    "enhanced_feature_extraction", 
    "model_training",
    "moea_optimization"
  ],
  "results_location": "$EXPERIMENT_DIR"
}
EOF

echo -e "${GREEN}✓ Generated experiment summary${NC}"

# Cleanup containers
echo -e "${YELLOW}Cleaning up containers...${NC}"
docker compose -f docker-compose.enhanced.yml down

echo -e "${GREEN}=== Experiment Complete ===${NC}"
echo -e "${GREEN}Results saved to: $EXPERIMENT_DIR${NC}"
echo ""
echo -e "${BLUE}To view results:${NC}"
echo -e "${BLUE}  ls -la $EXPERIMENT_DIR${NC}"
echo -e "${BLUE}  cat $EXPERIMENT_DIR/experiment_summary.json${NC}"
echo ""
echo -e "${BLUE}To run CPU vs GPU comparison:${NC}"
echo -e "${BLUE}  ./run_cpu_vs_gpu_comparison.sh${NC}"