#!/bin/bash
# Run CPU vs GPU MOEA comparison experiments

set -e

echo "=== Greenhouse MOEA CPU vs GPU Comparison ==="
echo "Starting at: $(date)"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose-clean.yml}"
RESULTS_DIR="moea_optimizer/results/comparison_$(date +%Y%m%d_%H%M%S)"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Function to check service health
check_service() {
    local service=$1
    echo -e "${BLUE}Checking $service health...${NC}"
    if docker compose -f "$COMPOSE_FILE" ps "$service" | grep -q "healthy\|running"; then
        echo -e "${GREEN}✓ $service is healthy${NC}"
        return 0
    else
        echo -e "${RED}✗ $service is not healthy${NC}"
        return 1
    fi
}

# Function to run experiment
run_experiment() {
    local optimizer_service=$1
    local experiment_name=$2
    
    echo -e "\n${BLUE}Running $experiment_name...${NC}"
    
    # Start the optimizer
    docker compose -f "$COMPOSE_FILE" up -d "$optimizer_service"
    
    # Wait for completion
    echo "Waiting for $optimizer_service to complete..."
    docker compose -f "$COMPOSE_FILE" wait "$optimizer_service"
    
    # Check exit code
    EXIT_CODE=$(docker compose -f "$COMPOSE_FILE" ps -q "$optimizer_service" | xargs docker inspect -f '{{.State.ExitCode}}')
    
    if [ "$EXIT_CODE" -eq 0 ]; then
        echo -e "${GREEN}✓ $experiment_name completed successfully${NC}"
        
        # Copy results
        docker compose -f "$COMPOSE_FILE" cp "$optimizer_service:/app/results/." "$RESULTS_DIR/$experiment_name/"
        
        # Get timing information
        docker compose -f "$COMPOSE_FILE" logs "$optimizer_service" | grep -E "runtime|completed" > "$RESULTS_DIR/$experiment_name/timing.log"
    else
        echo -e "${RED}✗ $experiment_name failed with exit code $EXIT_CODE${NC}"
        
        # Save error logs
        docker compose -f "$COMPOSE_FILE" logs "$optimizer_service" > "$RESULTS_DIR/$experiment_name/error.log"
    fi
    
    # Stop the service
    docker compose -f "$COMPOSE_FILE" stop "$optimizer_service"
}

# Main execution
echo -e "\n${BLUE}Starting infrastructure services...${NC}"
docker compose -f "$COMPOSE_FILE" up -d db redis pgadmin

# Wait for database
echo "Waiting for database to be ready..."
sleep 10
check_service "db" || exit 1

# Check if models exist
echo -e "\n${BLUE}Checking for trained models...${NC}"
if [ ! -d "model_builder/models" ] || [ -z "$(ls -A model_builder/models)" ]; then
    echo -e "${RED}No models found. Running model training first...${NC}"
    
    # Run full pipeline up to model building
    docker compose -f "$COMPOSE_FILE" up --build \
        rust_pipeline \
        preprocessing \
        era_detector \
        feature_extraction \
        model_builder
    
    # Wait for model builder to complete
    docker compose -f "$COMPOSE_FILE" wait model_builder
fi

# Run CPU experiment
echo -e "\n${GREEN}=== Running CPU MOEA Experiment ===${NC}"
START_CPU=$(date +%s)
run_experiment "moea_optimizer_cpu" "cpu_results"
END_CPU=$(date +%s)
CPU_DURATION=$((END_CPU - START_CPU))

# Run GPU experiment
echo -e "\n${GREEN}=== Running GPU MOEA Experiment ===${NC}"
START_GPU=$(date +%s)
run_experiment "moea_optimizer_gpu" "gpu_results"
END_GPU=$(date +%s)
GPU_DURATION=$((END_GPU - START_GPU))

# Generate comparison report
echo -e "\n${BLUE}Generating comparison report...${NC}"
cat > "$RESULTS_DIR/comparison_summary.txt" << EOF
MOEA CPU vs GPU Comparison Results
==================================
Experiment Date: $(date)

Execution Times:
- CPU: ${CPU_DURATION}s
- GPU: ${GPU_DURATION}s
- Speedup: $(echo "scale=2; $CPU_DURATION / $GPU_DURATION" | bc)x

Results Directory: $RESULTS_DIR

Directory Structure:
├── cpu_results/
│   ├── pareto_front.csv
│   ├── metrics.csv
│   ├── convergence_history.csv
│   └── timing.log
├── gpu_results/
│   ├── pareto_front.csv
│   ├── metrics.csv
│   ├── convergence_history.csv
│   └── timing.log
└── comparison_summary.txt
EOF

# Run statistical comparison if Python script exists
if [ -f "moea_optimizer/src/compare_results.py" ]; then
    echo -e "\n${BLUE}Running statistical comparison...${NC}"
    docker run --rm \
        -v "$(pwd)/$RESULTS_DIR:/results" \
        -v "$(pwd)/moea_optimizer/src:/src" \
        python:3.11-slim \
        python /src/compare_results.py \
            /results/cpu_results \
            /results/gpu_results \
            --output /results/statistical_comparison.html
fi

echo -e "\n${GREEN}=== Comparison Complete ===${NC}"
echo "Results saved to: $RESULTS_DIR"
echo "Completed at: $(date)"

# Optional: Open results in browser
if command -v xdg-open &> /dev/null; then
    xdg-open "$RESULTS_DIR/comparison_summary.txt"
elif command -v open &> /dev/null; then
    open "$RESULTS_DIR/comparison_summary.txt"
fi