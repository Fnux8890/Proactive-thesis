#\!/bin/bash
# CPU vs GPU Performance Comparison for MOEA Optimization
# Runs identical experiments with CPU and GPU configurations

set -e

# Configuration
START_DATE="${START_DATE:-2013-12-01}"
END_DATE="${END_DATE:-2014-02-28}"  # Smaller dataset for faster comparison
COMPARISON_NAME="cpu_vs_gpu_$(date +%Y%m%d_%H%M%S)"
POPULATION_SIZE="${POPULATION_SIZE:-50}"  # Smaller for faster testing
GENERATIONS="${GENERATIONS:-50}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== CPU vs GPU MOEA Performance Comparison ===${NC}"
echo -e "${BLUE}Comparison: ${COMPARISON_NAME}${NC}"
echo -e "${BLUE}Population: ${POPULATION_SIZE}, Generations: ${GENERATIONS}${NC}"
echo ""

# Create comparison directory
COMPARISON_DIR="experiments/comparisons/${COMPARISON_NAME}"
mkdir -p $COMPARISON_DIR

# Prepare configurations
create_moea_config() {
    local device=$1
    local config_file=$2
    
    cat > $config_file << EOM
[meta]
experiment_name = "moea_${device}_test"
description = "MOEA optimization test using ${device}"

[algorithm]
type = "NSGA-III"
population_size = ${POPULATION_SIZE}
n_generations = ${GENERATIONS}
crossover_probability = 0.9
crossover_eta = 15
mutation_probability = 0.1
mutation_eta = 20
n_reference_points = 12
use_gpu = $([ "$device" = "gpu" ] && echo "true" || echo "false")
cuda_device_id = 0

[objectives]
energy_consumption = { type = "minimize", weight = 0.5 }
plant_growth = { type = "maximize", weight = 0.5 }

[decision_variables]
temperature_setpoint = { min = 15.0, max = 30.0, type = "continuous" }
humidity_target = { min = 40.0, max = 90.0, type = "continuous" }
co2_target = { min = 400.0, max = 1200.0, type = "continuous" }
photoperiod_hours = { min = 8.0, max = 18.0, type = "continuous" }

[constraints]
temperature_bounds = [15.0, 30.0]
humidity_bounds = [40.0, 90.0]
co2_bounds = [400.0, 1200.0]
light_bounds = [0.0, 1000.0]

[output]
base_dir = "/app/results"
experiment_dir = "${device}_test"
save_interval = 0
save_population = true
save_pareto_front = true
save_history = true

[evaluation]
log_interval = 10
monitor_memory = true

n_runs = 1
base_seed = 42
EOM
}

echo -e "${YELLOW}Running common pipeline stages...${NC}"

export START_DATE
export END_DATE
export FEATURES_TABLE="comparison_features_${COMPARISON_NAME}"

# Run the pipeline up to model training
echo -e "${YELLOW}Starting pipeline (database → models)...${NC}"
docker compose -f docker-compose.enhanced.yml up -d db
sleep 15

# Check if we need data ingestion
echo -e "${YELLOW}Checking if data ingestion needed...${NC}"
docker compose -f docker-compose.enhanced.yml up rust_pipeline enhanced_sparse_pipeline model_builder

# Prepare MOEA configurations
echo -e "${YELLOW}Preparing MOEA configurations...${NC}"
mkdir -p ./moea_optimizer/config/comparison
create_moea_config "cpu" "./moea_optimizer/config/comparison/moea_config_cpu_test.toml"
create_moea_config "gpu" "./moea_optimizer/config/comparison/moea_config_gpu_test.toml"

# Function to run MOEA experiment
run_moea_experiment() {
    local device=$1
    local start_time=$(date +%s)
    
    echo -e "${YELLOW}Running MOEA experiment with ${device}...${NC}"
    
    # Set environment for this run
    export CONFIG_PATH="/app/config/comparison/moea_config_${device}_test.toml"
    export DEVICE=$([ "$device" = "gpu" ] && echo "cuda" || echo "cpu")
    
    # Create results directory
    mkdir -p "${COMPARISON_DIR}/${device}_results"
    
    # Run the experiment
    if [ "$device" = "gpu" ]; then
        # GPU version - use the GPU service
        docker compose -f docker-compose.enhanced.yml run --rm \
            -e CONFIG_PATH="$CONFIG_PATH" \
            -e DEVICE="cuda" \
            -v "${COMPARISON_DIR}/${device}_results:/app/results:rw" \
            -v "./moea_optimizer/config/comparison:/app/config/comparison:ro" \
            moea_optimizer python -m src.cli run --config "$CONFIG_PATH"
    else
        # CPU version - override the deploy section to disable GPU
        docker compose -f docker-compose.enhanced.yml run --rm \
            -e CONFIG_PATH="$CONFIG_PATH" \
            -e DEVICE="cpu" \
            -v "${COMPARISON_DIR}/${device}_results:/app/results:rw" \
            -v "./moea_optimizer/config/comparison:/app/config/comparison:ro" \
            --gpus= \
            moea_optimizer python -m src.cli run --config "$CONFIG_PATH"
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo -e "${GREEN}✓ ${device} experiment completed in ${duration} seconds${NC}"
    echo $duration > "${COMPARISON_DIR}/${device}_runtime.txt"
}

# Run CPU experiment
echo -e "${BLUE}=== Running CPU Experiment ===${NC}"
run_moea_experiment "cpu"

# Run GPU experiment
echo -e "${BLUE}=== Running GPU Experiment ===${NC}"
run_moea_experiment "gpu"

# Generate comparison report
echo -e "${YELLOW}Generating comparison report...${NC}"

cat > "${COMPARISON_DIR}/comparison_report.md" << EOM
# CPU vs GPU MOEA Performance Comparison

## Experiment Configuration
- Date Range: $START_DATE to $END_DATE
- Population Size: $POPULATION_SIZE
- Generations: $GENERATIONS
- Timestamp: $(date)

## Performance Results

### Runtime Comparison
EOM

if [ -f "${COMPARISON_DIR}/cpu_runtime.txt" ] && [ -f "${COMPARISON_DIR}/gpu_runtime.txt" ]; then
    CPU_TIME=$(cat "${COMPARISON_DIR}/cpu_runtime.txt")
    GPU_TIME=$(cat "${COMPARISON_DIR}/gpu_runtime.txt")
    
    # Calculate speedup (use bc if available, otherwise use awk)
    if command -v bc >/dev/null 2>&1; then
        SPEEDUP=$(echo "scale=2; $CPU_TIME / $GPU_TIME"  < /dev/null |  bc)
    else
        SPEEDUP=$(awk "BEGIN {printf \"%.2f\", $CPU_TIME / $GPU_TIME}")
    fi
    
    echo "- CPU Runtime: ${CPU_TIME} seconds" >> "${COMPARISON_DIR}/comparison_report.md"
    echo "- GPU Runtime: ${GPU_TIME} seconds" >> "${COMPARISON_DIR}/comparison_report.md"  
    echo "- Speedup: ${SPEEDUP}x" >> "${COMPARISON_DIR}/comparison_report.md"
    
    echo -e "${GREEN}Performance Summary:${NC}"
    echo -e "${GREEN}  CPU: ${CPU_TIME}s${NC}"
    echo -e "${GREEN}  GPU: ${GPU_TIME}s${NC}"
    echo -e "${GREEN}  Speedup: ${SPEEDUP}x${NC}"
fi

cat >> "${COMPARISON_DIR}/comparison_report.md" << 'EOM'

### Solution Quality
Check the respective results directories for:
- Pareto front solutions
- Convergence history  
- Hypervolume metrics

### Files Generated
- `cpu_results/` - CPU experiment outputs
- `gpu_results/` - GPU experiment outputs
- `moea_config_cpu_test.toml` - CPU configuration
- `moea_config_gpu_test.toml` - GPU configuration
- `comparison_report.md` - This report

### Analysis Commands
```bash
# View Pareto fronts
ls cpu_results/*/pareto_*.npy
ls gpu_results/*/pareto_*.npy

# Compare metrics  
cat cpu_results/*/metrics.json
cat gpu_results/*/metrics.json

# View convergence
head cpu_results/*/convergence.csv
head gpu_results/*/convergence.csv
```
EOM

# Cleanup
echo -e "${YELLOW}Cleaning up containers...${NC}"
docker compose -f docker-compose.enhanced.yml down

echo -e "${GREEN}=== Comparison Complete ===${NC}"
echo -e "${GREEN}Results saved to: $COMPARISON_DIR${NC}"
echo ""
echo -e "${BLUE}To view results:${NC}"
echo -e "${BLUE}  cat $COMPARISON_DIR/comparison_report.md${NC}"
echo -e "${BLUE}  ls -la $COMPARISON_DIR/cpu_results/    # CPU results${NC}"
echo -e "${BLUE}  ls -la $COMPARISON_DIR/gpu_results/    # GPU results${NC}"
