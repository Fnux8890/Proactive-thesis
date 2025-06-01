#!/bin/bash
# Full End-to-End Pipeline Experiment with 2013-2016 Data
# Runs complete pipeline with CPU/GPU MOEA comparison and evaluation

set -e

# Configuration
EXPERIMENT_NAME="full_pipeline_$(date +%Y%m%d_%H%M%S)"
START_DATE="2013-12-01"
END_DATE="2016-09-08"
BATCH_SIZE="48"
MIN_ERA_ROWS="200"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}  FULL PIPELINE EXPERIMENT: 2013-2016 DATASET${NC}"
echo -e "${BLUE}  CPU/GPU MOEA Comparison with Real-World Validation${NC}"
echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}Experiment: ${EXPERIMENT_NAME}${NC}"
echo -e "${BLUE}Date range: ${START_DATE} to ${END_DATE}${NC}"
echo -e "${BLUE}Expected duration: 2-4 hours${NC}"
echo ""

# Create experiment directory
EXPERIMENT_DIR="experiments/full_experiment"
mkdir -p $EXPERIMENT_DIR

# Function to check service health
wait_for_service() {
    local service=$1
    local max_attempts=60  # Longer timeout for full dataset
    local attempt=1
    
    echo -e "${YELLOW}Waiting for ${service} to be ready...${NC}"
    
    while [ $attempt -le $max_attempts ]; do
        if docker compose -f docker-compose.full-comparison.yml ps $service | grep -q "healthy\\|Up"; then
            echo -e "${GREEN}✓ ${service} is ready${NC}"
            return 0
        fi
        
        echo -e "${YELLOW}  Attempt ${attempt}/${max_attempts}...${NC}"
        sleep 15
        attempt=$((attempt + 1))
    done
    
    echo -e "${RED}✗ ${service} failed to start${NC}"
    return 1
}

# Function to monitor service progress
monitor_service() {
    local service=$1
    local description=$2
    local expected_duration_min=$3
    
    echo -e "${PURPLE}${description} started (expected: ~${expected_duration_min} minutes)${NC}"
    echo -e "${PURPLE}Monitoring ${service} progress...${NC}"
    
    # Start background monitoring
    (
        while docker compose -f docker-compose.full-comparison.yml ps $service | grep -q "running"; do
            # Show last few log lines every 30 seconds
            echo -e "${YELLOW}[$(date +%H:%M:%S)] ${service} status:${NC}"
            docker compose -f docker-compose.full-comparison.yml logs --tail=3 $service 2>/dev/null || true
            sleep 30
        done
    ) &
    
    local monitor_pid=$!
    
    # Wait for service to complete
    docker compose -f docker-compose.full-comparison.yml up $service
    local exit_code=$?
    
    # Stop monitoring
    kill $monitor_pid 2>/dev/null || true
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ ${description} completed successfully${NC}"
    else
        echo -e "${RED}✗ ${description} failed${NC}"
        docker compose -f docker-compose.full-comparison.yml logs --tail=20 $service
        exit 1
    fi
}

# Function to check disk space
check_disk_space() {
    local required_gb=$1
    local available_gb=$(df -BG --output=avail . | tail -1 | tr -d 'G')
    
    if [ "$available_gb" -lt "$required_gb" ]; then
        echo -e "${RED}⚠ Warning: Low disk space. Required: ${required_gb}GB, Available: ${available_gb}GB${NC}"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo -e "${GREEN}✓ Sufficient disk space: ${available_gb}GB available${NC}"
    fi
}

# Pre-flight checks
echo -e "${YELLOW}=== PRE-FLIGHT CHECKS ===${NC}"

# Check disk space (full dataset requires significant space)
check_disk_space 10

# Check GPU availability
if command -v nvidia-smi >/dev/null 2>&1; then
    echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo -e "${YELLOW}⚠ No NVIDIA GPU detected - GPU experiments will be skipped${NC}"
fi

# Check Docker Compose
if ! docker compose version >/dev/null 2>&1; then
    echo -e "${RED}✗ Docker Compose not found${NC}"
    exit 1
else
    echo -e "${GREEN}✓ Docker Compose available${NC}"
fi

# Check data files
if [ ! -d "../Data" ]; then
    echo -e "${RED}✗ Data directory not found: ../Data${NC}"
    exit 1
else
    data_files=$(find ../Data -name "*.csv" | wc -l)
    echo -e "${GREEN}✓ Data directory found with ${data_files} CSV files${NC}"
fi

echo ""

# Set environment variables
export START_DATE
export END_DATE
export BATCH_SIZE
export MIN_ERA_ROWS
export EXPERIMENT_NAME

echo -e "${BLUE}=== STARTING FULL PIPELINE EXPERIMENT ===${NC}"

# Stage 1: Infrastructure
echo -e "${YELLOW}Stage 1: Starting infrastructure...${NC}"
docker compose -f docker-compose.full-comparison.yml up -d db
wait_for_service db

# Stage 2: Data Ingestion (Full Dataset)
echo -e "${YELLOW}Stage 2: Data ingestion (full 2013-2016 dataset)...${NC}"
docker compose -f docker-compose.full-comparison.yml up rust_pipeline
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Data ingestion completed${NC}"
else
    echo -e "${RED}✗ Data ingestion failed${NC}"
    exit 1
fi

# Stage 3: Enhanced Sparse Feature Extraction
echo -e "${YELLOW}Stage 3: Enhanced sparse feature extraction...${NC}"
docker compose -f docker-compose.full-comparison.yml up enhanced_sparse_pipeline
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Enhanced feature extraction completed${NC}"
else
    echo -e "${RED}✗ Enhanced feature extraction failed${NC}"
    exit 1
fi

# Stage 4: Model Building with Full Features
echo -e "${YELLOW}Stage 4: Comprehensive model training...${NC}"
docker compose -f docker-compose.full-comparison.yml up model_builder
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Model training completed${NC}"
else
    echo -e "${RED}✗ Model training failed${NC}"
    exit 1
fi

# Stage 5A: CPU MOEA Optimization
echo -e "${YELLOW}Stage 5A: CPU MOEA optimization...${NC}"
docker compose -f docker-compose.full-comparison.yml up moea_optimizer_cpu &
CPU_PID=$!

# Stage 5B: GPU MOEA Optimization (parallel)
echo -e "${YELLOW}Stage 5B: GPU MOEA optimization...${NC}"
docker compose -f docker-compose.full-comparison.yml up moea_optimizer_gpu &
GPU_PID=$!

# Wait for both MOEA processes
wait $CPU_PID
CPU_EXIT=$?
wait $GPU_PID  
GPU_EXIT=$?

if [ $CPU_EXIT -eq 0 ] && [ $GPU_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓ Both MOEA optimizations completed${NC}"
else
    echo -e "${RED}✗ MOEA optimization failed (CPU: $CPU_EXIT, GPU: $GPU_EXIT)${NC}"
    exit 1
fi

# Stage 6: Results Evaluation
echo -e "${YELLOW}Stage 6: Comprehensive evaluation...${NC}"
docker compose -f docker-compose.full-comparison.yml up results_evaluator
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Results evaluation completed${NC}"
else
    echo -e "${RED}✗ Results evaluation failed${NC}"
    exit 1
fi

# Collect and organize results
echo -e "${YELLOW}=== COLLECTING EXPERIMENT RESULTS ===${NC}"

# Create timestamped results directory
RESULTS_DIR="$EXPERIMENT_DIR/${EXPERIMENT_NAME}"
mkdir -p $RESULTS_DIR

# Copy feature extraction results
if [ -d "./gpu_feature_extraction/checkpoints" ]; then
    cp -r ./gpu_feature_extraction/checkpoints "$RESULTS_DIR/"
    echo -e "${GREEN}✓ Copied feature extraction checkpoints${NC}"
fi

# Copy model artifacts
if [ -d "./model_builder/models" ]; then
    cp -r ./model_builder/models "$RESULTS_DIR/"
    echo -e "${GREEN}✓ Copied trained models${NC}"
fi

# Copy MOEA results
if [ -d "./experiments/full_experiment/moea_cpu" ]; then
    cp -r ./experiments/full_experiment/moea_cpu "$RESULTS_DIR/"
    echo -e "${GREEN}✓ Copied CPU MOEA results${NC}"
fi

if [ -d "./experiments/full_experiment/moea_gpu" ]; then
    cp -r ./experiments/full_experiment/moea_gpu "$RESULTS_DIR/"
    echo -e "${GREEN}✓ Copied GPU MOEA results${NC}"
fi

# Copy evaluation results
if [ -d "./experiments/full_experiment/evaluation_results" ]; then
    cp -r ./experiments/full_experiment/evaluation_results "$RESULTS_DIR/"
    echo -e "${GREEN}✓ Copied evaluation results${NC}"
fi

# Generate experiment summary
cat > "$RESULTS_DIR/experiment_summary.json" << EOF
{
  "experiment_name": "$EXPERIMENT_NAME",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "configuration": {
    "dataset": "full_2013_2016",
    "start_date": "$START_DATE",
    "end_date": "$END_DATE",
    "batch_size": $BATCH_SIZE,
    "min_era_rows": $MIN_ERA_ROWS
  },
  "pipeline_stages": [
    "full_data_ingestion",
    "enhanced_sparse_feature_extraction", 
    "comprehensive_model_training",
    "cpu_moea_optimization",
    "gpu_moea_optimization",
    "comprehensive_evaluation"
  ],
  "experiment_type": "cpu_gpu_comparison",
  "evaluation_framework": "lightgbm_validation",
  "results_location": "$RESULTS_DIR"
}
EOF

echo -e "${GREEN}✓ Generated experiment summary${NC}"

# Performance summary
echo -e "${YELLOW}=== GENERATING PERFORMANCE SUMMARY ===${NC}"

# Extract key metrics from evaluation results
if [ -f "$RESULTS_DIR/evaluation_results/comprehensive_evaluation_report.json" ]; then
    echo -e "${GREEN}✓ Evaluation report found${NC}"
    
    # Try to extract key metrics using python
    python3 -c "
import json
import sys

try:
    with open('$RESULTS_DIR/evaluation_results/comprehensive_evaluation_report.json', 'r') as f:
        report = json.load(f)
    
    print('\\n${BLUE}=== EXPERIMENT RESULTS SUMMARY ===${NC}')
    
    # Key findings
    if 'key_findings' in report:
        print('\\n${GREEN}Key Findings:${NC}')
        for finding in report['key_findings']:
            print(f'  ✓ {finding}')
    
    # Performance comparison
    if 'performance_comparison' in report:
        comp = report['performance_comparison']
        if 'computational_efficiency' in comp and 'speedup' in comp['computational_efficiency']:
            speedup = comp['computational_efficiency']['speedup']
            print(f'\\n${BLUE}GPU Speedup: {speedup:.1f}x${NC}')
    
    # Validation results
    validation = report.get('validation_results', {})
    for device in ['cpu', 'gpu']:
        if device in validation:
            perf = validation[device].get('real_world_performance', {})
            energy_imp = perf.get('energy_improvement_percent', 0)
            growth_imp = perf.get('growth_improvement_percent', 0)
            print(f'\\n${PURPLE}{device.upper()} Performance:${NC}')
            print(f'  Energy Improvement: {energy_imp:.1f}%')
            print(f'  Growth Improvement: {growth_imp:.1f}%')
            
            economic = validation[device].get('economic_impact', {})
            benefit = economic.get('net_economic_benefit_eur', 0)
            roi = economic.get('roi_percent', 0)
            print(f'  Economic Benefit: €{benefit:,.0f}/year')
            print(f'  ROI: {roi:.1f}%')

except Exception as e:
    print(f'Could not extract metrics: {e}')
" || echo -e "${YELLOW}⚠ Could not extract detailed metrics${NC}"
fi

# Cleanup containers
echo -e "${YELLOW}Cleaning up containers...${NC}"
docker compose -f docker-compose.full-comparison.yml down

echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  FULL PIPELINE EXPERIMENT COMPLETED SUCCESSFULLY${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""
echo -e "${BLUE}Results Location:${NC} $RESULTS_DIR"
echo ""
echo -e "${BLUE}Key Files Generated:${NC}"
echo -e "${BLUE}  • experiment_summary.json${NC}       - Experiment configuration and metadata"
echo -e "${BLUE}  • checkpoints/stage3_features.json${NC} - Feature extraction results"
echo -e "${BLUE}  • models/energy_consumption_model.pt${NC} - Energy prediction model"
echo -e "${BLUE}  • models/plant_growth_model.pt${NC}    - Growth prediction model"
echo -e "${BLUE}  • moea_cpu/pareto_*.npy${NC}          - CPU optimization results"
echo -e "${BLUE}  • moea_gpu/pareto_*.npy${NC}          - GPU optimization results"
echo -e "${BLUE}  • evaluation_results/evaluation_summary.md${NC} - Comprehensive evaluation"
echo ""
echo -e "${BLUE}Quick Analysis Commands:${NC}"
echo -e "${BLUE}  ls -la $RESULTS_DIR                   ${NC}# Browse all results"
echo -e "${BLUE}  cat $RESULTS_DIR/experiment_summary.json${NC} # View experiment config"
echo -e "${BLUE}  cat $RESULTS_DIR/evaluation_results/evaluation_summary.md${NC} # Read evaluation"
echo ""
echo -e "${BLUE}To answer your questions:${NC}"
echo -e "${BLUE}  1. MOEA containers: ✓ Separate CPU/GPU containers completed${NC}"
echo -e "${BLUE}  2. Full 2013-2016 data: ✓ Complete dataset processed${NC}"
echo -e "${BLUE}  3. Real-world evaluation: ✓ LightGBM validation completed${NC}"
echo -e "${BLUE}  4. CPU vs GPU comparison: ✓ Performance analysis available${NC}"
echo ""
echo -e "${GREEN}The evaluation framework validates MOEA solutions using trained LightGBM models${NC}"
echo -e "${GREEN}and assesses real-world performance including energy efficiency, plant growth,${NC}"
echo -e "${GREEN}economic impact, and operational feasibility.${NC}"