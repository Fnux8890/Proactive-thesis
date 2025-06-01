#!/bin/bash

# Multi-Run MOEA Experiment Script
# 
# This script executes multiple MOEA experiments for statistical analysis
# and comparison of CPU vs GPU performance on greenhouse optimization
#
# Usage:
#   ./run_multiple_experiments.sh --cpu-runs 5 --gpu-runs 5
#   ./run_multiple_experiments.sh --algorithm cpu --runs 10
#   ./run_multiple_experiments.sh --quick-test

set -e

# Default parameters
CPU_RUNS=3
GPU_RUNS=3
GENERATIONS=500
POPULATION=100
QUICK_MODE=false
ALGORITHM=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu-runs)
            CPU_RUNS="$2"
            shift 2
            ;;
        --gpu-runs)
            GPU_RUNS="$2"
            shift 2
            ;;
        --algorithm)
            ALGORITHM="$2"
            shift 2
            ;;
        --runs)
            CPU_RUNS="$2"
            GPU_RUNS="$2"
            shift 2
            ;;
        --generations)
            GENERATIONS="$2"
            shift 2
            ;;
        --population)
            POPULATION="$2"
            shift 2
            ;;
        --quick-test)
            QUICK_MODE=true
            GENERATIONS=100
            POPULATION=50
            CPU_RUNS=2
            GPU_RUNS=2
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --cpu-runs N         Number of CPU experiment runs (default: 3)"
            echo "  --gpu-runs N         Number of GPU experiment runs (default: 3)"
            echo "  --algorithm TYPE     Run only specific algorithm (cpu|gpu)"
            echo "  --runs N             Set both CPU and GPU runs to N"
            echo "  --generations N      MOEA generations per run (default: 500)"
            echo "  --population N       MOEA population size (default: 100)"
            echo "  --quick-test         Quick test mode (100 gen, 50 pop, 2 runs each)"
            echo "  --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --cpu-runs 5 --gpu-runs 5"
            echo "  $0 --algorithm cpu --runs 10"
            echo "  $0 --quick-test"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== MOEA Multi-Run Experiment Suite ===${NC}"
echo ""
echo "Configuration:"
if [[ -z "$ALGORITHM" || "$ALGORITHM" == "cpu" ]]; then
    echo "  CPU Runs: $CPU_RUNS"
fi
if [[ -z "$ALGORITHM" || "$ALGORITHM" == "gpu" ]]; then
    echo "  GPU Runs: $GPU_RUNS"
fi
echo "  Generations: $GENERATIONS"
echo "  Population: $POPULATION"
echo "  Quick Mode: $QUICK_MODE"
echo ""

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

# Check Docker Compose
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker not found. Please install Docker.${NC}"
    exit 1
fi

# Check Python and dependencies
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python3 not found. Please install Python3.${NC}"
    exit 1
fi

# Check if experiment tracker exists
if [[ ! -f "experiments/experiment_tracker.py" ]]; then
    echo -e "${RED}Error: Experiment tracker not found. Please ensure experiment_tracker.py exists.${NC}"
    exit 1
fi

# Check database connection
echo "Testing database connection..."
python3 -c "
import psycopg2
try:
    conn = psycopg2.connect('postgresql://postgres:postgres@localhost:5432/postgres')
    conn.close()
    print('✓ Database connection successful')
except Exception as e:
    print(f'✗ Database connection failed: {e}')
    exit(1)
"

if [[ $? -ne 0 ]]; then
    echo -e "${RED}Error: Database connection failed. Please ensure PostgreSQL is running.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Prerequisites check passed${NC}"
echo ""

# Create experiments directory
mkdir -p experiments/results
mkdir -p experiments/reports

# Install Python dependencies if needed
echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip3 install psycopg2-binary pandas numpy scipy matplotlib seaborn &> /dev/null || echo "Dependencies already installed"

# Function to run experiments for a specific algorithm
run_experiments() {
    local algorithm=$1
    local num_runs=$2
    
    echo -e "${BLUE}Starting $algorithm experiments ($num_runs runs)...${NC}"
    
    for ((i=1; i<=num_runs; i++)); do
        echo -e "${YELLOW}[$algorithm] Run $i/$num_runs${NC}"
        
        start_time=$(date +%s)
        
        # Run experiment using the tracker
        python3 experiments/experiment_tracker.py run \
            --algorithm "$algorithm" \
            --runs 1 \
            --generations "$GENERATIONS" \
            --population "$POPULATION"
        
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        
        echo -e "${GREEN}[$algorithm] Run $i completed in ${duration}s${NC}"
        echo ""
        
        # Brief pause between runs
        sleep 2
    done
}

# Start timestamp
experiment_start=$(date +%s)
echo -e "${BLUE}Experiment suite started at $(date)${NC}"
echo ""

# Run CPU experiments
if [[ -z "$ALGORITHM" || "$ALGORITHM" == "cpu" ]]; then
    run_experiments "cpu" "$CPU_RUNS"
fi

# Run GPU experiments  
if [[ -z "$ALGORITHM" || "$ALGORITHM" == "gpu" ]]; then
    run_experiments "gpu" "$GPU_RUNS"
fi

# Generate comprehensive analysis report
echo -e "${BLUE}Generating analysis report...${NC}"

report_filename="experiment_report_$(date +%Y%m%d_%H%M%S).md"

python3 experiments/experiment_tracker.py report --output "$report_filename"

# Calculate total experiment time
experiment_end=$(date +%s)
total_duration=$((experiment_end - experiment_start))
total_minutes=$((total_duration / 60))
total_seconds=$((total_duration % 60))

echo ""
echo -e "${GREEN}=== Experiment Suite Completed ===${NC}"
echo ""
echo "Summary:"
if [[ -z "$ALGORITHM" || "$ALGORITHM" == "cpu" ]]; then
    echo "  CPU Experiments: $CPU_RUNS runs"
fi
if [[ -z "$ALGORITHM" || "$ALGORITHM" == "gpu" ]]; then
    echo "  GPU Experiments: $GPU_RUNS runs"
fi
echo "  Total Duration: ${total_minutes}m ${total_seconds}s"
echo "  Report Generated: experiments/reports/$report_filename"
echo ""

# Show quick analysis
echo -e "${BLUE}Quick Analysis:${NC}"
python3 experiments/experiment_tracker.py analyze

echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "1. Review detailed report: experiments/reports/$report_filename"
echo "2. Access database results: psql -h localhost -U postgres -d postgres"
echo "3. Run additional experiments: ./run_multiple_experiments.sh --help"
echo "4. Generate custom reports: python3 experiments/experiment_tracker.py report"

echo ""
echo -e "${GREEN}✓ All experiments completed successfully!${NC}"