#!/bin/bash

# Enhanced Sparse Pipeline - CPU vs GPU Benchmark Experiments
# This script runs comprehensive benchmarks comparing:
# 1. Basic vs Enhanced pipeline modes
# 2. CPU vs GPU performance
# 3. Different batch sizes and data ranges

EXPERIMENT_NAME="enhanced_sparse_pipeline_benchmarks"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/${TIMESTAMP}"
LOGS_DIR="logs/${TIMESTAMP}"

# Create output directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$LOGS_DIR"

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}===========================================${NC}"
echo -e "${BLUE}Enhanced Sparse Pipeline Benchmark Suite${NC}"
echo -e "${BLUE}===========================================${NC}"
echo "Timestamp: $TIMESTAMP"
echo "Results: $RESULTS_DIR"
echo "Logs: $LOGS_DIR"
echo ""

# Configuration
declare -a BATCH_SIZES=(12 24 48 96)
declare -a DATE_RANGES=(
    "2014-01-01:2014-01-31"    # 1 month
    "2014-01-01:2014-03-31"    # 3 months
    "2014-01-01:2014-06-30"    # 6 months
    "2014-01-01:2014-12-31"    # 12 months
)
declare -a MODES=("basic" "enhanced")
declare -a DEVICES=("cpu" "gpu")

# Results file
RESULTS_FILE="$RESULTS_DIR/benchmark_results.csv"
echo "experiment_id,mode,device,batch_size,date_range,duration_months,total_time_s,feature_extraction_time_s,feature_count,features_per_second,gpu_util_avg,gpu_util_max,memory_used_mb,memory_peak_mb,cpu_percent,status" > "$RESULTS_FILE"

# Summary file
SUMMARY_FILE="$RESULTS_DIR/benchmark_summary.json"
echo "{" > "$SUMMARY_FILE"
echo "  \"timestamp\": \"$TIMESTAMP\"," >> "$SUMMARY_FILE"
echo "  \"experiments\": [" >> "$SUMMARY_FILE"

# Function to extract metrics from logs
extract_metrics() {
    local log_file=$1
    local mode=$2
    local device=$3
    
    # Extract timing
    local total_time=$(grep -E "(pipeline complete in|Sparse pipeline complete in)" "$log_file" | grep -oE "[0-9]+\.[0-9]+s" | sed 's/s//' | head -1)
    local feature_time=$(grep "Stage 3" "$log_file" | grep -oE "[0-9]+\.[0-9]+s" | sed 's/s//' | head -1)
    
    # Extract feature counts
    local feature_count=0
    if [ "$mode" = "enhanced" ]; then
        feature_count=$(grep "Total feature sets:" "$log_file" | grep -oE "[0-9]+" | tail -1)
    else
        feature_count=$(grep "Window features:" "$log_file" | grep -oE "[0-9]+" | tail -1)
    fi
    
    # Extract performance
    local features_per_sec=$(grep -E "(features?/second|feature sets/second)" "$log_file" | grep -oE "[0-9]+\.?[0-9]*" | tail -1)
    
    # GPU metrics (if available)
    local gpu_util_avg="0"
    local gpu_util_max="0"
    local memory_used="0"
    local memory_peak="0"
    
    if [ "$device" = "gpu" ] && [ -f "${log_file}.gpu" ]; then
        gpu_util_avg=$(awk '{ sum += $1; count++ } END { if (count > 0) print sum/count; else print 0 }' "${log_file}.gpu")
        gpu_util_max=$(awk '{ if ($1 > max) max = $1 } END { print max }' "${log_file}.gpu")
        memory_used=$(awk '{ sum += $2; count++ } END { if (count > 0) print sum/count; else print 0 }' "${log_file}.gpu")
        memory_peak=$(awk '{ if ($2 > max) max = $2 } END { print max }' "${log_file}.gpu")
    fi
    
    # CPU usage
    local cpu_percent=$(grep "CPU usage:" "$log_file" | grep -oE "[0-9]+\.?[0-9]*" | head -1)
    [ -z "$cpu_percent" ] && cpu_percent="0"
    
    echo "$total_time,$feature_time,$feature_count,$features_per_sec,$gpu_util_avg,$gpu_util_max,$memory_used,$memory_peak,$cpu_percent"
}

# Function to monitor GPU
monitor_gpu() {
    local pid=$1
    local output_file=$2
    
    while kill -0 $pid 2>/dev/null; do
        nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits >> "$output_file"
        sleep 1
    done
}

# Function to run a single experiment
run_experiment() {
    local experiment_id=$1
    local mode=$2
    local device=$3
    local batch_size=$4
    local start_date=$5
    local end_date=$6
    local duration_months=$7
    
    echo -e "\n${YELLOW}Experiment $experiment_id: $mode mode, $device, batch=$batch_size, $start_date to $end_date${NC}"
    
    local log_file="$LOGS_DIR/exp_${experiment_id}_${mode}_${device}_b${batch_size}.log"
    local gpu_log_file="${log_file}.gpu"
    
    # Prepare environment
    export SPARSE_MODE="true"
    export ENHANCED_MODE="false"
    export DISABLE_GPU="true"
    export SPARSE_BATCH_SIZE="$batch_size"
    export SPARSE_START_DATE="$start_date"
    export SPARSE_END_DATE="$end_date"
    
    if [ "$mode" = "enhanced" ]; then
        export ENHANCED_MODE="true"
    fi
    
    if [ "$device" = "gpu" ]; then
        export DISABLE_GPU="false"
    fi
    
    # Start GPU monitoring if using GPU
    local gpu_monitor_pid=""
    if [ "$device" = "gpu" ]; then
        monitor_gpu $$ "$gpu_log_file" &
        gpu_monitor_pid=$!
    fi
    
    # Run the experiment
    echo "Starting at $(date)..."
    local start_time=$(date +%s)
    
    # Docker command based on mode
    local docker_cmd=""
    if [ "$mode" = "enhanced" ]; then
        docker_cmd="docker compose -f docker-compose.sparse.yml run --rm sparse_pipeline --enhanced-mode --start-date $start_date --end-date $end_date --batch-size $batch_size"
    else
        docker_cmd="docker compose -f docker-compose.sparse.yml run --rm sparse_pipeline --sparse-mode --start-date $start_date --end-date $end_date --batch-size $batch_size"
    fi
    
    # Execute and capture output
    if $docker_cmd > "$log_file" 2>&1; then
        local status="success"
        echo -e "${GREEN}✓ Completed successfully${NC}"
    else
        local status="failed"
        echo -e "${RED}✗ Failed${NC}"
    fi
    
    local end_time=$(date +%s)
    local total_time=$((end_time - start_time))
    
    # Stop GPU monitoring
    if [ -n "$gpu_monitor_pid" ]; then
        kill $gpu_monitor_pid 2>/dev/null
        wait $gpu_monitor_pid 2>/dev/null
    fi
    
    # Extract metrics
    local metrics=$(extract_metrics "$log_file" "$mode" "$device")
    
    # Write to results file
    echo "$experiment_id,$mode,$device,$batch_size,$start_date:$end_date,$duration_months,$total_time,$metrics,$status" >> "$RESULTS_FILE"
    
    # Add to summary JSON
    if [ "$experiment_id" -gt 1 ]; then
        echo "," >> "$SUMMARY_FILE"
    fi
    echo -n "    {\"id\": $experiment_id, \"mode\": \"$mode\", \"device\": \"$device\", \"batch_size\": $batch_size, \"duration_months\": $duration_months, \"total_time\": $total_time, \"status\": \"$status\"}" >> "$SUMMARY_FILE"
    
    # Cool down period between experiments
    sleep 5
}

# Ensure database is running
echo -e "${BLUE}Ensuring database is ready...${NC}"
docker compose -f docker-compose.sparse.yml up -d db
sleep 10

# Run data ingestion if needed
echo -e "${BLUE}Checking if data ingestion is needed...${NC}"
if ! docker compose -f docker-compose.sparse.yml run --rm sparse_pipeline --sparse-mode --start-date 2014-01-01 --end-date 2014-01-01 --batch-size 24 > /dev/null 2>&1; then
    echo "Running data ingestion..."
    docker compose -f docker-compose.sparse.yml run --rm rust_pipeline
fi

# Main experiment loop
experiment_id=1

for date_range in "${DATE_RANGES[@]}"; do
    IFS=':' read -r start_date end_date <<< "$date_range"
    
    # Calculate duration in months
    start_year=$(echo $start_date | cut -d'-' -f1)
    start_month=$(echo $start_date | cut -d'-' -f2)
    end_year=$(echo $end_date | cut -d'-' -f1)
    end_month=$(echo $end_date | cut -d'-' -f2)
    duration_months=$(( (end_year - start_year) * 12 + end_month - start_month + 1 ))
    
    for batch_size in "${BATCH_SIZES[@]}"; do
        for mode in "${MODES[@]}"; do
            for device in "${DEVICES[@]}"; do
                # Skip enhanced CPU mode for large datasets (too slow)
                if [ "$mode" = "enhanced" ] && [ "$device" = "cpu" ] && [ $duration_months -gt 3 ]; then
                    echo -e "${YELLOW}Skipping enhanced CPU mode for $duration_months months (too slow)${NC}"
                    continue
                fi
                
                run_experiment $experiment_id "$mode" "$device" $batch_size "$start_date" "$end_date" $duration_months
                experiment_id=$((experiment_id + 1))
            done
        done
    done
done

# Close JSON summary
echo "" >> "$SUMMARY_FILE"
echo "  ]," >> "$SUMMARY_FILE"
echo "  \"total_experiments\": $((experiment_id - 1))" >> "$SUMMARY_FILE"
echo "}" >> "$SUMMARY_FILE"

# Generate performance report
echo -e "\n${BLUE}Generating performance report...${NC}"
python3 scripts/analyze_benchmark_results.py "$RESULTS_FILE" "$RESULTS_DIR"

echo -e "\n${GREEN}Benchmark suite completed!${NC}"
echo "Results saved to: $RESULTS_FILE"
echo "Summary saved to: $SUMMARY_FILE"
echo "Logs saved to: $LOGS_DIR"

# Show quick summary
echo -e "\n${BLUE}Quick Summary:${NC}"
echo "Total experiments: $((experiment_id - 1))"
echo "Successful: $(grep -c ",success$" "$RESULTS_FILE")"
echo "Failed: $(grep -c ",failed$" "$RESULTS_FILE")"

# Display top performers
echo -e "\n${BLUE}Top 5 Performers (by features/second):${NC}"
tail -n +2 "$RESULTS_FILE" | sort -t',' -k10 -nr | head -5 | while IFS=',' read -r id mode device batch range months time feat_time feat_count fps rest; do
    echo "  $mode/$device: ${fps} feat/s (batch=$batch, ${months}mo)"
done