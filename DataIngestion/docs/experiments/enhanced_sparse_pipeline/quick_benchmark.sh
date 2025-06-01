#!/bin/bash

# Quick Benchmark for Enhanced Sparse Pipeline
# Runs a subset of experiments for rapid validation

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/quick_${TIMESTAMP}"
LOGS_DIR="logs/quick_${TIMESTAMP}"

mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}==================================${NC}"
echo -e "${BLUE}Enhanced Pipeline Quick Benchmark${NC}"
echo -e "${BLUE}==================================${NC}"
echo "Results: $RESULTS_DIR"
echo ""

# Quick test configurations
RESULTS_FILE="$RESULTS_DIR/quick_results.csv"
echo "test_name,mode,device,batch_size,duration,total_time_s,feature_count,features_per_sec,gpu_util_avg,memory_mb" > "$RESULTS_FILE"

# Function to run quick test
run_quick_test() {
    local test_name=$1
    local mode=$2
    local device=$3
    local batch_size=$4
    local start_date=$5
    local end_date=$6
    
    echo -e "\n${YELLOW}Test: $test_name${NC}"
    echo "Mode: $mode, Device: $device, Batch: $batch_size"
    
    # Set environment
    export SPARSE_MODE="true"
    export ENHANCED_MODE="false"
    export DISABLE_GPU="true"
    
    if [ "$mode" = "enhanced" ]; then
        export ENHANCED_MODE="true"
    fi
    
    if [ "$device" = "gpu" ]; then
        export DISABLE_GPU="false"
    fi
    
    # Log file
    local log_file="$LOGS_DIR/${test_name}.log"
    local gpu_log="$LOGS_DIR/${test_name}.gpu"
    
    # GPU monitoring
    if [ "$device" = "gpu" ]; then
        nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 > "$gpu_log" &
        local gpu_pid=$!
    fi
    
    # Run test
    local start_time=$(date +%s)
    
    if [ "$mode" = "enhanced" ]; then
        docker compose -f docker-compose.sparse.yml run --rm sparse_pipeline \
            --enhanced-mode --start-date "$start_date" --end-date "$end_date" \
            --batch-size $batch_size > "$log_file" 2>&1
    else
        docker compose -f docker-compose.sparse.yml run --rm sparse_pipeline \
            --sparse-mode --start-date "$start_date" --end-date "$end_date" \
            --batch-size $batch_size > "$log_file" 2>&1
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Stop GPU monitoring
    if [ "$device" = "gpu" ]; then
        kill $gpu_pid 2>/dev/null
        wait $gpu_pid 2>/dev/null
    fi
    
    # Extract metrics
    local feature_count=$(grep -E "(Total feature sets:|Window features:)" "$log_file" | grep -oE "[0-9]+" | tail -1)
    local features_per_sec=$(grep -E "features?/second" "$log_file" | grep -oE "[0-9]+\.?[0-9]*" | tail -1)
    
    local gpu_util="0"
    local memory_mb="0"
    if [ "$device" = "gpu" ] && [ -f "$gpu_log" ]; then
        gpu_util=$(awk '{ sum += $1; count++ } END { if (count > 0) print sum/count; else print 0 }' "$gpu_log")
        memory_mb=$(awk '{ sum += $2; count++ } END { if (count > 0) print sum/count; else print 0 }' "$gpu_log")
    fi
    
    # Save results
    echo "$test_name,$mode,$device,$batch_size,$start_date-$end_date,$duration,$feature_count,$features_per_sec,$gpu_util,$memory_mb" >> "$RESULTS_FILE"
    
    echo -e "${GREEN}✓ Completed in ${duration}s${NC}"
    echo "  Features: $feature_count"
    echo "  Speed: $features_per_sec feat/s"
    [ "$device" = "gpu" ] && echo "  GPU Util: ${gpu_util}%"
}

# Ensure database is ready
echo -e "${BLUE}Starting database...${NC}"
docker compose -f docker-compose.sparse.yml up -d db
sleep 10

# Quick benchmark tests
echo -e "\n${BLUE}Running Quick Benchmarks...${NC}"

# Test 1: Basic GPU vs CPU (1 month)
run_quick_test "basic_gpu_1mo" "basic" "gpu" 24 "2014-01-01" "2014-01-31"
run_quick_test "basic_cpu_1mo" "basic" "cpu" 24 "2014-01-01" "2014-01-31"

# Test 2: Enhanced GPU vs CPU (1 month)
run_quick_test "enhanced_gpu_1mo" "enhanced" "gpu" 24 "2014-01-01" "2014-01-31"
run_quick_test "enhanced_cpu_1mo" "enhanced" "cpu" 24 "2014-01-01" "2014-01-31"

# Test 3: Enhanced GPU different batch sizes (1 month)
run_quick_test "enhanced_gpu_b12" "enhanced" "gpu" 12 "2014-01-01" "2014-01-31"
run_quick_test "enhanced_gpu_b48" "enhanced" "gpu" 48 "2014-01-01" "2014-01-31"

# Test 4: Enhanced GPU larger dataset (3 months)
run_quick_test "enhanced_gpu_3mo" "enhanced" "gpu" 24 "2014-01-01" "2014-03-31"

# Generate summary
echo -e "\n${BLUE}=== Quick Benchmark Summary ===${NC}"
echo ""

# Calculate speedups
basic_gpu_time=$(grep "basic_gpu_1mo" "$RESULTS_FILE" | cut -d',' -f6)
basic_cpu_time=$(grep "basic_cpu_1mo" "$RESULTS_FILE" | cut -d',' -f6)
enhanced_gpu_time=$(grep "enhanced_gpu_1mo" "$RESULTS_FILE" | cut -d',' -f6)
enhanced_cpu_time=$(grep "enhanced_cpu_1mo" "$RESULTS_FILE" | cut -d',' -f6)

if [ -n "$basic_gpu_time" ] && [ -n "$basic_cpu_time" ]; then
    basic_speedup=$(echo "scale=1; $basic_cpu_time / $basic_gpu_time" | bc)
    echo "Basic Mode GPU Speedup: ${basic_speedup}x"
fi

if [ -n "$enhanced_gpu_time" ] && [ -n "$enhanced_cpu_time" ]; then
    enhanced_speedup=$(echo "scale=1; $enhanced_cpu_time / $enhanced_gpu_time" | bc)
    echo "Enhanced Mode GPU Speedup: ${enhanced_speedup}x"
fi

# Feature comparison
basic_features=$(grep "basic_gpu_1mo" "$RESULTS_FILE" | cut -d',' -f7)
enhanced_features=$(grep "enhanced_gpu_1mo" "$RESULTS_FILE" | cut -d',' -f7)

if [ -n "$basic_features" ] && [ -n "$enhanced_features" ]; then
    feature_ratio=$(echo "scale=1; $enhanced_features / $basic_features" | bc)
    echo "Feature Enhancement: ${enhanced_features} vs ${basic_features} (${feature_ratio}x)"
fi

# GPU utilization
enhanced_gpu_util=$(grep "enhanced_gpu_1mo" "$RESULTS_FILE" | cut -d',' -f9)
echo "Enhanced GPU Utilization: ${enhanced_gpu_util}%"

echo ""
echo -e "${GREEN}Quick benchmark complete!${NC}"
echo "Full results: $RESULTS_FILE"

# Create simple visualization
echo -e "\n${BLUE}Performance Summary:${NC}"
tail -n +2 "$RESULTS_FILE" | while IFS=',' read -r name mode device batch range time features fps gpu_util mem; do
    printf "%-20s: %6s feat in %4ss = %6s feat/s" "$name" "$features" "$time" "$fps"
    [ "$device" = "gpu" ] && printf " (GPU: %s%%)" "$gpu_util"
    echo ""
done