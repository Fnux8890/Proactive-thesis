#!/bin/bash
# Quick GPU Test - Single month comparison between CPU and GPU
# Faster test for validating GPU functionality

set -e

echo "============================================"
echo "Quick GPU Test - Single Month Comparison"
echo "Date: $(date)"
echo "Test Period: May 2014 (1 month)"
echo "============================================"

# Configuration
TEST_MONTH="2014-05"
START_DATE="${TEST_MONTH}-01"
END_DATE="${TEST_MONTH}-31"
RESULTS_DIR="./docs/experiments/results/quick_tests"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create results directory
mkdir -p "$RESULTS_DIR"

# Clean up
echo ""
echo "Cleaning up previous runs..."
docker compose -f docker-compose.sparse.yml down -v || true
rm -rf ./gpu_feature_extraction/checkpoints/*

# Copy environment
cp .env.sparse .env

# Build container
echo ""
echo "Building sparse pipeline container..."
docker compose -f docker-compose.sparse.yml build sparse_pipeline

# Start database
echo ""
echo "Starting database..."
docker compose -f docker-compose.sparse.yml up -d db
sleep 10

# Run data ingestion once
echo ""
echo "Running data ingestion for May 2014..."
docker compose -f docker-compose.sparse.yml run --rm \
    -e DATA_SOURCE_PATH=/app/data \
    rust_pipeline > /tmp/ingestion_quick.log 2>&1

echo "Ingestion complete. Starting tests..."

# Function to run sparse pipeline and measure time
run_sparse_pipeline() {
    local mode=$1
    local disable_gpu=$2
    local log_file=$3
    
    echo ""
    echo "Running sparse pipeline in $mode mode..."
    
    # Clean checkpoints
    rm -rf ./gpu_feature_extraction/checkpoints/*
    
    # Measure time
    local start_time=$(date +%s.%N)
    
    docker compose -f docker-compose.sparse.yml run --rm \
        -e DISABLE_GPU=$disable_gpu \
        -e CUDA_VISIBLE_DEVICES=0 \
        sparse_pipeline \
        --sparse-mode \
        --start-date "$START_DATE" \
        --end-date "$END_DATE" \
        --batch-size 24 2>&1 | tee "$log_file"
    
    local end_time=$(date +%s.%N)
    local elapsed=$(echo "$end_time - $start_time" | bc)
    
    echo "$elapsed"
}

# Run CPU test
echo ""
echo "============================================"
echo "CPU Test"
echo "============================================"
CPU_TIME=$(run_sparse_pipeline "CPU" "true" "/tmp/cpu_quick.log")

# Extract CPU metrics
CPU_FEATURES=$(grep -oP "Window features: \K\d+" /tmp/cpu_quick.log || echo "0")
CPU_RATE=$(grep -oP "Performance: \K[0-9.]+" /tmp/cpu_quick.log || echo "0")
CPU_GPU_MSG=$(grep -c "GPU disabled by environment variable" /tmp/cpu_quick.log || echo "0")

echo ""
echo "CPU Results:"
echo "  Time: ${CPU_TIME}s"
echo "  Features: $CPU_FEATURES"
echo "  Rate: ${CPU_RATE} features/sec"
echo "  GPU Disabled: $([ $CPU_GPU_MSG -gt 0 ] && echo "Yes" || echo "No")"

# Run GPU test
echo ""
echo "============================================"
echo "GPU Test"
echo "============================================"
GPU_TIME=$(run_sparse_pipeline "GPU" "false" "/tmp/gpu_quick.log")

# Extract GPU metrics
GPU_FEATURES=$(grep -oP "Window features: \K\d+" /tmp/gpu_quick.log || echo "0")
GPU_RATE=$(grep -oP "Performance: \K[0-9.]+" /tmp/gpu_quick.log || echo "0")
GPU_INIT_MSG=$(grep -c "CUDA context initialized" /tmp/gpu_quick.log || echo "0")
GPU_DISABLED_MSG=$(grep -c "GPU disabled" /tmp/gpu_quick.log || echo "0")

echo ""
echo "GPU Results:"
echo "  Time: ${GPU_TIME}s"
echo "  Features: $GPU_FEATURES"
echo "  Rate: ${GPU_RATE} features/sec"
echo "  GPU Initialized: $([ $GPU_INIT_MSG -gt 0 ] && echo "Yes" || echo "No")"

# Calculate speedup
if [ $(echo "$GPU_TIME > 0" | bc) -eq 1 ]; then
    SPEEDUP=$(echo "scale=2; $CPU_TIME / $GPU_TIME" | bc)
    RATE_IMPROVEMENT=$(echo "scale=2; $GPU_RATE / $CPU_RATE" | bc)
else
    SPEEDUP="N/A"
    RATE_IMPROVEMENT="N/A"
fi

# Generate results JSON
RESULTS_FILE="$RESULTS_DIR/quick_test_${TIMESTAMP}.json"
cat > "$RESULTS_FILE" <<EOF
{
  "test": "quick_gpu_comparison",
  "date": "$(date -Iseconds)",
  "test_period": "$TEST_MONTH",
  "cpu": {
    "time_seconds": $CPU_TIME,
    "features": $CPU_FEATURES,
    "features_per_second": $CPU_RATE,
    "gpu_disabled": $([ $CPU_GPU_MSG -gt 0 ] && echo "true" || echo "false")
  },
  "gpu": {
    "time_seconds": $GPU_TIME,
    "features": $GPU_FEATURES,
    "features_per_second": $GPU_RATE,
    "gpu_initialized": $([ $GPU_INIT_MSG -gt 0 ] && echo "true" || echo "false")
  },
  "comparison": {
    "time_speedup": "$SPEEDUP",
    "rate_improvement": "$RATE_IMPROVEMENT",
    "gpu_working": $([ $GPU_INIT_MSG -gt 0 ] && [ $GPU_DISABLED_MSG -eq 0 ] && echo "true" || echo "false")
  }
}
EOF

# Generate quick report
REPORT_FILE="$RESULTS_DIR/quick_report_${TIMESTAMP}.md"
cat > "$REPORT_FILE" <<EOF
# Quick GPU Test Report

Date: $(date)
Test Period: May 2014 (1 month)

## Results Summary

| Metric | CPU | GPU | Improvement |
|--------|-----|-----|-------------|
| Execution Time | ${CPU_TIME}s | ${GPU_TIME}s | ${SPEEDUP}x |
| Features Extracted | $CPU_FEATURES | $GPU_FEATURES | - |
| Feature Rate | ${CPU_RATE} feat/s | ${GPU_RATE} feat/s | ${RATE_IMPROVEMENT}x |

## GPU Status

EOF

if [ $GPU_INIT_MSG -gt 0 ] && [ $GPU_DISABLED_MSG -eq 0 ]; then
    cat >> "$REPORT_FILE" <<EOF
✅ **GPU Successfully Activated**

The GPU was properly initialized and used for feature extraction. The performance improvement of ${SPEEDUP}x validates that GPU acceleration is working correctly.

### Next Steps:
1. Run full comparison test: \`./run_cpu_vs_gpu_comparison.sh\`
2. Test with larger datasets
3. Monitor GPU utilization during execution
EOF
else
    cat >> "$REPORT_FILE" <<EOF
❌ **GPU Not Activated**

The GPU was not successfully initialized. Please check:

1. **NVIDIA Drivers**: Run \`nvidia-smi\` to verify drivers are installed
2. **Docker GPU Runtime**: Test with \`docker run --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi\`
3. **Environment Variables**: Ensure DISABLE_GPU=false in .env
4. **Container Rebuild**: Run \`docker compose -f docker-compose.sparse.yml build --no-cache sparse_pipeline\`

### Debug Information:
- GPU Init Messages: $GPU_INIT_MSG
- GPU Disabled Messages: $GPU_DISABLED_MSG
- CPU Mode Confirmed: $([ $CPU_GPU_MSG -gt 0 ] && echo "Yes" || echo "No")
EOF
fi

# Print summary
echo ""
echo "============================================"
echo "Quick Test Summary"
echo "============================================"
echo "CPU Time: ${CPU_TIME}s"
echo "GPU Time: ${GPU_TIME}s"
echo "Speedup: ${SPEEDUP}x"
echo "GPU Working: $([ $GPU_INIT_MSG -gt 0 ] && [ $GPU_DISABLED_MSG -eq 0 ] && echo "Yes" || echo "No")"
echo ""
echo "Results saved to:"
echo "  JSON: $RESULTS_FILE"
echo "  Report: $REPORT_FILE"
echo ""
echo "Logs available at:"
echo "  CPU: /tmp/cpu_quick.log"
echo "  GPU: /tmp/gpu_quick.log"

# Cleanup
docker compose -f docker-compose.sparse.yml down