#!/bin/bash
# GPU Performance Test for Sparse Pipeline
# Tests the pipeline with GPU acceleration properly enabled

set -e

echo "============================================"
echo "GPU Performance Test for Sparse Pipeline"
echo "Date: $(date)"
echo "============================================"

# Check GPU availability
echo ""
echo "Checking GPU availability..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. GPU may not be available."
else
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
fi

# Create results directory
RESULTS_DIR="./docs/experiments/results/gpu_tests"
mkdir -p "$RESULTS_DIR"

# Test timestamp
TEST_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="$RESULTS_DIR/gpu_performance_${TEST_TIMESTAMP}.json"

# Clean up previous runs
echo ""
echo "Cleaning up previous runs..."
docker compose -f docker-compose.sparse.yml down -v || true
rm -rf ./gpu_feature_extraction/checkpoints/*

# Copy the sparse environment file
echo ""
echo "Setting up environment..."
cp .env.sparse .env

# Build with GPU support
echo ""
echo "Building containers with GPU support..."
docker compose -f docker-compose.sparse.yml build sparse_pipeline

# Function to measure execution time
measure_time() {
    local start_time=$(date +%s.%N)
    "$@"
    local end_time=$(date +%s.%N)
    echo "$(echo "$end_time - $start_time" | bc)"
}

# Initialize results JSON
cat > "$RESULTS_FILE" <<EOF
{
  "test": "gpu_performance",
  "date": "$(date -Iseconds)",
  "config": {
    "date_range": "2014-01-01 to 2014-07-01",
    "disable_gpu": "false",
    "gpu_check": "enabled"
  },
  "runs": [
EOF

# Run multiple tests
NUM_RUNS=3
echo ""
echo "Running $NUM_RUNS test iterations..."

for i in $(seq 1 $NUM_RUNS); do
    echo ""
    echo "==================== Run $i/$NUM_RUNS ===================="
    
    # Clean checkpoints between runs
    rm -rf ./gpu_feature_extraction/checkpoints/*
    
    # Start database
    docker compose -f docker-compose.sparse.yml up -d db
    sleep 10
    
    # Run ingestion
    echo "Running data ingestion..."
    INGESTION_TIME=$(measure_time docker compose -f docker-compose.sparse.yml up rust_pipeline 2>&1 | tee /tmp/ingestion_$i.log)
    
    # Extract ingestion metrics
    RECORDS=$(grep -oP "Processed \K\d+" /tmp/ingestion_$i.log | tail -1 || echo "0")
    
    # Run sparse pipeline with GPU
    echo "Running sparse GPU pipeline..."
    SPARSE_TIME=$(measure_time docker compose -f docker-compose.sparse.yml up sparse_pipeline 2>&1 | tee /tmp/sparse_$i.log)
    
    # Check if GPU was actually used
    GPU_USED=$(grep -c "CUDA context initialized" /tmp/sparse_$i.log || echo "0")
    GPU_DISABLED=$(grep -c "GPU disabled" /tmp/sparse_$i.log || echo "0")
    
    # Extract metrics
    HOURLY_POINTS=$(grep -oP "Hourly data points: \K\d+" /tmp/sparse_$i.log || echo "0")
    WINDOW_FEATURES=$(grep -oP "Window features: \K\d+" /tmp/sparse_$i.log || echo "0")
    MONTHLY_ERAS=$(grep -oP "Monthly eras: \K\d+" /tmp/sparse_$i.log || echo "0")
    FEATURES_PER_SEC=$(grep -oP "Performance: \K[0-9.]+" /tmp/sparse_$i.log || echo "0")
    
    # Calculate total time
    TOTAL_TIME=$(echo "$INGESTION_TIME + $SPARSE_TIME" | bc)
    
    # Add to JSON (with comma handling)
    if [ $i -gt 1 ]; then
        echo "," >> "$RESULTS_FILE"
    fi
    
    cat >> "$RESULTS_FILE" <<EOF
    {
      "run": $i,
      "gpu_status": {
        "cuda_initialized": $GPU_USED,
        "gpu_disabled": $GPU_DISABLED,
        "gpu_active": $([ $GPU_USED -gt 0 ] && [ $GPU_DISABLED -eq 0 ] && echo "true" || echo "false")
      },
      "timings": {
        "ingestion_seconds": $INGESTION_TIME,
        "sparse_pipeline_seconds": $SPARSE_TIME,
        "total_seconds": $TOTAL_TIME
      },
      "metrics": {
        "records_ingested": $RECORDS,
        "hourly_data_points": $HOURLY_POINTS,
        "window_features": $WINDOW_FEATURES,
        "monthly_eras": $MONTHLY_ERAS,
        "features_per_second": $FEATURES_PER_SEC
      }
    }
EOF
    
    # Clean up
    docker compose -f docker-compose.sparse.yml down
    
    # Short pause between runs
    sleep 5
done

# Close JSON array
cat >> "$RESULTS_FILE" <<EOF
  ]
}
EOF

# Generate summary report
echo ""
echo "============================================"
echo "Test Summary"
echo "============================================"

# Calculate averages
TOTAL_GPU_RUNS=$(jq '[.runs[] | select(.gpu_status.gpu_active == true)] | length' "$RESULTS_FILE")
AVG_SPARSE_TIME=$(jq '[.runs[] | .timings.sparse_pipeline_seconds] | add/length' "$RESULTS_FILE")
AVG_FEATURES_SEC=$(jq '[.runs[] | .metrics.features_per_second] | add/length' "$RESULTS_FILE")

echo "Total runs: $NUM_RUNS"
echo "GPU-enabled runs: $TOTAL_GPU_RUNS"
echo "Average sparse pipeline time: ${AVG_SPARSE_TIME}s"
echo "Average features/second: ${AVG_FEATURES_SEC}"

# Compare with CPU baseline if available
BASELINE_FILE="./docs/experiments/results/sparse_pipeline_baseline_20250531_173612.json"
if [ -f "$BASELINE_FILE" ]; then
    echo ""
    echo "Comparison with CPU baseline:"
    CPU_TIME=$(jq '.summary.sparse_mean' "$BASELINE_FILE")
    CPU_RATE=$(jq '.summary.features_per_sec_mean' "$BASELINE_FILE")
    
    SPEEDUP=$(echo "scale=2; $CPU_TIME / $AVG_SPARSE_TIME" | bc)
    RATE_IMPROVEMENT=$(echo "scale=2; $AVG_FEATURES_SEC / $CPU_RATE" | bc)
    
    echo "CPU baseline time: ${CPU_TIME}s"
    echo "GPU time: ${AVG_SPARSE_TIME}s"
    echo "Speedup: ${SPEEDUP}x"
    echo "Feature rate improvement: ${RATE_IMPROVEMENT}x"
fi

echo ""
echo "Results saved to: $RESULTS_FILE"
echo "Logs saved to: /tmp/sparse_*.log and /tmp/ingestion_*.log"

# Clean up
docker compose -f docker-compose.sparse.yml down -v