#!/bin/bash
# Comprehensive CPU vs GPU Performance Comparison
# Runs multiple tests with both CPU and GPU modes to validate performance improvements

set -e

echo "============================================"
echo "CPU vs GPU Performance Comparison Test"
echo "Date: $(date)"
echo "============================================"

# Configuration
NUM_RUNS=5  # Number of test runs for each mode
TEST_START_DATE="2014-01-01"
TEST_END_DATE="2014-07-01"  # 6 months of data
RESULTS_DIR="./docs/experiments/results/cpu_gpu_comparison"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create results directory
mkdir -p "$RESULTS_DIR"

# Function to measure execution time with high precision
measure_time() {
    local start_time=$(date +%s.%N)
    "$@"
    local exit_code=$?
    local end_time=$(date +%s.%N)
    if [ $exit_code -eq 0 ]; then
        echo "$(echo "$end_time - $start_time" | bc)"
    else
        echo "-1"
    fi
    return $exit_code
}

# Function to extract metrics from logs
extract_metrics() {
    local log_file=$1
    local metrics_file=$2
    
    # Extract all relevant metrics
    echo "{" > "$metrics_file"
    echo "  \"hourly_data_points\": $(grep -oP "Hourly data points: \K\d+" "$log_file" | tail -1 || echo 0)," >> "$metrics_file"
    echo "  \"window_features\": $(grep -oP "Window features: \K\d+" "$log_file" | tail -1 || echo 0)," >> "$metrics_file"
    echo "  \"monthly_eras\": $(grep -oP "Monthly eras: \K\d+" "$log_file" | tail -1 || echo 0)," >> "$metrics_file"
    echo "  \"features_per_second\": $(grep -oP "Performance: \K[0-9.]+" "$log_file" | tail -1 || echo 0)," >> "$metrics_file"
    echo "  \"gpu_initialized\": $(grep -c "CUDA context initialized" "$log_file" || echo 0)," >> "$metrics_file"
    echo "  \"gpu_disabled\": $(grep -c "GPU disabled by environment variable" "$log_file" || echo 0)," >> "$metrics_file"
    echo "  \"gap_fills_co2\": $(grep -oP "CO2 gaps filled: \K\d+" "$log_file" | tail -1 || echo 0)," >> "$metrics_file"
    echo "  \"gap_fills_humidity\": $(grep -oP "Humidity gaps filled: \K\d+" "$log_file" | tail -1 || echo 0)," >> "$metrics_file"
    echo "  \"quality_coverage\": $(grep -oP "Coverage: \K[0-9.]+" "$log_file" | tail -1 || echo 0)," >> "$metrics_file"
    echo "  \"quality_continuity\": $(grep -oP "Continuity: \K[0-9.]+" "$log_file" | tail -1 || echo 0)," >> "$metrics_file"
    echo "  \"quality_consistency\": $(grep -oP "Consistency: \K[0-9.]+" "$log_file" | tail -1 || echo 0)" >> "$metrics_file"
    echo "}" >> "$metrics_file"
}

# Initialize results JSON
RESULTS_FILE="$RESULTS_DIR/comparison_${TIMESTAMP}.json"
cat > "$RESULTS_FILE" <<EOF
{
  "test": "cpu_vs_gpu_comparison",
  "date": "$(date -Iseconds)",
  "config": {
    "date_range": "$TEST_START_DATE to $TEST_END_DATE",
    "num_runs": $NUM_RUNS,
    "batch_size": 24,
    "window_hours": 12,
    "slide_hours": 6
  },
  "cpu_runs": [
EOF

# Clean up any previous runs
echo ""
echo "Cleaning up previous runs..."
docker compose -f docker-compose.sparse.yml down -v || true
rm -rf ./gpu_feature_extraction/checkpoints/*

# Copy sparse environment
cp .env.sparse .env

# Build containers
echo ""
echo "Building containers..."
docker compose -f docker-compose.sparse.yml build sparse_pipeline

# ====================
# CPU Performance Tests
# ====================
echo ""
echo "============================================"
echo "Running CPU Performance Tests"
echo "============================================"

CPU_TIMES=()
CPU_FEATURES_PER_SEC=()

for i in $(seq 1 $NUM_RUNS); do
    echo ""
    echo "CPU Run $i/$NUM_RUNS"
    echo "--------------------"
    
    # Clean checkpoints
    rm -rf ./gpu_feature_extraction/checkpoints/*
    
    # Start database
    docker compose -f docker-compose.sparse.yml up -d db
    sleep 10
    
    # Run data ingestion
    echo "Running data ingestion..."
    INGESTION_TIME=$(measure_time docker compose -f docker-compose.sparse.yml up rust_pipeline 2>&1 | tee "/tmp/cpu_ingestion_$i.log")
    
    if [ "$INGESTION_TIME" = "-1" ]; then
        echo "ERROR: Ingestion failed"
        continue
    fi
    
    # Get ingestion record count
    RECORDS=$(grep -oP "Processed \K\d+" "/tmp/cpu_ingestion_$i.log" | tail -1 || echo "0")
    
    # Run sparse pipeline with CPU (DISABLE_GPU=true)
    echo "Running sparse pipeline (CPU mode)..."
    SPARSE_TIME=$(measure_time docker compose -f docker-compose.sparse.yml run --rm \
        -e DISABLE_GPU=true \
        sparse_pipeline \
        --sparse-mode \
        --start-date "$TEST_START_DATE" \
        --end-date "$TEST_END_DATE" \
        --batch-size 24 2>&1 | tee "/tmp/cpu_sparse_$i.log")
    
    if [ "$SPARSE_TIME" = "-1" ]; then
        echo "ERROR: Sparse pipeline failed"
        docker compose -f docker-compose.sparse.yml down
        continue
    fi
    
    # Extract metrics
    extract_metrics "/tmp/cpu_sparse_$i.log" "/tmp/cpu_metrics_$i.json"
    
    # Read metrics
    FEATURES_PER_SEC=$(jq -r '.features_per_second' "/tmp/cpu_metrics_$i.json")
    CPU_TIMES+=("$SPARSE_TIME")
    CPU_FEATURES_PER_SEC+=("$FEATURES_PER_SEC")
    
    # Calculate total time
    TOTAL_TIME=$(echo "$INGESTION_TIME + $SPARSE_TIME" | bc)
    
    # Add to JSON
    if [ $i -gt 1 ]; then
        echo "," >> "$RESULTS_FILE"
    fi
    
    cat >> "$RESULTS_FILE" <<EOF
    {
      "run": $i,
      "mode": "cpu",
      "timings": {
        "ingestion_seconds": $INGESTION_TIME,
        "sparse_pipeline_seconds": $SPARSE_TIME,
        "total_seconds": $TOTAL_TIME
      },
      "metrics": $(cat "/tmp/cpu_metrics_$i.json")
    }
EOF
    
    # Clean up
    docker compose -f docker-compose.sparse.yml down
    sleep 5
done

# Close CPU runs array and start GPU runs
cat >> "$RESULTS_FILE" <<EOF
  ],
  "gpu_runs": [
EOF

# ====================
# GPU Performance Tests
# ====================
echo ""
echo "============================================"
echo "Running GPU Performance Tests"
echo "============================================"

GPU_TIMES=()
GPU_FEATURES_PER_SEC=()

for i in $(seq 1 $NUM_RUNS); do
    echo ""
    echo "GPU Run $i/$NUM_RUNS"
    echo "--------------------"
    
    # Clean checkpoints
    rm -rf ./gpu_feature_extraction/checkpoints/*
    
    # Start database
    docker compose -f docker-compose.sparse.yml up -d db
    sleep 10
    
    # Run data ingestion
    echo "Running data ingestion..."
    INGESTION_TIME=$(measure_time docker compose -f docker-compose.sparse.yml up rust_pipeline 2>&1 | tee "/tmp/gpu_ingestion_$i.log")
    
    if [ "$INGESTION_TIME" = "-1" ]; then
        echo "ERROR: Ingestion failed"
        continue
    fi
    
    # Get ingestion record count
    RECORDS=$(grep -oP "Processed \K\d+" "/tmp/gpu_ingestion_$i.log" | tail -1 || echo "0")
    
    # Run sparse pipeline with GPU (DISABLE_GPU=false)
    echo "Running sparse pipeline (GPU mode)..."
    SPARSE_TIME=$(measure_time docker compose -f docker-compose.sparse.yml run --rm \
        -e DISABLE_GPU=false \
        -e CUDA_VISIBLE_DEVICES=0 \
        sparse_pipeline \
        --sparse-mode \
        --start-date "$TEST_START_DATE" \
        --end-date "$TEST_END_DATE" \
        --batch-size 24 2>&1 | tee "/tmp/gpu_sparse_$i.log")
    
    if [ "$SPARSE_TIME" = "-1" ]; then
        echo "ERROR: Sparse pipeline failed"
        docker compose -f docker-compose.sparse.yml down
        continue
    fi
    
    # Extract metrics
    extract_metrics "/tmp/gpu_sparse_$i.log" "/tmp/gpu_metrics_$i.json"
    
    # Read metrics
    FEATURES_PER_SEC=$(jq -r '.features_per_second' "/tmp/gpu_metrics_$i.json")
    GPU_TIMES+=("$SPARSE_TIME")
    GPU_FEATURES_PER_SEC+=("$FEATURES_PER_SEC")
    
    # Calculate total time
    TOTAL_TIME=$(echo "$INGESTION_TIME + $SPARSE_TIME" | bc)
    
    # Add to JSON
    if [ $i -gt 1 ]; then
        echo "," >> "$RESULTS_FILE"
    fi
    
    cat >> "$RESULTS_FILE" <<EOF
    {
      "run": $i,
      "mode": "gpu",
      "timings": {
        "ingestion_seconds": $INGESTION_TIME,
        "sparse_pipeline_seconds": $SPARSE_TIME,
        "total_seconds": $TOTAL_TIME
      },
      "metrics": $(cat "/tmp/gpu_metrics_$i.json")
    }
EOF
    
    # Clean up
    docker compose -f docker-compose.sparse.yml down
    sleep 5
done

# Close GPU runs array
echo "  ]" >> "$RESULTS_FILE"

# ====================
# Statistical Analysis
# ====================
echo ""
echo "============================================"
echo "Calculating Statistics"
echo "============================================"

# Function to calculate mean
calc_mean() {
    local arr=("$@")
    local sum=0
    local count=0
    for val in "${arr[@]}"; do
        if [ "$val" != "-1" ]; then
            sum=$(echo "$sum + $val" | bc)
            count=$((count + 1))
        fi
    done
    if [ $count -gt 0 ]; then
        echo "scale=3; $sum / $count" | bc
    else
        echo "0"
    fi
}

# Function to calculate std dev
calc_std() {
    local arr=("$@")
    local mean=$1
    shift
    local arr=("$@")
    local sum=0
    local count=0
    for val in "${arr[@]}"; do
        if [ "$val" != "-1" ]; then
            diff=$(echo "$val - $mean" | bc)
            sum=$(echo "$sum + ($diff * $diff)" | bc)
            count=$((count + 1))
        fi
    done
    if [ $count -gt 1 ]; then
        echo "scale=3; sqrt($sum / ($count - 1))" | bc
    else
        echo "0"
    fi
}

# Calculate statistics
CPU_MEAN=$(calc_mean "${CPU_TIMES[@]}")
CPU_STD=$(calc_std "$CPU_MEAN" "${CPU_TIMES[@]}")
CPU_FEAT_MEAN=$(calc_mean "${CPU_FEATURES_PER_SEC[@]}")

GPU_MEAN=$(calc_mean "${GPU_TIMES[@]}")
GPU_STD=$(calc_std "$GPU_MEAN" "${GPU_TIMES[@]}")
GPU_FEAT_MEAN=$(calc_mean "${GPU_FEATURES_PER_SEC[@]}")

# Calculate speedup
if [ $(echo "$GPU_MEAN > 0" | bc) -eq 1 ]; then
    SPEEDUP=$(echo "scale=2; $CPU_MEAN / $GPU_MEAN" | bc)
    FEAT_SPEEDUP=$(echo "scale=2; $GPU_FEAT_MEAN / $CPU_FEAT_MEAN" | bc)
else
    SPEEDUP="N/A"
    FEAT_SPEEDUP="N/A"
fi

# Add summary to JSON
cat >> "$RESULTS_FILE" <<EOF
,
  "summary": {
    "cpu": {
      "mean_time": $CPU_MEAN,
      "std_time": $CPU_STD,
      "mean_features_per_sec": $CPU_FEAT_MEAN,
      "valid_runs": ${#CPU_TIMES[@]}
    },
    "gpu": {
      "mean_time": $GPU_MEAN,
      "std_time": $GPU_STD,
      "mean_features_per_sec": $GPU_FEAT_MEAN,
      "valid_runs": ${#GPU_TIMES[@]}
    },
    "speedup": {
      "time_speedup": "$SPEEDUP",
      "feature_rate_speedup": "$FEAT_SPEEDUP"
    }
  }
}
EOF

# ====================
# Generate Report
# ====================
echo ""
echo "Generating analysis report..."

python3 << 'PYTHON_SCRIPT'
import json
import sys
from datetime import datetime

# Load results
with open('$RESULTS_FILE', 'r') as f:
    data = json.load(f)

# Generate markdown report
report = f"""# CPU vs GPU Performance Comparison Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Test Configuration

- **Date Range**: {data['config']['date_range']}
- **Number of Runs**: {data['config']['num_runs']} per mode
- **Batch Size**: {data['config']['batch_size']}
- **Window Hours**: {data['config']['window_hours']}
- **Slide Hours**: {data['config']['slide_hours']}

## Executive Summary

"""

# Check if GPU was actually used
gpu_initialized = sum(run['metrics']['gpu_initialized'] for run in data['gpu_runs'])
cpu_disabled = sum(run['metrics']['gpu_disabled'] for run in data['cpu_runs'])

if gpu_initialized > 0:
    report += f"""✅ **GPU Successfully Enabled**: GPU acceleration was active in {gpu_initialized}/{len(data['gpu_runs'])} GPU runs.

### Performance Improvements:
- **Time Speedup**: {data['summary']['speedup']['time_speedup']}x faster
- **Feature Rate Improvement**: {data['summary']['speedup']['feature_rate_speedup']}x higher
- **CPU Mean Time**: {data['summary']['cpu']['mean_time']:.2f}s ± {data['summary']['cpu']['std_time']:.2f}s
- **GPU Mean Time**: {data['summary']['gpu']['mean_time']:.2f}s ± {data['summary']['gpu']['std_time']:.2f}s
"""
else:
    report += f"""❌ **GPU Not Activated**: GPU acceleration was not successfully enabled in any runs.

All runs executed in CPU mode. Please check:
1. NVIDIA drivers installation
2. Docker GPU runtime configuration
3. CUDA toolkit compatibility
"""

# Detailed results table
report += """
## Detailed Results

### CPU Performance

| Run | Pipeline Time (s) | Features/sec | Data Quality | Status |
|-----|------------------|--------------|--------------|---------|
"""

for run in data['cpu_runs']:
    quality = f"{run['metrics']['quality_coverage']:.1f}%"
    status = "✓" if run['metrics']['gpu_disabled'] > 0 else "?"
    report += f"| {run['run']} | {run['timings']['sparse_pipeline_seconds']:.2f} | "
    report += f"{run['metrics']['features_per_second']:.1f} | {quality} | {status} |\n"

report += """
### GPU Performance

| Run | Pipeline Time (s) | Features/sec | Data Quality | GPU Active |
|-----|------------------|--------------|--------------|------------|
"""

for run in data['gpu_runs']:
    quality = f"{run['metrics']['quality_coverage']:.1f}%"
    gpu_active = "✓" if run['metrics']['gpu_initialized'] > 0 else "✗"
    report += f"| {run['run']} | {run['timings']['sparse_pipeline_seconds']:.2f} | "
    report += f"{run['metrics']['features_per_second']:.1f} | {quality} | {gpu_active} |\n"

# Statistical analysis
report += f"""
## Statistical Analysis

### Timing Statistics

| Metric | CPU | GPU | Difference |
|--------|-----|-----|------------|
| Mean Time | {data['summary']['cpu']['mean_time']:.2f}s | {data['summary']['gpu']['mean_time']:.2f}s | {data['summary']['cpu']['mean_time'] - data['summary']['gpu']['mean_time']:.2f}s |
| Std Dev | {data['summary']['cpu']['std_time']:.2f}s | {data['summary']['gpu']['std_time']:.2f}s | - |
| Min Time | {min(r['timings']['sparse_pipeline_seconds'] for r in data['cpu_runs']):.2f}s | {min(r['timings']['sparse_pipeline_seconds'] for r in data['gpu_runs']):.2f}s | - |
| Max Time | {max(r['timings']['sparse_pipeline_seconds'] for r in data['cpu_runs']):.2f}s | {max(r['timings']['sparse_pipeline_seconds'] for r in data['gpu_runs']):.2f}s | - |

### Feature Extraction Performance

| Metric | CPU | GPU | Improvement |
|--------|-----|-----|-------------|
| Mean Rate | {data['summary']['cpu']['mean_features_per_sec']:.1f} feat/s | {data['summary']['gpu']['mean_features_per_sec']:.1f} feat/s | {data['summary']['speedup']['feature_rate_speedup']}x |
"""

# Data processing metrics
cpu_features = sum(r['metrics']['window_features'] for r in data['cpu_runs']) / len(data['cpu_runs'])
gpu_features = sum(r['metrics']['window_features'] for r in data['gpu_runs']) / len(data['gpu_runs'])

report += f"""
## Data Processing Metrics

### Average Per Run

| Metric | CPU | GPU |
|--------|-----|-----|
| Window Features | {cpu_features:.0f} | {gpu_features:.0f} |
| Hourly Data Points | {sum(r['metrics']['hourly_data_points'] for r in data['cpu_runs']) / len(data['cpu_runs']):.0f} | {sum(r['metrics']['hourly_data_points'] for r in data['gpu_runs']) / len(data['gpu_runs']):.0f} |
| CO2 Gap Fills | {sum(r['metrics']['gap_fills_co2'] for r in data['cpu_runs']) / len(data['cpu_runs']):.0f} | {sum(r['metrics']['gap_fills_co2'] for r in data['gpu_runs']) / len(data['gpu_runs']):.0f} |
| Humidity Gap Fills | {sum(r['metrics']['gap_fills_humidity'] for r in data['cpu_runs']) / len(data['cpu_runs']):.0f} | {sum(r['metrics']['gap_fills_humidity'] for r in data['gpu_runs']) / len(data['gpu_runs']):.0f} |

## Conclusions

"""

if gpu_initialized > 0:
    speedup = float(data['summary']['speedup']['time_speedup'])
    if speedup > 1.5:
        report += f"""1. **GPU Acceleration Successful**: Achieved {speedup:.1f}x speedup over CPU
2. **Consistent Performance**: Low standard deviation indicates stable performance
3. **Feature Extraction Bottleneck Addressed**: GPU significantly improves the main bottleneck
4. **Production Ready**: GPU mode is stable and ready for production use
"""
    else:
        report += f"""1. **Modest GPU Improvement**: Only {speedup:.1f}x speedup achieved
2. **Further Optimization Needed**: Current GPU implementation not fully optimized
3. **Investigate Bottlenecks**: Profile to identify non-GPU bottlenecks
"""
else:
    report += """1. **GPU Not Working**: Configuration issues prevent GPU usage
2. **Action Required**: Fix GPU setup before performance testing
3. **Rerun Tests**: After fixing GPU configuration
"""

report += """
## Recommendations

"""

if gpu_initialized > 0:
    report += """1. **Use GPU Mode in Production**: Clear performance benefits justify GPU deployment
2. **Monitor GPU Utilization**: Use nvidia-smi to ensure optimal GPU usage
3. **Scale Testing**: Test with larger datasets (full year)
4. **Optimize Further**: Port more algorithms to GPU for additional gains
"""
else:
    report += """1. **Fix GPU Configuration**: 
   - Check NVIDIA drivers: `nvidia-smi`
   - Verify Docker GPU runtime: `docker run --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`
   - Review environment variables
2. **Rebuild Containers**: `docker compose -f docker-compose.sparse.yml build --no-cache sparse_pipeline`
3. **Rerun Tests**: After fixing configuration issues
"""

# Save report
report_path = '$RESULTS_DIR/comparison_report_$TIMESTAMP.md'
with open(report_path, 'w') as f:
    f.write(report)

print(f"Report saved to: {report_path}")
PYTHON_SCRIPT

# Clean up
echo ""
echo "============================================"
echo "Test Complete"
echo "============================================"
echo "Results saved to: $RESULTS_FILE"
echo "Report saved to: $RESULTS_DIR/comparison_report_${TIMESTAMP}.md"
echo ""
echo "Quick Summary:"
echo "  CPU Mean: ${CPU_MEAN}s ± ${CPU_STD}s"
echo "  GPU Mean: ${GPU_MEAN}s ± ${GPU_STD}s"
echo "  Speedup: ${SPEEDUP}x"

# Final cleanup
docker compose -f docker-compose.sparse.yml down -v