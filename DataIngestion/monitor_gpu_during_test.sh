#!/bin/bash
# GPU Monitoring Script
# Tracks GPU utilization during sparse pipeline execution

set -e

echo "============================================"
echo "GPU Monitoring During Test Execution"
echo "Date: $(date)"
echo "============================================"

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. NVIDIA drivers may not be installed."
    exit 1
fi

# Configuration
MONITOR_INTERVAL=1  # seconds
OUTPUT_DIR="./docs/experiments/results/gpu_monitoring"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUTPUT_DIR"

# Output files
MONITOR_LOG="$OUTPUT_DIR/gpu_monitor_${TIMESTAMP}.log"
UTILIZATION_CSV="$OUTPUT_DIR/gpu_utilization_${TIMESTAMP}.csv"
SUMMARY_FILE="$OUTPUT_DIR/gpu_summary_${TIMESTAMP}.txt"

# Function to monitor GPU
monitor_gpu() {
    echo "timestamp,gpu_util_percent,memory_used_mb,memory_total_mb,temp_c,power_w" > "$UTILIZATION_CSV"
    
    while true; do
        TIMESTAMP=$(date +%s.%N)
        
        # Get GPU stats
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
                   --format=csv,noheader,nounits | while IFS=',' read -r gpu_util mem_used mem_total temp power; do
            
            # Trim whitespace
            gpu_util=$(echo $gpu_util | xargs)
            mem_used=$(echo $mem_used | xargs)
            mem_total=$(echo $mem_total | xargs)
            temp=$(echo $temp | xargs)
            power=$(echo $power | xargs)
            
            # Write to CSV
            echo "$TIMESTAMP,$gpu_util,$mem_used,$mem_total,$temp,$power" >> "$UTILIZATION_CSV"
            
            # Display current stats
            printf "\r[%s] GPU: %3s%% | Mem: %s/%s MB | Temp: %s°C | Power: %sW" \
                   "$(date +%H:%M:%S)" "$gpu_util" "$mem_used" "$mem_total" "$temp" "$power"
        done
        
        sleep $MONITOR_INTERVAL
    done
}

# Function to run test with monitoring
run_monitored_test() {
    local test_name=$1
    local disable_gpu=$2
    
    echo ""
    echo "Starting $test_name test with GPU monitoring..."
    
    # Start GPU monitoring in background
    monitor_gpu &
    MONITOR_PID=$!
    
    # Give monitor time to start
    sleep 2
    
    # Run the actual test
    echo ""
    echo "Executing sparse pipeline..."
    
    # Clean checkpoints
    rm -rf ./gpu_feature_extraction/checkpoints/*
    
    # Ensure database is running
    docker compose -f docker-compose.sparse.yml up -d db
    sleep 10
    
    # Run ingestion if needed
    docker compose -f docker-compose.sparse.yml up rust_pipeline > /dev/null 2>&1
    
    # Run sparse pipeline
    START_TIME=$(date +%s)
    docker compose -f docker-compose.sparse.yml run --rm \
        -e DISABLE_GPU=$disable_gpu \
        -e CUDA_VISIBLE_DEVICES=0 \
        sparse_pipeline \
        --sparse-mode \
        --start-date "2014-05-01" \
        --end-date "2014-05-31" \
        --batch-size 24 2>&1 | tee "$MONITOR_LOG"
    END_TIME=$(date +%s)
    
    # Stop monitoring
    kill $MONITOR_PID 2>/dev/null || true
    wait $MONITOR_PID 2>/dev/null || true
    
    # Calculate duration
    DURATION=$((END_TIME - START_TIME))
    
    echo ""
    echo "Test completed in ${DURATION} seconds"
    
    # Analyze results
    analyze_gpu_usage "$test_name" "$DURATION"
}

# Function to analyze GPU usage
analyze_gpu_usage() {
    local test_name=$1
    local duration=$2
    
    echo ""
    echo "Analyzing GPU usage for $test_name..."
    
    # Calculate statistics from CSV
    python3 << PYTHON_EOF
import csv
import statistics

# Read GPU utilization data
gpu_utils = []
mem_used = []
temps = []
powers = []

with open('$UTILIZATION_CSV', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            gpu_utils.append(float(row['gpu_util_percent']))
            mem_used.append(float(row['memory_used_mb']))
            temps.append(float(row['temp_c']))
            if row['power_w'] and row['power_w'] != '[N/A]':
                powers.append(float(row['power_w']))
        except (ValueError, KeyError):
            continue

# Calculate statistics
if gpu_utils:
    avg_gpu = statistics.mean(gpu_utils)
    max_gpu = max(gpu_utils)
    std_gpu = statistics.stdev(gpu_utils) if len(gpu_utils) > 1 else 0
    
    avg_mem = statistics.mean(mem_used)
    max_mem = max(mem_used)
    
    avg_temp = statistics.mean(temps)
    max_temp = max(temps)
    
    avg_power = statistics.mean(powers) if powers else 0
    max_power = max(powers) if powers else 0
    
    # Determine if GPU was actually used
    gpu_active = max_gpu > 5  # Consider GPU active if utilization > 5%
    
    # Write summary
    with open('$SUMMARY_FILE', 'w') as f:
        f.write(f"GPU Monitoring Summary - {test_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test Duration: {duration} seconds\n")
        f.write(f"Data Points: {len(gpu_utils)}\n")
        f.write(f"GPU Active: {'Yes' if gpu_active else 'No'}\n\n")
        
        f.write("GPU Utilization:\n")
        f.write(f"  Average: {avg_gpu:.1f}%\n")
        f.write(f"  Maximum: {max_gpu:.1f}%\n")
        f.write(f"  Std Dev: {std_gpu:.1f}%\n\n")
        
        f.write("Memory Usage:\n")
        f.write(f"  Average: {avg_mem:.0f} MB\n")
        f.write(f"  Maximum: {max_mem:.0f} MB\n\n")
        
        f.write("Temperature:\n")
        f.write(f"  Average: {avg_temp:.1f}°C\n")
        f.write(f"  Maximum: {max_temp:.1f}°C\n\n")
        
        if powers:
            f.write("Power Consumption:\n")
            f.write(f"  Average: {avg_power:.1f}W\n")
            f.write(f"  Maximum: {max_power:.1f}W\n")
    
    # Print summary
    print(f"\nGPU Usage Summary:")
    print(f"  GPU Active: {'Yes' if gpu_active else 'No'}")
    print(f"  Average Utilization: {avg_gpu:.1f}%")
    print(f"  Maximum Utilization: {max_gpu:.1f}%")
    print(f"  Average Memory: {avg_mem:.0f} MB")
    print(f"  Maximum Memory: {max_mem:.0f} MB")
    
    # Create utilization plot
    if gpu_active:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Read timestamps
        timestamps = []
        with open('$UTILIZATION_CSV', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                timestamps.append(float(row['timestamp']))
        
        # Convert to relative time
        start_time = timestamps[0]
        rel_times = [(t - start_time) for t in timestamps]
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # GPU utilization plot
        ax1.plot(rel_times, gpu_utils, 'b-', linewidth=1)
        ax1.fill_between(rel_times, gpu_utils, alpha=0.3)
        ax1.set_ylabel('GPU Utilization (%)')
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'GPU Usage During {test_name} Test')
        
        # Memory usage plot
        ax2.plot(rel_times, mem_used, 'r-', linewidth=1)
        ax2.fill_between(rel_times, mem_used, alpha=0.3)
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_xlabel('Time (seconds)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('$OUTPUT_DIR/gpu_usage_plot_$TIMESTAMP.png', dpi=150)
        print(f"\n  Plot saved: $OUTPUT_DIR/gpu_usage_plot_$TIMESTAMP.png")
else:
    print("\nNo GPU utilization data collected.")
    with open('$SUMMARY_FILE', 'w') as f:
        f.write("No GPU utilization data collected.\n")
        f.write("GPU may not be active or monitoring failed.\n")

PYTHON_EOF
}

# Main menu
echo ""
echo "Select test to monitor:"
echo "1) Quick GPU test (1 month)"
echo "2) CPU baseline (1 month)"
echo "3) GPU performance (1 month)"
echo "4) Full comparison (6 months)"
echo "5) Custom test"

read -p "Enter choice (1-5): " choice

case $choice in
    1)
        echo ""
        echo "Running quick GPU test with monitoring..."
        run_monitored_test "Quick_GPU" "false"
        ;;
    2)
        echo ""
        echo "Running CPU baseline with monitoring..."
        run_monitored_test "CPU_Baseline" "true"
        ;;
    3)
        echo ""
        echo "Running GPU performance test with monitoring..."
        run_monitored_test "GPU_Performance" "false"
        ;;
    4)
        echo ""
        echo "This will run a full 6-month test. Continue? (y/n)"
        read -p "> " confirm
        if [ "$confirm" = "y" ]; then
            # Modify dates for full test
            sed -i 's/2014-05-31/2014-07-01/g' "$0"
            run_monitored_test "Full_GPU" "false"
        fi
        ;;
    5)
        read -p "Enter start date (YYYY-MM-DD): " start_date
        read -p "Enter end date (YYYY-MM-DD): " end_date
        read -p "Disable GPU? (true/false): " disable_gpu
        
        # Run custom test
        echo ""
        echo "Starting custom test..."
        monitor_gpu &
        MONITOR_PID=$!
        
        docker compose -f docker-compose.sparse.yml run --rm \
            -e DISABLE_GPU=$disable_gpu \
            sparse_pipeline \
            --sparse-mode \
            --start-date "$start_date" \
            --end-date "$end_date" \
            --batch-size 24 2>&1 | tee "$MONITOR_LOG"
        
        kill $MONITOR_PID 2>/dev/null || true
        analyze_gpu_usage "Custom" "0"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

# Cleanup
docker compose -f docker-compose.sparse.yml down

echo ""
echo "============================================"
echo "Monitoring Complete"
echo "============================================"
echo "Results saved to:"
echo "  CSV: $UTILIZATION_CSV"
echo "  Summary: $SUMMARY_FILE"
echo "  Log: $MONITOR_LOG"
echo ""

# Display final summary
cat "$SUMMARY_FILE"