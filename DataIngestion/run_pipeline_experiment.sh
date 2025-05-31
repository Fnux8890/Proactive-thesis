#!/bin/bash
# Pipeline Performance Experiment
# Measures execution time and resource usage for sparse pipeline

echo "===== SPARSE PIPELINE PERFORMANCE EXPERIMENT ====="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo ""

# Configuration
EXPERIMENT_NAME="sparse_pipeline_baseline"
RESULTS_DIR="docs/experiments/results"
NUM_RUNS=${NUM_RUNS:-3}
DATE_RANGE_START="2014-01-01"
DATE_RANGE_END="2014-07-01"

# Create results directory
mkdir -p $RESULTS_DIR

# Function to measure execution time
measure_time() {
    local start_time=$(date +%s.%N)
    "$@"
    local end_time=$(date +%s.%N)
    echo "$(echo "$end_time - $start_time" | bc)"
}

# Function to get container stats
get_container_stats() {
    local container=$1
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" $container 2>/dev/null || echo "N/A"
}

# Clean environment
echo "Cleaning environment..."
docker compose -f docker-compose.sparse.yml down -v
rm -rf gpu_feature_extraction/checkpoints/*

# Start database
echo "Starting database..."
docker compose -f docker-compose.sparse.yml up -d db redis
sleep 10

# Results file
RESULTS_FILE="$RESULTS_DIR/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).json"

echo "{" > $RESULTS_FILE
echo "  \"experiment\": \"$EXPERIMENT_NAME\"," >> $RESULTS_FILE
echo "  \"date\": \"$(date -Iseconds)\"," >> $RESULTS_FILE
echo "  \"config\": {" >> $RESULTS_FILE
echo "    \"date_range\": \"$DATE_RANGE_START to $DATE_RANGE_END\"," >> $RESULTS_FILE
echo "    \"num_runs\": $NUM_RUNS," >> $RESULTS_FILE
echo "    \"disable_gpu\": \"${DISABLE_GPU:-false}\"" >> $RESULTS_FILE
echo "  }," >> $RESULTS_FILE
echo "  \"runs\": [" >> $RESULTS_FILE

# Run experiment multiple times
for run in $(seq 1 $NUM_RUNS); do
    echo -e "\n===== RUN $run/$NUM_RUNS ====="
    
    # Clear data between runs
    docker exec dataingestion-db-1 psql -U postgres -d postgres -c "TRUNCATE TABLE sensor_data CASCADE;" >/dev/null 2>&1
    
    # Stage 1: Rust Ingestion
    echo "Stage 1: Data Ingestion..."
    INGESTION_TIME=$(measure_time docker compose -f docker-compose.sparse.yml up rust_pipeline 2>&1 | tee /tmp/ingestion_$run.log | grep -E "(Pipeline completed successfully in|total_pipeline)" | tail -1 | grep -oP '\d+\.\d+s' | sed 's/s//')
    INGESTION_RECORDS=$(grep -oP "Records Inserted: \K\d+" /tmp/ingestion_$run.log || echo "0")
    
    # Stage 2-4: Sparse Pipeline
    echo "Stage 2-4: Sparse Pipeline..."
    SPARSE_START=$(date +%s.%N)
    docker run --rm \
        --network container:dataingestion-db-1 \
        -e DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres \
        -e RUST_LOG=gpu_feature_extraction=info \
        -e SPARSE_MODE=true \
        -e DISABLE_GPU=${DISABLE_GPU:-false} \
        -v $(pwd)/gpu_feature_extraction/checkpoints:/tmp/gpu_sparse_pipeline:rw \
        dataingestion-gpu-sparse-pipeline:latest \
        --sparse-mode \
        --start-date $DATE_RANGE_START \
        --end-date $DATE_RANGE_END \
        --batch-size 24 2>&1 | tee /tmp/sparse_$run.log
    SPARSE_END=$(date +%s.%N)
    SPARSE_TIME=$(echo "$SPARSE_END - $SPARSE_START" | bc)
    
    # Extract metrics from sparse pipeline log
    HOURLY_POINTS=$(grep -oP "Hourly data points: \K\d+" /tmp/sparse_$run.log || echo "0")
    WINDOW_FEATURES=$(grep -oP "Window features: \K\d+" /tmp/sparse_$run.log || echo "0")
    MONTHLY_ERAS=$(grep -oP "Monthly eras: \K\d+" /tmp/sparse_$run.log || echo "0")
    FEATURES_PER_SEC=$(grep -oP "Performance: \K[\d.]+" /tmp/sparse_$run.log || echo "0")
    
    # Get checkpoint file sizes
    CHECKPOINT_SIZE=$(du -sh gpu_feature_extraction/checkpoints 2>/dev/null | cut -f1 || echo "0")
    
    # Write run results
    if [ $run -gt 1 ]; then echo "," >> $RESULTS_FILE; fi
    echo "    {" >> $RESULTS_FILE
    echo "      \"run\": $run," >> $RESULTS_FILE
    echo "      \"timings\": {" >> $RESULTS_FILE
    echo "        \"ingestion_seconds\": ${INGESTION_TIME:-0}," >> $RESULTS_FILE
    echo "        \"sparse_pipeline_seconds\": $SPARSE_TIME," >> $RESULTS_FILE
    echo "        \"total_seconds\": $(echo "${INGESTION_TIME:-0} + $SPARSE_TIME" | bc)" >> $RESULTS_FILE
    echo "      }," >> $RESULTS_FILE
    echo "      \"metrics\": {" >> $RESULTS_FILE
    echo "        \"records_ingested\": $INGESTION_RECORDS," >> $RESULTS_FILE
    echo "        \"hourly_data_points\": $HOURLY_POINTS," >> $RESULTS_FILE
    echo "        \"window_features\": $WINDOW_FEATURES," >> $RESULTS_FILE
    echo "        \"monthly_eras\": $MONTHLY_ERAS," >> $RESULTS_FILE
    echo "        \"features_per_second\": $FEATURES_PER_SEC," >> $RESULTS_FILE
    echo "        \"checkpoint_size\": \"$CHECKPOINT_SIZE\"" >> $RESULTS_FILE
    echo "      }" >> $RESULTS_FILE
    echo -n "    }" >> $RESULTS_FILE
done

echo "" >> $RESULTS_FILE
echo "  ]" >> $RESULTS_FILE
echo "}" >> $RESULTS_FILE

# Calculate statistics
echo -e "\n\n===== EXPERIMENT SUMMARY ====="
python3 - << EOF
import json
import statistics

with open('$RESULTS_FILE', 'r') as f:
    data = json.load(f)

runs = data['runs']
ingestion_times = [r['timings']['ingestion_seconds'] for r in runs]
sparse_times = [r['timings']['sparse_pipeline_seconds'] for r in runs]
total_times = [r['timings']['total_seconds'] for r in runs]
features_per_sec = [r['metrics']['features_per_second'] for r in runs]

print(f"Number of runs: {len(runs)}")
print(f"\nIngestion Time:")
print(f"  Mean: {statistics.mean(ingestion_times):.2f}s")
print(f"  Std Dev: {statistics.stdev(ingestion_times) if len(ingestion_times) > 1 else 0:.2f}s")
print(f"\nSparse Pipeline Time:")
print(f"  Mean: {statistics.mean(sparse_times):.2f}s")
print(f"  Std Dev: {statistics.stdev(sparse_times) if len(sparse_times) > 1 else 0:.2f}s")
print(f"\nTotal Pipeline Time:")
print(f"  Mean: {statistics.mean(total_times):.2f}s")
print(f"  Std Dev: {statistics.stdev(total_times) if len(total_times) > 1 else 0:.2f}s")
print(f"\nFeature Extraction Rate:")
print(f"  Mean: {statistics.mean(features_per_sec):.1f} features/second")

# Save summary
summary = {
    "summary": {
        "ingestion_mean": statistics.mean(ingestion_times),
        "ingestion_std": statistics.stdev(ingestion_times) if len(ingestion_times) > 1 else 0,
        "sparse_mean": statistics.mean(sparse_times),
        "sparse_std": statistics.stdev(sparse_times) if len(sparse_times) > 1 else 0,
        "total_mean": statistics.mean(total_times),
        "total_std": statistics.stdev(total_times) if len(total_times) > 1 else 0,
        "features_per_sec_mean": statistics.mean(features_per_sec)
    }
}

# Update the results file with summary
with open('$RESULTS_FILE', 'r') as f:
    original = json.load(f)
original.update(summary)
with open('$RESULTS_FILE', 'w') as f:
    json.dump(original, f, indent=2)
EOF

echo -e "\nResults saved to: $RESULTS_FILE"
echo "Logs saved to: /tmp/ingestion_*.log and /tmp/sparse_*.log"

# Cleanup
docker compose -f docker-compose.sparse.yml down