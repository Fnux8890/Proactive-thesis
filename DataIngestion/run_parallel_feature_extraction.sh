#!/bin/bash
set -e

echo "============================================"
echo "Parallel Feature Extraction Pipeline"
echo "============================================"

# Check if running on Google Cloud or locally
if [ -z "$GOOGLE_CLOUD_PROJECT" ]; then
    echo "Running locally..."
    
    # First run the data ingestion pipeline up to era detection
    echo "Step 1: Running data ingestion and preprocessing..."
    docker compose up -d db
    sleep 10  # Wait for DB to be ready
    
    echo "Step 2: Running rust pipeline for data ingestion..."
    docker compose up rust_pipeline --exit-code-from rust_pipeline
    
    echo "Step 3: Running preprocessing..."
    docker compose up preprocessing --exit-code-from preprocessing
    
    echo "Step 4: Running era detection..."
    docker compose up era_detection --exit-code-from era_detection
    
    echo "Step 5: Starting parallel feature extraction..."
    # Use the parallel docker-compose file that extends the main one
    docker compose -f docker-compose.yml -f docker-compose.parallel-feature.yml up \
        redis pgbouncer parallel-coordinator \
        parallel-gpu-worker-0 parallel-gpu-worker-1 parallel-gpu-worker-2 parallel-gpu-worker-3 \
        parallel-cpu-worker-0 parallel-cpu-worker-1 parallel-cpu-worker-2 parallel-cpu-worker-3 \
        parallel-cpu-worker-4 parallel-cpu-worker-5 parallel-cpu-worker-6 parallel-cpu-worker-7 \
        --exit-code-from parallel-coordinator
    
else
    echo "Running on Google Cloud..."
    
    # On Google Cloud, we assume the full pipeline needs to run
    echo "Step 1: Building all images..."
    docker compose -f docker-compose.yml -f docker-compose.parallel-feature.yml build
    
    echo "Step 2: Starting database..."
    docker compose up -d db
    sleep 30  # Give more time on cloud
    
    echo "Step 3: Running full pipeline with parallel feature extraction..."
    # Run everything in sequence
    docker compose up rust_pipeline --exit-code-from rust_pipeline
    docker compose up preprocessing --exit-code-from preprocessing
    docker compose up era_detection --exit-code-from era_detection
    
    # Now run parallel feature extraction
    docker compose -f docker-compose.yml -f docker-compose.parallel-feature.yml up \
        redis pgbouncer parallel-coordinator \
        parallel-gpu-worker-0 parallel-gpu-worker-1 parallel-gpu-worker-2 parallel-gpu-worker-3 \
        parallel-cpu-worker-0 parallel-cpu-worker-1 parallel-cpu-worker-2 parallel-cpu-worker-3 \
        parallel-cpu-worker-4 parallel-cpu-worker-5 parallel-cpu-worker-6 parallel-cpu-worker-7 \
        --exit-code-from parallel-coordinator
fi

echo "============================================"
echo "Parallel Feature Extraction Complete!"
echo "============================================"

# Check results
echo "Checking feature extraction results..."
docker compose exec -T db psql -U postgres -d postgres -c "
SELECT 
    era_id,
    compartment_id,
    COUNT(DISTINCT feature_name) as num_features,
    MIN(created_at) as first_feature,
    MAX(created_at) as last_feature
FROM tsfresh_features
GROUP BY era_id, compartment_id
ORDER BY era_id, compartment_id
LIMIT 20;
"

echo "Total features extracted:"
docker compose exec -T db psql -U postgres -d postgres -c "
SELECT COUNT(*) as total_features FROM tsfresh_features;
"