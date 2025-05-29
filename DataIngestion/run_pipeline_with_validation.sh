#!/bin/bash
# Run the complete pipeline with validation between stages
# This ensures data flows correctly through each stage

set -e  # Exit on any error

echo "=== Starting Data Pipeline with Validation ==="
echo "This will run each stage and validate data before proceeding"
echo ""

# Function to wait for service to complete
wait_for_service() {
    local service=$1
    local max_wait=${2:-600}  # Default 10 minutes
    local elapsed=0
    
    echo "Waiting for $service to complete..."
    
    while [ $elapsed -lt $max_wait ]; do
        # Check if service exited
        status=$(docker compose ps --format json $service 2>/dev/null | jq -r '.[0].State' 2>/dev/null || echo "unknown")
        
        if [ "$status" = "exited" ]; then
            # Check exit code
            exit_code=$(docker compose ps --format json $service 2>/dev/null | jq -r '.[0].ExitCode' 2>/dev/null || echo "1")
            if [ "$exit_code" = "0" ]; then
                echo "✅ $service completed successfully"
                return 0
            else
                echo "❌ $service failed with exit code $exit_code"
                docker compose logs --tail=50 $service
                return 1
            fi
        fi
        
        sleep 10
        elapsed=$((elapsed + 10))
        echo -n "."
    done
    
    echo ""
    echo "⚠️  Timeout waiting for $service"
    return 1
}

# Function to run validation
validate_stage() {
    echo ""
    echo "Running validation..."
    docker run --rm \
        --network greenhouse-pipeline \
        -e DB_HOST=db \
        -e DB_USER=postgres \
        -e DB_PASSWORD=postgres \
        -e DB_NAME=postgres \
        -v $(pwd)/scripts:/scripts \
        python:3.11-slim \
        bash -c "pip install pandas sqlalchemy psycopg2-binary && python /scripts/validate_pipeline_data.py"
}

# 0. Start database if not running
echo "=== Starting Database ==="
docker compose up -d db
sleep 10  # Wait for DB to be ready

# 1. Data Ingestion
echo ""
echo "=== Stage 1: Data Ingestion ==="
docker compose up -d rust_pipeline
wait_for_service rust_pipeline 1200  # 20 minutes max

# Validate
validate_stage

# 2. Preprocessing
echo ""
echo "=== Stage 2: Preprocessing ==="
docker compose up -d preprocessing
wait_for_service preprocessing 1800  # 30 minutes max

# Validate
validate_stage

# 3. Era Detection
echo ""
echo "=== Stage 3: Era Detection ==="
docker compose up -d era_detector
wait_for_service era_detector 600  # 10 minutes max

# Validate
validate_stage

# 4. Feature Extraction
echo ""
echo "=== Stage 4: Feature Extraction ==="
# Set parallel processing parameters
export BATCH_SIZE=200
export N_JOBS=-1
docker compose up -d feature_extraction
wait_for_service feature_extraction 3600  # 60 minutes max

# Validate
validate_stage

# 5. Create Synthetic Targets (if needed)
echo ""
echo "=== Stage 5: Creating Target Variables ==="
docker compose run --rm model_builder python -m src.utils.create_synthetic_targets

# Final validation
echo ""
echo "=== Final Validation ==="
validate_stage

echo ""
echo "=== Pipeline Complete! ==="
echo "Next steps:"
echo "1. Train models: docker compose up model_builder"
echo "2. Run MOEA optimization: docker compose up moea_optimizer_gpu"
echo ""
echo "To deploy to cloud:"
echo "1. cd terraform/parallel-feature"
echo "2. terraform apply"