# PowerShell script to run the complete pipeline with validation
# This ensures data flows correctly through each stage

$ErrorActionPreference = "Stop"

Write-Host "=== Starting Data Pipeline with Validation ===" -ForegroundColor Green
Write-Host "This will run each stage and validate data before proceeding" -ForegroundColor Green
Write-Host ""

# Function to wait for service to complete
function Wait-ForService {
    param(
        [string]$ServiceName,
        [int]$MaxWaitSeconds = 600
    )
    
    $elapsed = 0
    Write-Host "Waiting for $ServiceName to complete..." -NoNewline
    
    while ($elapsed -lt $MaxWaitSeconds) {
        # Check if service exited
        $status = docker compose ps --format json $ServiceName 2>$null | ConvertFrom-Json
        
        if ($status -and $status.State -eq "exited") {
            if ($status.ExitCode -eq 0) {
                Write-Host ""
                Write-Host "✅ $ServiceName completed successfully" -ForegroundColor Green
                return $true
            } else {
                Write-Host ""
                Write-Host "❌ $ServiceName failed with exit code $($status.ExitCode)" -ForegroundColor Red
                docker compose logs --tail=50 $ServiceName
                return $false
            }
        }
        
        Start-Sleep -Seconds 10
        $elapsed += 10
        Write-Host "." -NoNewline
    }
    
    Write-Host ""
    Write-Host "⚠️  Timeout waiting for $ServiceName" -ForegroundColor Yellow
    return $false
}

# Function to run validation
function Validate-Stage {
    Write-Host ""
    Write-Host "Running validation..." -ForegroundColor Yellow
    
    docker run --rm `
        --network greenhouse-pipeline `
        -e DB_HOST=db `
        -e DB_USER=postgres `
        -e DB_PASSWORD=postgres `
        -e DB_NAME=postgres `
        -v ${PWD}/scripts:/scripts `
        python:3.11-slim `
        bash -c "pip install pandas sqlalchemy psycopg2-binary && python /scripts/validate_pipeline_data.py"
}

# 0. Start database if not running
Write-Host "=== Starting Database ===" -ForegroundColor Cyan
docker compose up -d db
Start-Sleep -Seconds 10

# 1. Data Ingestion
Write-Host ""
Write-Host "=== Stage 1: Data Ingestion ===" -ForegroundColor Cyan
docker compose up -d rust_pipeline
if (-not (Wait-ForService -ServiceName "rust_pipeline" -MaxWaitSeconds 1200)) {
    Write-Host "Pipeline failed at Data Ingestion" -ForegroundColor Red
    exit 1
}
Validate-Stage

# 2. Preprocessing
Write-Host ""
Write-Host "=== Stage 2: Preprocessing ===" -ForegroundColor Cyan
docker compose up -d preprocessing
if (-not (Wait-ForService -ServiceName "preprocessing" -MaxWaitSeconds 1800)) {
    Write-Host "Pipeline failed at Preprocessing" -ForegroundColor Red
    exit 1
}
Validate-Stage

# 3. Era Detection
Write-Host ""
Write-Host "=== Stage 3: Era Detection ===" -ForegroundColor Cyan
docker compose up -d era_detector
if (-not (Wait-ForService -ServiceName "era_detector" -MaxWaitSeconds 600)) {
    Write-Host "Pipeline failed at Era Detection" -ForegroundColor Red
    exit 1
}
Validate-Stage

# 4. Feature Extraction
Write-Host ""
Write-Host "=== Stage 4: Feature Extraction ===" -ForegroundColor Cyan
$env:BATCH_SIZE = "200"
$env:N_JOBS = "-1"
docker compose up -d feature_extraction
if (-not (Wait-ForService -ServiceName "feature_extraction" -MaxWaitSeconds 3600)) {
    Write-Host "Pipeline failed at Feature Extraction" -ForegroundColor Red
    exit 1
}
Validate-Stage

# 5. Create Synthetic Targets
Write-Host ""
Write-Host "=== Stage 5: Creating Target Variables ===" -ForegroundColor Cyan
docker compose run --rm model_builder python -m src.utils.create_synthetic_targets

# Final validation
Write-Host ""
Write-Host "=== Final Validation ===" -ForegroundColor Cyan
Validate-Stage

Write-Host ""
Write-Host "=== Pipeline Complete! ===" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Train models: docker compose up model_builder"
Write-Host "2. Run MOEA optimization: docker compose up moea_optimizer_gpu"
Write-Host ""
Write-Host "To deploy to cloud:" -ForegroundColor Yellow
Write-Host "1. cd terraform/parallel-feature"
Write-Host "2. terraform apply"