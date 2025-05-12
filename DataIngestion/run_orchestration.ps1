# --- Script Parameters ---
param(
    [Parameter(Mandatory=$false)] # Make the parameter optional
    [string]$SingleDate # Expects date string like YYYY-MM-DD
)

# Set error preference to stop execution on non-terminating errors, similar to 'set -e'
$ErrorActionPreference = 'Stop'

# Get the directory where the script is located
$ScriptRoot = $PSScriptRoot

# Navigate to the script's directory
Set-Location $ScriptRoot
Write-Host "Changed location to: $(Get-Location)"
if ($SingleDate) {
    Write-Host "Single date specified: $SingleDate"
} else {
    Write-Host "No single date specified, running for full range."
}

# --- Load Environment Variables from .env file ---
$EnvFilePath = Join-Path $ScriptRoot ".env"
if (Test-Path $EnvFilePath) {
    Write-Host "Loading environment variables from .env..."
    try {
        Get-Content $EnvFilePath | ForEach-Object {
            $line = $_.Trim()
            # Skip comments and empty lines
            if ($line -and $line -notmatch '^\s*#') {
                # Split only on the first '='
                $parts = $line.Split('=', 2)
                if ($parts.Length -eq 2) {
                    $key = $parts[0].Trim()
                    $value = $parts[1].Trim()
                    # Set environment variable for the current process
                    [System.Environment]::SetEnvironmentVariable($key, $value, 'Process')
                    # Write-Host "Loaded: $key=***" # Uncomment for debugging, hides value
                } else {
                    Write-Warning "Skipping malformed line in .env: $line"
                }
            }
        }
    } catch {
        Write-Error "Failed to load environment variables from .env: $($_.Exception.Message)"
        exit 1 # Exit if loading fails
    }
} else {
    Write-Host ".env file not found. Skipping environment variable loading."
}

# Add an explicit check after attempting to load .env
Write-Host "DEBUG: INITIAL_RUN_DATE in host script is: [$($env:INITIAL_RUN_DATE)]"

# --- Define Services ---
# Services that need to be *healthy* (have healthchecks defined in docker-compose.yml)
$CORE_SERVICES_WITH_HEALTHCHECK = @("db", "orion")
# All services to start initially (including those without healthchecks)
$ALL_SERVICES = @("db", "orion", "mlflow-server", "data-prep", "rust_pipeline", "pgadmin")

# --- Prefect Deployment Details (Not used directly in script, for info) ---
# $DEPLOYMENT_NAME = "feature-etl-deployment" # Name defined in prefect.yaml
# $FLOW_NAME = "main-feature-flow" # Flow name defined in Python script
# $FLOW_PATH = "/app/src/flow_main.py:main_feature_flow" # Path inside the run container
# $WORK_POOL = "default-process-pool" # Work pool name - IMPORTANT: Update in prefect.yaml if needed

# --- Build & Start Services ---
Write-Host "Building necessary images (forcing data-prep rebuild for diagnosis)..."
try {
    # Re-enabled --no-cache for data-prep to ensure clean pip install for diagnosis
    docker compose build --no-cache data-prep
    # Build any other services if needed (usually relies on cache)
    docker compose build
} catch {
    Write-Error "Docker build failed: $($_.Exception.Message)"
    exit 1
}

Write-Host "Starting background services (db, redis, orion, mlflow-server, data-prep, rust_pipeline, pgadmin)..."
try {
    # Use --force-recreate to ensure containers use the newly built images
    # Pass the specific services defined in $ALL_SERVICES to the up command
    docker compose up --force-recreate --wait --detach $ALL_SERVICES 
    # Check health after wait (optional but good practice)
    $UnhealthyServices = docker compose ps --filter "status=running" --filter "health=unhealthy" --format "{{.Name}}"
    if ($UnhealthyServices) {
        Write-Warning "The following services are running but unhealthy: $($UnhealthyServices -join ', ')"
        # Decide whether to exit or continue
        # exit 1
    }
    Write-Host "Core services with healthchecks appear healthy."
} catch {
    Write-Error "Docker compose up failed: $($_.Exception.Message)"
    exit 1
}

# --- Prefect Deployment (One-shot Service) ---
# Add a small delay to allow Orion to fully initialize after healthcheck passes
Write-Host "Waiting 5 seconds for Orion service to settle..."
Start-Sleep -Seconds 5

Write-Host "Deploying Prefect flow using prefect-deployer (reads prefect.yaml)..."
# The command inside bash -c runs within the Linux container, so it remains unchanged
# Use `run` which respects the env_file defined for prefect-deployer
docker compose run --rm prefect-deployer bash -c "prefect --no-prompt deploy --name feature-etl-deployment"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Prefect deployment failed."
    exit 1
}

# 4) Run the Python script to trigger flow runs
# Construct the base command
$pythonCommandBase = "python /app/src/run_all_dates.py"
# Add the date argument if provided
$pythonCommand = $pythonCommandBase
if ($SingleDate) {
    $pythonCommand = "$pythonCommandBase --date $SingleDate"
    Write-Host "Executing run_all_dates.py to trigger flow run for single date: $SingleDate..."
} else {
    Write-Host "Executing run_all_dates.py to trigger flow runs for the full date range..."
}

# Add commands to check versions inside the container
Write-Host "Checking prefect and pydantic versions inside prefect-deployer container..."
docker compose run --rm prefect-deployer bash -c "pip show prefect pydantic"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Could not check pip versions inside container."
}
Write-Host "--- Starting run_all_dates.py --- "
Write-Host "Executing command inside container: $pythonCommand"
docker compose run --rm prefect-deployer bash -c $pythonCommand # Execute the constructed command
if ($LASTEXITCODE -ne 0) {
    Write-Error "Execution of run_all_dates.py failed."
    exit 1
}

# --- NEW: Trigger single flow run for INITIAL_RUN_DATE --- #
# Write-Host "Triggering single flow run for date: [$($env:INITIAL_RUN_DATE)]..."
# if (-not $env:INITIAL_RUN_DATE) {
#     Write-Error "INITIAL_RUN_DATE environment variable is not set. Cannot trigger single run."
#     exit 1
# }
#
# $FlowName = "main-feature-flow"
# $DeploymentName = "feature-etl-deployment"
# $ParamName = "run_date_str"
# $ParamValue = $env:INITIAL_RUN_DATE
#
# # Construct the correct identifier and command
# $DeploymentIdentifier = "'${FlowName}/${DeploymentName}'"
# $PrefectCommand = "prefect deployment run ${DeploymentIdentifier} --param ${ParamName}=${ParamValue}"
#
# # Execute the command using docker compose run
# Write-Host "Executing: docker compose run --rm prefect-deployer bash -c "$PrefectCommand""
# docker compose run --rm prefect-deployer bash -c $PrefectCommand
# if ($LASTEXITCODE -ne 0) {
#     Write-Error "Triggering single Prefect flow run failed."
#     exit 1
# }

Write-Host "Orchestration script finished. Flow runs submitted. Monitor progress via Prefect UI or agent logs." # Updated final message


