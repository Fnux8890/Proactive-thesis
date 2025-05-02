# Set error preference to stop execution on non-terminating errors, similar to 'set -e'
$ErrorActionPreference = 'Stop'

# Get the directory where the script is located
$ScriptRoot = $PSScriptRoot

# Navigate to the script's directory
Set-Location $ScriptRoot
Write-Host "Changed location to: $(Get-Location)"

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
$CORE_SERVICES_WITH_HEALTHCHECK = @("db", "redis", "orion")
# All services to start initially (including those without healthchecks)
$ALL_SERVICES = @("db", "redis", "orion", "mlflow-server", "data-prep", "rust_pipeline", "pgadmin")

# --- Prefect Deployment Details (Not used directly in script, for info) ---
# $DEPLOYMENT_NAME = "feature-etl-deployment" # Name defined in prefect.yaml
# $FLOW_NAME = "main-feature-flow" # Flow name defined in Python script
# $FLOW_PATH = "/app/src/flow_main.py:main_feature_flow" # Path inside the run container
# $WORK_POOL = "default-process-pool" # Work pool name - IMPORTANT: Update in prefect.yaml if needed

# 1) Start all required background services
Write-Host "Starting background services ($($ALL_SERVICES -join ', '))..."
# Pass array elements as separate arguments to docker compose
docker compose up -d $ALL_SERVICES
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to start Docker Compose services."
    exit 1
}

# 2) Explicitly wait for core services with healthchecks to be healthy
Write-Host "Waiting for services with healthchecks ($($CORE_SERVICES_WITH_HEALTHCHECK -join ', ')) to become healthy..."

foreach ($service in $CORE_SERVICES_WITH_HEALTHCHECK) {
    Write-Host "Waiting for $service..."
    $isHealthy = $false
    while (-not $isHealthy) {
        # Get container ID quietly
        $containerId = docker compose ps -q $service 2>$null

        if (-not $containerId) {
            Write-Warning "Container for service '$service' not found yet or already stopped. Waiting..."
            Start-Sleep -Seconds 5
            continue # Skip inspect if no ID
        }

        # Check status first
        $stateStatus = (docker inspect -f '{{.State.Status}}' $containerId 2>$null).Trim()
        if ($LASTEXITCODE -ne 0) {
             Write-Warning "Could not inspect state for $service (Container ID: $containerId). Might be starting..."
             Start-Sleep -Seconds 5
             continue
        }

        if ($stateStatus -eq 'exited') {
            Write-Error "ERROR: Service $service (Container ID: $containerId) exited unexpectedly!"
            # Attempt to get logs
            docker compose logs $service
            exit 1
        }

        # Check health status
        $healthStatus = (docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}no-healthcheck{{end}}' $containerId 2>$null).Trim()
         if ($LASTEXITCODE -ne 0) {
             Write-Warning "Could not inspect health for $service (Container ID: $containerId). State: $stateStatus"
             # Don't continue here if state is 'running', maybe healthcheck is pending
         }

        if ($healthStatus -eq 'healthy') {
            $isHealthy = $true
        } else {
            # Write-Host "$service status: $stateStatus, health: $healthStatus. Waiting..." # Verbose
            Start-Sleep -Seconds 5
        }
    }
    Write-Host "$service is healthy."
}
Write-Host "Core services with healthchecks are healthy."

# 3) Deploy Prefect flow using the prefect-deployer service
Write-Host "Deploying Prefect flow using prefect-deployer (reads prefect.yaml)..."
# The command inside bash -c runs within the Linux container, so it remains unchanged
# Use `run` which respects the env_file defined for prefect-deployer
docker compose run --rm prefect-deployer bash -c "prefect --no-prompt deploy --name feature-etl-deployment"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Prefect deployment failed."
    exit 1
}

# 4) Run the Python script to trigger flow runs for all dates
Write-Host "Executing run_all_dates.py to trigger flow runs for the full date range..."
# Add commands to check versions inside the container
Write-Host "Checking prefect and pydantic versions inside prefect-deployer container..."
docker compose run --rm prefect-deployer bash -c "pip show prefect pydantic"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Could not check pip versions inside container."
}
Write-Host "--- Starting run_all_dates.py --- "
# The command inside python runs within the Linux container
docker compose run --rm prefect-deployer python /app/src/run_all_dates.py
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

Write-Host "Orchestration script finished. All flow runs submitted. Monitor progress via Prefect UI or agent logs." # Updated final message

# Optional: Stop services
# Write-Host "Stopping Docker Compose services..."
# docker compose down
