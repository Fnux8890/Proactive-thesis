#!/usr/bin/env bash
set -e
# Navigate to this script's directory (DataIngestion) where docker-compose.yml lives
cd "$(dirname "$0")"

# Source the .env file to load variables into this script's environment
if [ -f .env ]; then
  echo "Loading environment variables from .env..."
  export $(grep -v '^#' .env | xargs)
fi

# Add an explicit check after sourcing .env
echo "DEBUG: INITIAL_RUN_DATE in host script is: [$INITIAL_RUN_DATE]"

# Define services needed
# Services that need to be *healthy* (have healthchecks defined in docker-compose.yml)
CORE_SERVICES_WITH_HEALTHCHECK="db redis orion"
# All services to start initially (including those without healthchecks)
ALL_SERVICES="db redis orion mlflow-server data-prep rust_pipeline pgadmin"

DEPLOYMENT_NAME="feature-etl-deployment"
FLOW_NAME="main-feature-flow"
FLOW_PATH="/app/src/flow_main.py:main_feature_flow" # Path inside the run container
WORK_POOL="default-process-pool"
YAML_OUTPUT_PATH="/app/feature-etl-deployment.yaml" # Path inside the run container

# 1) Start all required background services
echo "Starting background services ($ALL_SERVICES)..."
docker compose up -d $ALL_SERVICES

# 2) Explicitly wait for core services with healthchecks to be healthy
echo "Waiting for services with healthchecks ($CORE_SERVICES_WITH_HEALTHCHECK) to become healthy..."
# Use `docker compose ps` with filters to check health status in a loop
for service in $CORE_SERVICES_WITH_HEALTHCHECK; do
  echo "Waiting for $service..."
  while [[ "$(docker compose ps -q $service | xargs docker inspect -f '{{.State.Health.Status}}' 2>/dev/null || echo 'starting')" != "healthy" ]]; do
    if [[ "$(docker compose ps -q $service | xargs docker inspect -f '{{.State.Status}}' 2>/dev/null || echo 'starting')" == "exited" ]]; then
      echo "ERROR: Service $service exited unexpectedly!"
      docker compose logs $service
      exit 1
    fi
    sleep 5
  done
  echo "$service is healthy."
done
echo "Core services with healthchecks are healthy."

# 3) Deploy Prefect flow using the new v3 CLI (single command)
echo "Deploying Prefect flow using prefect-deployer (reads prefect.yaml)..."
# Use `run` which respects the env_file defined for prefect-deployer
docker compose run --rm prefect-deployer bash -c \
  "prefect --no-prompt deploy --name feature-etl-deployment" # Disable prompts and target the deployment

# 3.5) Trigger the Prefect Flow Run directly -> REPLACED with run_all_dates trigger
# echo "Triggering Prefect flow run for $FLOW_NAME/$DEPLOYMENT_NAME ..."
# if [ -n "$INITIAL_RUN_DATE" ]; then
#   echo "Using INITIAL_RUN_DATE=$INITIAL_RUN_DATE"
#   docker compose run --rm prefect-deployer prefect deployment run "$FLOW_NAME/$DEPLOYMENT_NAME" --param run_date_str="$INITIAL_RUN_DATE"
# else
#   echo "No INITIAL_RUN_DATE provided; running with defaults"
#   docker compose run --rm prefect-deployer prefect deployment run "$FLOW_NAME/$DEPLOYMENT_NAME"
# fi

# 4) Run the Python script to trigger flow runs for all dates
#    Run this using the prefect-deployer service to ensure access to:
#    - Prefect client library
#    - Database environment variables (via docker compose env_file or environment section)
#    - The Python script itself (assuming simulation_data_prep is mounted)
echo "Executing run_all_dates.py to trigger flow runs for the full date range..."
# Add commands to check versions inside the container
echo "Checking prefect and pydantic versions inside prefect-deployer container..."
docker compose run --rm prefect-deployer bash -c "pip show prefect pydantic"
echo "--- Starting run_all_dates.py --- "
docker compose run --rm prefect-deployer python /app/src/run_all_dates.py

echo "Orchestration script finished. Flow runs submitted. Monitor progress via Prefect UI or agent logs."
# You might want to add a 'docker compose down' here if you want to stop everything afterwards 