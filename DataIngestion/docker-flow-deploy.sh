#!/usr/bin/env bash
set -euo pipefail

# Default Prefect API URL if not provided
: "${PREFECT_API_URL:=http://host.docker.internal:4200/api}"
# Default run date parameter
: "${RUN_DATE:=auto}"

# Directory containing your flow code
FLOW_DIR="$(pwd)/simulation_data_prep/src"
# Deployment settings
DEPLOYMENT_NAME="feature-etl-deployment"
POOL="default-process-pool"
QUEUE="default"

echo "Deploying flow 'main_feature_flow' to Prefect at $PREFECT_API_URL..."
docker run --rm -it \
  -e PREFECT_API_URL="$PREFECT_API_URL" \
  -v "$FLOW_DIR":/flows \
  prefecthq/prefect:2-latest \
  prefect deploy /flows/flow.py:main_feature_flow \
    --name "$DEPLOYMENT_NAME" \
    --pool "$POOL" -q "$QUEUE" \
    --param "run_date_str=$RUN_DATE"

echo "Deployment completed successfully."