#!/usr/bin/env bash
# one-shot test runner (from DataIngestion/ directory)
set -e
echo "Running simulation_data_prep unit tests in fresh container…"
docker compose run --rm test
