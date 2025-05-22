"""Configuration helpers for feature extraction."""

from __future__ import annotations

import os
from pathlib import Path


def env(key: str, default: str) -> str:
    """Return the environment variable value or a default."""
    return os.getenv(key, default)


DB_USER = env("DB_USER", "postgres")
DB_PASSWORD = env("DB_PASSWORD", "postgres")
DB_HOST = env("DB_HOST", "db")
DB_PORT = env("DB_PORT", "5432")
DB_NAME = env("DB_NAME", "postgres")

# Table name for database storage of selected features
FEATURES_TABLE = env("FEATURES_TABLE", "tsfresh_selected_features")

# GPU flag used throughout the pipeline
USE_GPU_FLAG = env("USE_GPU", "false").lower() == "true"

# Paths used by the extraction script (can be overridden via env)
CONSOLIDATED_DATA_FILE_PATH = Path(
    env("CONSOLIDATED_DATA_FILE_PATH", "consolidated.json")
)
ERA_DEFINITIONS_DIR_PATH = Path(
    env("ERA_DEFINITIONS_DIR_PATH", "era_definitions")
)
OUTPUT_PATH = Path(env("OUTPUT_PATH", "output/features.parquet"))
SELECTED_OUTPUT_PATH = Path(
    env("SELECTED_OUTPUT_PATH", "output/features_selected.parquet")
)

# Supervised feature selection settings
USE_SUPERVISED_SELECTION = env("USE_SUPERVISED_SELECTION", "false").lower() == "true"
FDR_LEVEL = float(env("FDR_LEVEL", "0.05"))
N_JOBS = int(env("N_JOBS", "0"))
CHUNKSIZE_ENV = env("CHUNKSIZE", "")
CHUNKSIZE = int(CHUNKSIZE_ENV) if CHUNKSIZE_ENV else None

# Name of the target column for supervised selection
TARGET_COLUMN = env("TARGET_COLUMN", "energy_kWh_next_hour")
