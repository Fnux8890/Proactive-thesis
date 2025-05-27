"""Configuration handling with Pydantic validation."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import ValidationError
from pydantic_settings import BaseSettings

from .models import PreprocessingConfig

logger = logging.getLogger(__name__)


class EnvironmentConfig(BaseSettings):
    """Environment configuration from environment variables."""

    # Database settings
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "greenhouse_db"
    db_user: str = "postgres"
    db_password: str = "postgres"

    # Processing settings
    batch_size: int = 10000
    log_level: str = "INFO"

    # Storage settings
    storage_type: str = "timescale_columnar"
    output_table: str = "preprocessed_greenhouse_data"

    class Config:
        """Pydantic settings configuration."""

        env_prefix = "PREPROCESSING_"
        case_sensitive = False


def load_and_validate_config(config_path: Path) -> PreprocessingConfig:
    """Load and validate configuration from JSON file.

    Args:
        config_path: Path to configuration JSON file

    Returns:
        Validated PreprocessingConfig object

    Raises:
        SystemExit: If configuration is invalid
    """
    try:
        # Load environment config
        env_config = EnvironmentConfig()

        # Load JSON config
        with open(config_path) as f:
            json_config = json.load(f)

        # Merge environment and JSON configs
        # Environment variables take precedence
        if "database" not in json_config:
            json_config["database"] = {}

        json_config["database"].update(
            {
                "host": env_config.db_host,
                "port": env_config.db_port,
                "database": env_config.db_name,
                "user": env_config.db_user,
                "password": env_config.db_password,
            }
        )

        # Apply other environment overrides
        json_config["batch_size"] = env_config.batch_size
        json_config["storage_type"] = env_config.storage_type
        json_config["output_table"] = env_config.output_table

        # Validate with Pydantic
        config = PreprocessingConfig(**json_config)

        logger.info("Configuration loaded and validated successfully")
        return config

    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}")
        raise SystemExit(1) from None

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file: {e}")
        raise SystemExit(1) from e

    except ValidationError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise SystemExit(1) from e


def get_database_url(config: PreprocessingConfig) -> str:
    """Get database connection URL from config.

    Args:
        config: Preprocessing configuration

    Returns:
        PostgreSQL connection URL
    """
    db = config.database
    return f"postgresql://{db['user']}:{db['password']}@{db['host']}:{db['port']}/{db['database']}"
