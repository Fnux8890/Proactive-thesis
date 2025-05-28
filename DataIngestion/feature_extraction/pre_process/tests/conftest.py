"""Pytest configuration and fixtures."""

from __future__ import annotations

import os
import sys
from collections.abc import Generator
from pathlib import Path

import pandas as pd
import pytest
from hypothesis import settings
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return path to test data directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def sample_greenhouse_df() -> pd.DataFrame:
    """Create a sample greenhouse DataFrame for testing."""
    dates = pd.date_range("2023-01-01", periods=100, freq="5min")

    return pd.DataFrame(
        {
            "timestamp": dates,
            "air_temp_c": 20 + 5 * pd.Series(range(100)).apply(lambda x: pd.np.sin(x / 10)),
            "relative_humidity_percent": 60
            + 10 * pd.Series(range(100)).apply(lambda x: pd.np.cos(x / 8)),
            "co2_measured_ppm": 400 + 50 * pd.Series(range(100)).apply(lambda x: pd.np.sin(x / 15)),
            "light_intensity_umol": 100
            + 200 * pd.Series(range(100)).apply(lambda x: max(0, pd.np.sin(x / 12))),
            "radiation_w_m2": 50
            + 100 * pd.Series(range(100)).apply(lambda x: max(0, pd.np.cos(x / 20))),
        }
    )


@pytest.fixture
def sample_config() -> dict:
    """Create a sample configuration dictionary."""
    return {
        "common_settings": {
            "time_col": "timestamp",
            "id_col": "entity_id",
        },
        "preprocessing": {
            "outlier_rules": [
                {"column": "air_temp_c", "min_value": -10, "max_value": 50, "clip": True},
                {
                    "column": "relative_humidity_percent",
                    "min_value": 0,
                    "max_value": 100,
                    "clip": True,
                },
                {"column": "co2_measured_ppm", "min_value": 0, "max_value": 5000, "clip": True},
            ],
            "imputation_rules": [
                {"column": "air_temp_c", "strategy": "linear"},
                {"column": "relative_humidity_percent", "strategy": "forward_fill", "limit": 3},
                {"column": "co2_measured_ppm", "strategy": "mean"},
            ],
        },
        "segmentation": {
            "min_gap_hours": 24,
        },
    }


@pytest.fixture
def sqlite_engine() -> Generator[Engine, None, None]:
    """Create an in-memory SQLite engine for testing."""
    engine = create_engine("sqlite:///:memory:")
    yield engine
    engine.dispose()


@pytest.fixture(autouse=True)
def setup_logging(caplog):
    """Set up logging for tests."""
    caplog.set_level("INFO")


# Hypothesis settings


settings.register_profile("dev", max_examples=10)
settings.register_profile("ci", max_examples=100)

# Use dev profile by default, CI can override with HYPOTHESIS_PROFILE env var
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))
