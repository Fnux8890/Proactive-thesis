"""Enhanced Feature Extraction Pipeline with Clean Architecture.

This module implements a modular, maintainable feature extraction pipeline using
software engineering best practices and design patterns:

- Repository Pattern for data access
- Factory Pattern for feature extractors
- Pipeline Pattern for data transformations
- Strategy Pattern for feature selection methods
- Dependency Injection for better testability

The pipeline integrates era detection results from multiple tables and performs
efficient feature extraction using tsfresh.

Usage:
    python extract_features_enhanced.py

Environment variables:
    DB_* - Database connection parameters
    FEATURES_TABLE - Destination table for features
    USE_GPU - Enable GPU acceleration (true/false)
    BATCH_SIZE - Number of eras to process in parallel
"""

from __future__ import annotations

import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

import pandas as pd
import sqlalchemy
from sqlalchemy import text
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters

from . import config
from .db_utils import SQLAlchemyPostgresConnector

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if os.getenv("DEBUG", "").lower() == "true" else logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class EraDefinition:
    """Represents a detected era with metadata."""

    era_id: str
    signal_name: str
    level: str  # A, B, or C
    stage: str  # PELT, BOCPD, or HMM
    start_time: datetime
    end_time: datetime
    rows: int

    @classmethod
    def from_db_row(cls, row: dict[str, Any]) -> EraDefinition:
        """Create from database row."""
        return cls(
            era_id=f"{row['signal_name']}_{row['level']}_{row['stage']}_{row['era_id']}",
            signal_name=row["signal_name"],
            level=row["level"],
            stage=row["stage"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            rows=row["rows"],
        )


@dataclass
class FeatureExtractionConfig:
    """Configuration for feature extraction."""

    batch_size: int = 1000
    use_gpu: bool = False
    feature_set: str = "efficient"  # minimal, efficient, comprehensive
    n_jobs: int = -1
    chunk_size: int | None = None
    era_level: str = "B"  # Which era detection level to use
    min_era_rows: int = 100  # Minimum rows per era to process


# ============================================================================
# Abstract Base Classes and Protocols
# ============================================================================


class DataRepository(Protocol):
    """Protocol for data access operations."""

    def fetch_sensor_data(
        self, start_time: datetime, end_time: datetime, columns: list[str] | None = None
    ) -> pd.DataFrame:
        """Fetch sensor data for time range."""
        ...

    def fetch_era_definitions(self, level: str, signal_name: str | None = None) -> list[EraDefinition]:
        """Fetch era definitions from database."""
        ...

    def save_features(self, features_df: pd.DataFrame, table_name: str, if_exists: str = "append") -> None:
        """Save features to database."""
        ...


class FeatureExtractor(ABC):
    """Abstract base class for feature extractors."""

    @abstractmethod
    def extract(self, data: pd.DataFrame, era_id: str) -> pd.DataFrame:
        """Extract features from data."""
        pass

    @abstractmethod
    def get_required_columns(self) -> list[str]:
        """Get list of required input columns."""
        pass


class DataTransformer(ABC):
    """Abstract base class for data transformers."""

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data."""
        pass


# ============================================================================
# Concrete Implementations
# ============================================================================


class TimescaleDBRepository:
    """Repository implementation for TimescaleDB."""

    def __init__(self, connector: SQLAlchemyPostgresConnector):
        self.connector = connector
        self.engine = connector.engine

    def fetch_sensor_data(
        self, start_time: datetime, end_time: datetime, columns: list[str] | None = None
    ) -> pd.DataFrame:
        """Fetch sensor data from preprocessed_features table."""
        if columns:
            # Direct column access for hybrid table
            cols_sql = ", ".join([f'"{col}"' for col in columns if col != "time"])
            query = f"""
            SELECT
                time,
                {cols_sql}
            FROM preprocessed_features
            WHERE time >= :start_time
            AND time < :end_time
            AND time IS NOT NULL
            ORDER BY time
            """
        else:
            # Fetch all columns
            query = """
            SELECT *
            FROM preprocessed_features
            WHERE time >= :start_time
            AND time < :end_time
            AND time IS NOT NULL
            ORDER BY time
            """

        logger.info(f"Fetching sensor data from {start_time} to {end_time}")
        df = pd.read_sql(text(query), self.engine, params={"start_time": start_time, "end_time": end_time})
        logger.info(f"Fetched {len(df)} rows with {len(df.columns)} columns")
        return df

    def fetch_era_definitions(self, level: str, signal_name: str | None = None) -> list[EraDefinition]:
        """Fetch era definitions from era_labels_level_* tables."""
        table_name = f"era_labels_level_{level.lower()}"

        query = f"""
        SELECT
            signal_name,
            level,
            stage,
            era_id,
            start_time,
            end_time,
            rows
        FROM {table_name}
        WHERE 1=1
        """

        params = {}
        if signal_name:
            query += " AND signal_name = :signal_name"
            params["signal_name"] = signal_name

        query += " ORDER BY signal_name, start_time"

        logger.info(f"Fetching era definitions from {table_name}")
        result = self.connector.fetch_data_to_pandas(text(query), **params)

        eras = [EraDefinition.from_db_row(row) for _, row in result.iterrows()]
        logger.info(f"Fetched {len(eras)} era definitions")
        return eras

    def fetch_external_data(self) -> dict[str, pd.DataFrame]:
        """Fetch external data (weather, energy prices, phenotypes)."""
        external_data = {}

        # Fetch weather data
        weather_query = """
        SELECT
            time,
            temperature_2m,
            precipitation,
            solar_radiation
        FROM external_weather_data
        WHERE time IS NOT NULL
        ORDER BY time
        """
        external_data["weather"] = pd.read_sql(text(weather_query), self.engine)

        # Fetch energy prices
        energy_query = """
        SELECT
            time,
            price_dk1,
            price_dk2
        FROM energy_prices
        WHERE time IS NOT NULL
        ORDER BY time
        """
        external_data["energy"] = pd.read_sql(text(energy_query), self.engine)

        # Fetch phenotype data (static)
        phenotype_query = """
        SELECT * FROM phenotypes
        """
        external_data["phenotypes"] = pd.read_sql(text(phenotype_query), self.engine)

        return external_data

    def save_features(self, features_df: pd.DataFrame, table_name: str, if_exists: str = "append") -> None:
        """Save features to database."""
        logger.info(f"Saving {len(features_df)} rows to {table_name}")

        # Feature tables don't need to be hypertables - they're not time-series data
        # They're derived features for each era, stored as regular tables
        self.connector.write_dataframe(features_df, table_name, if_exists=if_exists, index=False)

        # Create indices for better query performance
        if if_exists == "replace":
            with self.engine.begin() as conn:
                # Create index on era_id if it exists
                if 'era_id' in features_df.columns:
                    conn.execute(sqlalchemy.text(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_era_id ON {table_name}(era_id)"))
                # Create index on signal_name if it exists
                if 'signal_name' in features_df.columns:
                    conn.execute(sqlalchemy.text(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_signal ON {table_name}(signal_name)"))
                # Create index on level if it exists
                if 'level' in features_df.columns:
                    conn.execute(sqlalchemy.text(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_level ON {table_name}(level)"))

                logger.info(f"Created indices for {table_name}")


class TsfreshLongFormatTransformer(DataTransformer):
    """Transforms wide sensor data to tsfresh long format."""

    def __init__(self, value_columns: list[str]):
        self.value_columns = value_columns

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform to long format expected by tsfresh."""
        if "era_id" not in data.columns:
            raise ValueError("Data must have 'era_id' column")

        if "time" not in data.columns:
            raise ValueError("Data must have 'time' column")

        # Filter to only numeric columns
        numeric_cols = [
            col for col in self.value_columns if col in data.columns and pd.api.types.is_numeric_dtype(data[col])
        ]

        if not numeric_cols:
            raise ValueError("No numeric columns found for transformation")

        logger.info(f"Melting {len(numeric_cols)} numeric columns to long format")

        # Melt to long format
        long_df = data.melt(id_vars=["era_id", "time"], value_vars=numeric_cols, var_name="kind", value_name="value")

        # Remove NaN values
        long_df = long_df.dropna(subset=["value"])

        # Ensure proper types
        long_df["era_id"] = long_df["era_id"].astype(str)
        long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")

        logger.info(f"Transformed to long format: {len(long_df)} rows")
        return long_df


class TsfreshFeatureExtractor(FeatureExtractor):
    """Feature extractor using tsfresh."""

    def __init__(self, feature_set: str = "efficient", n_jobs: int = -1):
        self.feature_set = feature_set
        # Fix n_jobs for tsfresh: -1 means use all CPUs, but tsfresh might need positive number
        if n_jobs == -1:
            import multiprocessing

            self.n_jobs = multiprocessing.cpu_count()
        else:
            self.n_jobs = max(1, n_jobs)
        self.fc_parameters = self._get_fc_parameters()

    def _get_fc_parameters(self):
        """Get feature calculation parameters based on feature set."""
        if self.feature_set == "minimal":
            return MinimalFCParameters()
        elif self.feature_set == "efficient":
            return EfficientFCParameters()
        else:
            # For comprehensive, use default (all features)
            return None

    def extract(self, data: pd.DataFrame, era_id: str) -> pd.DataFrame:
        """Extract features using tsfresh."""
        if data.empty:
            logger.warning(f"Empty data for era {era_id}")
            return pd.DataFrame({"era_id": [era_id]})

        try:
            features = extract_features(
                data,
                column_id="era_id",
                column_sort="time",
                column_kind="kind",
                column_value="value",
                default_fc_parameters=self.fc_parameters,
                n_jobs=self.n_jobs,
                disable_progressbar=True,
            )

            if features.empty:
                logger.warning(f"No features extracted for era {era_id}")
                return pd.DataFrame({"era_id": [era_id]})

            # Debug: log the structure
            logger.debug(f"Features index name: {features.index.name}")
            logger.debug(f"Features columns: {list(features.columns)[:5]}...")

            # Reset index to get era_id as column
            features = features.reset_index()

            # The index might be named 'era_id' or 'id' depending on tsfresh version
            if 'index' in features.columns:
                features.rename(columns={"index": "era_id"}, inplace=True)
            elif 'id' in features.columns:
                features.rename(columns={"id": "era_id"}, inplace=True)
            elif 'era_id' not in features.columns:
                # If era_id is still not in columns, something's wrong
                logger.error(f"Could not find id column in features. Columns: {list(features.columns)}")
                features["era_id"] = era_id

            logger.info(f"Extracted {len(features.columns) - 1} features for era {era_id}")
            return features

        except Exception as e:
            logger.error(f"Feature extraction failed for era {era_id}: {e}")
            return pd.DataFrame({"era_id": [era_id]})

    def get_required_columns(self) -> list[str]:
        """Get required columns - determined dynamically."""
        return []  # Tsfresh works with any numeric columns


class FeatureExtractionPipeline:
    """Main pipeline orchestrator using dependency injection."""

    def __init__(
        self,
        repository: DataRepository,
        extractor: FeatureExtractor,
        transformer: DataTransformer,
        config: FeatureExtractionConfig,
    ):
        self.repository = repository
        self.extractor = extractor
        self.transformer = transformer
        self.config = config

    def run(self) -> pd.DataFrame:
        """Run the complete feature extraction pipeline."""
        start_time = time.time()

        # Fetch era definitions
        eras = self.repository.fetch_era_definitions(level=self.config.era_level)

        if not eras:
            logger.error("No era definitions found")
            return pd.DataFrame()

        # Filter eras by minimum rows
        eras = [era for era in eras if era.rows >= self.config.min_era_rows]
        logger.info(f"Processing {len(eras)} eras with >= {self.config.min_era_rows} rows")

        # Process eras in batches
        all_features = []
        for i in range(0, len(eras), self.config.batch_size):
            batch_eras = eras[i : i + self.config.batch_size]
            logger.info(f"Processing batch {i // self.config.batch_size + 1} with {len(batch_eras)} eras")

            batch_features = self._process_era_batch(batch_eras)

            if not batch_features.empty:
                logger.info(f"Batch features shape: {batch_features.shape}")
                if 'era_id' not in batch_features.columns:
                    logger.error(f"ERROR: Batch missing era_id! Columns: {list(batch_features.columns)[:10]}...")
                all_features.append(batch_features)
            else:
                logger.warning(f"Batch {i // self.config.batch_size + 1} returned empty features")

        if not all_features:
            logger.warning("No features extracted")
            return pd.DataFrame()

        # Combine all features
        logger.info(f"Combining {len(all_features)} batches")
        final_features = pd.concat(all_features, ignore_index=True)

        if 'era_id' not in final_features.columns:
            logger.error(f"FINAL ERROR: Combined features missing era_id! Columns: {list(final_features.columns)[:10]}...")

        elapsed = time.time() - start_time
        logger.info(f"Pipeline completed in {elapsed:.2f} seconds")
        logger.info(f"Final features shape: {final_features.shape}")

        return final_features

    def _process_era_batch(self, eras: list[EraDefinition]) -> pd.DataFrame:
        """Process a batch of eras."""
        batch_features = []

        for era in eras:
            try:
                # Fetch sensor data for era
                sensor_data = self.repository.fetch_sensor_data(start_time=era.start_time, end_time=era.end_time)

                if sensor_data.empty:
                    logger.warning(f"No data for era {era.era_id}")
                    continue

                # Add era_id column
                sensor_data["era_id"] = era.era_id

                # Transform to long format
                long_data = self.transformer.transform(sensor_data)

                # Extract features
                features = self.extractor.extract(long_data, era.era_id)

                # Add era metadata
                features["signal_name"] = era.signal_name
                features["level"] = era.level
                features["stage"] = era.stage
                features["start_time"] = era.start_time
                features["end_time"] = era.end_time
                features["era_rows"] = era.rows

                batch_features.append(features)

            except Exception as e:
                logger.error(f"Error processing era {era.era_id}: {e}")
                continue

        if not batch_features:
            return pd.DataFrame()

        # Concatenate all features
        result = pd.concat(batch_features, ignore_index=True)

        # Debug log
        logger.debug(f"Concatenated features shape: {result.shape}")
        logger.debug(f"Concatenated features columns: {list(result.columns)[:10]}...")

        return result


class OptimalSignalSelector:
    """Selects optimal signals based on coverage and importance."""

    # Optimal signals from analysis
    PRIMARY_SIGNALS = [
        "dli_sum",  # 100% - Primary light metric
        "radiation_w_m2",  # 4.2% - Solar radiation
        "outside_temp_c",  # 3.8% - External temperature
        "co2_measured_ppm",  # 3.8% - Growth indicator
        "air_temp_middle_c",  # 3.5% - Internal climate
        "air_temp_c",  # Alternative temperature signal
        "relative_humidity_percent",  # Humidity control
        "light_intensity_umol",  # Light intensity
    ]

    SECONDARY_SIGNALS = [
        "pipe_temp_1_c",  # 3.5% - Heating system
        "curtain_1_percent",  # 3.7% - Light control
        "humidity_deficit_g_m3",  # 3.5% - Humidity control
        "heating_setpoint_c",  # Heating control
        "vpd_hpa",  # Vapor pressure deficit
        "total_lamps_on",  # Artificial lighting status
    ]

    @classmethod
    def get_optimal_columns(cls, available_columns: list[str]) -> list[str]:
        """Get optimal columns that are available in the data."""
        optimal = []

        # Add primary signals first
        for signal in cls.PRIMARY_SIGNALS:
            if signal in available_columns:
                optimal.append(signal)

        # Add secondary signals
        for signal in cls.SECONDARY_SIGNALS:
            if signal in available_columns:
                optimal.append(signal)

        return optimal


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """Main entry point for enhanced feature extraction."""
    # Load configuration
    extraction_config = FeatureExtractionConfig(
        batch_size=int(os.getenv("BATCH_SIZE", "100")),
        use_gpu=config.USE_GPU_FLAG,
        feature_set=os.getenv("FEATURE_SET", "efficient"),
        n_jobs=int(os.getenv("N_JOBS", "-1")),
        era_level=os.getenv("ERA_LEVEL", "B"),
        min_era_rows=int(os.getenv("MIN_ERA_ROWS", "100")),
    )

    logger.info(f"Starting feature extraction with config: {extraction_config}")

    # Create repository
    connector = SQLAlchemyPostgresConnector(
        user=config.DB_USER,
        password=config.DB_PASSWORD,
        host=config.DB_HOST,
        port=config.DB_PORT,
        db_name=config.DB_NAME,
    )
    repository = TimescaleDBRepository(connector)

    try:
        # Get available columns from a sample query
        sample_df = repository.fetch_sensor_data(start_time=datetime(2013, 12, 1), end_time=datetime(2013, 12, 2))

        if sample_df.empty:
            logger.error("No data available in preprocessed_features")
            return 1

        available_columns = list(sample_df.columns)
        logger.info(f"Available columns: {len(available_columns)}")

        # Select optimal columns
        value_columns = OptimalSignalSelector.get_optimal_columns(available_columns)
        logger.info(f"Selected {len(value_columns)} optimal signals for processing")

        # Create components
        transformer = TsfreshLongFormatTransformer(value_columns)
        extractor = TsfreshFeatureExtractor(feature_set=extraction_config.feature_set, n_jobs=extraction_config.n_jobs)

        # Create and run pipeline
        pipeline = FeatureExtractionPipeline(
            repository=repository, extractor=extractor, transformer=transformer, config=extraction_config
        )

        features_df = pipeline.run()

        if features_df.empty:
            logger.error("No features extracted")
            return 1

        # Save features to database
        features_table = config.FEATURES_TABLE
        logger.info(f"Features DataFrame columns: {list(features_df.columns)[:10]}...")
        logger.info(f"Features DataFrame shape: {features_df.shape}")
        if 'era_id' not in features_df.columns:
            logger.error("ERROR: 'era_id' column missing from features DataFrame!")
            logger.info(f"Available columns: {list(features_df.columns)}")
        repository.save_features(features_df, features_table, if_exists="replace")

        logger.info(f"Successfully saved {len(features_df)} feature rows to {features_table}")
        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1
    finally:
        # Clean up
        if connector.engine:
            connector.engine.dispose()


if __name__ == "__main__":
    sys.exit(main())
