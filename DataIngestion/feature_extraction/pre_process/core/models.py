"""Data models and type definitions for the preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from typing_extensions import TypedDict
import pandas as pd
from pydantic import BaseModel, Field, field_validator


# Enums
class OutlierMethod(str, Enum):
    """Outlier detection methods."""

    IQR = "iqr"
    ZSCORE = "zscore"
    MAD = "mad"
    WINSORIZE = "winsorize"


class ImputationMethod(str, Enum):
    """Imputation methods for missing values."""

    FORWARD_FILL = "ffill"
    BACKWARD_FILL = "bfill"
    LINEAR = "linear"
    SPLINE = "spline"
    MEAN = "mean"
    MEDIAN = "median"
    ZERO = "zero"


class StorageType(str, Enum):
    """Storage backend types."""

    TIMESCALE_COLUMNAR = "timescale_columnar"
    TIMESCALE_JSONB = "timescale_jsonb"
    PARQUET = "parquet"


# TypedDicts for configuration
class OutlierConfig(TypedDict):
    """Configuration for outlier handling."""

    method: str
    threshold: float
    columns: list[str] | None
    do_not_clip_columns: list[str] | None


class ImputationConfig(TypedDict):
    """Configuration for imputation."""

    method: str
    limit: int | None
    order: int | None


class ResamplingConfig(TypedDict):
    """Configuration for resampling."""

    frequency: str
    aggregation: dict[str, str]


class DatabaseConfig(TypedDict):
    """Database connection configuration."""

    host: str
    port: int
    database: str
    user: str
    password: str


# Dataclasses for structured data
@dataclass
class TimeSegment:
    """Represents a continuous time segment in the data."""

    segment_id: int
    start_time: datetime
    end_time: datetime
    duration_minutes: float
    row_count: int
    has_gaps: bool = False
    gap_duration_minutes: float = 0.0

    @property
    def is_valid(self) -> bool:
        """Check if segment meets minimum requirements."""
        return self.duration_minutes >= 60 and self.row_count >= 10


@dataclass
class ProcessingMetrics:
    """Metrics collected during preprocessing."""

    total_rows_input: int = 0
    total_rows_output: int = 0
    outliers_detected: int = 0
    outliers_clipped: int = 0
    missing_values_filled: int = 0
    segments_created: int = 0
    processing_time_seconds: float = 0.0
    memory_usage_mb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for logging."""
        return {
            "total_rows_input": self.total_rows_input,
            "total_rows_output": self.total_rows_output,
            "outliers_detected": self.outliers_detected,
            "outliers_clipped": self.outliers_clipped,
            "missing_values_filled": self.missing_values_filled,
            "segments_created": self.segments_created,
            "processing_time_seconds": self.processing_time_seconds,
            "memory_usage_mb": self.memory_usage_mb,
            "efficiency_ratio": self.total_rows_output / max(self.total_rows_input, 1),
        }


@dataclass
class ColumnMetadata:
    """Metadata for a data column."""

    name: str
    dtype: str
    missing_count: int = 0
    missing_percentage: float = 0.0
    unique_values: int = 0
    min_value: float | None = None
    max_value: float | None = None
    mean_value: float | None = None
    std_value: float | None = None


# Pydantic models for configuration validation
class PreprocessingConfig(BaseModel):
    """Main preprocessing configuration with validation."""

    # Database settings
    database: DatabaseConfig

    # Processing settings
    start_date: datetime | None = None
    end_date: datetime | None = None
    batch_size: int = Field(default=10000, ge=100, le=100000)

    # Outlier handling
    outlier_config: OutlierConfig = Field(
        default_factory=lambda: {"method": "iqr", "threshold": 1.5, "do_not_clip_columns": []}
    )

    # Imputation settings
    imputation_config: ImputationConfig = Field(
        default_factory=lambda: {"method": "linear", "limit": 5}
    )

    # Resampling settings
    resampling_config: ResamplingConfig = Field(
        default_factory=lambda: {"frequency": "5min", "aggregation": {"default": "mean"}}
    )

    # Storage settings
    storage_type: StorageType = StorageType.TIMESCALE_COLUMNAR
    output_table: str = "preprocessed_greenhouse_data"

    # Feature engineering flags
    calculate_derived_features: bool = True
    include_era_features: bool = True
    include_light_synthesis: bool = True

    @field_validator("outlier_config")
    def validate_outlier_config(cls, v: OutlierConfig) -> OutlierConfig:
        """Validate outlier configuration."""
        if v["method"] not in ["iqr", "zscore", "mad", "winsorize"]:
            raise ValueError(f"Invalid outlier method: {v['method']}")
        if v["threshold"] <= 0:
            raise ValueError("Outlier threshold must be positive")
        return v

    @field_validator("imputation_config")
    def validate_imputation_config(cls, v: ImputationConfig) -> ImputationConfig:
        """Validate imputation configuration."""
        valid_methods = ["ffill", "bfill", "linear", "spline", "mean", "median", "zero"]
        if v["method"] not in valid_methods:
            raise ValueError(f"Invalid imputation method: {v['method']}")
        return v

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
        arbitrary_types_allowed = True


@dataclass
class ProcessingResult:
    """Result of preprocessing operation."""

    success: bool
    data: pd.DataFrame | None = None
    metrics: ProcessingMetrics = field(default_factory=ProcessingMetrics)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    segments: list[TimeSegment] = field(default_factory=list)

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
        self.success = False

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)
