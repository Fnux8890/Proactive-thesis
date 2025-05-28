"""Processing steps for data preprocessing pipeline."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from .models import (
    ColumnMetadata,
    ProcessingMetrics,
    TimeSegment,
)

logger = logging.getLogger(__name__)


class OutlierHandler:
    """Handles outlier detection and clipping based on configured rules."""

    def __init__(
        self, rules: list[dict[str, Any]], rules_cfg_dict: dict[str, Any] | None = None
    ) -> None:
        """Initialize outlier handler with rules and configuration.

        Args:
            rules: List of outlier rules with columns and thresholds
            rules_cfg_dict: Additional configuration including do_not_clip_columns
        """
        self.rules = rules
        if rules_cfg_dict is None:
            rules_cfg_dict = {}
        self.no_clip_columns: set[str] = set(rules_cfg_dict.get("do_not_clip_columns", []))
        self.metrics = ProcessingMetrics()
        logger.info(
            f"OutlierHandler initialized with {len(rules)} rules. "
            f"No-clip columns: {self.no_clip_columns}"
        )

    def clip_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clip outliers in specified columns based on rules.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with outliers clipped
        """
        logger.info("Starting outlier clipping...")
        df_processed = df.copy()

        for rule in self.rules:
            col: str = rule["column"]

            if col in self.no_clip_columns:
                logger.info(
                    f"Skipping outlier clipping for column '{col}' as it is in no_clip_columns."
                )
                continue

            if col in df_processed.columns:
                min_val: float | None = rule.get("min_value")
                max_val: float | None = rule.get("max_value")

                if rule.get("clip", False):
                    # Count outliers before clipping
                    outliers_low = 0
                    outliers_high = 0

                    if min_val is not None:
                        outliers_low = int((df_processed[col] < min_val).sum())
                        self.metrics.outliers_detected += outliers_low
                    if max_val is not None:
                        outliers_high = int((df_processed[col] > max_val).sum())
                        self.metrics.outliers_detected += outliers_high

                    logger.info(f"Clipping outliers in column '{col}' to [{min_val}, {max_val}]")
                    df_processed[col] = df_processed[col].clip(lower=min_val, upper=max_val)
                    self.metrics.outliers_clipped += outliers_low + outliers_high
            else:
                logger.warning(f"Column '{col}' for outlier rule not found in DataFrame.")

        logger.info("Outlier clipping completed.")
        return df_processed


class ImputationHandler:
    """Handles missing value imputation based on configured strategies."""

    def __init__(self, rules: list[dict[str, Any]]) -> None:
        """Initialize imputation handler with rules.

        Args:
            rules: List of imputation rules with columns and strategies
        """
        self.rules = rules
        self.metrics = ProcessingMetrics()
        logger.info(f"ImputationHandler initialized with {len(rules)} rules.")

    def impute_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values based on configured strategies.

        Args:
            df: Input DataFrame with missing values

        Returns:
            DataFrame with imputed values
        """
        logger.info("Starting data imputation...")
        df_processed = df.copy()

        for rule in self.rules:
            col: str = rule["column"]
            strategy: str = rule.get("strategy", "")

            if col not in df_processed.columns:
                logger.warning(f"Column '{col}' for imputation rule not found in DataFrame.")
                continue

            # Count missing values before imputation
            missing_before = df_processed[col].isna().sum()

            logger.info(f"Applying imputation for column '{col}' using strategy '{strategy}'")

            if strategy == "forward_fill":
                limit: int | None = rule.get("limit")
                df_processed[col] = df_processed[col].ffill(limit=limit)

            elif strategy in ["backward_fill", "bfill"]:
                limit = rule.get("limit")
                df_processed[col] = df_processed[col].bfill(limit=limit)

            elif strategy == "linear":
                limit_val: int | None = rule.get("limit")
                if pd.api.types.is_numeric_dtype(df_processed[col]):
                    df_processed[col] = df_processed[col].interpolate(
                        method="linear", limit_direction="both", limit=limit_val
                    )
                else:
                    logger.warning(
                        f"Column '{col}' is not numeric "
                        f"(dtype: {df_processed[col].dtype}). "
                        f"Skipping linear interpolation."
                    )

            elif strategy == "mean":
                if pd.api.types.is_numeric_dtype(df_processed[col]):
                    mean_val = df_processed[col].mean()
                    df_processed[col] = df_processed[col].fillna(mean_val)
                else:
                    logger.warning(f"Column '{col}' is not numeric. Skipping mean imputation.")

            elif strategy == "median":
                if pd.api.types.is_numeric_dtype(df_processed[col]):
                    median_val = df_processed[col].median()
                    df_processed[col] = df_processed[col].fillna(median_val)
                else:
                    logger.warning(f"Column '{col}' is not numeric. Skipping median imputation.")

            else:
                logger.warning(
                    f"Unknown imputation strategy '{strategy}' for column '{col}'. Skipping."
                )

            # Count missing values filled
            missing_after = df_processed[col].isna().sum()
            self.metrics.missing_values_filled += missing_before - missing_after

        logger.info("Data imputation completed.")
        return df_processed


class DataSegmenter:
    """Segments time series data based on temporal gaps."""

    def __init__(
        self, era_config: dict[str, Any], common_config: dict[str, Any] | None = None
    ) -> None:
        """Initialize data segmenter with configuration.

        Args:
            era_config: Era detection configuration
            common_config: Common configuration settings
        """
        self.era_config = era_config
        self.common_config = common_config if common_config is not None else {}
        self.metrics = ProcessingMetrics()

        # Try to get time_col from era_config first, then common_config, then default
        self.time_col: str = self.era_config.get(
            "time_col", self.common_config.get("time_col", "timestamp")
        )

        # Get segmentation specific config from era_config, default to 24 hours gap
        segmentation_settings = self.era_config.get("segmentation", {})
        self.min_gap_for_new_segment: pd.Timedelta = pd.Timedelta(
            segmentation_settings.get("min_gap_hours", 24), unit="h"
        )

        logger.info(
            f"DataSegmenter initialized. Time column: '{self.time_col}'. "
            f"Min gap for new segment: {self.min_gap_for_new_segment}"
        )

    def segment_by_availability(self, df: pd.DataFrame) -> list[pd.DataFrame]:
        """Segment DataFrame based on time gaps.

        Args:
            df: Input DataFrame with time column

        Returns:
            List of DataFrame segments
        """
        logger.info("Starting data segmentation by availability...")

        # Check if time_col is a regular column or the index
        if self.time_col in df.columns:
            df_sorted = df.sort_values(by=self.time_col).copy()
            time_data = pd.to_datetime(df_sorted[self.time_col])
        elif df.index.name == self.time_col:
            # If time_col is the index, ensure it's a DatetimeIndex and use it
            if not isinstance(df.index, pd.DatetimeIndex):
                print(
                    f"Warning: Index '{self.time_col}' is not a DatetimeIndex. Attempting conversion."
                )
                try:
                    df.index = pd.to_datetime(
                        df.index, utc=True
                    )  # Assuming UTC from previous steps
                except Exception as e:
                    print(
                        f"Error converting index '{self.time_col}' to DatetimeIndex: {e}. Returning single segment."
                    )
                    return [df]
            df_sorted = df.sort_index().copy()  # Sort by index if it's time
            time_data = df_sorted.index
        else:
            print(
                f"Warning: Time column or index '{self.time_col}' not found. Returning single segment."
            )
            return [df]

        if df_sorted.empty:
            print("Input DataFrame is empty. Returning empty list of segments.")
            return []

        segments = []
        current_segment_start_idx = 0
        for i in range(1, len(df_sorted)):
            # Access elements directly if time_data is an Index, or via .iloc if it's a Series
            if isinstance(time_data, pd.Index):  # Covers DatetimeIndex
                current_time = time_data[i]
                previous_time = time_data[i - 1]
            else:  # It's a Series
                current_time = time_data.iloc[i]
                previous_time = time_data.iloc[i - 1]

            time_diff = current_time - previous_time
            if time_diff > self.min_gap_for_new_segment:
                segments.append(df_sorted.iloc[current_segment_start_idx:i])
                current_segment_start_idx = i

        # Add the last segment
        segments.append(df_sorted.iloc[current_segment_start_idx:])

        # Update metrics
        self.metrics.segments_created = len(segments)

        logger.info(f"Data segmentation completed. Found {len(segments)} segments.")

        segment_info: list[TimeSegment] = []
        for i, seg_df in enumerate(segments):
            if not seg_df.empty:
                # Access time data correctly whether it's a column or index
                idx_for_print = (
                    seg_df.index
                    if isinstance(seg_df.index, pd.DatetimeIndex)
                    and seg_df.index.name == self.time_col
                    else pd.to_datetime(seg_df[self.time_col])
                )

                start_time = idx_for_print[0]
                end_time = idx_for_print[-1]
                duration = (end_time - start_time).total_seconds() / 60  # minutes

                segment = TimeSegment(
                    segment_id=i + 1,
                    start_time=start_time,
                    end_time=end_time,
                    duration_minutes=duration,
                    row_count=len(seg_df),
                )
                segment_info.append(segment)

                logger.info(
                    f"  Segment {i + 1}: {len(seg_df)} rows, from {start_time} to {end_time}"
                )
            else:
                logger.info(f"  Segment {i + 1}: Empty")

        # Store segment info in metrics for later use
        self.segment_info = segment_info

        return [s for s in segments if not s.empty]  # Filter out empty segments

    def get_segment_info(self) -> list[TimeSegment]:
        """Get information about created segments.

        Returns:
            List of TimeSegment objects with segment metadata
        """
        return getattr(self, "segment_info", [])


def analyze_column_metadata(df: pd.DataFrame) -> list[ColumnMetadata]:
    """Analyze DataFrame columns and return metadata.

    Args:
        df: Input DataFrame

    Returns:
        List of ColumnMetadata objects
    """
    metadata_list: list[ColumnMetadata] = []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            metadata = ColumnMetadata(
                name=col,
                dtype=str(df[col].dtype),
                missing_count=df[col].isna().sum(),
                missing_percentage=df[col].isna().sum() / len(df) * 100,
                unique_values=df[col].nunique(),
                min_value=float(df[col].min()) if not df[col].isna().all() else None,
                max_value=float(df[col].max()) if not df[col].isna().all() else None,
                mean_value=float(df[col].mean()) if not df[col].isna().all() else None,
                std_value=float(df[col].std()) if not df[col].isna().all() else None,
            )
        else:
            metadata = ColumnMetadata(
                name=col,
                dtype=str(df[col].dtype),
                missing_count=df[col].isna().sum(),
                missing_percentage=df[col].isna().sum() / len(df) * 100,
                unique_values=df[col].nunique(),
            )

        metadata_list.append(metadata)

    return metadata_list


# Example Usage (for testing this module directly)
if __name__ == "__main__":
    # Sample DataFrame
    data = {
        "timestamp": pd.to_datetime(
            [
                "2023-01-01 00:00:00",
                "2023-01-01 01:00:00",
                "2023-01-01 02:00:00",  # Segment 1
                "2023-01-03 03:00:00",
                "2023-01-03 04:00:00",  # Segment 2 (after >24h gap)
                "2023-01-03 04:05:00",  # Still Segment 2
            ]
        ),
        "temperature": [10, 11, None, 100, 15, 16],
        "humidity": [50, 52, 51, 50, -10, 53],
    }
    sample_df = pd.DataFrame(data)

    # Sample Config
    sample_config = {
        "time_col": "timestamp",
        "preprocessing": {
            "outlier_rules": [
                {"column": "temperature", "min_value": 0, "max_value": 50, "clip": True},
                {"column": "humidity", "min_value": 0, "max_value": 100, "clip": True},
            ],
            "imputation_rules": [
                {"column": "temperature", "strategy": "linear"},
                {"column": "humidity", "strategy": "forward_fill", "limit": 1},
            ],
        },
        "segmentation": {"min_gap_hours": 24},
    }

    print("--- Testing OutlierHandler ---")
    outlier_handler = OutlierHandler(sample_config["preprocessing"]["outlier_rules"])
    df_after_outliers = outlier_handler.clip_outliers(sample_df.copy())
    print("DataFrame after outlier handling:")
    print(df_after_outliers)

    print("\n--- Testing ImputationHandler ---")
    imputation_handler = ImputationHandler(sample_config["preprocessing"]["imputation_rules"])
    df_after_imputation = imputation_handler.impute_data(df_after_outliers.copy())
    print("DataFrame after imputation:")
    print(df_after_imputation)

    print("\n--- Testing DataSegmenter ---")
    # Pass both era_config (as sample_config) and a common_config example
    segmenter = DataSegmenter(era_config=sample_config, common_config={"time_col": "timestamp"})
    segments = segmenter.segment_by_availability(df_after_imputation.copy())
    for i, seg in enumerate(segments):
        print(f"\nSegment {i + 1}:")
        print(seg)
