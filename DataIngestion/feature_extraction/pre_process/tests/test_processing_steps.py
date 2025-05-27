"""Tests for processing steps."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import (
    DataSegmenter,
    ImputationHandler,
    OutlierHandler,
    analyze_column_metadata,
)


class TestOutlierHandler:
    """Test the OutlierHandler class."""

    def test_basic_outlier_clipping(self):
        """Test basic outlier clipping functionality."""
        df = pd.DataFrame(
            {
                "temp": [10, 20, 100, 30, -10],
                "humidity": [50, 60, 70, 80, 150],
            }
        )

        rules = [
            {"column": "temp", "min_value": 0, "max_value": 50, "clip": True},
            {"column": "humidity", "min_value": 0, "max_value": 100, "clip": True},
        ]

        handler = OutlierHandler(rules)
        result = handler.clip_outliers(df)

        # Check clipping
        assert result["temp"].min() >= 0
        assert result["temp"].max() <= 50
        assert result["humidity"].min() >= 0
        assert result["humidity"].max() <= 100

        # Check specific values
        assert result["temp"].tolist() == [10, 20, 50, 30, 0]
        assert result["humidity"].tolist() == [50, 60, 70, 80, 100]

        # Check metrics
        assert handler.metrics.outliers_detected == 3  # 100, -10, 150
        assert handler.metrics.outliers_clipped == 3

    def test_do_not_clip_columns(self):
        """Test that specified columns are not clipped."""
        df = pd.DataFrame(
            {
                "temperature": [10, 11, 100, 15, 16],
                "humidity": [50, 52, 51, 53, 200],
                "special_col": [-1000, 1000, 2000, -500, 3000],
            }
        )

        rules = [
            {"column": "temperature", "min_value": 0, "max_value": 50, "clip": True},
            {"column": "humidity", "min_value": 0, "max_value": 100, "clip": True},
            {"column": "special_col", "min_value": -100, "max_value": 100, "clip": True},
        ]

        rules_cfg = {"do_not_clip_columns": ["special_col"]}

        handler = OutlierHandler(rules, rules_cfg)
        result = handler.clip_outliers(df)

        # Temperature and humidity should be clipped
        assert result["temperature"].max() == 50
        assert result["humidity"].max() == 100

        # Special column should NOT be clipped
        assert result["special_col"].min() == -1000
        assert result["special_col"].max() == 3000
        assert result["special_col"].tolist() == [-1000, 1000, 2000, -500, 3000]

    def test_missing_column_warning(self, caplog):
        """Test warning when column not found."""
        df = pd.DataFrame({"temp": [10, 20, 30]})
        rules = [{"column": "humidity", "min_value": 0, "max_value": 100, "clip": True}]

        handler = OutlierHandler(rules)
        result = handler.clip_outliers(df)

        # Should return unchanged DataFrame
        pd.testing.assert_frame_equal(result, df)

        # Should log warning
        assert "not found in DataFrame" in caplog.text

    def test_no_clip_flag(self):
        """Test that clip=False rules are ignored."""
        df = pd.DataFrame({"temp": [10, 100, -10]})
        rules = [{"column": "temp", "min_value": 0, "max_value": 50, "clip": False}]

        handler = OutlierHandler(rules)
        result = handler.clip_outliers(df)

        # Should not clip
        pd.testing.assert_frame_equal(result, df)
        assert handler.metrics.outliers_clipped == 0


class TestImputationHandler:
    """Test the ImputationHandler class."""

    def test_forward_fill(self):
        """Test forward fill imputation."""
        df = pd.DataFrame(
            {
                "temp": [10, np.nan, np.nan, 20, np.nan],
                "humidity": [50, 60, np.nan, 70, 80],
            }
        )

        rules = [
            {"column": "temp", "strategy": "forward_fill", "limit": 1},
            {"column": "humidity", "strategy": "forward_fill"},
        ]

        handler = ImputationHandler(rules)
        result = handler.impute_data(df)

        # Check forward fill with limit
        assert result["temp"].tolist()[0:3] == [10, 10, np.nan]  # Only 1 forward fill
        assert not pd.isna(result["temp"].iloc[1])  # First NaN filled
        assert pd.isna(result["temp"].iloc[2])  # Second NaN not filled (limit=1)

        # Check forward fill without limit
        assert result["humidity"].tolist() == [50, 60, 60, 70, 80]

        # Check metrics
        assert handler.metrics.missing_values_filled == 2

    def test_backward_fill(self):
        """Test backward fill imputation."""
        df = pd.DataFrame({"temp": [np.nan, 10, np.nan, 20]})

        rules = [{"column": "temp", "strategy": "backward_fill"}]

        handler = ImputationHandler(rules)
        result = handler.impute_data(df)

        assert result["temp"].tolist() == [10, 10, 20, 20]
        assert handler.metrics.missing_values_filled == 2

    def test_linear_interpolation(self):
        """Test linear interpolation."""
        df = pd.DataFrame(
            {
                "temp": [10.0, np.nan, np.nan, 40.0],
                "text": ["a", np.nan, "c", "d"],  # Non-numeric
            }
        )

        rules = [
            {"column": "temp", "strategy": "linear"},
            {"column": "text", "strategy": "linear"},  # Should skip
        ]

        handler = ImputationHandler(rules)
        result = handler.impute_data(df)

        # Check linear interpolation
        assert result["temp"].tolist() == [10.0, 20.0, 30.0, 40.0]

        # Non-numeric column should be unchanged
        assert pd.isna(result["text"].iloc[1])

    def test_mean_imputation(self):
        """Test mean imputation."""
        df = pd.DataFrame({"temp": [10, 20, np.nan, 30, np.nan]})

        rules = [{"column": "temp", "strategy": "mean"}]

        handler = ImputationHandler(rules)
        result = handler.impute_data(df)

        # Mean of [10, 20, 30] = 20
        assert result["temp"].tolist() == [10, 20, 20, 30, 20]
        assert handler.metrics.missing_values_filled == 2

    def test_median_imputation(self):
        """Test median imputation."""
        df = pd.DataFrame({"temp": [10, 20, np.nan, 30, np.nan, 40]})

        rules = [{"column": "temp", "strategy": "median"}]

        handler = ImputationHandler(rules)
        result = handler.impute_data(df)

        # Median of [10, 20, 30, 40] = 25
        assert result["temp"].tolist() == [10, 20, 25, 30, 25, 40]

    def test_invalid_strategy_warning(self, caplog):
        """Test warning for invalid strategy."""
        df = pd.DataFrame({"temp": [10, np.nan, 20]})
        rules = [{"column": "temp", "strategy": "invalid_strategy"}]

        handler = ImputationHandler(rules)
        result = handler.impute_data(df)

        # Should not change data
        assert pd.isna(result["temp"].iloc[1])
        assert "Unknown imputation strategy" in caplog.text

    @given(
        data_frames(
            columns=[
                column("numeric", dtype=float),
                column("integers", dtype=int),
            ],
            rows=st.tuples(
                st.floats(min_value=-100, max_value=100, allow_nan=True),
                st.integers(min_value=0, max_value=100),
            ),
        )
    )
    def test_imputation_reduces_nans(self, df: pd.DataFrame):
        """Property: imputation should never increase NaN count."""
        rules = [
            {"column": "numeric", "strategy": "mean"},
            {"column": "integers", "strategy": "forward_fill"},
        ]

        handler = ImputationHandler(rules)

        nans_before = df.isna().sum().sum()
        result = handler.impute_data(df)
        nans_after = result.isna().sum().sum()

        assert nans_after <= nans_before


class TestDataSegmenter:
    """Test the DataSegmenter class."""

    def test_basic_segmentation(self):
        """Test basic time-based segmentation."""
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2023-01-01 00:00:00",
                        "2023-01-01 01:00:00",
                        # 25 hour gap
                        "2023-01-02 02:00:00",
                        "2023-01-02 03:00:00",
                    ]
                ),
                "value": [1, 2, 3, 4],
            }
        )

        config = {
            "time_col": "timestamp",
            "segmentation": {"min_gap_hours": 24},
        }

        segmenter = DataSegmenter(config)
        segments = segmenter.segment_by_availability(df)

        assert len(segments) == 2
        assert len(segments[0]) == 2
        assert len(segments[1]) == 2

        # Check segment info
        segment_info = segmenter.get_segment_info()
        assert len(segment_info) == 2
        assert segment_info[0].row_count == 2
        assert segment_info[1].row_count == 2

    def test_time_as_index(self):
        """Test segmentation with time as index."""
        times = pd.to_datetime(
            [
                "2023-01-01 00:00:00",
                "2023-01-01 01:00:00",
                "2023-01-02 02:00:00",  # >24h gap
            ]
        )
        df = pd.DataFrame({"value": [1, 2, 3]}, index=times)
        df.index.name = "timestamp"

        config = {"time_col": "timestamp"}
        segmenter = DataSegmenter(config)
        segments = segmenter.segment_by_availability(df)

        assert len(segments) == 2
        assert len(segments[0]) == 2
        assert len(segments[1]) == 1

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame({"timestamp": pd.Series([], dtype="datetime64[ns]")})

        config = {"time_col": "timestamp"}
        segmenter = DataSegmenter(config)
        segments = segmenter.segment_by_availability(df)

        assert len(segments) == 0

    def test_single_row(self):
        """Test with single row DataFrame."""
        df = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2023-01-01")],
                "value": [1],
            }
        )

        config = {"time_col": "timestamp"}
        segmenter = DataSegmenter(config)
        segments = segmenter.segment_by_availability(df)

        assert len(segments) == 1
        assert len(segments[0]) == 1


class TestAnalyzeColumnMetadata:
    """Test the analyze_column_metadata function."""

    def test_numeric_columns(self):
        """Test metadata for numeric columns."""
        df = pd.DataFrame(
            {
                "temp": [10, 20, 30, np.nan, 50],
                "humidity": [50, 60, 70, 80, 90],
            }
        )

        metadata = analyze_column_metadata(df)

        assert len(metadata) == 2

        # Check temperature metadata
        temp_meta = next(m for m in metadata if m.name == "temp")
        assert temp_meta.missing_count == 1
        assert temp_meta.missing_percentage == 20.0
        assert temp_meta.min_value == 10
        assert temp_meta.max_value == 50
        assert temp_meta.mean_value == pytest.approx(27.5)

        # Check humidity metadata (no missing)
        humidity_meta = next(m for m in metadata if m.name == "humidity")
        assert humidity_meta.missing_count == 0
        assert humidity_meta.missing_percentage == 0.0
        assert humidity_meta.unique_values == 5

    def test_non_numeric_columns(self):
        """Test metadata for non-numeric columns."""
        df = pd.DataFrame(
            {
                "category": ["A", "B", "A", np.nan, "C"],
                "id": ["x1", "x2", "x3", "x4", "x5"],
            }
        )

        metadata = analyze_column_metadata(df)

        cat_meta = next(m for m in metadata if m.name == "category")
        assert cat_meta.missing_count == 1
        assert cat_meta.unique_values == 3  # A, B, C (not counting NaN)
        assert cat_meta.min_value is None  # Non-numeric
        assert cat_meta.max_value is None


def test_outlier_handler_with_empty_no_clip_list():
    """Test with empty do_not_clip_columns list."""
    df = pd.DataFrame(
        {
            "temperature": [10, 11, 100, 15, 16],
            "humidity": [50, 52, 51, 53, 200],
        }
    )

    rules = [
        {"column": "temperature", "min_value": 0, "max_value": 50, "clip": True},
        {"column": "humidity", "min_value": 0, "max_value": 100, "clip": True},
    ]

    rules_cfg = {"do_not_clip_columns": []}

    handler = OutlierHandler(rules, rules_cfg)
    result = handler.clip_outliers(df)

    # All columns should be clipped
    assert result["temperature"].max() == 50
    assert result["humidity"].max() == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
