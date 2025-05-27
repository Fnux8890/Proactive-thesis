"""Tests for island detection functionality."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest
from hypothesis import given
from hypothesis.extra.pandas import column, data_frames, range_indexes

from core import (
    detect_islands_with_metadata,
    filter_valid_segments,
    label_islands,
    merge_small_gaps,
)


class TestLabelIslands:
    """Test the label_islands function."""

    def test_empty_series(self):
        """Test with empty series."""
        result = label_islands(pd.Series([], dtype="datetime64[ns]"))
        assert len(result) == 0
        assert result.dtype == "int64"

    def test_single_point(self):
        """Test with single data point."""
        ts = pd.Series([pd.Timestamp("2023-01-01")])
        result = label_islands(ts)
        assert result.tolist() == [0]

    def test_continuous_data_no_gaps(self):
        """Test continuous data with no gaps."""
        # Data every 5 minutes
        times = pd.date_range("2023-01-01", periods=10, freq="5min")
        result = label_islands(pd.Series(times), gap="15min")
        assert result.nunique() == 1
        assert result.tolist() == [0] * 10

    def test_single_large_gap(self):
        """Test data with one large gap."""
        times = pd.to_datetime(
            [
                "2023-01-01 00:00:00",
                "2023-01-01 00:05:00",
                "2023-01-01 00:10:00",
                # 1 hour gap
                "2023-01-01 01:10:00",
                "2023-01-01 01:15:00",
            ]
        )
        result = label_islands(pd.Series(times), gap="30min")
        assert result.tolist() == [0, 0, 0, 1, 1]

    def test_multiple_gaps(self):
        """Test data with multiple gaps."""
        times = pd.to_datetime(
            [
                "2023-01-01 00:00:00",
                "2023-01-01 00:05:00",
                # 20 min gap
                "2023-01-01 00:25:00",
                "2023-01-01 00:30:00",
                # 40 min gap
                "2023-01-01 01:10:00",
            ]
        )
        result = label_islands(pd.Series(times), gap="15min")
        assert result.tolist() == [0, 0, 1, 1, 2]

    def test_gap_at_boundary(self):
        """Test gap exactly at threshold."""
        times = pd.to_datetime(
            [
                "2023-01-01 00:00:00",
                "2023-01-01 00:15:00",  # Exactly 15 min gap
                "2023-01-01 00:30:01",  # Just over 15 min gap
            ]
        )
        result = label_islands(pd.Series(times), gap="15min")
        # First gap is exactly 15min (not a new island)
        # Second gap is > 15min (new island)
        assert result.tolist() == [0, 0, 1]

    @given(
        data_frames(
            columns=[column("times", dtype="datetime64[ns]")],
            index=range_indexes(min_size=2, max_size=100),
        )
    )
    def test_monotonic_property(self, df: pd.DataFrame):
        """Property: island IDs should be monotonically increasing."""
        result = label_islands(df["times"].sort_values())

        # Check monotonicity
        assert (result.diff().dropna() >= 0).all()

        # Check integer type
        assert result.dtype == "int64"

        # Check starts from 0
        if len(result) > 0:
            assert result.min() == 0


class TestDetectIslandsWithMetadata:
    """Test the detect_islands_with_metadata function."""

    def test_basic_detection(self):
        """Test basic island detection with metadata."""
        df = pd.DataFrame(
            {
                "time": pd.date_range("2023-01-01", periods=20, freq="5min"),
                "value": range(20),
            }
        )

        # Add a gap
        df = pd.concat(
            [df.iloc[:10], df.iloc[10:].assign(time=lambda x: x["time"] + pd.Timedelta("1h"))]
        ).reset_index(drop=True)

        result_df, segments = detect_islands_with_metadata(df, time_col="time", gap="30min")

        # Check segment_id column added
        assert "segment_id" in result_df.columns

        # Check we have 2 segments
        assert len(segments) == 2

        # Check segment metadata
        seg1, seg2 = segments
        assert seg1.row_count == 10
        assert seg2.row_count == 10
        assert seg1.segment_id == 0
        assert seg2.segment_id == 1
        assert not seg1.has_gaps
        assert not seg2.has_gaps

    def test_min_island_size_filtering(self):
        """Test filtering of small islands."""
        df = pd.DataFrame(
            {
                "time": pd.to_datetime(
                    [
                        "2023-01-01 00:00:00",
                        "2023-01-01 00:05:00",
                        # Large gap
                        "2023-01-01 02:00:00",
                        "2023-01-01 02:05:00",
                        "2023-01-01 02:10:00",
                    ]
                ),
                "value": range(5),
            }
        )

        result_df, segments = detect_islands_with_metadata(
            df, time_col="time", gap="30min", min_island_size=3
        )

        # Only second segment meets minimum size
        assert len(segments) == 1
        assert segments[0].segment_id == 1
        assert segments[0].row_count == 3

    def test_time_as_index(self):
        """Test with time as DataFrame index."""
        times = pd.date_range("2023-01-01", periods=10, freq="5min")
        df = pd.DataFrame({"value": range(10)}, index=times)
        df.index.name = "timestamp"

        result_df, segments = detect_islands_with_metadata(df, time_col="timestamp")

        assert "segment_id" in result_df.columns
        assert len(segments) == 1


class TestFilterValidSegments:
    """Test segment filtering functionality."""

    def test_filter_by_duration(self):
        """Test filtering segments by duration."""
        from core import TimeSegment

        # Create test data
        df = pd.DataFrame(
            {
                "segment_id": [0, 0, 0, 1, 1, 2, 2, 2, 2, 2],
                "value": range(10),
            }
        )

        segments = [
            TimeSegment(
                segment_id=0,
                start_time=datetime(2023, 1, 1, 0, 0),
                end_time=datetime(2023, 1, 1, 0, 30),  # 30 min
                duration_minutes=30,
                row_count=3,
            ),
            TimeSegment(
                segment_id=1,
                start_time=datetime(2023, 1, 1, 1, 0),
                end_time=datetime(2023, 1, 1, 1, 10),  # 10 min
                duration_minutes=10,
                row_count=2,
            ),
            TimeSegment(
                segment_id=2,
                start_time=datetime(2023, 1, 1, 2, 0),
                end_time=datetime(2023, 1, 1, 4, 0),  # 2 hours
                duration_minutes=120,
                row_count=5,
            ),
        ]

        # Filter for segments >= 1 hour
        filtered_df, valid_segments = filter_valid_segments(
            df, segments, min_duration_hours=1.0, min_rows=3
        )

        # Only segment 2 meets criteria
        assert len(valid_segments) == 1
        assert valid_segments[0].segment_id == 2
        assert len(filtered_df) == 5
        assert filtered_df["segment_id"].unique().tolist() == [2]


class TestMergeSmallGaps:
    """Test gap merging functionality."""

    def test_merge_nearby_segments(self):
        """Test merging segments with small gaps."""
        df = pd.DataFrame(
            {
                "time": pd.to_datetime(
                    [
                        "2023-01-01 00:00:00",
                        "2023-01-01 00:05:00",
                        # 20 min gap (will be separate with 15min threshold)
                        "2023-01-01 00:25:00",
                        "2023-01-01 00:30:00",
                        # 10 min gap
                        "2023-01-01 00:40:00",
                    ]
                ),
                "value": range(5),
                "segment_id": [0, 0, 1, 1, 2],  # Original segments
            }
        )

        # Merge with 30min gap threshold
        result = merge_small_gaps(df, time_col="time", merge_gap="30min", original_gap="15min")

        # All should be merged into one segment
        assert result["segment_id"].nunique() == 1
        assert "sub_segment_id" in result.columns
        assert result["sub_segment_id"].tolist() == [0, 0, 1, 1, 2]


@pytest.mark.parametrize(
    "gap_str,expected_td",
    [
        ("15min", timedelta(minutes=15)),
        ("1h", timedelta(hours=1)),
        ("1d", timedelta(days=1)),
        ("30s", timedelta(seconds=30)),
    ],
)
def test_gap_string_parsing(gap_str: str, expected_td: timedelta):
    """Test that gap strings are parsed correctly."""
    times = pd.Series(
        [
            pd.Timestamp("2023-01-01 00:00:00"),
            pd.Timestamp("2023-01-01 00:00:00") + expected_td + timedelta(seconds=1),
        ]
    )

    result = label_islands(times, gap=gap_str)
    # Should detect as separate islands since gap > threshold
    assert result.tolist() == [0, 1]
