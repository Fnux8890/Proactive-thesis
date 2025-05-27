"""Island detection for time series data with gap identification."""

from __future__ import annotations

import logging

import pandas as pd

from .models import TimeSegment

logger = logging.getLogger(__name__)


def label_islands(ts: pd.Series, gap: str | pd.Timedelta = "15min") -> pd.Series:
    """Label continuous time segments (islands) based on temporal gaps.

    This function identifies contiguous segments in a time series by detecting
    gaps larger than a specified threshold. Each segment gets a unique integer ID.

    Args:
        ts: Time series data (must be datetime type)
        gap: Maximum allowed gap between consecutive points.
             Can be a string like '15min' or a pd.Timedelta object.

    Returns:
        Series of integer labels identifying each island/segment

    Example:
        >>> times = pd.to_datetime(['2023-01-01 00:00', '2023-01-01 00:05',
        ...                         '2023-01-01 01:00', '2023-01-01 01:05'])
        >>> labels = label_islands(pd.Series(times), gap='30min')
        >>> print(labels.tolist())
        [0, 0, 1, 1]
    """
    if len(ts) == 0:
        return pd.Series([], dtype="int64")

    # Convert gap to Timedelta if string
    gap_td = pd.Timedelta(gap) if isinstance(gap, str) else gap

    # Calculate time differences
    time_diffs = ts.diff()

    # Identify where new islands start (gap > threshold)
    # First element is always start of first island
    new_island = (time_diffs > gap_td) | ts.isna()

    # Cumulative sum gives unique ID to each island
    island_ids = new_island.cumsum()

    return island_ids.astype("int64")


def detect_islands_with_metadata(
    df: pd.DataFrame,
    time_col: str,
    gap: str | pd.Timedelta = "15min",
    min_island_size: int = 10,
) -> tuple[pd.DataFrame, list[TimeSegment]]:
    """Detect islands and add segment metadata to DataFrame.

    Args:
        df: Input DataFrame
        time_col: Name of the time column
        gap: Maximum allowed gap between consecutive points
        min_island_size: Minimum number of points for valid island

    Returns:
        Tuple of:
        - DataFrame with added 'segment_id' column
        - List of TimeSegment objects with metadata
    """
    df = df.copy()

    # Ensure time column is datetime
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col])
        time_series = df[time_col]
    elif df.index.name == time_col:
        df.index = pd.to_datetime(df.index)
        time_series = pd.Series(df.index, index=df.index)
    else:
        raise ValueError(f"Time column '{time_col}' not found in DataFrame")

    # Label islands
    df["segment_id"] = label_islands(time_series, gap=gap)

    # Collect segment metadata
    segments: list[TimeSegment] = []

    for segment_id in df["segment_id"].unique():
        segment_mask = df["segment_id"] == segment_id
        segment_df = df[segment_mask]

        if len(segment_df) < min_island_size:
            logger.warning(
                f"Segment {segment_id} has only {len(segment_df)} points, "
                f"below minimum of {min_island_size}"
            )
            continue

        # Get time data
        segment_times = segment_df[time_col] if time_col in segment_df.columns else segment_df.index

        start_time = segment_times.min()
        end_time = segment_times.max()
        duration_minutes = (end_time - start_time).total_seconds() / 60

        # Check for internal gaps
        time_diffs = segment_times.diff()
        gap_td = pd.Timedelta(gap) if isinstance(gap, str) else gap
        has_gaps = (time_diffs > gap_td).any()

        if has_gaps:
            gap_duration = time_diffs[time_diffs > gap_td].sum().total_seconds() / 60
        else:
            gap_duration = 0.0

        segment = TimeSegment(
            segment_id=int(segment_id),
            start_time=start_time,
            end_time=end_time,
            duration_minutes=duration_minutes,
            row_count=len(segment_df),
            has_gaps=has_gaps,
            gap_duration_minutes=gap_duration,
        )

        segments.append(segment)

    logger.info(
        f"Detected {len(segments)} valid segments "
        f"(filtered {df['segment_id'].nunique() - len(segments)} small segments)"
    )

    return df, segments


def filter_valid_segments(
    df: pd.DataFrame,
    segments: list[TimeSegment],
    min_duration_hours: float = 1.0,
    min_rows: int = 10,
) -> tuple[pd.DataFrame, list[TimeSegment]]:
    """Filter DataFrame to keep only valid segments.

    Args:
        df: DataFrame with segment_id column
        segments: List of TimeSegment metadata
        min_duration_hours: Minimum segment duration in hours
        min_rows: Minimum number of rows per segment

    Returns:
        Tuple of filtered DataFrame and valid segments
    """
    valid_segments = [
        seg
        for seg in segments
        if seg.duration_minutes >= min_duration_hours * 60 and seg.row_count >= min_rows
    ]

    valid_segment_ids = {seg.segment_id for seg in valid_segments}

    filtered_df = df[df["segment_id"].isin(valid_segment_ids)].copy()

    logger.info(
        f"Filtered to {len(valid_segments)} valid segments "
        f"({len(filtered_df)} rows from original {len(df)} rows)"
    )

    return filtered_df, valid_segments


def merge_small_gaps(
    df: pd.DataFrame,
    time_col: str,
    merge_gap: str | pd.Timedelta = "30min",
) -> pd.DataFrame:
    """Merge segments that have small gaps between them.

    This is useful when you want to be more lenient about gaps and
    merge nearby segments into larger continuous blocks.

    Args:
        df: DataFrame with segment_id column
        time_col: Name of time column
        merge_gap: Gap size below which segments are merged
        original_gap: Original gap size used for detection

    Returns:
        DataFrame with updated segment_id values
    """
    df = df.copy()

    # First detect with larger gap
    df["segment_id_merged"] = label_islands(
        df[time_col] if time_col in df.columns else pd.Series(df.index), gap=merge_gap
    )

    # Keep original fine-grained segments as sub-segments
    df["sub_segment_id"] = df["segment_id"]
    df["segment_id"] = df["segment_id_merged"]

    logger.info(
        f"Merged segments: {df['sub_segment_id'].nunique()} sub-segments "
        f"into {df['segment_id'].nunique()} main segments"
    )

    return df
