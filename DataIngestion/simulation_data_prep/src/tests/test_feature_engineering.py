#!/usr/bin/env -S uv run --isolated
# /// script
# dependencies = ["polars", "pytest", "numpy", "pandas"]
# ///

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal
import numpy as np
import pytest
import logging
from pathlib import Path
from datetime import datetime, date, time, timedelta
import sys

# --- Adjust sys.path to allow importing from src ---
current_file_dir = Path(__file__).resolve().parent
src_dir = current_file_dir.parent
project_root_parent = src_dir.parent
if str(project_root_parent) not in sys.path:
    sys.path.insert(0, str(project_root_parent))
# --- End path adjustment ---

# Import the function to test
from src.feature_engineering import create_time_features

# --- Setup (if needed, e.g., logger) ---
logger = logging.getLogger(__name__)
# You could add basicConfig here if detailed logs are needed for debugging tests
# logging.basicConfig(level=logging.DEBUG)

# --- Test Fixtures (if needed) ---
@pytest.fixture
def sample_time_df() -> pl.DataFrame:
    """Creates a sample DataFrame with specific timestamps."""
    timestamps = [
        datetime(2023, 1, 1, 0, 0, 0),  # Start of year, midnight
        datetime(2023, 3, 15, 12, 30, 0), # Mid-month, noonish (Wednesday)
        datetime(2023, 12, 31, 23, 59, 59),# End of year, near midnight (Sunday)
        datetime(2024, 2, 29, 6, 0, 0),  # Leap year day, morning (Thursday)
    ]
    return pl.DataFrame({"event_time": timestamps})

# --- Unit Tests for create_time_features ---

def test_create_time_features_basic(sample_time_df):
    """Tests the correctness of basic time feature generation."""
    time_col_name = "event_time"
    out_df = create_time_features(sample_time_df, time_col=time_col_name)

    # Expected outputs for the specific timestamps
    # Timestamp 1: 2023-01-01 00:00:00 (Sun - DoW=7 in Polars, 6 for calc; Month=1; DoY=1)
    # Timestamp 2: 2023-03-15 12:30:00 (Wed - DoW=3; Month=3; DoY=74)
    # Timestamp 3: 2023-12-31 23:59:59 (Sun - DoW=7; Month=12; DoY=365)
    # Timestamp 4: 2024-02-29 06:00:00 (Thu - DoW=4; Month=2; DoY=60)
    expected_hours = pl.Series("hour_of_day", [0, 12, 23, 6])
    expected_dow = pl.Series("day_of_week", [7, 3, 7, 4]) # Polars: 1-7 Mon-Sun
    expected_month = pl.Series("month_of_year", [1, 3, 12, 2])
    expected_doy = pl.Series("day_of_year", [1, 74, 365, 60])
    
    # Expected cyclical features (approximate)
    # Hour Sin/Cos
    exp_hr_sin = np.sin(2 * np.pi * np.array([0, 12, 23, 6]) / 24.0)
    exp_hr_cos = np.cos(2 * np.pi * np.array([0, 12, 23, 6]) / 24.0)
    # DoW Sin/Cos (using 0-6 for calculation)
    exp_dow_sin = np.sin(2 * np.pi * np.array([6, 2, 6, 3]) / 7.0)
    exp_dow_cos = np.cos(2 * np.pi * np.array([6, 2, 6, 3]) / 7.0)
    # Month Sin/Cos
    exp_month_sin = np.sin(2 * np.pi * np.array([1, 3, 12, 2]) / 12.0)
    exp_month_cos = np.cos(2 * np.pi * np.array([1, 3, 12, 2]) / 12.0)

    # Assertions
    assert "hour_of_day" in out_df.columns
    assert "day_of_week" in out_df.columns
    assert "month_of_year" in out_df.columns
    assert "day_of_year" in out_df.columns
    assert "hour_sin" in out_df.columns
    assert "hour_cos" in out_df.columns
    assert "dayofweek_sin" in out_df.columns
    assert "dayofweek_cos" in out_df.columns
    assert "month_sin" in out_df.columns
    assert "month_cos" in out_df.columns

    assert_series_equal(out_df["hour_of_day"], expected_hours, check_dtypes=False)
    assert_series_equal(out_df["day_of_week"], expected_dow, check_dtypes=False)
    assert_series_equal(out_df["month_of_year"], expected_month, check_dtypes=False)
    assert_series_equal(out_df["day_of_year"], expected_doy, check_dtypes=False)
    
    # Check cyclical features approximately
    assert_series_equal(out_df["hour_sin"], pl.Series("hour_sin", exp_hr_sin), rtol=1e-6)
    assert_series_equal(out_df["hour_cos"], pl.Series("hour_cos", exp_hr_cos), rtol=1e-6)
    assert_series_equal(out_df["dayofweek_sin"], pl.Series("dayofweek_sin", exp_dow_sin), rtol=1e-6)
    assert_series_equal(out_df["dayofweek_cos"], pl.Series("dayofweek_cos", exp_dow_cos), rtol=1e-6)
    assert_series_equal(out_df["month_sin"], pl.Series("month_sin", exp_month_sin), rtol=1e-6)
    assert_series_equal(out_df["month_cos"], pl.Series("month_cos", exp_month_cos), rtol=1e-6)

def test_create_time_features_empty_df():
    """Tests behavior with an empty input DataFrame."""
    empty_df = pl.DataFrame({"event_time": []}, schema={"event_time": pl.Datetime})
    out_df = create_time_features(empty_df, time_col="event_time")
    
    # Expect an empty DataFrame but with all the feature columns added
    assert out_df.is_empty()
    expected_cols = [
        "event_time", "hour_of_day", "day_of_week", "day_of_month", 
        "month_of_year", "year", "day_of_year", "week_of_year",
        "hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos",
        "month_sin", "month_cos"
    ]
    assert sorted(out_df.columns) == sorted(expected_cols)

def test_create_time_features_string_input_valid():
    """Tests if the function correctly handles string inputs that can be parsed."""
    timestamps_str = [
        "2023-01-01T00:00:00", 
        "2023-03-15T12:30:00"
    ]
    df_str = pl.DataFrame({"time_str": timestamps_str})
    out_df = create_time_features(df_str, time_col="time_str")

    # Check if conversion happened and a basic feature is correct
    assert out_df["time_str"].dtype == pl.Datetime # Check conversion
    assert "hour_of_day" in out_df.columns
    expected_hours = pl.Series("hour_of_day", [0, 12])
    assert_series_equal(out_df["hour_of_day"], expected_hours, check_dtypes=False)

def test_create_time_features_string_input_invalid():
    """Tests if the function handles unparseable string inputs gracefully."""
    timestamps_str = [
        "2023-01-01T00:00:00", 
        "Invalid Date String", # Unparseable string
        "2023-03-15T12:30:00"
    ]
    df_str = pl.DataFrame({"time_str": timestamps_str})
    # The function logs errors and returns the original df if conversion fails
    # (based on the implementation seen earlier)
    out_df = create_time_features(df_str, time_col="time_str")
    
    # Assert that the original DataFrame is returned without the time features
    # because the conversion would have failed.
    assert_frame_equal(out_df, df_str) 
    # Alternatively, check that feature columns were NOT added
    assert "hour_of_day" not in out_df.columns

def test_create_time_features_missing_time_col():
    """Tests that the function returns the original df if time_col is missing."""
    df_no_time = pl.DataFrame({"value": [1, 2, 3]})
    # Function should log an error and return the input df unchanged
    out_df = create_time_features(df_no_time, time_col="non_existent_time")
    assert_frame_equal(out_df, df_no_time)
    assert "hour_of_day" not in out_df.columns

def test_create_time_features_wrong_time_col_type():
    """Tests handling of a time column with an unconvertible numeric type."""
    # Use integer seconds since epoch as an example non-datetime, non-string type
    epoch_seconds = [
        int(datetime(2023, 1, 1, 0, 0, 0).timestamp()),
        int(datetime(2023, 3, 15, 12, 30, 0).timestamp()),
    ]
    df_int_time = pl.DataFrame({"epoch_secs": epoch_seconds})
    # Function will try str.to_datetime() on integers, which fails.
    # It should log an error and return the input df unchanged.
    out_df = create_time_features(df_int_time, time_col="epoch_secs")
    assert_frame_equal(out_df, df_int_time)
    assert "hour_of_day" not in out_df.columns

def test_create_time_features_custom_col_name(sample_time_df):
    """Tests using a non-default time column name."""
    # sample_time_df fixture already uses "event_time"
    time_col_name = "event_time"
    out_df = create_time_features(sample_time_df, time_col=time_col_name)
    
    # Basic check to ensure features were created
    assert "hour_of_day" in out_df.columns
    assert out_df.shape[0] == sample_time_df.shape[0] # Row count shouldn't change
    # Verify one feature for correctness
    expected_hours = pl.Series("hour_of_day", [0, 12, 23, 6])
    assert_series_equal(out_df["hour_of_day"], expected_hours, check_dtypes=False)

if __name__ == "__main__":
    import pytest
    import sys
    # Exit with the appropriate code from pytest
    raise SystemExit(pytest.main([__file__] + sys.argv[1:]))
