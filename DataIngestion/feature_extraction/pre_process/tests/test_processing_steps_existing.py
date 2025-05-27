import sys
from pathlib import Path

import pandas as pd
from pandas.testing import assert_series_equal
from processing_steps import OutlierHandler

# Add the pre_process directory to sys.path to allow importing OutlierHandler
# This assumes the tests directory is d:\GitKraken\Proactive-thesis\DataIngestion\feature_extraction\tests
# and processing_steps.py is in d:\GitKraken\Proactive-thesis\DataIngestion\feature_extraction\pre_process
PRE_PROCESS_DIR = Path(__file__).parent.parent / "pre_process"
sys.path.insert(0, str(PRE_PROCESS_DIR))

# Basic outlier rules for testing. These rules would apply to 'some_other_col'.
# For 'par_synth_umol', the 'do_not_clip_columns' should override any potential rules.
TEST_OUTLIER_RULES = [
    {"column": "some_other_col", "min_value": 0, "max_value": 100, "clip": True},
    # A hypothetical rule for par_synth_umol that *should be ignored* due to do_not_clip_columns
    {"column": "par_synth_umol", "min_value": 0, "max_value": 2500, "clip": True},
]


def test_outlier_handler_do_not_clip():
    """
    Tests that OutlierHandler does not clip columns specified in 'do_not_clip_columns',
    even if there are rules defined for them.
    """
    df_input = pd.DataFrame(
        {
            "par_synth_umol": [
                -999.0,
                0.0,
                2000.0,
                100000.0,
            ],  # Values outside hypothetical rule [0, 2500]
            "some_other_col": [-10, 50, 150, 99],
        }
    )

    # Configuration for OutlierHandler, specifying par_synth_umol not to be clipped.
    # This 'rules_cfg_dict' mimics what would be loaded from preprocess_config.json.
    rules_cfg = {
        "do_not_clip_columns": ["par_synth_umol"],
        # Other parts of outlier_configs from preprocess_config.json (not directly used by this test logic)
        "default_method": "iqr",
        "default_threshold": 1.5,
        "column_specific": {},
    }

    handler = OutlierHandler(rules=TEST_OUTLIER_RULES, rules_cfg_dict=rules_cfg)
    df_output = handler.clip_outliers(df_input.copy())

    # Assert that 'par_synth_umol' remains unchanged because it's in do_not_clip_columns
    assert_series_equal(
        df_input["par_synth_umol"],
        df_output["par_synth_umol"],
        check_dtype=False,
        err_msg="par_synth_umol should not be clipped.",
    )

    # Assert that 'some_other_col' *was* clipped according to TEST_OUTLIER_RULES
    expected_other_col = pd.Series([0.0, 50.0, 100.0, 99.0], name="some_other_col")
    assert_series_equal(
        expected_other_col,
        df_output["some_other_col"],
        check_dtype=False,
        err_msg="some_other_col should be clipped according to its rules.",
    )
