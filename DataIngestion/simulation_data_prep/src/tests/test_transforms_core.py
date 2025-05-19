#!/usr/bin/env -S uv run --isolated
# /// script
# dependencies = ["pytest", "polars", "numpy", "pydantic", "prefect"]
# ///

import pytest
import polars as pl
from polars.testing import assert_frame_equal
import numpy as np
from datetime import datetime, date, timedelta
import logging
import sys
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

# --- Adjust sys.path to allow importing from src and transforms ---
current_file_dir = Path(__file__).resolve().parent # .../tests
src_dir = current_file_dir.parent # .../src
simulation_data_prep_dir = src_dir.parent # .../simulation_data_prep

# Add the simulation_data_prep directory (parent of src and transforms) to the path
if str(simulation_data_prep_dir) not in sys.path:
    sys.path.insert(0, str(simulation_data_prep_dir))
# --- End path adjustment ---

# --- Debug Import ---
import src.config 
print("--- DEBUG: Contents of src.config ---")
try:
    print(dir(src.config))
except Exception as e:
    print(f"Could not print dir(src.config): {e}")
print("--- END DEBUG ---")
# --- End Debug Import ---

# Use imports relative to simulation_data_prep dir (e.g., src.config)
try:
    from transforms.core import transform_features
    # Import all necessary config models used in fixtures
    from src.config import (
        PlantConfig, DataProcessingConfig, GddParameters, GddProfile, 
        DifParameters, FeatureParameters, AdvancedFeatureParameters, 
        ObjectiveFunctionParameters, EnergyPowerRatings, FixedSetpoints, 
        OptimalRange, StressThresholds, ActuatorSummaryParameters, Metadata, 
        OptimalConditions, PhotoperiodParameters, LampGroupDetail, 
        DBConnectionConfig, OutlierDetectionConfig, ImputationConfig, 
        ColumnProcessingLists, DataSegment, SdInduction, HeatDelay, LampGroupPower,
        # New/Updated config models for segment-aware features
        SegmentFeatureConfig, AdvancedFeatureParametersPlaceholder,
        RollingWindowConfig,
        NightStressFlagDetail, TempTarget, VpdTarget 
    )
    # Ensure pydantic imports needed by inline test are present if definition removed
    from pydantic import BaseModel, Field 
except ImportError as e: 
    print(f"Import Error after path adjustment: {e}")
    raise

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Helper Fixtures --- 

@pytest.fixture
def sample_polars_dataframe() -> pl.DataFrame:
    """Creates a sample Polars DataFrame with time and basic sensor data."""
    start_dt = datetime(2014, 3, 1, 0, 0, 0)
    n_periods = 24 * 3 # 3 days of hourly data
    interval = "1h"
    # Calculate end_dt instead of using None
    end_dt = start_dt + timedelta(hours=n_periods - 1)
    base_df = pl.DataFrame({
        "time": pl.datetime_range(start_dt, end_dt, interval=interval, eager=True) # Use calculated end_dt
    })
    # Ensure exactly n_periods if range generation is slightly off
    if len(base_df) != n_periods:
        base_df = base_df.head(n_periods)
        
    df = base_df.with_columns([
        (15 + 10 * np.sin(np.pi * (pl.col("time").dt.ordinal_day() + pl.col("time").dt.hour() / 24.0) / 182.5)).alias("air_temp_c"),
        (70 + 20 * np.sin(np.pi * pl.col("time").dt.hour() / 12.0)).alias("relative_humidity_percent"),
        (pl.lit(None, dtype=pl.Float64)).alias("outside_temp_c"), # Add outside temp for delta testing
        (pl.when(pl.col("time").dt.hour().is_between(6, 18)).then(400 * np.sin(np.pi * (pl.col("time").dt.hour() - 6) / 12.0)).otherwise(0).clip(lower_bound=0) * 2.1).alias("light_intensity_umol"), # Approx umol from W/m2
        (pl.when(pl.col("time").dt.hour().is_between(6, 18)).then(400 * np.sin(np.pi * (pl.col("time").dt.hour() - 6) / 12.0)).otherwise(0).clip(lower_bound=0)).alias("radiation_w_m2"),
        (400 + 100 * np.sin(np.pi * pl.col("time").dt.hour() / 6.0) + np.random.rand(len(base_df)) * 50).alias("co2_measured_ppm"), # Use len(base_df)
        pl.lit(500.0).alias("co2_required_ppm"),
        pl.lit(0).cast(pl.Int8).alias("lamp_status_1"), # Example lamp status
    ])
    # Simulate some NaNs post-imputation (unlikely but for testing)
    # df = df.with_row_count().with_columns(pl.when(pl.col("row_nr") % 10 == 0).then(None).otherwise(pl.col("air_temp_c")).alias("air_temp_c")).drop("row_nr")
    return df

@pytest.fixture
def sample_plant_config() -> PlantConfig:
    """Creates a sample PlantConfig object for testing."""
    # Create minimal nested structures needed by transform_features
    metadata = Metadata(plant_species="test_plant", profile_name="test_prof", version="1.0", description="Test", source_document_id="test_doc")
    gdd_profile = GddProfile(t_base_celsius=10.0, t_cap_celsius=30.0)
    gdd_params = GddParameters(crop_profile="test_gdd", profiles={"test_gdd": gdd_profile})
    dif_params = DifParameters(day_definition="fixed_time", lamp_status_columns=[], fixed_time_day_start_hour=6, fixed_time_day_end_hour=18)
    lamp_groups = {"lamp_status_1": LampGroupDetail(count=10, ppf_umol_s=50, power_kw=0.1)}
    actuator_params = ActuatorSummaryParameters(percent_columns_for_average=[], percent_columns_for_changes=[], binary_columns_for_on_time=[])
    
    # Create valid instances for ObjectiveFunctionParameters fields
    mock_energy = EnergyPowerRatings(ventilation_passive=True, lamp_group_power_kw=LampGroupPower(root={}))
    mock_setpoints = FixedSetpoints() # All fields are optional, so default is fine
    objective_params = ObjectiveFunctionParameters(
        energy_power_ratings_kw=mock_energy, 
        fixed_setpoints=mock_setpoints, 
        optimal_ranges={}
    )
    
    mock_heat_delay = HeatDelay(onset_risk=30, significant_impact=35) # Assuming HeatDelay is needed
    stress_params = StressThresholds(
        heat_delay_night_temp_celsius_sd=mock_heat_delay, 
        low_temp_cam_induction_celsius=5.0
    )
    
    # Add sample optimal conditions
    opt_conditions = OptimalConditions(
        temperature_celsius={"growth": TempTarget(min=18.0, max=25.0)},
        vpd_kpa={"growth": VpdTarget(min=0.8, max=1.2)},
        dli_mol_m2_day={}
    )
    
    # Advanced feature params in PlantConfig might be minimal now, as specifics move to DataProcessingConfig
    adv_feat_params_plant = AdvancedFeatureParameters()
    
    # Need to mock all required fields even if empty/default
    return PlantConfig(
        plant_profile_metadata=metadata,
        gdd_parameters=gdd_params,
        optimal_conditions=opt_conditions, # Use populated optimal conditions
        photoperiod_parameters=PhotoperiodParameters(classification="DN", critical_night_length_hours=10, inductive_night_length_hours_target=11, sd_induction_duration_weeks=SdInduction(value=1), night_interruption_inhibits_flowering=False),
        dif_parameters=dif_params,
        stress_thresholds=stress_params, # Use valid instance
        actuator_summary_parameters=actuator_params,
        feature_parameters=FeatureParameters(), # Keep this as FeatureParameters() - segment config overrides
        objective_function_parameters=objective_params, # Use valid instance
        advanced_feature_parameters=adv_feat_params_plant, # Keep this as default - segment config overrides
        lamp_groups=lamp_groups,
        column_uuid_mapping={},
        data_frequency_minutes=60 # Set data frequency to 60 minutes (since interval="1h")
    )

@pytest.fixture
def sample_data_proc_config() -> DataProcessingConfig:
    """Creates a sample DataProcessingConfig object."""
    
    # Define feature configurations for the global/default case
    global_fp = FeatureParameters(
        delta_cols={"temp_delta_in_out": ["air_temp_c", "outside_temp_c"]},
        rate_of_change_cols=["air_temp_c"],
        rolling_average_cols={
            "air_temp_c": RollingWindowConfig(window_minutes=180, min_periods=2) 
        }
    )
    global_afp = AdvancedFeatureParametersPlaceholder(
        rolling_std_dev_cols={
            "relative_humidity_percent": RollingWindowConfig(window_minutes=240, min_periods=3)
        },
        lag_features={
            "co2_measured_ppm": 180 # Lag by 180 minutes (will convert to 3 periods)
        },
        availability_flags_for_cols=["air_temp_c", "missing_col_test"], # Test available and missing source
        night_stress_flags={
            "low_temp_stress": NightStressFlagDetail(
                input_temp_col="air_temp_c", 
                threshold_config_key="low_temp_cam_induction_celsius", # Points to float in PlantConfig.stress_thresholds
                stress_type="low"
            ),
            "high_temp_stress": NightStressFlagDetail(
                input_temp_col="air_temp_c",
                threshold_config_key="heat_delay_night_temp_celsius_sd", # Points to HeatDelay object
                threshold_sub_key="significant_impact", # Points to field within HeatDelay
                stress_type="high"
            )
        }
    )
    
    global_segment_config = SegmentFeatureConfig(
        feature_parameters=global_fp,
        advanced_feature_parameters=global_afp,
        active_optimal_condition_keys=["temperature_celsius"] # Only activate temp optimal range features
    )

    # Assume default ColumnProcessingLists is sufficient for this test
    return DataProcessingConfig(
        db_connection=DBConnectionConfig(), # Use defaults
        data_segments=[], # Not used by transform_features directly
        outlier_detection=OutlierDetectionConfig(), # Use defaults
        imputation=ImputationConfig(), # Use defaults
        column_lists=ColumnProcessingLists(), # Use defaults
        global_feature_config=global_segment_config,
        segment_feature_configs={} # No specific segments defined for this default test config
    )

# --- Test Cases --- 

def test_transform_features_basic_run(sample_polars_dataframe): # Removed fixtures
    """Test if the function runs without errors and adds some expected columns."""
    logger.info("--- Starting Test: Basic Run (Inline Config) --- ")
    input_df = sample_polars_dataframe

    # --- Define Configs Inline --- 
    # Minimal PlantConfig needed for this test
    plant_cfg = PlantConfig(
        plant_profile_metadata=Metadata(plant_species="inline_plant", profile_name="inline_prof", version="1.0", description="Test", source_document_id="inline_doc"),
        gdd_parameters=GddParameters(crop_profile="test_gdd", profiles={"test_gdd": GddProfile(t_base_celsius=10.0, t_cap_celsius=30.0)}),
        optimal_conditions=OptimalConditions(
            temperature_celsius={"growth": TempTarget(min=18.0, max=25.0)},
            vpd_kpa={}, dli_mol_m2_day={}),
        photoperiod_parameters=PhotoperiodParameters(classification="DN", critical_night_length_hours=10, inductive_night_length_hours_target=11, sd_induction_duration_weeks=SdInduction(value=1), night_interruption_inhibits_flowering=False),
        dif_parameters=DifParameters(day_definition="fixed_time", lamp_status_columns=[], fixed_time_day_start_hour=6, fixed_time_day_end_hour=18),
        stress_thresholds=StressThresholds(
            heat_delay_night_temp_celsius_sd=HeatDelay(onset_risk=30, significant_impact=35), 
            low_temp_cam_induction_celsius=5.0
        ),
        actuator_summary_parameters=ActuatorSummaryParameters(percent_columns_for_average=[], percent_columns_for_changes=[], binary_columns_for_on_time=[]),
        feature_parameters=FeatureParameters(), # Base, overridden by segment/global
        objective_function_parameters=ObjectiveFunctionParameters(
            energy_power_ratings_kw=EnergyPowerRatings(ventilation_passive=True, lamp_group_power_kw=LampGroupPower(root={})),
            fixed_setpoints=FixedSetpoints(), 
            optimal_ranges={}
        ),
        advanced_feature_parameters=AdvancedFeatureParameters(), # Base, overridden by segment/global
        lamp_groups={"lamp_status_1": LampGroupDetail(count=10, ppf_umol_s=50, power_kw=0.1)},
        data_frequency_minutes=60
    )

    # Minimal DataProcessingConfig with global config needed
    global_fp = FeatureParameters(
        delta_cols={"temp_delta_in_out": ["air_temp_c", "outside_temp_c"]},
        rate_of_change_cols=["air_temp_c"],
        rolling_average_cols={
            "air_temp_c": { "window_minutes": 180, "min_periods": 2 } 
        }
    )
    global_afp = AdvancedFeatureParametersPlaceholder(
        rolling_std_dev_cols={
            "relative_humidity_percent": { "window_minutes": 240, "min_periods": 3 }
        },
        lag_features={"co2_measured_ppm": 180},
        availability_flags_for_cols=["air_temp_c"],
        night_stress_flags={
            "low_temp_stress": NightStressFlagDetail(input_temp_col="air_temp_c", threshold_config_key="low_temp_cam_induction_celsius", stress_type="low"),
            "high_temp_stress": NightStressFlagDetail(input_temp_col="air_temp_c",threshold_config_key="heat_delay_night_temp_celsius_sd", threshold_sub_key="significant_impact", stress_type="high")
        }
    )
    global_segment_config = SegmentFeatureConfig(
        feature_parameters=global_fp,
        advanced_feature_parameters=global_afp,
        active_optimal_condition_keys=["temperature_celsius"] 
    )
    data_proc_cfg = DataProcessingConfig(
        db_connection=DBConnectionConfig(), 
        data_segments=[], 
        outlier_detection=OutlierDetectionConfig(), 
        imputation=ImputationConfig(), 
        column_lists=ColumnProcessingLists(), 
        global_feature_config=global_segment_config,
        segment_feature_configs={}
    )
    # --- End Inline Configs ---
    
    result_df = transform_features(input_df.clone(), plant_cfg, data_proc_cfg, segment_name="TestSegmentInline")
    
    assert isinstance(result_df, pl.DataFrame)
    assert result_df.shape[0] == input_df.shape[0]
    assert result_df.shape[1] > input_df.shape[1]
    
    # Check for a few expected feature columns (names depend on calculator functions)
    # Update based on the inline global_segment_config
    expected_cols = [
        "time", "date", "hour_of_day", "day_of_week", "hour_sin", "hour_cos", 
        "vpd_kpa", 
        "supplemental_ppf_umol_s", "lamp_power_kw", "ppfd_total", 
        "DLI_mol_m2_d", 
        "GDD_daily", "GDD_cumulative", 
        "DIF_daily", 
        "CO2_diff_ppm", 
        "air_temp_c_RoC_per_s", 
        "air_temp_c_rolling_avg_180m", 
        "relative_humidity_percent_rolling_std_240m",
        "co2_measured_ppm_lag_3p", 
        "air_temp_c_is_available", 
        "air_temp_c_night_low_stress_low_temp_cam_induction_celsius_direct_low",
        "air_temp_c_night_high_stress_heat_delay_night_temp_celsius_sd_significant_impact_high",
        "air_temp_c_dist_opt_mid",
        "air_temp_c_in_opt_range",
        "temp_delta_in_out" 
    ]
    missing_cols = [col for col in expected_cols if col not in result_df.columns]
    assert not missing_cols, f"Missing expected feature columns: {missing_cols}"
    logger.info(f"Output columns confirmed (first few): {result_df.columns[:15]}")
    logger.info("Result: PASSED")

def test_transform_features_missing_inputs(sample_polars_dataframe, sample_plant_config, sample_data_proc_config):
    """Test that features are skipped if input columns are missing."""
    logger.info("--- Starting Test: Missing Inputs --- ")
    # Drop columns needed for VPD and CO2 difference
    input_df = sample_polars_dataframe.drop(["relative_humidity_percent", "co2_required_ppm"])
    plant_cfg = sample_plant_config
    data_proc_cfg = sample_data_proc_config

    result_df = transform_features(input_df.clone(), plant_cfg, data_proc_cfg, segment_name="TestMissing")

    assert "vpd_kpa" not in result_df.columns, "VPD should not be calculated if RH is missing."
    assert "CO2_diff_ppm" not in result_df.columns, "CO2 diff should not be calculated if required CO2 is missing."
    assert "air_temp_c_RoC_per_s" in result_df.columns, "RoC for air_temp_c should still be calculated."
    logger.info("Verified that features with missing inputs were skipped.")
    logger.info("Result: PASSED")

def test_transform_features_empty_input(sample_plant_config, sample_data_proc_config):
    """Test behavior with an empty input DataFrame."""
    logger.info("--- Starting Test: Empty Input --- ")
    input_df = pl.DataFrame({"time": pl.Series([], dtype=pl.Datetime)}) # Empty frame with time column
    plant_cfg = sample_plant_config
    data_proc_cfg = sample_data_proc_config

    result_df = transform_features(input_df.clone(), plant_cfg, data_proc_cfg, segment_name="TestEmpty")
    
    assert result_df.is_empty()
    logger.info("Empty input DataFrame resulted in empty output DataFrame, as expected.")
    logger.info("Result: PASSED")

# --- New Tests for Segment Logic and Specific Features ---

def test_transform_features_segment_specific(sample_polars_dataframe, sample_plant_config, sample_data_proc_config):
    """Test that segment-specific configurations override global ones."""
    logger.info("--- Starting Test: Segment Specific Config ---")
    input_df = sample_polars_dataframe
    plant_cfg = sample_plant_config
    # Modify the data proc config to add a segment-specific config
    data_proc_cfg = sample_data_proc_config.model_copy(deep=True)
    
    segment_fp = FeatureParameters(rate_of_change_cols=["relative_humidity_percent"]) # Only RH RoC
    segment_afp = AdvancedFeatureParametersPlaceholder(availability_flags_for_cols=["relative_humidity_percent"]) # Only RH availability
    segment_config = SegmentFeatureConfig(feature_parameters=segment_fp, advanced_feature_parameters=segment_afp)
    data_proc_cfg.segment_feature_configs["SegmentA"] = segment_config

    # Run for SegmentA - should use segment config
    result_df_a = transform_features(input_df.clone(), plant_cfg, data_proc_cfg, segment_name="SegmentA")
    # Assert based on SegmentA config
    assert "relative_humidity_percent_RoC_per_s" in result_df_a.columns
    assert "air_temp_c_RoC_per_s" not in result_df_a.columns # This was global
    assert "relative_humidity_percent_is_available" in result_df_a.columns
    assert "air_temp_c_is_available" not in result_df_a.columns # This was global
    logger.info("SegmentA processing used segment-specific config as expected.")

    # Run for SegmentB - should use global config
    result_df_b = transform_features(input_df.clone(), plant_cfg, data_proc_cfg, segment_name="SegmentB")
    # Assert based on global config (from fixture)
    assert "relative_humidity_percent_RoC_per_s" not in result_df_b.columns # Not in global
    assert "air_temp_c_RoC_per_s" in result_df_b.columns
    assert "relative_humidity_percent_is_available" not in result_df_b.columns # Not in global availability list
    assert "air_temp_c_is_available" in result_df_b.columns
    logger.info("SegmentB processing used global config as expected.")
    logger.info("Result: PASSED")

def test_rolling_features_min_periods_logic(sample_plant_config, sample_data_proc_config):
    """Test min_periods logic in rolling calculations."""
    logger.info("--- Starting Test: Rolling Min Periods --- ")
    # Create specific data: 5 hourly points, ask for 3h window (180m) with min_periods=3
    start_dt = datetime(2024, 1, 1, 0) 
    df = pl.DataFrame({
        "time": pl.datetime_range(start_dt, start_dt + timedelta(hours=4), interval="1h", eager=True),
        "value": [1.0, 2.0, None, 4.0, 5.0] # Null in the middle
    })
    plant_cfg = sample_plant_config # Needs data_frequency_minutes=60
    data_proc_cfg = sample_data_proc_config.model_copy(deep=True)
    
    # Configure rolling average with min_periods=3 for a 180min window
    data_proc_cfg.global_feature_config.feature_parameters.rolling_average_cols = {
        "value": { "window_minutes": 180, "min_periods": 3 }
    }
    # Clear other features to focus the test
    data_proc_cfg.global_feature_config.feature_parameters.rate_of_change_cols = []
    data_proc_cfg.global_feature_config.advanced_feature_parameters = AdvancedFeatureParametersPlaceholder()

    result_df = transform_features(df, plant_cfg, data_proc_cfg, segment_name="MinPeriodTest")
    
    expected_rolling_avg_col = "value_rolling_avg_180m"
    assert expected_rolling_avg_col in result_df.columns
    # Window definition: 180m = 3 hours. Polars `rolling` period is closed on the right by default?
    # Let's re-check Polars `rolling` `closed` default. It's 'right'. So, window ending at time `t` includes `(t-period, t]`.
    # For `closed='left'` (which we used in calculator): `[t-period, t)`
    # time[0] (00:00): Window is [-3h, 0h). Empty -> NaN
    # time[1] (01:00): Window is [-2h, 1h). Includes time[0]. 1 point -> NaN (min_periods=3)
    # time[2] (02:00): Window is [-1h, 2h). Includes time[0], time[1]. 2 points -> NaN (min_periods=3)
    # time[3] (03:00): Window is [0h, 3h). Includes time[0], time[1], time[2](null). 2 valid points -> NaN (min_periods=3)
    # time[4] (04:00): Window is [1h, 4h). Includes time[1], time[2](null), time[3]. 2 valid points -> NaN (min_periods=3)
    # Let's adjust input data or min_periods to get a valid value
    # Try min_periods=2
    data_proc_cfg.global_feature_config.feature_parameters.rolling_average_cols["value"] = { "window_minutes": 180, "min_periods": 2 }
    result_df = transform_features(df, plant_cfg, data_proc_cfg, segment_name="MinPeriodTest2")
    expected_values = [None, None, 1.5, 1.5, 3.0] # [None, None, mean(1,2), mean(1,2), mean(2,4)] - check calculation
    # time[2] includes [0, 1]. Mean(1,2) = 1.5
    # time[3] includes [0, 1, 2(null)]. Mean(1,2) = 1.5
    # time[4] includes [1, 2(null), 3]. Mean(2,4) = 3.0

    assert result_df[expected_rolling_avg_col].to_list() == expected_values, \
        f"Expected rolling avg: {expected_values}, Got: {result_df[expected_rolling_avg_col].to_list()}"
    logger.info("Rolling average with min_periods=2 produced expected NaNs and values.")
    logger.info("Result: PASSED")

def test_availability_flags_generation(sample_polars_dataframe, sample_plant_config, sample_data_proc_config):
    """Test the generation of _is_available flags."""
    logger.info("--- Starting Test: Availability Flags --- ")
    # Make a column partially null
    input_df = sample_polars_dataframe.with_columns(
        pl.when(pl.col("time").dt.day() == 1).then(None).otherwise(pl.col("air_temp_c")).alias("air_temp_c")
    )
    plant_cfg = sample_plant_config
    data_proc_cfg = sample_data_proc_config # Uses global config which asks for air_temp_c_is_available

    result_df = transform_features(input_df.clone(), plant_cfg, data_proc_cfg, segment_name="AvailabilityTest")

    flag_col = "air_temp_c_is_available"
    assert flag_col in result_df.columns
    assert result_df[flag_col].dtype == pl.Int8
    # Check values: 0 where original was null (day 1), 1 otherwise
    expected_flags_expr = pl.when(input_df["time"].dt.day() == 1).then(0).otherwise(1).cast(pl.Int8)
    # Evaluate the expression on the input_df to get the expected Series/DataFrame
    expected_flags_df = input_df.select(expected_flags_expr.alias(flag_col))
    assert_frame_equal(result_df.select(flag_col), expected_flags_df)
    
    # Check flag for a column not present in input
    missing_flag_col = "missing_col_test_is_available"
    # Currently core.py logs a warning and skips the column. Check it's NOT present.
    assert missing_flag_col not in result_df.columns
    logger.info("Availability flags generated correctly for present and missing source columns.")
    logger.info("Result: PASSED")

def test_night_stress_flags_logic(sample_plant_config, sample_data_proc_config):
    """Test high and low night stress flag logic."""
    logger.info("--- Starting Test: Night Stress Flags --- ")
    # Config (from fixture) uses air_temp_c, night=18:00-06:00, 
    # low_thresh=5.0, high_thresh=35.0
    plant_cfg = sample_plant_config 
    data_proc_cfg = sample_data_proc_config # Uses global config
    low_thresh = plant_cfg.stress_thresholds.low_temp_cam_induction_celsius
    high_thresh = plant_cfg.stress_thresholds.heat_delay_night_temp_celsius_sd.significant_impact
    
    # Create data spanning day/night with temps crossing thresholds
    times = [
        datetime(2024, 1, 1, 4), # Night, Low Temp (4 < 5)
        datetime(2024, 1, 1, 5), # Night, OK Temp (10)
        datetime(2024, 1, 1, 10), # Day, Low Temp (ignored)
        datetime(2024, 1, 1, 20), # Night, OK Temp (25)
        datetime(2024, 1, 1, 22), # Night, High Temp (40 > 35)
        datetime(2024, 1, 2, 7),  # Day, High Temp (ignored)
    ]
    temps = [4.0, 10.0, 3.0, 25.0, 40.0, 38.0]
    df = pl.DataFrame({"time": times, "air_temp_c": temps})

    result_df = transform_features(df, plant_cfg, data_proc_cfg, segment_name="NightStressTest")

    low_stress_col = "air_temp_c_night_low_stress_low_temp_cam_induction_celsius_direct_low"
    high_stress_col = "air_temp_c_night_high_stress_heat_delay_night_temp_celsius_sd_significant_impact_high"

    assert low_stress_col in result_df.columns
    assert high_stress_col in result_df.columns

    expected_low_stress = [1, 0, 0, 0, 0, 0] # Only first point is night and < 5
    expected_high_stress = [0, 0, 0, 0, 1, 0] # Only fifth point is night and > 35
    
    assert result_df[low_stress_col].to_list() == expected_low_stress
    assert result_df[high_stress_col].to_list() == expected_high_stress
    logger.info("Night stress flags (high and low) generated correctly.")
    logger.info("Result: PASSED")

def test_lag_features_period_based(sample_plant_config, sample_data_proc_config):
    """Test period-based lag feature calculation."""
    logger.info("--- Starting Test: Lag Features (Periods) --- ")
    plant_cfg = sample_plant_config # data_frequency_minutes = 60
    data_proc_cfg = sample_data_proc_config # global config asks for lag co2_measured_ppm: 180 -> 3 periods

    df = pl.DataFrame({
        "time": pl.datetime_range(datetime(2024,1,1,0), datetime(2024,1,1,5), interval="1h", eager=True),
        "co2_measured_ppm": [400, 410, 420, 430, 440, 450]
    })

    result_df = transform_features(df, plant_cfg, data_proc_cfg, segment_name="LagTest")

    lag_col = "co2_measured_ppm_lag_3p"
    assert lag_col in result_df.columns
    
    # Expect 3 leading NaNs, then shifted values
    expected_lag_values = [None, None, None, 400.0, 410.0, 420.0]
    # Cast actual result to Float64 for comparison if needed, or just compare values
    actual_series = result_df[lag_col]
    # Handle potential float precision issues in comparison
    actual_list = actual_series.to_list()
    assert len(actual_list) == len(expected_lag_values)
    for i, (actual, expected) in enumerate(zip(actual_list, expected_lag_values)):
        if expected is None:
            assert actual is None, f"Mismatch at index {i}: Expected None, got {actual}"
        else:
            assert actual is not None and abs(actual - expected) < 1e-9, f"Mismatch at index {i}: Expected {expected}, got {actual}"

    logger.info(f"Period-based lag feature '{lag_col}' generated correctly.")
    logger.info("Result: PASSED")

# --- Remove pyarrow dependency from script header --- 
# Done at the top of this edit block.

# --- TODO Section Review ---
# - Test segment-aware logic (Added test_transform_features_segment_specific)
# - Test specific values for some calculated features (Partially covered by new tests, could add more value checks)
# - Test different configurations for features (Partially covered by segment test)
# - Test handling of data with NaNs remaining after cleaning (Tested availability flag, min_periods)
# - Test Actuator Summary calculations more thoroughly when implemented in Polars (Still Pending full calculator implementation)

if __name__ == "__main__":
    import pytest
    import sys
    # Exit with the appropriate code from pytest
    raise SystemExit(pytest.main([__file__] + sys.argv[1:]))
