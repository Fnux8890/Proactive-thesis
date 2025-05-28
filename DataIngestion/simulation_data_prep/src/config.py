from __future__ import annotations # Ensure this is at the top
import json
import os # Added for os.getenv
from pydantic import BaseModel, Field, RootModel
from typing import List, Dict, Optional, Union, Any, Tuple # Added Tuple

# --- Define RollingWindowConfig at the top ---
class RollingWindowConfig(BaseModel):
    window_minutes: int
    min_periods: Optional[int] = None
    _comment: Optional[str] = None
# --- End definition ---

# Define specific sub-models for clarity and type safety

class Metadata(BaseModel):
    plant_species: str
    profile_name: str
    version: str
    description: str
    source_document_id: str
    _comment_cultivar: Optional[str] = None

class GddProfile(BaseModel):
    t_base_celsius: float
    t_cap_celsius: float
    _comment_t_base: Optional[str] = None
    _comment_t_cap: Optional[str] = None

class GddParameters(BaseModel):
    crop_profile: str
    profiles: Dict[str, GddProfile]

class TempTarget(BaseModel):
    min: Optional[float] = None
    max: Optional[float] = None
    _comment: Optional[str] = None

class DliTarget(BaseModel):
    min: Optional[float] = None
    max: Optional[float] = None
    _comment: Optional[str] = None

class VpdTarget(BaseModel):
    min: float
    max: float
    _comment: Optional[str] = None

class OptimalConditions(BaseModel):
    _comment_stages: Optional[str] = None
    temperature_celsius: Dict[str, TempTarget]
    # Accept any mapping for DLI, including comments, numeric values, and nested structures
    dli_mol_m2_day: Dict[str, Any]
    vpd_kpa: Dict[str, VpdTarget]

class SdInduction(BaseModel):
    value: Optional[float] = None
    _comment: Optional[str] = None

class PhotoperiodParameters(BaseModel):
    classification: str
    _comment_classification: Optional[str] = None
    critical_night_length_hours: float
    _comment_cnl: Optional[str] = None
    inductive_night_length_hours_target: float
    _comment_inductive_ln: Optional[str] = None
    sd_induction_duration_weeks: SdInduction
    night_interruption_inhibits_flowering: bool
    _comment_ni: Optional[str] = None

class DifParameters(BaseModel):
    _comment_dif: Optional[str] = None
    day_definition: str
    _comment_day_def: Optional[str] = None
    lamp_status_columns: List[str]
    fixed_time_day_start_hour: int
    fixed_time_day_end_hour: int
    target_dif_celsius: Optional[float] = None
    _comment_target: Optional[str] = None

class HeatDelay(BaseModel):
    onset_risk: float
    significant_impact: float
    _comment: Optional[str] = None

class StressThresholds(BaseModel):
    heat_delay_night_temp_celsius_sd: HeatDelay
    low_temp_cam_induction_celsius: float
    _comment_cam: Optional[str] = None

# --- Removed old definition location ---
# # --- Restore definition ---
# # class RollingWindowConfig(BaseModel):
# #     window_minutes: int
# #     min_periods: Optional[int] = None
# #     _comment: Optional[str] = None
# # --- End restored definition ---

class ActuatorSummaryParameters(BaseModel):
    percent_columns_for_average: List[str]
    percent_columns_for_changes: List[str]
    binary_columns_for_on_time: List[str]
    _comment_on_time: Optional[str] = None

class FeatureParameters(BaseModel):
    delta_cols: Optional[Dict[str, List[str]]] = None
    rate_of_change_cols: Optional[List[str]] = None
    # Use forward reference string for the type hint
    rolling_average_cols: Optional[Dict[str, "RollingWindowConfig"]] = None
    _comment_humidity_delta: Optional[str] = None

class OptimalRange(BaseModel):
    lower: float
    upper: float

# Define NightStressFlagDetail EARLIER
class NightStressFlagDetail(BaseModel):
    input_temp_col: str
    threshold_config_key: str
    threshold_sub_key: Optional[str] = None # Make sub_key optional for direct float thresholds
    stress_type: str = Field(default="high", pattern="^(high|low)$") # 'high' or 'low'
    output_col_suffix: Optional[str] = None # Optional suffix for the output column name

class AdvancedFeatureParametersPlaceholder(BaseModel):
    """A placeholder mirroring AdvancedFeatureParameters, to avoid cyclic imports if defined too early,
    or to be used if the full AdvancedFeatureParameters is too complex for simple segment overrides.
    For now, let's assume we can reuse/redefine the necessary parts.
    Actual fields from AdvancedFeatureParameters will be used in SegmentFeatureConfig.
    """
    # Use forward reference string for the type hint
    rolling_std_dev_cols: Optional[Dict[str, "RollingWindowConfig"]] = None
    lag_features: Optional[Dict[str, Any]] = None
    distance_from_optimal_midpoint: Optional[Dict[str, str]] = None
    in_optimal_range_flag: Optional[Dict[str, str]] = None
    # Use forward reference for NightStressFlagDetail too
    night_stress_flags: Optional[Dict[str, Union[str, "NightStressFlagDetail"]]] = None
    availability_flags_for_cols: Optional[List[str]] = None

class SegmentFeatureConfig(BaseModel):
    """Defines feature calculation parameters specific to a data segment."""
    feature_parameters: Optional[FeatureParameters] = Field(default_factory=FeatureParameters) # Reuse existing
    advanced_feature_parameters: Optional[AdvancedFeatureParametersPlaceholder] = Field(default_factory=AdvancedFeatureParametersPlaceholder)
    # We might also want to specify which optimal_conditions keys from PlantConfig are relevant for this segment
    active_optimal_condition_keys: Optional[List[str]] = None # e.g., ["temperature_celsius", "vpd_kpa"]

class LampGroupPower(RootModel[Dict[str, Optional[float]]]):
    root: Dict[str, Optional[float]] # Root model for lamp group power key/value mapping

class EnergyPowerRatings(BaseModel):
    _comment_heating: Optional[str] = None
    heating_system_power_kw: Optional[float] = None
    _comment_ventilation: Optional[str] = None
    ventilation_passive: bool
    ventilation_system_power_kw: Optional[float] = None
    air_conditioning_power_kw: Optional[float] = None
    _comment_lighting: Optional[str] = None
    lamp_group_power_kw: LampGroupPower

class FixedSetpoints(BaseModel):
    _comment: Optional[str] = None
    temperature_celsius: Optional[float] = None
    relative_humidity_percent: Optional[float] = None
    _comment_light_mad: Optional[str] = None
    light_par_umol_m2_s: Optional[float] = None

class ObjectiveFunctionParameters(BaseModel):
    _comment: Optional[str] = None
    energy_power_ratings_kw: EnergyPowerRatings
    fixed_setpoints: FixedSetpoints
    optimal_ranges: Dict[str, Union[OptimalRange, str]] # Accommodate potential comments/other types
    _comment_humidity_delta: Optional[str] = None

class AdvancedFeatureParameters(BaseModel):
    _comment: Optional[str] = None
    # Accept any mapping for advanced feature parameters, including comments
    rolling_std_dev_cols: Optional[Dict[str, Any]] = None
    lag_features: Optional[Dict[str, Any]] = None
    distance_from_optimal_midpoint: Optional[Dict[str, str]] = None
    in_optimal_range_flag: Optional[Dict[str, str]] = None
    night_stress_flags: Optional[Dict[str, Union[str, NightStressFlagDetail]]] = None

class LampGroupDetail(BaseModel):
    count: int
    ppf_umol_s: float
    power_kw: float

# Main Configuration Model for Plant Biology / Simulation
class PlantConfig(BaseModel):
    plant_profile_metadata: Metadata
    gdd_parameters: GddParameters
    optimal_conditions: OptimalConditions
    photoperiod_parameters: PhotoperiodParameters
    dif_parameters: DifParameters
    stress_thresholds: StressThresholds
    actuator_summary_parameters: ActuatorSummaryParameters
    feature_parameters: FeatureParameters
    objective_function_parameters: ObjectiveFunctionParameters
    advanced_feature_parameters: AdvancedFeatureParameters
    lamp_groups: Optional[Dict[str, LampGroupDetail]] = None
    column_uuid_mapping: Optional[Dict[str, List[str]]] = Field(default_factory=dict)
    data_frequency_minutes: int = Field(default=10, description="Frequency of the resampled data in minutes, used for lag period calculation.")

# Function to load the plant configuration
def load_plant_config(path: str = "/app/plant_config.json") -> PlantConfig:
    with open(path, 'r') as f:
        data = json.load(f)
    return PlantConfig.model_validate(data)

# --- New Models for Data Processing Configuration ---

class DBConnectionConfig(BaseModel):
    host: str = Field(default_factory=lambda: os.getenv("DB_HOST", "db"))
    port: int = Field(default_factory=lambda: int(os.getenv("DB_PORT", "5432")))
    user: str = Field(default_factory=lambda: os.getenv("DB_USER", "postgres"))
    password: str = Field(default_factory=lambda: os.getenv("DB_PASSWORD", "postgres"))
    dbname: str = Field(default_factory=lambda: os.getenv("DB_NAME", "postgres"))

# Define structure for individual outlier rule
class OutlierRule(BaseModel):
    column: str
    method: str # e.g., 'iqr', 'zscore_rolling', 'domain'
    params: Dict[str, Any] = Field(default_factory=dict) # e.g., {"factor": 1.5}, {"window_size": 168, "threshold": 3.0}, {"lower_bound": 0}
    handling_strategy: str = 'to_nan' # e.g., 'to_nan', 'clip'
    clip_bounds: Optional[Tuple[Optional[float], Optional[float]]] = None # Used if strategy is 'clip'

class OutlierDetectionConfig(BaseModel):
    # Option 1: A single list of rules applied to all segments
    rules: List[OutlierRule] = Field(default_factory=list)
    # Option 2 (More flexible): Segment-specific rules + optional global rules
    # segment_rules: Optional[Dict[str, List[OutlierRule]]] = None # Keyed by segment name
    # global_rules: List[OutlierRule] = Field(default_factory=list) # Applied if no segment specific rules exist

class ImputationStrategy(BaseModel):
    method: str # e.g., 'linear', 'time', 'polynomial', 'locf', 'mean', 'median', 'zero', 'forward_fill', 'backward_fill'
    order: Optional[int] = None 
    limit_direction: Optional[str] = None 
    limit_area: Optional[str] = None

class ImputationConfig(BaseModel):
    default_strategy: Optional[ImputationStrategy] = Field(default_factory=lambda: ImputationStrategy(method='linear')) # Default to linear
    # Option 1: Column strategies apply globally
    column_specific_strategies: Dict[str, ImputationStrategy] = Field(default_factory=dict)
    # Option 2: Segment-specific overrides
    # segment_column_strategies: Optional[Dict[str, Dict[str, ImputationStrategy]]] = None # Keyed by segment name, then column name

class ColumnProcessingLists(BaseModel):
    # Add potential numeric cols list based on actual DB schema analysis
    potential_numeric_cols: List[str] = Field(default_factory=lambda: [
        'air_temp_c', 'air_temp_middle_c', 'outside_temp_c', 
        'relative_humidity_percent', 'humidity_deficit_g_m3', 
        'radiation_w_m2', 'light_intensity_lux', 'light_intensity_umol', 'outside_light_w_m2',
        'co2_measured_ppm', 'co2_required_ppm', 'co2_dosing_status', 'co2_status',
        'vent_pos_1_percent', 'vent_pos_2_percent', 
        'vent_lee_afd3_percent', 'vent_wind_afd3_percent', 'vent_lee_afd4_percent', 'vent_wind_afd4_percent',
        'curtain_1_percent', 'curtain_2_percent', 'curtain_3_percent', 'curtain_4_percent',
        'window_1_percent', 'window_2_percent',
        'heating_setpoint_c', 'pipe_temp_1_c', 'pipe_temp_2_c', 'flow_temp_1_c', 'flow_temp_2_c',
        'temperature_forecast_c', 'sun_radiation_forecast_w_m2', 
        'temperature_actual_c', 'sun_radiation_actual_w_m2',
        'vpd_hpa',
        'humidity_deficit_afd3_g_m3', 'relative_humidity_afd3_percent', 
        'humidity_deficit_afd4_g_m3', 'relative_humidity_afd4_percent',
        'behov', 'timer_on', 'timer_off', 'dli_sum', 
        'lampe_timer_on', 'lampe_timer_off', 'value' 
    ])
    columns_to_exclude: List[str] = Field(default_factory=lambda: [
        'source_system', 'source_file', 'format_type', 'uuid', 'lamp_group', 
        'status_str', 'oenske_ekstra_lys' # Exclude metadata and pure text columns
    ])
    columns_for_lag_features: List[str] = Field(default_factory=list)
    lag_periods: List[int] = Field(default_factory=lambda: [1, 3, 6, 12, 24]) 
    
    columns_for_rolling_window: List[str] = Field(default_factory=list)
    rolling_window_sizes: List[int] = Field(default_factory=lambda: [3, 6, 12, 24])
    rolling_window_aggregations: List[str] = Field(default_factory=lambda: ['mean', 'median', 'std'])

    target_variable: Optional[str] = None
    datetime_column: str = "time" # Default name for the main time column

# New Model for Data Segments
class DataSegment(BaseModel):
    name: str
    start_date: str # ISO Format string e.g., "2014-01-01T00:00:00Z"
    end_date: str   # ISO Format string e.g., "2014-08-31T23:59:59Z"
    description: Optional[str] = None
    # Potentially add segment-specific overrides here later if needed
    # E.g., applicable_feature_list: Optional[List[str]] = None

# Main Data Processing Config Model
class DataProcessingConfig(BaseModel):
    db_connection: DBConnectionConfig = Field(default_factory=DBConnectionConfig)
    data_segments: List[DataSegment] = Field(default_factory=list)
    outlier_detection: OutlierDetectionConfig = Field(default_factory=OutlierDetectionConfig)
    imputation: ImputationConfig = Field(default_factory=ImputationConfig)
    column_lists: ColumnProcessingLists = Field(default_factory=ColumnProcessingLists)
    
    # Segment-specific and global feature configurations
    segment_feature_configs: Dict[str, SegmentFeatureConfig] = Field(default_factory=dict)
    global_feature_config: Optional[SegmentFeatureConfig] = Field(default_factory=SegmentFeatureConfig) # Provides defaults

    # These might become deprecated if segments define the full range
    # processing_start_date: Optional[str] = None 
    # processing_end_date: Optional[str] = None   

# Function to load the data processing configuration
def load_data_processing_config(path: str = "/app/src/data_processing_config.json") -> DataProcessingConfig:
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return DataProcessingConfig.model_validate(data)
    except FileNotFoundError:
        print(f"INFO: Data processing config file not found at {path}. Using default configuration.")
        return DataProcessingConfig()
    except json.JSONDecodeError as e:
        print(f"ERROR: Could not decode JSON from {path}. Error: {e}. Using default configuration.")
        return DataProcessingConfig()
    except Exception as e:
        print(f"ERROR: Unexpected error loading data processing config from {path}. Error: {e}. Using default configuration.")
        return DataProcessingConfig()


# Example Usage (keep commented out in final script)
# if __name__ == "__main__":
#     try:
#         plant_conf = load_plant_config() # Assumes plant_config.json exists at /app/
#         print("PlantConfig loaded successfully!")
#         print(f"GDD Base Temp: {plant_conf.gdd_parameters.profiles['kalanchoe_model_specific'].t_base_celsius}")
#
#         data_proc_conf = load_data_processing_config() # Will use defaults if file not found
#         print("\nDataProcessingConfig loaded successfully (or defaulted)!")
#         print(f"DB Host (from env or default): {data_proc_conf.db_connection.host}")
#         print(f"Default imputation method: {data_proc_conf.imputation.default_strategy.method if data_proc_conf.imputation.default_strategy else 'None'}")
#         print(f"Target variable: {data_proc_conf.column_lists.target_variable}")
#
#     except Exception as e:
#         print(f"Error in example usage: {e}")

# Placeholder for actual configuration loading (e.g., from JSON, YAML, or environment variables)
# These would ideally be loaded from a file or environment variables
DB_HOST = "localhost"
DB_PORT = 5432
DB_USER = "user"
DB_PASSWORD = "password"
DB_NAME = "mydb"

PROCESSING_START_DATE = "2023-01-01 00:00:00" # Example
PROCESSING_END_DATE = "2024-01-01 00:00:00"   # Example

TARGET_VARIABLE = "your_target_column" # TODO: Specify your actual target variable

# Outlier Detection
IQR_FACTOR = 1.5
Z_SCORE_WINDOW = 24
Z_SCORE_THRESHOLD = 3.0
# COLUMNS_FOR_OUTLIER_DETECTION = ["sensor_value_1", "sensor_value_2"] # TODO: Specify columns

# Imputation
# IMPUTATION_STRATEGIES = {
#     "sensor_value_1": "linear",  # Interpolate
#     "sensor_value_2": "mean",
#     "another_sensor": "locf"   # Last observation carried forward
# } # TODO: Define strategies

# Feature Engineering
TIME_COLUMN = "time" # Assuming 'time' is the name of your datetime column
# COLUMNS_FOR_LAG_FEATURES = ["sensor_value_1", TARGET_VARIABLE] # TODO: Specify columns
# LAG_PERIODS = [1, 3, 6, 12, 24] # Example lag periods (e.g., hours if data is hourly)

# COLUMNS_FOR_ROLLING_WINDOW = ["sensor_value_1", TARGET_VARIABLE] # TODO: Specify columns
# ROLLING_WINDOW_SIZES = [6, 12, 24] # Example window sizes
# ROLLING_WINDOW_AGGREGATIONS = ["mean", "std", "min", "max"] # Example aggregation functions

# Feature Selection
VARIANCE_THRESHOLD = 0.01 # For VarianceThreshold
CORR_TARGET_THRESHOLD = 0.05 # Minimum absolute correlation with target
INTER_FEATURE_CORR_THRESHOLD = 0.95 # Maximum inter-feature correlation to drop one
TOP_N_FEATURES = 20 # For model-based feature selection

# Output paths
OUTPUT_DIR = "DataIngestion/simulation_data_prep/output"
UNSCALED_FEATURES_FILE = "features_unscaled.parquet"
SCALER_FILE = "fitted_scaler.joblib" # This will be saved by train_surrogate.py
