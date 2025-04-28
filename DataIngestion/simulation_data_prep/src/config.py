import json
from pydantic import BaseModel, Field, RootModel
from typing import List, Dict, Optional, Union, Any

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

class ActuatorSummaryParameters(BaseModel):
    percent_columns_for_average: List[str]
    percent_columns_for_changes: List[str]
    binary_columns_for_on_time: List[str]
    _comment_on_time: Optional[str] = None

class FeatureParameters(BaseModel):
    delta_cols: Optional[Dict[str, List[str]]] = None
    rate_of_change_cols: Optional[List[str]] = None
    rolling_average_cols: Optional[Dict[str, int]] = None
    _comment_humidity_delta: Optional[str] = None

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

class OptimalRange(BaseModel):
    lower: float
    upper: float

class ObjectiveFunctionParameters(BaseModel):
    _comment: Optional[str] = None
    energy_power_ratings_kw: EnergyPowerRatings
    fixed_setpoints: FixedSetpoints
    optimal_ranges: Dict[str, Union[OptimalRange, str]] # Accommodate potential comments/other types

class NightStressFlagDetail(BaseModel):
    input_temp_col: str
    threshold_config_key: str
    threshold_sub_key: str

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

# Main Configuration Model
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

# Function to load the configuration
def load_config(path: str = "/app/plant_config.json") -> PlantConfig:
    with open(path, 'r') as f:
        data = json.load(f)
    # Use model_validate for Pydantic v2 compatibility
    return PlantConfig.model_validate(data)

# Example Usage (keep commented out in final script)
# if __name__ == "__main__":
#     try:
#         config = load_config()
#         print("Config loaded successfully!")
#         print(f"GDD Base Temp: {config.gdd_parameters.profiles['kalanchoe_model_specific'].t_base_celsius}")
#         print(f"Lamp Power Keys: {list(config.objective_function_parameters.energy_power_ratings_kw.lamp_group_power_kw.root.keys())}")
#     except Exception as e:
#         print(f"Error loading or parsing config: {e}")
