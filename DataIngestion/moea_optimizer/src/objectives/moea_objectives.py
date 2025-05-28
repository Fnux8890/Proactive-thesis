"""Configuration for MOEA optimization objectives and their relationships.

This module defines the optimization objectives for the Multi-Objective
Evolutionary Algorithm (MOEA) and maps them to surrogate models.
"""

from dataclasses import dataclass, field
from enum import Enum


class ObjectiveType(Enum):
    """Types of optimization objectives."""
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class PhenotypeFeature(Enum):
    """Available phenotype features from literature data."""
    # Growth characteristics
    GROWTH_RATE = "growth_rate"  # mm/day
    FINAL_HEIGHT = "final_height"  # mm
    LEAF_AREA = "leaf_area"  # cm²
    STEM_DIAMETER = "stem_diameter"  # mm

    # Flowering characteristics
    TIME_TO_FLOWER = "time_to_flower"  # days
    FLOWER_COUNT = "flower_count"  # number
    FLOWER_SIZE = "flower_size"  # cm

    # Environmental responses
    LIGHT_SATURATION = "light_saturation"  # μmol/m²/s
    OPTIMAL_TEMPERATURE = "optimal_temperature"  # °C
    WATER_USE_EFFICIENCY = "water_use_efficiency"  # g/L

    # Quality metrics
    COMPACTNESS = "compactness"  # ratio
    UNIFORMITY = "uniformity"  # coefficient
    MARKET_VALUE = "market_value"  # relative scale


@dataclass
class OptimizationObjective:
    """Definition of an optimization objective."""
    name: str
    description: str
    type: ObjectiveType
    unit: str
    # Feature engineering for this objective
    feature_expression: str | None = None
    # Phenotype features that influence this objective
    related_phenotypes: list[PhenotypeFeature] = field(default_factory=list)
    # Sensor signals that are important for this objective
    important_signals: list[str] = field(default_factory=list)
    # Weight range for multi-objective optimization
    weight_range: tuple[float, float] = (0.0, 1.0)


# Define standard objectives for greenhouse optimization
OBJECTIVES = {
    "energy_consumption": OptimizationObjective(
        name="energy_consumption",
        description="Total energy consumption for climate control",
        type=ObjectiveType.MINIMIZE,
        unit="kWh",
        feature_expression="heating_energy + cooling_energy + lighting_energy",
        important_signals=[
            "pipe_temp_1_c",
            "pipe_temp_2_c",
            "heating_setpoint_c",
            "total_lamps_on",
            "vent_pos_1_percent",
            "vent_pos_2_percent"
        ],
        weight_range=(0.2, 0.8)
    ),

    "plant_growth": OptimizationObjective(
        name="plant_growth",
        description="Plant growth rate and development",
        type=ObjectiveType.MAXIMIZE,
        unit="g/day",
        feature_expression="biomass_accumulation_rate",
        related_phenotypes=[
            PhenotypeFeature.GROWTH_RATE,
            PhenotypeFeature.LEAF_AREA,
            PhenotypeFeature.STEM_DIAMETER
        ],
        important_signals=[
            "dli_sum",
            "co2_measured_ppm",
            "air_temp_c",
            "relative_humidity_percent",
            "vpd_hpa"
        ],
        weight_range=(0.3, 0.9)
    ),

    "water_usage": OptimizationObjective(
        name="water_usage",
        description="Total water consumption for irrigation",
        type=ObjectiveType.MINIMIZE,
        unit="L",
        feature_expression="irrigation_volume + evapotranspiration",
        related_phenotypes=[
            PhenotypeFeature.WATER_USE_EFFICIENCY
        ],
        important_signals=[
            "relative_humidity_percent",
            "vpd_hpa",
            "radiation_w_m2",
            "air_temp_c"
        ],
        weight_range=(0.1, 0.5)
    ),

    "crop_quality": OptimizationObjective(
        name="crop_quality",
        description="Overall crop quality and market value",
        type=ObjectiveType.MAXIMIZE,
        unit="quality_index",
        feature_expression="compactness * uniformity * flower_quality",
        related_phenotypes=[
            PhenotypeFeature.COMPACTNESS,
            PhenotypeFeature.UNIFORMITY,
            PhenotypeFeature.FLOWER_COUNT,
            PhenotypeFeature.FLOWER_SIZE,
            PhenotypeFeature.MARKET_VALUE
        ],
        important_signals=[
            "dli_sum",
            "light_intensity_umol",
            "air_temp_c",
            "humidity_deficit_g_m3"
        ],
        weight_range=(0.2, 0.7)
    ),

    "production_time": OptimizationObjective(
        name="production_time",
        description="Time from planting to harvest",
        type=ObjectiveType.MINIMIZE,
        unit="days",
        related_phenotypes=[
            PhenotypeFeature.TIME_TO_FLOWER,
            PhenotypeFeature.GROWTH_RATE
        ],
        important_signals=[
            "dli_sum",
            "air_temp_c",
            "co2_measured_ppm"
        ],
        weight_range=(0.1, 0.6)
    ),

    "climate_stability": OptimizationObjective(
        name="climate_stability",
        description="Variance in climate conditions",
        type=ObjectiveType.MINIMIZE,
        unit="variance",
        feature_expression="temp_variance + humidity_variance + co2_variance",
        important_signals=[
            "air_temp_c",
            "relative_humidity_percent",
            "co2_measured_ppm",
            "vent_pos_1_percent",
            "vent_pos_2_percent"
        ],
        weight_range=(0.1, 0.4)
    )
}


@dataclass
class MOEAConfiguration:
    """Configuration for Multi-Objective Evolutionary Algorithm."""
    # Selected objectives for optimization
    objectives: list[str] = field(default_factory=lambda: ["energy_consumption", "plant_growth"])

    # Population settings
    population_size: int = 100
    n_generations: int = 200

    # Genetic operators
    crossover_probability: float = 0.9
    crossover_eta: float = 15  # Distribution index for SBX crossover
    mutation_probability: float = 0.1
    mutation_eta: float = 20  # Distribution index for polynomial mutation

    # Algorithm selection
    algorithm: str = "NSGA-II"  # Options: NSGA-II, NSGA-III, MOEA/D

    # Constraints
    temperature_bounds: tuple[float, float] = (15.0, 30.0)  # °C
    humidity_bounds: tuple[float, float] = (40.0, 90.0)  # %
    co2_bounds: tuple[float, float] = (400.0, 1200.0)  # ppm
    light_bounds: tuple[float, float] = (0.0, 1000.0)  # μmol/m²/s

    # Reference points for NSGA-III
    n_reference_points: int = 12

    # Parallel evaluation
    n_parallel_evaluations: int = 4
    use_gpu: bool = True


def get_objective_features(objective_name: str) -> dict[str, list[str]]:
    """Get the important features for a specific objective.

    Returns:
        Dictionary with 'signals' and 'phenotypes' lists
    """
    if objective_name not in OBJECTIVES:
        raise ValueError(f"Unknown objective: {objective_name}")

    obj = OBJECTIVES[objective_name]
    return {
        "signals": obj.important_signals,
        "phenotypes": [p.value for p in obj.related_phenotypes]
    }


def create_composite_objective(
    name: str,
    objectives: list[str],
    weights: list[float] | None = None
) -> OptimizationObjective:
    """Create a composite objective from multiple objectives.

    Args:
        name: Name for the composite objective
        objectives: List of objective names to combine
        weights: Optional weights for each objective (defaults to equal)

    Returns:
        New OptimizationObjective combining the specified objectives
    """
    if weights is None:
        weights = [1.0 / len(objectives)] * len(objectives)

    if len(weights) != len(objectives):
        raise ValueError("Number of weights must match number of objectives")

    # Collect all unique signals and phenotypes
    all_signals = set()
    all_phenotypes = set()

    for obj_name in objectives:
        if obj_name not in OBJECTIVES:
            raise ValueError(f"Unknown objective: {obj_name}")
        obj = OBJECTIVES[obj_name]
        all_signals.update(obj.important_signals)
        all_phenotypes.update(obj.related_phenotypes)

    # Create weighted expression
    expressions = []
    for obj_name, weight in zip(objectives, weights, strict=False):
        obj = OBJECTIVES[obj_name]
        sign = "-" if obj.type == ObjectiveType.MAXIMIZE else ""
        expressions.append(f"{sign}{weight:.2f} * {obj_name}")

    feature_expression = " + ".join(expressions)

    return OptimizationObjective(
        name=name,
        description=f"Composite objective: {', '.join(objectives)}",
        type=ObjectiveType.MINIMIZE,  # Always minimize composite
        unit="composite_score",
        feature_expression=feature_expression,
        related_phenotypes=list(all_phenotypes),
        important_signals=list(all_signals),
        weight_range=(0.0, 1.0)
    )


# Predefined composite objectives for common use cases
COMPOSITE_OBJECTIVES = {
    "sustainable_production": create_composite_objective(
        "sustainable_production",
        ["energy_consumption", "water_usage", "plant_growth"],
        [0.3, 0.2, 0.5]
    ),

    "quality_focused": create_composite_objective(
        "quality_focused",
        ["crop_quality", "climate_stability", "production_time"],
        [0.5, 0.3, 0.2]
    ),

    "efficient_growth": create_composite_objective(
        "efficient_growth",
        ["energy_consumption", "plant_growth", "production_time"],
        [0.3, 0.5, 0.2]
    )
}
