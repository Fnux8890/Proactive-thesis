"""Configuration loader for MOEA optimization.

This module handles loading and parsing of TOML configuration files
for the MOEA optimizer.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Fallback for older Python

logger = logging.getLogger(__name__)


@dataclass
class ObjectiveConfig:
    """Configuration for a single objective."""
    name: str
    description: str
    type: str  # minimize or maximize
    unit: str
    weight_range: tuple[float, float]
    model_path: str
    scaler_path: str | None = None
    feature_expression: str | None = None
    important_signals: list[str] = field(default_factory=list)
    phenotype_features: list[str] = field(default_factory=list)


@dataclass
class DecisionVariable:
    """Configuration for a decision variable."""
    name: str
    description: str
    unit: str
    bounds: tuple[float, float]
    resolution: float


@dataclass
class ConstraintConfig:
    """Configuration for constraints."""
    environmental: dict[str, float]
    operational: dict[str, float]
    economic: dict[str, float] | None = None


@dataclass
class AlgorithmConfig:
    """Configuration for the optimization algorithm."""
    type: str
    population_size: int
    n_generations: int
    # Selection
    selection_method: str
    tournament_size: int
    # Crossover
    crossover_method: str
    crossover_probability: float
    crossover_eta: float
    # Mutation
    mutation_method: str
    mutation_probability: float
    mutation_eta: float
    # Other
    constraint_handling: str
    penalty_factor: float
    # Parallel execution
    parallel_enable: bool
    parallel_workers: int
    parallel_batch_size: int
    use_gpu: bool
    # Termination criteria
    termination: dict[str, Any]
    # NSGA-III specific
    n_reference_points: int = 12  # Default for NSGA-III
    verbose: bool = True  # Verbose output during optimization


@dataclass
class OutputConfig:
    """Configuration for output handling."""
    save_interval: int
    save_population: bool
    save_pareto_front: bool
    save_convergence_history: bool
    plot_interval: int
    plot_objectives: bool
    plot_decision_variables: bool
    plot_convergence: bool
    formats: dict[str, str | list[str]]
    base_dir: str = "results"
    experiment_dir: str = "experiment"
    save_history: bool = True  # Save optimization history


@dataclass
class ScenarioConfig:
    """Configuration for optimization scenario."""
    name: str
    description: str
    start_date: str
    duration_days: int
    location: str
    weather_scenario: str
    objective_weights: dict[str, float] | None = None
    constraints: dict[str, float] | None = None


@dataclass
class MOEAConfig:
    """Complete MOEA configuration."""
    meta: dict[str, str]
    objectives: list[ObjectiveConfig]
    composite_objectives: dict[str, dict[str, Any]]
    decision_variables: list[DecisionVariable]
    constraints: ConstraintConfig
    algorithm: AlgorithmConfig
    output: OutputConfig
    scenario: ScenarioConfig
    validation: dict[str, Any]

    def get_objective(self, name: str) -> ObjectiveConfig | None:
        """Get objective configuration by name."""
        for obj in self.objectives:
            if obj.name == name:
                return obj
        return None

    def get_active_objectives(self) -> list[ObjectiveConfig]:
        """Get objectives that should be optimized based on scenario."""
        if self.scenario.objective_weights:
            # Filter objectives based on scenario weights
            active_names = list(self.scenario.objective_weights.keys())
            return [obj for obj in self.objectives if obj.name in active_names]
        return self.objectives


class ConfigLoader:
    """Loads and validates MOEA configuration from TOML files."""

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

    def load(self) -> MOEAConfig:
        """Load and parse the TOML configuration."""
        logger.info(f"Loading MOEA configuration from {self.config_path}")

        with open(self.config_path, 'rb') as f:
            data = tomllib.load(f)

        # Parse objectives
        objectives = []
        for obj_data in data.get('objectives', []):
            objectives.append(ObjectiveConfig(
                name=obj_data['name'],
                description=obj_data['description'],
                type=obj_data['type'],
                unit=obj_data['unit'],
                weight_range=tuple(obj_data['weight_range']),
                model_path=obj_data['model_path'],
                scaler_path=obj_data.get('scaler_path'),
                feature_expression=obj_data.get('feature_expression'),
                important_signals=obj_data.get('important_signals', []),
                phenotype_features=obj_data.get('phenotype_features', [])
            ))

        # Parse decision variables
        decision_variables = []
        for var_data in data.get('decision_variables', []):
            decision_variables.append(DecisionVariable(
                name=var_data['name'],
                description=var_data['description'],
                unit=var_data['unit'],
                bounds=tuple(var_data['bounds']),
                resolution=var_data['resolution']
            ))

        # Parse constraints
        constraints = ConstraintConfig(
            environmental=data['constraints']['environmental'],
            operational=data['constraints']['operational'],
            economic=data['constraints'].get('economic')
        )

        # Parse algorithm configuration
        alg_data = data['algorithm']
        algorithm = AlgorithmConfig(
            type=alg_data['type'],
            population_size=alg_data['population_size'],
            n_generations=alg_data['n_generations'],
            selection_method=alg_data['selection_method'],
            tournament_size=alg_data['tournament_size'],
            crossover_method=alg_data['crossover_method'],
            crossover_probability=alg_data['crossover_probability'],
            crossover_eta=alg_data['crossover_eta'],
            mutation_method=alg_data['mutation_method'],
            mutation_probability=alg_data['mutation_probability'],
            mutation_eta=alg_data['mutation_eta'],
            constraint_handling=alg_data['constraint_handling'],
            penalty_factor=alg_data['penalty_factor'],
            n_reference_points=alg_data.get('n_reference_points', 12),
            verbose=alg_data.get('verbose', True),
            parallel_enable=alg_data['parallel']['enable'],
            parallel_workers=alg_data['parallel']['n_workers'],
            parallel_batch_size=alg_data['parallel']['batch_size'],
            use_gpu=alg_data['parallel']['use_gpu'],
            termination=alg_data['termination']
        )

        # Parse output configuration
        out_data = data['output']
        output = OutputConfig(
            save_interval=out_data['save_interval'],
            save_population=out_data['save_population'],
            save_pareto_front=out_data['save_pareto_front'],
            save_convergence_history=out_data['save_convergence_history'],
            plot_interval=out_data['plot_interval'],
            plot_objectives=out_data['plot_objectives'],
            plot_decision_variables=out_data['plot_decision_variables'],
            plot_convergence=out_data['plot_convergence'],
            formats=out_data['formats'],
            base_dir=out_data.get('base_dir', 'results'),
            experiment_dir=out_data.get('experiment_dir', 'experiment'),
            save_history=out_data.get('save_history', True)
        )

        # Parse scenario configuration
        scen_data = data['scenario']
        scenario = ScenarioConfig(
            name=scen_data['name'],
            description=scen_data['description'],
            start_date=scen_data['start_date'],
            duration_days=scen_data['duration_days'],
            location=scen_data['location'],
            weather_scenario=scen_data['weather_scenario'],
            objective_weights=scen_data.get('objective_weights'),
            constraints=scen_data.get('constraints')
        )

        # Create complete configuration
        config = MOEAConfig(
            meta=data['meta'],
            objectives=objectives,
            composite_objectives=data.get('composite_objectives', {}),
            decision_variables=decision_variables,
            constraints=constraints,
            algorithm=algorithm,
            output=output,
            scenario=scenario,
            validation=data.get('validation', {})
        )

        logger.info(f"Loaded configuration with {len(objectives)} objectives and "
                   f"{len(decision_variables)} decision variables")

        return config

    def validate(self, config: MOEAConfig) -> bool:
        """Validate the configuration for consistency."""
        logger.info("Validating MOEA configuration")

        # Check that all model paths exist
        for obj in config.objectives:
            model_path = Path(obj.model_path)
            if not model_path.exists():
                logger.warning(f"Model file not found for objective '{obj.name}': {model_path}")

            if obj.scaler_path:
                scaler_path = Path(obj.scaler_path)
                if not scaler_path.exists():
                    logger.warning(f"Scaler file not found for objective '{obj.name}': {scaler_path}")

        # Check decision variable bounds
        for var in config.decision_variables:
            if var.bounds[0] >= var.bounds[1]:
                logger.error(f"Invalid bounds for variable '{var.name}': {var.bounds}")
                return False

            if var.resolution <= 0:
                logger.error(f"Invalid resolution for variable '{var.name}': {var.resolution}")
                return False

        # Check algorithm parameters
        if config.algorithm.population_size < 10:
            logger.warning("Small population size may lead to poor convergence")

        if config.algorithm.crossover_probability < 0 or config.algorithm.crossover_probability > 1:
            logger.error(f"Invalid crossover probability: {config.algorithm.crossover_probability}")
            return False

        if config.algorithm.mutation_probability < 0 or config.algorithm.mutation_probability > 1:
            logger.error(f"Invalid mutation probability: {config.algorithm.mutation_probability}")
            return False

        # Check scenario weights if specified
        if config.scenario.objective_weights:
            total_weight = sum(config.scenario.objective_weights.values())
            if abs(total_weight - 1.0) > 0.001:
                logger.warning(f"Scenario objective weights sum to {total_weight}, not 1.0")

        logger.info("Configuration validation completed")
        return True


def load_config(config_path: str | Path) -> MOEAConfig:
    """Convenience function to load and validate configuration."""
    loader = ConfigLoader(config_path)
    config = loader.load()

    if not loader.validate(config):
        raise ValueError("Configuration validation failed")

    return config
