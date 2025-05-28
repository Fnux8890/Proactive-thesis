"""MOEA problem definition for greenhouse climate optimization."""

import logging
from typing import Any

import numpy as np

from ..objectives.base import ObjectiveFunction, ObjectiveType
from .config_loader import MOEAConfig

logger = logging.getLogger(__name__)


class MOEAProblem:
    """Defines the multi-objective optimization problem."""

    def __init__(
        self,
        config: MOEAConfig,
        objectives: list[ObjectiveFunction],
        context_provider: Any | None = None
    ):
        self.config = config
        self.objectives = objectives
        self.context_provider = context_provider

        # Extract decision variables
        self.decision_variables = config.decision_variables
        self.n_variables = len(self.decision_variables)

        # Extract bounds
        self.lower_bounds = np.array([var.bounds[0] for var in self.decision_variables])
        self.upper_bounds = np.array([var.bounds[1] for var in self.decision_variables])

        # Number of objectives
        self.n_objectives = len(objectives)

        # Constraint functions
        self.constraints = self._build_constraints()
        self.n_constraints = len(self.constraints)

        # Normalization bounds (will be updated during optimization)
        self.objective_bounds = [(np.inf, -np.inf) for _ in range(self.n_objectives)]

        logger.info(f"Created MOEA problem with {self.n_objectives} objectives, "
                   f"{self.n_variables} variables, and {self.n_constraints} constraints")

    def _build_constraints(self) -> list[callable]:
        """Build constraint functions from configuration."""
        constraints = []

        # Environmental constraints
        env_constraints = self.config.constraints.environmental

        # Temperature constraints
        if 'min_temperature' in env_constraints:
            constraints.append(
                lambda x: self._get_var_value(x, 'temperature_setpoint') - env_constraints['min_temperature']
            )
        if 'max_temperature' in env_constraints:
            constraints.append(
                lambda x: env_constraints['max_temperature'] - self._get_var_value(x, 'temperature_setpoint')
            )

        # Humidity constraints
        if 'min_humidity' in env_constraints:
            constraints.append(
                lambda x: self._get_var_value(x, 'humidity_setpoint') - env_constraints['min_humidity']
            )
        if 'max_humidity' in env_constraints:
            constraints.append(
                lambda x: env_constraints['max_humidity'] - self._get_var_value(x, 'humidity_setpoint')
            )

        # CO2 constraints
        if 'min_co2' in env_constraints:
            constraints.append(
                lambda x: self._get_var_value(x, 'co2_setpoint') - env_constraints['min_co2']
            )
        if 'max_co2' in env_constraints:
            constraints.append(
                lambda x: env_constraints['max_co2'] - self._get_var_value(x, 'co2_setpoint')
            )

        # VPD constraint
        if 'max_vpd' in env_constraints:
            def vpd_constraint(x):
                temp = self._get_var_value(x, 'temperature_setpoint')
                rh = self._get_var_value(x, 'humidity_setpoint')
                svp = 0.611 * np.exp(17.27 * temp / (temp + 237.3))
                vpd = svp * (1 - rh / 100)
                return env_constraints['max_vpd'] - vpd
            constraints.append(vpd_constraint)

        # Operational constraints
        op_constraints = self.config.constraints.operational

        # Daily light integral constraint
        if 'max_daily_light_integral' in op_constraints:
            def dli_constraint(x):
                intensity = self._get_var_value(x, 'light_intensity')
                hours = self._get_var_value(x, 'light_hours')
                dli = intensity * hours * 3.6 / 1000  # mol/m²/day
                return op_constraints['max_daily_light_integral'] - dli
            constraints.append(dli_constraint)

        return constraints

    def _get_var_value(self, x: np.ndarray, var_name: str) -> float:
        """Get value of a specific variable from solution vector."""
        for i, var in enumerate(self.decision_variables):
            if var.name == var_name:
                return x[i]
        # Return default if not found
        return 0.0

    def _array_to_dict(self, x: np.ndarray) -> dict[str, float]:
        """Convert solution array to decision variable dictionary."""
        return {var.name: x[i] for i, var in enumerate(self.decision_variables)}

    def evaluate(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate objectives and constraints for a solution.

        Args:
            x: Solution vector (decision variables)

        Returns:
            objectives: Array of objective values
            constraints: Array of constraint values (negative means violated)
        """
        # Convert to dictionary
        decision_vars = self._array_to_dict(x)

        # Get context if available
        context = self.context_provider.get_context() if self.context_provider else None

        # Evaluate objectives
        objectives = np.zeros(self.n_objectives)
        for i, obj in enumerate(self.objectives):
            try:
                value = obj.evaluate_with_cache(decision_vars, context)
                objectives[i] = value

                # Update bounds for normalization
                self.objective_bounds[i] = (
                    min(self.objective_bounds[i][0], value),
                    max(self.objective_bounds[i][1], value)
                )
            except Exception as e:
                logger.error(f"Error evaluating objective '{obj.name}': {e}")
                objectives[i] = np.inf if obj.type == ObjectiveType.MINIMIZE else -np.inf

        # Evaluate constraints
        constraints = np.zeros(self.n_constraints)
        for i, constraint_fn in enumerate(self.constraints):
            try:
                constraints[i] = constraint_fn(x)
            except Exception as e:
                logger.error(f"Error evaluating constraint {i}: {e}")
                constraints[i] = -np.inf  # Violation

        return objectives, constraints

    def evaluate_batch(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate a batch of solutions.

        Args:
            X: Array of solutions, shape (n_solutions, n_variables)

        Returns:
            objectives: Array of objective values, shape (n_solutions, n_objectives)
            constraints: Array of constraint values, shape (n_solutions, n_constraints)
        """
        n_solutions = X.shape[0]
        objectives = np.zeros((n_solutions, self.n_objectives))
        constraints = np.zeros((n_solutions, self.n_constraints))

        for i, x in enumerate(X):
            objectives[i], constraints[i] = self.evaluate(x)

        return objectives, constraints

    def repair_solution(self, x: np.ndarray) -> np.ndarray:
        """Repair a solution to satisfy bounds and resolution constraints."""
        # Clip to bounds
        x_repaired = np.clip(x, self.lower_bounds, self.upper_bounds)

        # Apply resolution constraints
        for i, var in enumerate(self.decision_variables):
            if var.resolution > 0:
                x_repaired[i] = np.round(x_repaired[i] / var.resolution) * var.resolution

        return x_repaired

    def random_solution(self) -> np.ndarray:
        """Generate a random feasible solution."""
        x = np.zeros(self.n_variables)

        for i, var in enumerate(self.decision_variables):
            # Generate random value within bounds
            x[i] = np.random.uniform(var.bounds[0], var.bounds[1])

            # Apply resolution
            if var.resolution > 0:
                x[i] = np.round(x[i] / var.resolution) * var.resolution

        return x

    def get_pareto_optimal_set(self, objectives: np.ndarray) -> np.ndarray:
        """Find Pareto optimal solutions from a set of objective values.

        Args:
            objectives: Array of objective values, shape (n_solutions, n_objectives)

        Returns:
            Boolean array indicating Pareto optimal solutions
        """
        n_solutions = objectives.shape[0]
        is_pareto = np.ones(n_solutions, dtype=bool)

        for i in range(n_solutions):
            if not is_pareto[i]:
                continue

            # Check if solution i is dominated by any other solution
            for j in range(n_solutions):
                if i == j:
                    continue

                # Check if j dominates i
                dominates = True
                for k in range(self.n_objectives):
                    if self.objectives[k].type == ObjectiveType.MINIMIZE:
                        if objectives[j, k] > objectives[i, k]:
                            dominates = False
                            break
                    else:  # MAXIMIZE
                        if objectives[j, k] < objectives[i, k]:
                            dominates = False
                            break

                if dominates:
                    # Check strict domination (at least one objective is strictly better)
                    strictly_better = False
                    for k in range(self.n_objectives):
                        if self.objectives[k].type == ObjectiveType.MINIMIZE:
                            if objectives[j, k] < objectives[i, k]:
                                strictly_better = True
                                break
                        else:  # MAXIMIZE
                            if objectives[j, k] > objectives[i, k]:
                                strictly_better = True
                                break

                    if strictly_better:
                        is_pareto[i] = False
                        break

        return is_pareto

    def get_statistics(self) -> dict[str, Any]:
        """Get problem statistics."""
        stats = {
            "n_objectives": self.n_objectives,
            "n_variables": self.n_variables,
            "n_constraints": self.n_constraints,
            "objective_names": [obj.name for obj in self.objectives],
            "variable_names": [var.name for var in self.decision_variables],
            "objective_bounds": self.objective_bounds
        }

        # Add objective statistics
        for i, obj in enumerate(self.objectives):
            stats[f"objective_{i}_stats"] = obj.get_statistics()

        return stats
