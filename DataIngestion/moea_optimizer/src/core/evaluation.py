"""Evaluation metrics for multi-objective optimization.

This module provides implementations of common performance metrics
for evaluating the quality of Pareto fronts, including:
- Hypervolume (HV)
- Inverted Generational Distance Plus (IGD+)
- Epsilon Indicator
- Spacing
- Maximum Spread
"""

import logging

import numpy as np
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Collection of performance metrics for multi-objective optimization."""

    def __init__(self, ref_point: np.ndarray | None = None, pf: np.ndarray | None = None):
        """Initialize performance metrics.

        Args:
            ref_point: Reference point for hypervolume calculation
            pf: True Pareto front for IGD+ calculation
        """
        self.ref_point = ref_point
        self.pf = pf

        # Initialize metric calculators
        self.hv_calc = HV(ref_point=ref_point) if ref_point is not None else None
        self.igd_plus_calc = IGDPlus(pf) if pf is not None else None

    def hypervolume(self, F: np.ndarray) -> float:
        """Calculate hypervolume indicator.

        Args:
            F: Objective values (n_solutions x n_objectives)

        Returns:
            Hypervolume value
        """
        if self.hv_calc is None:
            raise ValueError("Reference point not set for hypervolume calculation")

        if len(F) == 0:
            return 0.0

        try:
            return self.hv_calc(F)
        except Exception as e:
            logger.warning(f"Hypervolume calculation failed: {e}")
            return 0.0

    def igd_plus(self, F: np.ndarray) -> float:
        """Calculate Inverted Generational Distance Plus.

        Args:
            F: Objective values (n_solutions x n_objectives)

        Returns:
            IGD+ value
        """
        if self.igd_plus_calc is None:
            raise ValueError("True Pareto front not set for IGD+ calculation")

        if len(F) == 0:
            return float('inf')

        try:
            return self.igd_plus_calc(F)
        except Exception as e:
            logger.warning(f"IGD+ calculation failed: {e}")
            return float('inf')

    def epsilon_indicator(self, F: np.ndarray, F_true: np.ndarray | None = None) -> float:
        """Calculate additive epsilon indicator.

        Args:
            F: Obtained Pareto front
            F_true: True Pareto front (uses self.pf if not provided)

        Returns:
            Epsilon indicator value
        """
        if F_true is None:
            F_true = self.pf

        if F_true is None:
            raise ValueError("True Pareto front not provided for epsilon indicator")

        if len(F) == 0:
            return float('inf')

        # For each solution in F_true, find minimum translation needed
        eps_values = []
        for f_true in F_true:
            # Find minimum epsilon such that exists f in F with f + eps dominates f_true
            min_eps = float('inf')
            for f in F:
                # Calculate epsilon for this pair
                eps = np.max(f - f_true)
                min_eps = min(min_eps, eps)
            eps_values.append(min_eps)

        return np.max(eps_values)

    def spacing(self, F: np.ndarray) -> float:
        """Calculate spacing metric (uniformity of distribution).

        Args:
            F: Objective values

        Returns:
            Spacing value (lower is better)
        """
        if len(F) <= 1:
            return 0.0

        # Calculate distances to nearest neighbor for each point
        distances = cdist(F, F)
        np.fill_diagonal(distances, np.inf)
        min_distances = distances.min(axis=1)

        # Calculate spacing
        mean_dist = min_distances.mean()
        spacing = np.sqrt(np.sum((min_distances - mean_dist) ** 2) / len(F))

        return spacing

    def maximum_spread(self, F: np.ndarray) -> float:
        """Calculate maximum spread (extent of Pareto front).

        Args:
            F: Objective values

        Returns:
            Maximum spread value (higher is better)
        """
        if len(F) == 0:
            return 0.0

        # Calculate spread in each objective
        spreads = F.max(axis=0) - F.min(axis=0)

        # Normalize by ideal spread if available
        if self.pf is not None:
            ideal_spreads = self.pf.max(axis=0) - self.pf.min(axis=0)
            spreads = spreads / (ideal_spreads + 1e-10)

        # Return product of spreads
        return np.prod(spreads)

    def calculate_all(self, F: np.ndarray) -> dict:
        """Calculate all available metrics.

        Args:
            F: Objective values

        Returns:
            Dictionary of metric names to values
        """
        metrics = {}

        # Hypervolume
        if self.ref_point is not None:
            try:
                metrics['hypervolume'] = self.hypervolume(F)
            except Exception as e:
                logger.warning(f"Failed to calculate hypervolume: {e}")
                metrics['hypervolume'] = None

        # IGD+
        if self.pf is not None:
            try:
                metrics['igd_plus'] = self.igd_plus(F)
            except Exception as e:
                logger.warning(f"Failed to calculate IGD+: {e}")
                metrics['igd_plus'] = None

            # Epsilon indicator
            try:
                metrics['epsilon_indicator'] = self.epsilon_indicator(F)
            except Exception as e:
                logger.warning(f"Failed to calculate epsilon indicator: {e}")
                metrics['epsilon_indicator'] = None

        # Spacing (always available)
        try:
            metrics['spacing'] = self.spacing(F)
        except Exception as e:
            logger.warning(f"Failed to calculate spacing: {e}")
            metrics['spacing'] = None

        # Maximum spread
        try:
            metrics['maximum_spread'] = self.maximum_spread(F)
        except Exception as e:
            logger.warning(f"Failed to calculate maximum spread: {e}")
            metrics['maximum_spread'] = None

        # Additional statistics
        if len(F) > 0:
            metrics['n_solutions'] = len(F)
            metrics['mean_objectives'] = F.mean(axis=0).tolist()
            metrics['std_objectives'] = F.std(axis=0).tolist()

        return metrics


class ConvergenceTracker:
    """Track convergence metrics over generations."""

    def __init__(self, metrics: PerformanceMetrics):
        """Initialize convergence tracker.

        Args:
            metrics: Performance metrics calculator
        """
        self.metrics = metrics
        self.history = {
            'generation': [],
            'n_evaluations': [],
            'hypervolume': [],
            'igd_plus': [],
            'spacing': [],
            'n_solutions': []
        }

    def update(self, generation: int, n_evaluations: int, F: np.ndarray) -> None:
        """Update convergence history.

        Args:
            generation: Current generation number
            n_evaluations: Total function evaluations
            F: Current Pareto front approximation
        """
        self.history['generation'].append(generation)
        self.history['n_evaluations'].append(n_evaluations)

        # Calculate metrics
        results = self.metrics.calculate_all(F)

        self.history['hypervolume'].append(results.get('hypervolume', None))
        self.history['igd_plus'].append(results.get('igd_plus', None))
        self.history['spacing'].append(results.get('spacing', None))
        self.history['n_solutions'].append(results.get('n_solutions', 0))

    def get_dataframe(self):
        """Get convergence history as pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self.history)

    def is_converged(
        self,
        window: int = 20,
        tolerance: float = 0.001,
        metric: str = 'hypervolume'
    ) -> bool:
        """Check if optimization has converged.

        Args:
            window: Number of generations to check
            tolerance: Relative improvement threshold
            metric: Metric to use for convergence check

        Returns:
            True if converged
        """
        if metric not in self.history:
            return False

        values = self.history[metric]
        if len(values) < window:
            return False

        # Get recent values
        recent = values[-window:]
        if None in recent:
            return False

        # Check relative improvement
        if metric in ['igd_plus', 'spacing']:  # Minimization metrics
            best_old = min(recent[:window//2])
            best_new = min(recent[window//2:])
            improvement = (best_old - best_new) / (best_old + 1e-10)
        else:  # Maximization metrics
            best_old = max(recent[:window//2])
            best_new = max(recent[window//2:])
            improvement = (best_new - best_old) / (best_old + 1e-10)

        return improvement < tolerance


def get_reference_point_for_problem(problem_name: str, n_obj: int) -> list[float]:
    """Get default reference point for known problems.

    Args:
        problem_name: Name of the problem
        n_obj: Number of objectives

    Returns:
        Reference point coordinates
    """
    # Default reference points for common problems
    ref_points = {
        'DTLZ1': [11.0] * n_obj,
        'DTLZ2': [2.5] * n_obj,
        'DTLZ3': [2.5] * n_obj,
        'DTLZ4': [2.5] * n_obj,
        'DTLZ5': [2.5] * n_obj,
        'DTLZ6': [2.5] * n_obj,
        'DTLZ7': [1.0] * (n_obj - 1) + [2.0 * n_obj],
        'WFG1': [3.0] * n_obj,
        'WFG2': [3.0] * n_obj,
        'WFG3': [3.0] * n_obj,
        'WFG4': [3.0] * n_obj,
        'WFG5': [3.0] * n_obj,
        'WFG6': [3.0] * n_obj,
        'WFG7': [3.0] * n_obj,
        'WFG8': [3.0] * n_obj,
        'WFG9': [3.0] * n_obj,
    }

    return ref_points.get(problem_name.upper(), [1.1] * n_obj)


# Example usage
if __name__ == "__main__":
    # Create random Pareto front for testing
    np.random.seed(42)
    F = np.random.rand(50, 3)

    # Create metrics calculator
    ref_point = np.array([1.1, 1.1, 1.1])
    metrics = PerformanceMetrics(ref_point=ref_point)

    # Calculate all metrics
    results = metrics.calculate_all(F)

    print("Performance Metrics:")
    for name, value in results.items():
        print(f"  {name}: {value}")

    # Test convergence tracking
    tracker = ConvergenceTracker(metrics)
    for gen in range(10):
        F_gen = np.random.rand(50 + gen * 5, 3)
        tracker.update(gen, gen * 100, F_gen)

    print("\nConvergence History:")
    print(tracker.get_dataframe())
