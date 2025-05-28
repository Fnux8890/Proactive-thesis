"""Base classes for objective functions."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import numpy as np


class ObjectiveType(Enum):
    """Type of optimization objective."""
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class ObjectiveFunction(ABC):
    """Abstract base class for objective functions."""

    def __init__(
        self,
        name: str,
        description: str,
        obj_type: ObjectiveType,
        unit: str,
        weight_range: tuple = (0.0, 1.0)
    ):
        self.name = name
        self.description = description
        self.type = obj_type
        self.unit = unit
        self.weight_range = weight_range
        self._evaluation_count = 0
        self._evaluation_cache: dict[str, float] = {}

    @abstractmethod
    def evaluate(self, decision_variables: dict[str, float], context: dict[str, Any] | None = None) -> float:
        """Evaluate the objective function.

        Args:
            decision_variables: Dictionary of decision variable values
            context: Optional context information (e.g., weather data, time of day)

        Returns:
            Objective function value
        """
        pass

    def evaluate_with_cache(self, decision_variables: dict[str, float], context: dict[str, Any] | None = None) -> float:
        """Evaluate with caching for repeated evaluations."""
        # Create cache key from decision variables
        cache_key = "_".join(f"{k}:{v:.4f}" for k, v in sorted(decision_variables.items()))

        if cache_key in self._evaluation_cache:
            return self._evaluation_cache[cache_key]

        # Evaluate and cache
        value = self.evaluate(decision_variables, context)
        self._evaluation_cache[cache_key] = value
        self._evaluation_count += 1

        return value

    def normalize(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize objective value to [0, 1] range."""
        if max_val == min_val:
            return 0.5

        normalized = (value - min_val) / (max_val - min_val)

        # Invert if maximizing (so lower is better for all objectives)
        if self.type == ObjectiveType.MAXIMIZE:
            normalized = 1.0 - normalized

        return np.clip(normalized, 0.0, 1.0)

    def get_statistics(self) -> dict[str, Any]:
        """Get evaluation statistics."""
        cache_values = list(self._evaluation_cache.values())

        if not cache_values:
            return {
                "evaluation_count": self._evaluation_count,
                "cache_size": 0,
                "min_value": None,
                "max_value": None,
                "mean_value": None,
                "std_value": None
            }

        return {
            "evaluation_count": self._evaluation_count,
            "cache_size": len(self._evaluation_cache),
            "min_value": np.min(cache_values),
            "max_value": np.max(cache_values),
            "mean_value": np.mean(cache_values),
            "std_value": np.std(cache_values)
        }

    def clear_cache(self) -> None:
        """Clear the evaluation cache."""
        self._evaluation_cache.clear()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', type={self.type.value})"
