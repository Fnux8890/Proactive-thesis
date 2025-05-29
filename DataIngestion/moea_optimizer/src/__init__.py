"""Multi-Objective Evolutionary Algorithm (MOEA) Optimizer for Greenhouse Climate Control.

This package implements various MOEA algorithms for optimizing greenhouse
climate control strategies with multiple conflicting objectives such as
energy efficiency, plant growth, water usage, and crop quality.
"""

__version__ = "1.0.0"
__author__ = "Greenhouse AI Team"

# Import what actually exists
from .core import ConfigLoader, MOEAConfig, MOEAProblem
from .objectives import ObjectiveFunction, ObjectiveType, SurrogateObjective

__all__ = [
    # Core classes
    'ConfigLoader',
    'MOEAConfig',
    'MOEAProblem',
    # Objectives
    'ObjectiveFunction',
    'ObjectiveType',
    'SurrogateObjective'
]
