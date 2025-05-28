"""Multi-Objective Evolutionary Algorithm (MOEA) Optimizer for Greenhouse Climate Control.

This package implements various MOEA algorithms for optimizing greenhouse
climate control strategies with multiple conflicting objectives such as
energy efficiency, plant growth, water usage, and crop quality.
"""

__version__ = "1.0.0"
__author__ = "Greenhouse AI Team"

from .algorithms import MOEAD, NSGA2, NSGA3, SPEA2, get_algorithm
from .core import MOEAProblem, MOEASolver, Population, Solution
from .objectives import CompositeObjective, ObjectiveFunction, ObjectiveManager, SurrogateObjective

__all__ = [
    'MOEAD',
    # Algorithms
    'NSGA2',
    'NSGA3',
    'SPEA2',
    'CompositeObjective',
    # Core classes
    'MOEAProblem',
    'MOEASolver',
    # Objectives
    'ObjectiveFunction',
    'ObjectiveManager',
    'Population',
    'Solution',
    'SurrogateObjective',
    'get_algorithm'
]
