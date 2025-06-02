"""
Algorithm Strategy Pattern Interface

This module provides the abstract base class and factory for MOEA algorithms,
implementing the Strategy design pattern for algorithm selection and execution.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MOEAAlgorithm(ABC):
    """Abstract base class for Multi-Objective Evolutionary Algorithms.
    
    This interface defines the contract that all MOEA implementations must follow,
    enabling polymorphic algorithm usage and easy algorithm switching.
    """
    
    @abstractmethod
    def run(self, problem: Any, seed: Optional[int] = None, callback: Optional[Any] = None) -> Dict[str, Any]:
        """Run the optimization algorithm.
        
        Args:
            problem: Problem instance to optimize
            seed: Random seed for reproducibility
            callback: Optional callback for monitoring progress
            
        Returns:
            Dictionary containing optimization results
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the algorithm name.
        
        Returns:
            String identifier for the algorithm
        """
        pass
    
    @abstractmethod
    def get_device(self) -> str:
        """Get the computational device used by the algorithm.
        
        Returns:
            Device identifier (e.g., 'cpu', 'cuda:0')
        """
        pass
    
    @abstractmethod
    def save_results(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Save optimization results to disk.
        
        Args:
            results: Results dictionary from run()
            output_dir: Directory to save results
        """
        pass
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get algorithm capabilities and features.
        
        Returns:
            Dictionary describing algorithm capabilities
        """
        return {
            "name": self.get_name(),
            "device": self.get_device(),
            "supports_constraints": False,
            "supports_mixed_variables": False,
            "parallel_evaluation": False,
        }


class AlgorithmFactory:
    """Factory class for creating MOEA algorithm instances.
    
    This factory implements the Strategy pattern by providing a central
    point for algorithm creation and configuration.
    """
    
    # Registry of available algorithms
    _algorithms = {}
    
    @classmethod
    def register_algorithm(cls, name: str, algorithm_class: type, device_type: str = "any"):
        """Register an algorithm implementation.
        
        Args:
            name: Unique name for the algorithm
            algorithm_class: Class that implements MOEAAlgorithm
            device_type: Required device type ('cpu', 'gpu', 'any')
        """
        cls._algorithms[name] = {
            "class": algorithm_class,
            "device_type": device_type
        }
        logger.info(f"Registered algorithm: {name} (device: {device_type})")
    
    @classmethod
    def create_algorithm(cls, algorithm_name: str, config: Any, **kwargs) -> MOEAAlgorithm:
        """Create an algorithm instance using the factory.
        
        Args:
            algorithm_name: Name of the algorithm to create
            config: Configuration object for the algorithm
            **kwargs: Additional arguments for algorithm creation
            
        Returns:
            MOEAAlgorithm instance
            
        Raises:
            ValueError: If algorithm name is not registered
        """
        if algorithm_name not in cls._algorithms:
            available = list(cls._algorithms.keys())
            raise ValueError(f"Unknown algorithm: {algorithm_name}. Available: {available}")
        
        algorithm_info = cls._algorithms[algorithm_name]
        algorithm_class = algorithm_info["class"]
        
        try:
            return algorithm_class(config, **kwargs)
        except Exception as e:
            logger.error(f"Failed to create algorithm {algorithm_name}: {e}")
            raise
    
    @classmethod
    def get_available_algorithms(cls, device_type: Optional[str] = None) -> Dict[str, Dict]:
        """Get list of available algorithms.
        
        Args:
            device_type: Filter by device type ('cpu', 'gpu')
            
        Returns:
            Dictionary of available algorithms and their info
        """
        if device_type is None:
            return cls._algorithms.copy()
        
        filtered = {}
        for name, info in cls._algorithms.items():
            if info["device_type"] in [device_type, "any"]:
                filtered[name] = info
        
        return filtered
    
    @classmethod
    def auto_select_algorithm(cls, config: Any) -> str:
        """Automatically select the best algorithm based on configuration.
        
        Args:
            config: Configuration object
            
        Returns:
            Name of the selected algorithm
        """
        # Determine device preference
        use_gpu = getattr(config.algorithm, 'use_gpu', False) if hasattr(config, 'algorithm') else False
        device_type = "gpu" if use_gpu else "cpu"
        
        # Get available algorithms for device
        available = cls.get_available_algorithms(device_type)
        
        if not available:
            # Fallback to any device
            available = cls.get_available_algorithms()
            logger.warning(f"No {device_type} algorithms available, using fallback")
        
        if not available:
            raise RuntimeError("No algorithms available")
        
        # Priority order for algorithm selection
        priority_order = [
            "tensornsga3",  # Prefer TensorNSGA3 if available
            "nsga3_gpu",    # Then GPU NSGA-III
            "nsga3_cpu",    # Then CPU NSGA-III
        ]
        
        # Select highest priority available algorithm
        for algorithm_name in priority_order:
            if algorithm_name in available:
                logger.info(f"Auto-selected algorithm: {algorithm_name}")
                return algorithm_name
        
        # Fallback to first available
        selected = list(available.keys())[0]
        logger.info(f"Auto-selected fallback algorithm: {selected}")
        return selected


class AlgorithmContext:
    """Context class for the Strategy pattern.
    
    This class maintains a reference to one of the concrete strategies
    and delegates the work to the strategy object.
    """
    
    def __init__(self, algorithm: MOEAAlgorithm):
        """Initialize with a specific algorithm strategy.
        
        Args:
            algorithm: MOEAAlgorithm implementation to use
        """
        self._algorithm = algorithm
        logger.info(f"Algorithm context initialized with: {algorithm.get_name()}")
    
    def set_algorithm(self, algorithm: MOEAAlgorithm) -> None:
        """Change the algorithm strategy at runtime.
        
        Args:
            algorithm: New MOEAAlgorithm implementation to use
        """
        logger.info(f"Switching algorithm from {self._algorithm.get_name()} to {algorithm.get_name()}")
        self._algorithm = algorithm
    
    def run_optimization(self, problem: Any, seed: Optional[int] = None, callback: Optional[Any] = None) -> Dict[str, Any]:
        """Execute optimization using the current algorithm strategy.
        
        Args:
            problem: Problem instance to optimize
            seed: Random seed for reproducibility
            callback: Optional callback for monitoring progress
            
        Returns:
            Dictionary containing optimization results
        """
        logger.info(f"Running optimization with {self._algorithm.get_name()}")
        return self._algorithm.run(problem, seed, callback)
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get information about the current algorithm.
        
        Returns:
            Dictionary with algorithm information
        """
        return self._algorithm.get_capabilities()