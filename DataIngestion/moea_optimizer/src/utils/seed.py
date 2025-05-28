"""Random seed utilities for reproducible experiments.

This module provides utilities for setting random seeds across
different libraries (NumPy, PyTorch, Python random) to ensure
reproducible results in MOEA experiments.
"""

import logging
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_all_seeds(seed: int, worker_id: int = 0, cuda_deterministic: bool = True) -> None:
    """Set random seeds for all libraries to ensure reproducibility.

    Args:
        seed: Base random seed
        worker_id: Worker ID for parallel execution (added to base seed)
        cuda_deterministic: Whether to enforce deterministic CUDA operations
    """
    effective_seed = seed + worker_id

    # Python random
    random.seed(effective_seed)
    logger.debug(f"Set Python random seed to {effective_seed}")

    # NumPy
    np.random.seed(effective_seed)
    logger.debug(f"Set NumPy random seed to {effective_seed}")

    # PyTorch (if available)
    try:
        import torch

        torch.manual_seed(effective_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(effective_seed)
            torch.cuda.manual_seed_all(effective_seed)

            if cuda_deterministic:
                # This can impact performance but ensures reproducibility
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                logger.debug("Set CUDA to deterministic mode")

        logger.debug(f"Set PyTorch random seed to {effective_seed}")

    except ImportError:
        logger.debug("PyTorch not available, skipping PyTorch seed setting")


def get_rng(seed: int | None = None) -> np.random.Generator:
    """Create a NumPy random number generator with optional seed.

    Args:
        seed: Random seed (if None, uses default initialization)

    Returns:
        NumPy random generator instance
    """
    if seed is not None:
        return np.random.default_rng(seed)
    return np.random.default_rng()


class SeedManager:
    """Context manager for temporary seed setting."""

    def __init__(self, seed: int):
        """Initialize seed manager.

        Args:
            seed: Random seed to use within context
        """
        self.seed = seed
        self.old_python_state = None
        self.old_numpy_state = None
        self.old_torch_state = None
        self.old_cuda_state = None

    def __enter__(self):
        """Save current random states and set new seed."""
        # Save Python random state
        self.old_python_state = random.getstate()

        # Save NumPy random state
        self.old_numpy_state = np.random.get_state()

        # Save PyTorch states if available
        try:
            import torch
            self.old_torch_state = torch.get_rng_state()
            if torch.cuda.is_available():
                self.old_cuda_state = torch.cuda.get_rng_state_all()
        except ImportError:
            pass

        # Set new seeds
        set_all_seeds(self.seed)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore previous random states."""
        # Restore Python random state
        random.setstate(self.old_python_state)

        # Restore NumPy random state
        np.random.set_state(self.old_numpy_state)

        # Restore PyTorch states if available
        try:
            import torch
            if self.old_torch_state is not None:
                torch.set_rng_state(self.old_torch_state)
            if self.old_cuda_state is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(self.old_cuda_state)
        except ImportError:
            pass
