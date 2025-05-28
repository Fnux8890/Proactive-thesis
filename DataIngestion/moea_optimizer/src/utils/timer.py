"""Timer utilities for performance measurement.

This module provides timing utilities that properly handle
CUDA synchronization for accurate GPU timing measurements.
"""

import logging
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class Timer:
    """Simple timer for measuring execution time."""

    def __init__(self, name: str = "Timer", cuda_sync: bool = False):
        """Initialize timer.

        Args:
            name: Name for this timer
            cuda_sync: Whether to synchronize CUDA before timing
        """
        self.name = name
        self.cuda_sync = cuda_sync
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.elapsed: float | None = None

    def start(self) -> None:
        """Start the timer."""
        if self.cuda_sync:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except ImportError:
                pass

        self.start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer and return elapsed time."""
        if self.cuda_sync:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except ImportError:
                pass

        self.end_time = time.perf_counter()

        if self.start_time is None:
            raise RuntimeError("Timer was not started")

        self.elapsed = self.end_time - self.start_time
        return self.elapsed

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        if self.elapsed is not None:
            logger.debug(f"{self.name} took {self.elapsed:.4f} seconds")


class MultiTimer:
    """Timer for tracking multiple named timing measurements."""

    def __init__(self, cuda_sync: bool = False):
        """Initialize multi-timer.

        Args:
            cuda_sync: Whether to synchronize CUDA for all timings
        """
        self.cuda_sync = cuda_sync
        self.timings: dict[str, list[float]] = {}
        self._active_timers: dict[str, float] = {}

    def start(self, name: str) -> None:
        """Start timing for a named operation."""
        if self.cuda_sync:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except ImportError:
                pass

        self._active_timers[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        """Stop timing for a named operation and record elapsed time."""
        if name not in self._active_timers:
            raise ValueError(f"Timer '{name}' was not started")

        if self.cuda_sync:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except ImportError:
                pass

        elapsed = time.perf_counter() - self._active_timers[name]

        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(elapsed)

        del self._active_timers[name]
        return elapsed

    @contextmanager
    def time(self, name: str):
        """Context manager for timing a named operation."""
        self.start(name)
        try:
            yield self
        finally:
            self.stop(name)

    def get_stats(self, name: str) -> dict[str, float]:
        """Get timing statistics for a named operation."""
        if name not in self.timings:
            raise ValueError(f"No timings recorded for '{name}'")

        times = self.timings[name]
        return {
            'count': len(times),
            'total': sum(times),
            'mean': sum(times) / len(times),
            'min': min(times),
            'max': max(times),
            'std': np.std(times) if len(times) > 1 else 0.0
        }

    def get_all_stats(self) -> dict[str, dict[str, float]]:
        """Get timing statistics for all operations."""
        return {name: self.get_stats(name) for name in self.timings}

    def reset(self, name: str | None = None) -> None:
        """Reset timing data."""
        if name is not None:
            if name in self.timings:
                del self.timings[name]
            if name in self._active_timers:
                del self._active_timers[name]
        else:
            self.timings.clear()
            self._active_timers.clear()

    def report(self) -> str:
        """Generate a timing report."""
        if not self.timings:
            return "No timings recorded"

        lines = ["Timing Report:", "-" * 50]

        for name, stats in self.get_all_stats().items():
            lines.append(
                f"{name}: "
                f"count={stats['count']}, "
                f"total={stats['total']:.4f}s, "
                f"mean={stats['mean']:.4f}s, "
                f"min={stats['min']:.4f}s, "
                f"max={stats['max']:.4f}s"
            )

        return "\n".join(lines)


# Global timer instance for convenience
_global_timer = MultiTimer()


def start_timer(name: str, cuda_sync: bool = False) -> None:
    """Start global timer for named operation."""
    if cuda_sync:
        _global_timer.cuda_sync = cuda_sync
    _global_timer.start(name)


def stop_timer(name: str) -> float:
    """Stop global timer for named operation."""
    return _global_timer.stop(name)


@contextmanager
def time_block(name: str, cuda_sync: bool = False):
    """Context manager for timing a code block using global timer."""
    if cuda_sync:
        _global_timer.cuda_sync = cuda_sync
    with _global_timer.time(name):
        yield


def get_timing_report() -> str:
    """Get report from global timer."""
    return _global_timer.report()


def reset_global_timer() -> None:
    """Reset global timer."""
    _global_timer.reset()


# Import numpy for statistics
try:
    import numpy as np
except ImportError:
    # Fallback implementation without numpy
    def np_std(values):
        if len(values) <= 1:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5

    # Replace np.std reference
    import sys
    sys.modules[__name__].np = type('np', (), {'std': np_std})
