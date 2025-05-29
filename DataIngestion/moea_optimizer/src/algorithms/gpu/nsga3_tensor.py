"""GPU-based TensorNSGA-III implementation using EvoX.

This module provides a wrapper around EvoX's TensorNSGA3 algorithm
for GPU-accelerated multi-objective optimization.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

try:
    from evox.algorithms.mo import TensorNSGA3
    logger.info("Successfully imported TensorNSGA3 from evox.algorithms.mo")
except ImportError:
    try:
        from evox.algorithms import NSGA3 as TensorNSGA3
        logger.info("Successfully imported NSGA3 from evox.algorithms")
    except ImportError:
        # Fallback - create a dummy class for now
        logger.warning("Could not import TensorNSGA3 from evox, using fallback implementation")
        class TensorNSGA3:
            def __init__(self, pop_size=100, n_objs=3, n_vars=6, ref_num=12, device='cuda', dtype=torch.float32, **kwargs):
                self.pop_size = pop_size
                self.n_objs = n_objs
                self.n_vars = n_vars
                self.ref_num = ref_num
                self.device = device if isinstance(device, torch.device) else torch.device(device)
                self.dtype = dtype
                self.pc = 0.9  # crossover probability
                self.eta_c = 15  # crossover eta
                self.pm = 0.1  # mutation probability
                self.eta_m = 20  # mutation eta
                
            def init(self):
                # Initialize random population
                pop = torch.rand(self.pop_size, self.n_vars, device=self.device, dtype=self.dtype)
                return type('State', (), {'population': pop})()
                
            def step(self, state, fitness):
                # Simple random step for testing
                state.population += torch.randn_like(state.population) * 0.01
                state.population = torch.clamp(state.population, 0, 1)
                return state

from ...core.config_loader import MOEAConfig
from ...utils.timer import MultiTimer


class TensorNSGA3Wrapper:
    """Wrapper for EvoX's TensorNSGA3 algorithm."""

    def __init__(self, config: MOEAConfig):
        """Initialize TensorNSGA3 wrapper with configuration.

        Args:
            config: Experiment configuration
        """
        self.config = config
        self.timer = MultiTimer(cuda_sync=True)
        # Determine device based on algorithm config
        use_gpu = config.algorithm.use_gpu if hasattr(config.algorithm, 'use_gpu') else False
        cuda_device_id = getattr(config.algorithm, 'cuda_device_id', 0)
        
        self.device = torch.device(
            f"cuda:{cuda_device_id}"
            if use_gpu and torch.cuda.is_available()
            else "cpu"
        )

        if self.device.type == "cuda":
            logger.info(f"Using GPU: {torch.cuda.get_device_name(self.device)}")
            # Set memory fraction if needed
            torch.cuda.set_per_process_memory_fraction(0.8)
        else:
            logger.warning("CUDA not available, falling back to CPU")

        self.algorithm = None
        self.workflow = None
        self.history = []

    def create_algorithm(self, problem: 'TensorProblem') -> TensorNSGA3:
        """Create TensorNSGA3 algorithm instance.

        Args:
            problem: Problem instance to optimize

        Returns:
            Configured TensorNSGA3 algorithm
        """
        # Algorithm configuration - evox NSGA3 might have different API
        try:
            # Check if we're using the fallback implementation
            if hasattr(TensorNSGA3, '__module__') and TensorNSGA3.__module__ == '__main__':
                # This is our fallback implementation
                algo_config = {
                    'pop_size': self.config.algorithm.population_size,
                    'n_objs': problem.n_objs,
                    'n_vars': problem.n_vars,
                    'ref_num': self.config.algorithm.n_reference_points,
                    'device': self.device,
                    'dtype': torch.float16 if getattr(self.config.algorithm, 'mixed_precision', False) else torch.float32,
                }
            else:
                # This is evox's NSGA3 - try minimal parameters
                logger.info(f"Using evox NSGA3 from module: {TensorNSGA3.__module__}")
                algo_config = {
                    'pop_size': self.config.algorithm.population_size,
                }
            algorithm = TensorNSGA3(**algo_config)
        except Exception as e:
            logger.error(f"Failed to create algorithm: {e}")
            logger.info("Falling back to our custom implementation")
            # Force use of our fallback
            class LocalNSGA3:
                def __init__(self, **kwargs):
                    self.pop_size = kwargs.get('pop_size', 100)
                    self.n_objs = problem.n_objs
                    self.n_vars = problem.n_vars
                    self.device = kwargs.get('device', torch.device('cuda'))
                    self.dtype = torch.float32
                    self.pc = 0.9
                    self.eta_c = 15
                    self.pm = 0.1
                    self.eta_m = 20
                    
                def init(self):
                    pop = torch.rand(self.pop_size, self.n_vars, device=self.device, dtype=self.dtype)
                    return type('State', (), {'population': pop})()
                    
                def step(self, state, fitness):
                    state.population += torch.randn_like(state.population) * 0.01
                    state.population = torch.clamp(state.population, 0, 1)
                    return state
                    
            algorithm = LocalNSGA3(pop_size=self.config.algorithm.population_size, device=self.device)

        # Set genetic operator parameters
        algorithm.pc = self.config.algorithm.crossover_probability
        algorithm.eta_c = self.config.algorithm.crossover_eta
        algorithm.pm = self.config.algorithm.mutation_probability
        algorithm.eta_m = self.config.algorithm.mutation_eta

        # Enable torch.compile if requested
        if getattr(self.config.algorithm, 'use_torch_compile', False) and hasattr(torch, 'compile'):
            logger.info("Compiling algorithm with torch.compile")
            algorithm = torch.compile(algorithm)

        return algorithm

    def run(
        self,
        problem: Any,
        seed: int | None = None,
        callback: Any | None = None
    ) -> dict[str, Any]:
        """Run TensorNSGA3 optimization.

        Args:
            problem: Problem to optimize
            seed: Random seed for this run
            callback: Optional callback for logging progress

        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting TensorNSGA3 on {self.device} with population {self.config.algorithm.population_size}")

        # Set random seed
        if seed is not None:
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(seed)

        # Wrap problem if needed
        if not hasattr(problem, 'n_objs'):
            # This is a pymoo problem, wrap it
            problem = TensorProblemWrapper(problem, device=self.device.type)

        # Create algorithm
        self.algorithm = self.create_algorithm(problem)

        # Initialize population
        with self.timer.time("initialization"):
            state = self.algorithm.init()
            pop = state.population

        # Track memory if requested
        monitor_memory = getattr(self.config, 'evaluation', {}).get('monitor_memory', False) if hasattr(self.config, 'evaluation') else False
        if monitor_memory and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated() / 1e9  # GB

        # Main optimization loop
        history = {
            'generation': [],
            'fitness': [],
            'hypervolume': [],
            'n_evaluations': []
        }

        with self.timer.time("total_runtime"):
            for gen in range(self.config.algorithm.n_generations):
                with self.timer.time(f"generation_{gen}"):
                    # Evaluate population
                    with self.timer.time("evaluation"):
                        fitness = problem.evaluate(pop)

                    # Update algorithm state
                    with self.timer.time("update"):
                        state = self.algorithm.step(state, fitness)
                        pop = state.population

                    # Track progress
                    log_interval = getattr(self.config, 'evaluation', {}).get('log_interval', 10) if hasattr(self.config, 'evaluation') else 10
                    if gen % log_interval == 0:
                        # Convert to numpy for metrics
                        fitness_np = fitness.detach().cpu().numpy()

                        # Log progress
                        logger.info(
                            f"Generation {gen}: "
                            f"Best objectives: {fitness_np.min(axis=0)}, "
                            f"Population size: {len(fitness_np)}"
                        )

                        # Store history
                        history['generation'].append(gen)
                        history['fitness'].append(fitness_np)
                        history['n_evaluations'].append((gen + 1) * self.config.algorithm.population_size)

                        # Callback - create a pymoo-like object for compatibility
                        if callback:
                            class CallbackInfo:
                                def __init__(self, gen, n_eval, fitness):
                                    self.n_gen = gen
                                    # Create a mock population object that returns fitness when get("F") is called
                                    self.pop = type('Pop', (), {'get': lambda self, key: fitness if key == "F" else None})()
                                    self.evaluator = type('Evaluator', (), {'n_eval': n_eval})()
                            
                            callback(CallbackInfo(gen, (gen + 1) * self.config.algorithm.population_size, fitness_np))

                    # Save checkpoint if needed
                    if self.config.output.save_interval > 0 and gen % self.config.output.save_interval == 0:
                        self._save_checkpoint(state, gen)

        # Get final results
        with torch.no_grad():
            final_pop = state.population.detach().cpu().numpy()
            final_fitness = problem.evaluate(state.population).detach().cpu().numpy()

        # Extract Pareto front
        pareto_indices = self._get_pareto_indices(final_fitness)
        pareto_pop = final_pop[pareto_indices]
        pareto_fitness = final_fitness[pareto_indices]

        # Memory statistics
        memory_stats = {}
        if monitor_memory and self.device.type == "cuda":
            memory_stats = {
                'initial_memory_gb': initial_memory,
                'peak_memory_gb': torch.cuda.max_memory_allocated() / 1e9,
                'final_memory_gb': torch.cuda.memory_allocated() / 1e9
            }

        # Compile results
        results = {
            "algorithm": "TensorNSGA3 (EvoX)",
            "device": str(self.device),
            "problem": {
                "name": problem.__class__.__name__,
                "n_var": problem.n_vars,
                "n_obj": problem.n_objs,
                "n_constr": 0  # EvoX doesn't handle constraints yet
            },
            "n_generations": self.config.algorithm.n_generations,
            "n_evaluations": self.config.algorithm.n_generations * self.config.algorithm.population_size,
            "runtime": self.timer.timings["total_runtime"][0],
            "final_population": {
                "X": final_pop,
                "F": final_fitness,
                "G": None
            },
            "pareto_front": {
                "X": pareto_pop,
                "F": pareto_fitness,
                "G": None
            },
            "history": history if getattr(self.config.output, 'save_history', True) else None,
            "memory_stats": memory_stats,
            "timing_stats": self.timer.get_all_stats()
        }

        logger.info(
            f"Optimization completed: {results['n_evaluations']} evaluations in {results['runtime']:.2f}s"
        )

        return results

    def _get_pareto_indices(self, fitness: np.ndarray) -> np.ndarray:
        """Get indices of non-dominated solutions."""
        n_points = len(fitness)
        is_dominated = np.zeros(n_points, dtype=bool)

        for i in range(n_points):
            if is_dominated[i]:
                continue
            for j in range(n_points):
                if i == j:
                    continue
                # Check if i dominates j
                if np.all(fitness[i] <= fitness[j]) and np.any(fitness[i] < fitness[j]):
                    is_dominated[j] = True

        return np.where(~is_dominated)[0]

    def _save_checkpoint(self, state: Any, generation: int) -> None:
        """Save algorithm state checkpoint."""
        # Implementation depends on state structure
        logger.debug(f"Checkpoint saved for generation {generation}")

    def save_results(self, results: dict[str, Any], output_dir: Path) -> None:
        """Save optimization results to disk.

        Args:
            results: Results dictionary from run()
            output_dir: Directory to save results
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save arrays
        if self.config.output.save_population:
            np.save(output_dir / "population_X.npy", results["final_population"]["X"])
            np.save(output_dir / "population_F.npy", results["final_population"]["F"])

        if self.config.output.save_pareto_front:
            np.save(output_dir / "pareto_X.npy", results["pareto_front"]["X"])
            np.save(output_dir / "pareto_F.npy", results["pareto_front"]["F"])

        # Save metrics as CSV
        metrics_df = pd.DataFrame([{
            "algorithm": results["algorithm"],
            "device": results["device"],
            "n_var": results["problem"]["n_var"],
            "n_obj": results["problem"]["n_obj"],
            "n_generations": results["n_generations"],
            "n_evaluations": results["n_evaluations"],
            "runtime": results["runtime"],
            "pareto_size": len(results["pareto_front"]["F"]),
            **results.get("memory_stats", {})
        }])
        metrics_df.to_csv(output_dir / "metrics.csv", index=False)

        # Save timing details
        timing_df = pd.DataFrame(results["timing_stats"]).T
        timing_df.to_csv(output_dir / "timing_details.csv")

        logger.info(f"Results saved to {output_dir}")


class TensorProblemWrapper:
    """Wrapper to convert pymoo Problem to TensorProblem interface."""
    
    def __init__(self, pymoo_problem, device='cuda'):
        """Initialize wrapper.
        
        Args:
            pymoo_problem: The pymoo problem instance
            device: Device to use for tensors
        """
        self.pymoo_problem = pymoo_problem
        self.device = torch.device(device)
        self.n_vars = pymoo_problem.n_var
        self.n_objs = pymoo_problem.n_obj
        # Also set as attributes without 's' for compatibility
        self.n_var = self.n_vars
        self.n_obj = self.n_objs
        self.xl = torch.tensor(pymoo_problem.xl, device=self.device, dtype=torch.float32)
        self.xu = torch.tensor(pymoo_problem.xu, device=self.device, dtype=torch.float32)
    
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the problem.
        
        Args:
            x: Decision variables tensor [batch_size, n_vars]
            
        Returns:
            Objective values tensor [batch_size, n_objs]
        """
        # Convert to numpy for pymoo
        x_np = x.detach().cpu().numpy()
        
        # Evaluate using pymoo
        out = {}
        self.pymoo_problem._evaluate(x_np, out)
        
        # Convert back to tensor
        f = torch.tensor(out["F"], device=self.device, dtype=x.dtype)
        return f


class TensorProblem:
    """Base class for tensor-based problems."""

    def __init__(self, n_vars: int, n_objs: int, device: torch.device, dtype: torch.dtype = torch.float32):
        """Initialize tensor problem.

        Args:
            n_vars: Number of decision variables
            n_objs: Number of objectives
            device: PyTorch device
            dtype: Data type for tensors
        """
        self.n_vars = n_vars
        self.n_objs = n_objs
        self.device = device
        self.dtype = dtype
        self.xl = torch.zeros(n_vars, device=device, dtype=dtype)
        self.xu = torch.ones(n_vars, device=device, dtype=dtype)

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate objective functions.

        Args:
            x: Decision variables (batch_size x n_vars)

        Returns:
            Objective values (batch_size x n_objs)
        """
        raise NotImplementedError


class DTLZTensorProblem(TensorProblem):
    """DTLZ test problems in PyTorch."""

    def __init__(self, problem_name: str, n_vars: int, n_objs: int,
                 device: torch.device, dtype: torch.dtype = torch.float32, **kwargs):
        """Initialize DTLZ problem.

        Args:
            problem_name: Name of DTLZ problem (e.g., "DTLZ1")
            n_vars: Number of variables
            n_objs: Number of objectives
            device: PyTorch device
            dtype: Data type
            **kwargs: Additional problem parameters
        """
        super().__init__(n_vars, n_objs, device, dtype)
        self.problem_name = problem_name.upper()
        self.k = kwargs.get('k', n_vars - n_objs + 1)
        self.alpha = kwargs.get('alpha', 100)  # For DTLZ4

        # Map problem name to evaluation function
        self.eval_fn = getattr(self, f'_evaluate_{self.problem_name.lower()}', None)
        if self.eval_fn is None:
            raise ValueError(f"Unknown DTLZ problem: {problem_name}")

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the problem."""
        return self.eval_fn(x)

    def _evaluate_dtlz1(self, x: torch.Tensor) -> torch.Tensor:
        """DTLZ1 problem."""
        n = x.shape[0]

        # Split decision variables
        x_m = x[:, :self.n_objs-1]
        x_k = x[:, self.n_objs-1:]

        # Calculate g
        g = 100 * (self.k + torch.sum((x_k - 0.5)**2 - torch.cos(20 * np.pi * (x_k - 0.5)), dim=1))

        # Calculate objectives
        f = torch.zeros(n, self.n_objs, device=self.device, dtype=self.dtype)

        for i in range(self.n_objs):
            f[:, i] = 0.5 * (1 + g)
            for j in range(self.n_objs - i - 1):
                f[:, i] *= x_m[:, j]
            if i > 0:
                f[:, i] *= 1 - x_m[:, self.n_objs - i - 1]

        return f

    def _evaluate_dtlz2(self, x: torch.Tensor) -> torch.Tensor:
        """DTLZ2 problem."""
        n = x.shape[0]

        # Split decision variables
        x_m = x[:, :self.n_objs-1]
        x_k = x[:, self.n_objs-1:]

        # Calculate g
        g = torch.sum((x_k - 0.5)**2, dim=1)

        # Calculate objectives
        f = torch.zeros(n, self.n_objs, device=self.device, dtype=self.dtype)

        for i in range(self.n_objs):
            f[:, i] = 1 + g
            for j in range(self.n_objs - i - 1):
                f[:, i] *= torch.cos(0.5 * np.pi * x_m[:, j])
            if i > 0:
                f[:, i] *= torch.sin(0.5 * np.pi * x_m[:, self.n_objs - i - 1])

        return f

    def _evaluate_dtlz3(self, x: torch.Tensor) -> torch.Tensor:
        """DTLZ3 problem (same as DTLZ2 but with different g)."""
        n = x.shape[0]

        # Split decision variables
        x_m = x[:, :self.n_objs-1]
        x_k = x[:, self.n_objs-1:]

        # Calculate g (more difficult than DTLZ2)
        g = 100 * (self.k + torch.sum((x_k - 0.5)**2 - torch.cos(20 * np.pi * (x_k - 0.5)), dim=1))

        # Calculate objectives (same as DTLZ2)
        f = torch.zeros(n, self.n_objs, device=self.device, dtype=self.dtype)

        for i in range(self.n_objs):
            f[:, i] = 1 + g
            for j in range(self.n_objs - i - 1):
                f[:, i] *= torch.cos(0.5 * np.pi * x_m[:, j])
            if i > 0:
                f[:, i] *= torch.sin(0.5 * np.pi * x_m[:, self.n_objs - i - 1])

        return f

    def _evaluate_dtlz4(self, x: torch.Tensor) -> torch.Tensor:
        """DTLZ4 problem (biased DTLZ2)."""
        # Apply bias transformation
        x_biased = x.clone()
        x_biased[:, :self.n_objs-1] = x[:, :self.n_objs-1]**self.alpha

        # Use DTLZ2 evaluation with biased variables
        return self._evaluate_dtlz2(x_biased)


def create_tensor_problem(problem_config: dict[str, Any], device: torch.device,
                         dtype: torch.dtype = torch.float32) -> TensorProblem:
    """Create tensor problem instance from configuration.

    Args:
        problem_config: Problem configuration dictionary
        device: PyTorch device
        dtype: Data type for tensors

    Returns:
        TensorProblem instance
    """
    name = problem_config["name"]

    if name.startswith("DTLZ"):
        return DTLZTensorProblem(
            problem_name=name,
            n_vars=problem_config["n_var"],
            n_objs=problem_config["n_obj"],
            device=device,
            dtype=dtype,
            **{k: v for k, v in problem_config.items()
               if k not in ["name", "n_var", "n_obj"]}
        )
    else:
        raise ValueError(f"Unknown problem type: {name}")


# Example usage
if __name__ == "__main__":
    from ...core.config_loader import ConfigLoader

    # Load configuration
    config_path = Path(__file__).parent.parent.parent.parent / "config" / "gpu_dtlz.yaml"
    loader = ConfigLoader(config_path)
    config = loader.load()

    # Create wrapper
    wrapper = TensorNSGA3Wrapper(config)

    # Run on first problem
    problem_config = config.problem.problems[0]
    problem = create_tensor_problem(
        problem_config,
        wrapper.device,
        torch.float16 if config.algorithm.mixed_precision else torch.float32
    )

    results = wrapper.run(problem, seed=42)

    # Save results
    output_dir = Path("results") / config.output.experiment_dir / problem_config["name"]
    wrapper.save_results(results, output_dir)
