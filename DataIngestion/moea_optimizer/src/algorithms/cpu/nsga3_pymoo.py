"""CPU-based NSGA-III implementation using pymoo.

This module provides a wrapper around pymoo's NSGA-III algorithm
for multi-objective optimization on CPU.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions

from ...core.config_loader import MOEAConfig
from ...utils.timer import MultiTimer

logger = logging.getLogger(__name__)


class PymooNSGA3Wrapper:
    """Wrapper for pymoo's NSGA-III algorithm."""

    def __init__(self, config: MOEAConfig):
        """Initialize NSGA-III wrapper with configuration.

        Args:
            config: Experiment configuration
        """
        self.config = config
        self.timer = MultiTimer(cuda_sync=False)
        self.algorithm = None
        self.result = None
        self.history = []

    def create_algorithm(self, problem: Problem) -> NSGA3:
        """Create NSGA-III algorithm instance.

        Args:
            problem: Problem instance to optimize

        Returns:
            Configured NSGA-III algorithm
        """
        # Get reference directions
        n_obj = problem.n_obj
        ref_dirs = get_reference_directions(
            "energy",  # or "das-dennis" based on config
            n_obj,
            self.config.algorithm.n_reference_points,
        )

        # Configure operators
        sampling = FloatRandomSampling()

        crossover = SBX(
            prob=self.config.algorithm.crossover_probability,
            eta=self.config.algorithm.crossover_eta
        )

        mutation = PM(
            prob=self.config.algorithm.mutation_probability,
            eta=self.config.algorithm.mutation_eta
        )

        # Create algorithm
        algorithm = NSGA3(
            ref_dirs=ref_dirs,
            pop_size=self.config.algorithm.population_size,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=True,
        )

        return algorithm

    def run(
        self,
        problem: Problem,
        seed: int | None = None,
        callback: Any | None = None
    ) -> dict[str, Any]:
        """Run NSGA-III optimization.

        Args:
            problem: Problem to optimize
            seed: Random seed for this run
            callback: Optional callback for logging progress

        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting NSGA-III optimization with population size {self.config.algorithm.population_size}")

        # Create algorithm
        self.algorithm = self.create_algorithm(problem)

        # Configure termination
        termination = get_termination("n_gen", self.config.algorithm.n_generations)

        # Start timing
        with self.timer.time("total_runtime"):
            # Run optimization
            self.result = minimize(
                problem,
                self.algorithm,
                termination,
                seed=seed,
                callback=callback,
                save_history=self.config.output.save_history,
                verbose=self.config.algorithm.verbose
            )

        # Extract results
        results = {
            "algorithm": "NSGA-III (pymoo)",
            "problem": {
                "name": problem.__class__.__name__,
                "n_var": problem.n_var,
                "n_obj": problem.n_obj,
                "n_constr": problem.n_constr
            },
            "n_generations": self.result.algorithm.n_gen,
            "n_evaluations": self.result.algorithm.evaluator.n_eval,
            "runtime": self.timer.timings["total_runtime"][0],
            "final_population": {
                "X": self.result.pop.get("X"),  # Decision variables
                "F": self.result.pop.get("F"),  # Objective values
                "G": self.result.pop.get("G") if problem.n_constr > 0 else None  # Constraints
            },
            "pareto_front": {
                "X": self.result.X,
                "F": self.result.F,
                "G": self.result.G if problem.n_constr > 0 else None
            },
            "history": self.result.history if self.config.output.save_history else None,
            "exec_time": self.result.exec_time,
            "timing_stats": self.timer.get_all_stats()
        }

        logger.info(f"Optimization completed: {results['n_evaluations']} evaluations in {results['runtime']:.2f}s")

        return results

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
            "n_var": results["problem"]["n_var"],
            "n_obj": results["problem"]["n_obj"],
            "n_generations": results["n_generations"],
            "n_evaluations": results["n_evaluations"],
            "runtime": results["runtime"],
            "pareto_size": len(results["pareto_front"]["F"])
        }])
        metrics_df.to_csv(output_dir / "metrics.csv", index=False)

        # Save timing details
        timing_df = pd.DataFrame(results["timing_stats"]).T
        timing_df.to_csv(output_dir / "timing_details.csv")

        logger.info(f"Results saved to {output_dir}")

    def get_callback(self, log_interval: int = 10) -> Any:
        """Create callback for progress logging.

        Args:
            log_interval: Generation interval for logging

        Returns:
            Callback function
        """
        from pymoo.core.callback import Callback

        class ProgressCallback(Callback):
            def __init__(self, timer: MultiTimer):
                super().__init__()
                self.timer = timer
                self.gen_times = []

            def notify(self, algorithm):
                gen = algorithm.n_gen

                # Time each generation
                if gen > 1:
                    self.timer.stop(f"generation_{gen-1}")
                self.timer.start(f"generation_{gen}")

                # Log progress at intervals
                if gen % log_interval == 0:
                    pop = algorithm.pop
                    logger.info(
                        f"Generation {gen}: "
                        f"Best objectives: {pop.get('F').min(axis=0)}, "
                        f"Evaluations: {algorithm.evaluator.n_eval}"
                    )

        return ProgressCallback(self.timer)


class GreenhouseProblem(Problem):
    """Greenhouse optimization problem wrapper for pymoo."""

    def __init__(
        self,
        n_var: int,
        n_obj: int,
        objectives: list,
        decision_variables: list,
        constraints: dict,
        **kwargs
    ):
        # Extract bounds from decision variables
        xl = np.array([dv.bounds[0] for dv in decision_variables]) if decision_variables else np.zeros(n_var)
        xu = np.array([dv.bounds[1] for dv in decision_variables]) if decision_variables else np.ones(n_var)
        
        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_constr=0,  # TODO: Add constraint handling
            xl=xl,
            xu=xu,
            **kwargs
        )
        
        self.objectives = objectives
        self.decision_variables = decision_variables
        self.constraints = constraints

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate the greenhouse optimization objectives.
        
        For now, this is a placeholder that returns random values.
        In production, this would call the actual greenhouse models.
        """
        # TODO: Replace with actual model evaluation
        # This would typically:
        # 1. Convert decision variables to control setpoints
        # 2. Run greenhouse simulation or surrogate models
        # 3. Calculate objective values (energy, growth, etc.)
        
        # Placeholder: return random objective values
        n_pop = x.shape[0]
        f = np.random.rand(n_pop, self.n_obj)
        
        # Make it somewhat realistic (minimize first objective, maximize others)
        f[:, 0] = np.sum(x**2, axis=1)  # Energy (minimize)
        for i in range(1, self.n_obj):
            f[:, i] = -np.sum(x * (i+1), axis=1)  # Other objectives (maximize -> minimize negative)
        
        out["F"] = f


class DTLZProblem(Problem):
    """Wrapper for DTLZ test problems."""

    def __init__(self, problem_name: str, n_var: int, n_obj: int, **kwargs):
        """Initialize DTLZ problem.

        Args:
            problem_name: Name of DTLZ problem (e.g., "DTLZ1")
            n_var: Number of variables
            n_obj: Number of objectives
            **kwargs: Additional problem-specific parameters
        """
        from pymoo.problems import get_problem

        self.problem_name = problem_name
        self.base_problem = get_problem(problem_name.lower(), n_var=n_var, n_obj=n_obj, **kwargs)

        super().__init__(
            n_var=self.base_problem.n_var,
            n_obj=self.base_problem.n_obj,
            n_constr=self.base_problem.n_constr,
            xl=self.base_problem.xl,
            xu=self.base_problem.xu,
            type_var=self.base_problem.type_var,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate the problem."""
        self.base_problem._evaluate(x, out, *args, **kwargs)


def create_problem(problem_config: dict[str, Any]) -> Problem:
    """Create problem instance from configuration.

    Args:
        problem_config: Problem configuration dictionary

    Returns:
        Problem instance
    """
    name = problem_config["name"]

    if name.startswith("DTLZ"):
        return DTLZProblem(
            problem_name=name,
            n_var=problem_config["n_var"],
            n_obj=problem_config["n_obj"],
            **{k: v for k, v in problem_config.items()
               if k not in ["name", "n_var", "n_obj"]}
        )
    elif name == "GreenhouseOptimization":
        # For now, create a simple test problem with the correct dimensions
        # In production, this would connect to the actual greenhouse models
        return GreenhouseProblem(
            n_var=problem_config["n_var"],
            n_obj=problem_config["n_obj"],
            objectives=problem_config.get("objectives", []),
            decision_variables=problem_config.get("decision_variables", []),
            constraints=problem_config.get("constraints", {})
        )
    else:
        raise ValueError(f"Unknown problem type: {name}")


# Example usage
if __name__ == "__main__":
    from ...core.config_loader import ConfigLoader

    # Load configuration
    config_path = Path(__file__).parent.parent.parent.parent / "config" / "cpu_dtlz.yaml"
    loader = ConfigLoader(config_path)
    config = loader.load()

    # Create wrapper
    wrapper = PymooNSGA3Wrapper(config)

    # Run on first problem
    problem_config = config.problem.problems[0]
    problem = create_problem(problem_config)

    results = wrapper.run(problem, seed=42)

    # Save results
    output_dir = Path("results") / config.output.experiment_dir / problem_config["name"]
    wrapper.save_results(results, output_dir)
