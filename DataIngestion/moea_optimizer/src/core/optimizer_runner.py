"""Main optimizer runner for executing MOEA experiments.

This module provides the main entry point for running optimization
experiments with both CPU and GPU implementations.
"""

import json
import logging
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super().default(obj)

from ..algorithms.cpu.nsga3_pymoo import PymooNSGA3Wrapper, create_problem
from ..core.config_loader import ConfigLoader
from ..core.evaluation import ConvergenceTracker, PerformanceMetrics
from ..utils.seed import set_all_seeds
from ..utils.timer import MultiTimer

logger = logging.getLogger(__name__)


class OptimizationRunner:
    """Main runner for optimization experiments."""

    def __init__(self, config_path: Path):
        """Initialize optimization runner.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.loader = ConfigLoader(self.config_path)
        self.config = self.loader.load()
        self.results = []
        self.timer = MultiTimer()

    def run_single_problem(
        self,
        problem_config: dict[str, Any],
        seed: int,
        run_id: int
    ) -> dict[str, Any]:
        """Run optimization on a single problem instance.

        Args:
            problem_config: Problem configuration
            seed: Random seed for this run
            run_id: Run identifier

        Returns:
            Results dictionary
        """
        problem_name = problem_config["name"]
        logger.info(f"Running {problem_name} with seed {seed} (run {run_id})")

        # Set random seeds
        set_all_seeds(seed, worker_id=run_id)

        # Create problem
        problem = create_problem(problem_config)

        # Get reference point for metrics
        from ..core.evaluation import get_reference_point_for_problem
        ref_point = get_reference_point_for_problem(problem_name, problem.n_obj)
        ref_point = np.array(ref_point)

        # Create metrics calculator
        metrics_calc = PerformanceMetrics(ref_point=ref_point)

        # Create convergence tracker
        tracker = ConvergenceTracker(metrics_calc)

        # Create callback for tracking
        def callback_fn(algorithm):
            gen = algorithm.n_gen
            log_interval = getattr(self.config, 'evaluation', {}).get('log_interval', 10) if hasattr(self.config, 'evaluation') else 10
            if gen % log_interval == 0:
                # Get current Pareto front approximation
                F = algorithm.pop.get("F")
                n_eval = algorithm.evaluator.n_eval
                tracker.update(gen, n_eval, F)

        # Choose algorithm based on whether GPU is enabled
        use_gpu = self.config.algorithm.use_gpu if hasattr(self.config.algorithm, 'use_gpu') else False
        if not use_gpu:
            wrapper = PymooNSGA3Wrapper(self.config)
        else:
            from ..algorithms.gpu.nsga3_tensor import TensorNSGA3Wrapper
            wrapper = TensorNSGA3Wrapper(self.config)

        # Run optimization
        with self.timer.time(f"{problem_name}_run_{run_id}"):
            results = wrapper.run(problem, seed=seed, callback=callback_fn)

        # Calculate final metrics
        final_metrics = metrics_calc.calculate_all(results["pareto_front"]["F"])

        # Add metadata
        results.update({
            "problem_name": problem_name,
            "run_id": run_id,
            "seed": seed,
            "final_metrics": final_metrics,
            "convergence_history": tracker.get_dataframe().to_dict('records')
        })

        return results

    def run_experiment(self) -> None:
        """Run complete experiment based on configuration."""
        logger.info(f"Starting experiment: {self.config.meta.get('experiment_name', 'unnamed')}")

        # Create output directory
        output_base = Path(self.config.output.base_dir) / self.config.output.experiment_dir
        output_base.mkdir(parents=True, exist_ok=True)

        # Save configuration
        with open(output_base / "config.yaml", 'w') as f:
            import yaml
            yaml.dump(self.loader._raw_config if hasattr(self.loader, '_raw_config') else {}, f)

        # Run experiments
        all_results = []

        # Create a single problem configuration from the MOEA config
        problem_config = {
            "name": "GreenhouseOptimization",
            "n_var": len(self.config.decision_variables),
            "n_obj": len(self.config.get_active_objectives()),
            "objectives": self.config.objectives,
            "decision_variables": self.config.decision_variables,
            "constraints": self.config.constraints
        }

        problem_name = problem_config["name"]
        problem_results = []

        # Multiple runs with different seeds (default to 1 run if not specified)
        n_runs = getattr(self.config, 'n_runs', 1)
        base_seed = getattr(self.config, 'base_seed', 42)
        
        for run_id in range(n_runs):
            seed = base_seed + run_id
            
            try:
                results = self.run_single_problem(problem_config, seed, run_id)
                problem_results.append(results)

                # Save individual run results
                if self.config.output.save_interval > 0:
                    run_dir = output_base / problem_name / f"run_{run_id}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    self._save_run_results(results, run_dir)

            except Exception as e:
                logger.error(f"Failed to run {problem_name} (run {run_id}): {e}")
                continue

        # Aggregate results for this problem
        if problem_results:
            aggregated = self._aggregate_results(problem_results)
            aggregated["problem_name"] = problem_name
            all_results.append(aggregated)

        # Save aggregated results
        self._save_aggregated_results(all_results, output_base)

        # Generate report
        self._generate_report(all_results, output_base)

        logger.info(f"Experiment completed. Results saved to {output_base}")

    def _save_run_results(self, results: dict[str, Any], output_dir: Path) -> None:
        """Save results from a single run."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save Pareto front
        np.save(output_dir / "pareto_F.npy", results["pareto_front"]["F"])
        np.save(output_dir / "pareto_X.npy", results["pareto_front"]["X"])

        # Save metrics
        with open(output_dir / "metrics.json", 'w') as f:
            json.dump(results["final_metrics"], f, indent=2, cls=NumpyEncoder)

        # Save convergence history
        if results.get("convergence_history"):
            pd.DataFrame(results["convergence_history"]).to_csv(
                output_dir / "convergence.csv", index=False
            )

    def _aggregate_results(self, results_list: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate results from multiple runs."""
        # Extract metrics from all runs
        all_metrics = {}
        for metric_name in ["hypervolume", "igd_plus", "spacing", "n_solutions"]:
            values = []
            for result in results_list:
                if metric_name in result["final_metrics"]:
                    value = result["final_metrics"][metric_name]
                    if value is not None:
                        values.append(value)

            if values:
                all_metrics[metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "values": values
                }

        # Runtime statistics
        runtimes = [r["runtime"] for r in results_list]

        return {
            "n_runs": len(results_list),
            "metrics": all_metrics,
            "runtime": {
                "mean": np.mean(runtimes),
                "std": np.std(runtimes),
                "total": np.sum(runtimes)
            }
        }

    def _save_aggregated_results(self, results: list[dict[str, Any]], output_dir: Path) -> None:
        """Save aggregated results from all problems."""
        # Create summary DataFrame
        summary_data = []
        for result in results:
            row = {
                "problem": result["problem_name"],
                "n_runs": result["n_runs"],
                "runtime_mean": result["runtime"]["mean"],
                "runtime_std": result["runtime"]["std"]
            }

            # Add metric statistics
            for metric_name, stats in result["metrics"].items():
                row[f"{metric_name}_mean"] = stats["mean"]
                row[f"{metric_name}_std"] = stats["std"]

            summary_data.append(row)

        # Save as CSV
        pd.DataFrame(summary_data).to_csv(output_dir / "summary.csv", index=False)

        # Save complete results as JSON
        with open(output_dir / "complete_results.json", 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)

    def _generate_report(self, results: list[dict[str, Any]], output_dir: Path) -> None:
        """Generate experiment report."""
        report_lines = [
            f"# Experiment Report: {self.config.meta.get('experiment_name', 'unnamed')}",
            f"\nDescription: {self.config.meta.get('description', 'N/A')}",
            "\n## Configuration",
            f"- Algorithm: {self.config.algorithm.type}",
            f"- Population size: {self.config.algorithm.population_size}",
            f"- Generations: {self.config.algorithm.n_generations}",
            f"- Runs: {getattr(self.config, 'n_runs', 1)}",
            "\n## Results Summary\n"
        ]

        # Add results table
        if results:
            df = pd.DataFrame([{
                "Problem": r["problem_name"],
                "Runs": r["n_runs"],
                "HV Mean": f"{r['metrics'].get('hypervolume', {}).get('mean', 'N/A'):.4f}"
                          if 'hypervolume' in r['metrics'] else 'N/A',
                "HV Std": f"{r['metrics'].get('hypervolume', {}).get('std', 'N/A'):.4f}"
                         if 'hypervolume' in r['metrics'] else 'N/A',
                "Runtime": f"{r['runtime']['mean']:.2f}s"
            } for r in results])

            report_lines.append(df.to_markdown(index=False))

        # Add timing report
        report_lines.extend([
            "\n## Timing Details",
            f"\nTotal experiment runtime: {self.timer.get_all_stats()}",
        ])

        # Save report
        with open(output_dir / "report.md", 'w') as f:
            f.write("\n".join(report_lines))


@click.command()
@click.option(
    '--config', '-c',
    type=click.Path(exists=True),
    required=True,
    help='Path to configuration file'
)
@click.option(
    '--log-level', '-l',
    type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
    default='INFO',
    help='Logging level'
)
def main(config: str, log_level: str):
    """Run MOEA optimization experiment."""
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create and run experiment
    runner = OptimizationRunner(Path(config))
    runner.run_experiment()


if __name__ == "__main__":
    main()
