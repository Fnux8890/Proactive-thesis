"""Visualization utilities for multi-objective optimization results.

This module provides functions for creating various plots to visualize
optimization results, including Pareto fronts, convergence curves,
parallel coordinates, and statistical comparisons.
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class MOEAVisualizer:
    """Visualization tools for MOEA results."""

    def __init__(self, figsize: tuple[int, int] = (10, 8), dpi: int = 300):
        """Initialize visualizer.

        Args:
            figsize: Default figure size
            dpi: DPI for saved figures
        """
        self.figsize = figsize
        self.dpi = dpi

    def plot_pareto_front_2d(
        self,
        fronts: dict[str, np.ndarray],
        obj_names: list[str] | None = None,
        title: str = "Pareto Front Comparison",
        save_path: Path | None = None
    ) -> plt.Figure:
        """Plot 2D Pareto fronts for comparison.

        Args:
            fronts: Dictionary mapping algorithm names to objective arrays
            obj_names: Names of objectives
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if obj_names is None:
            obj_names = ["Objective 1", "Objective 2"]

        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot each front
        for _i, (name, front) in enumerate(fronts.items()):
            if front.shape[1] < 2:
                logger.warning(f"Front {name} has less than 2 objectives, skipping")
                continue

            ax.scatter(front[:, 0], front[:, 1], label=name, alpha=0.7, s=50)

        ax.set_xlabel(obj_names[0], fontsize=12)
        ax.set_ylabel(obj_names[1], fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")

        return fig

    def plot_pareto_front_3d(
        self,
        fronts: dict[str, np.ndarray],
        obj_names: list[str] | None = None,
        title: str = "3D Pareto Front",
        save_path: Path | None = None,
        view_angles: tuple[float, float] = (30, 45)
    ) -> plt.Figure:
        """Plot 3D Pareto fronts.

        Args:
            fronts: Dictionary mapping algorithm names to objective arrays
            obj_names: Names of objectives
            title: Plot title
            save_path: Path to save figure
            view_angles: Elevation and azimuth angles for 3D view

        Returns:
            Matplotlib figure
        """
        if obj_names is None:
            obj_names = ["Objective 1", "Objective 2", "Objective 3"]

        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Plot each front
        for _i, (name, front) in enumerate(fronts.items()):
            if front.shape[1] < 3:
                logger.warning(f"Front {name} has less than 3 objectives, skipping")
                continue

            ax.scatter(front[:, 0], front[:, 1], front[:, 2],
                      label=name, alpha=0.7, s=50)

        ax.set_xlabel(obj_names[0], fontsize=12)
        ax.set_ylabel(obj_names[1], fontsize=12)
        ax.set_zlabel(obj_names[2], fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()

        # Set viewing angle
        ax.view_init(elev=view_angles[0], azim=view_angles[1])

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")

        return fig

    def plot_convergence_curves(
        self,
        histories: dict[str, pd.DataFrame],
        metric: str = "hypervolume",
        title: str = "Convergence Comparison",
        save_path: Path | None = None,
        log_scale: bool = False
    ) -> plt.Figure:
        """Plot convergence curves for multiple algorithms.

        Args:
            histories: Dictionary mapping algorithm names to convergence DataFrames
            metric: Metric to plot
            title: Plot title
            save_path: Path to save figure
            log_scale: Whether to use log scale for y-axis

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot each algorithm's convergence
        for name, history in histories.items():
            if metric not in history.columns:
                logger.warning(f"Metric {metric} not found for {name}, skipping")
                continue

            # Filter out None values
            data = history[['generation', metric]].dropna()

            ax.plot(data['generation'], data[metric], label=name, linewidth=2)

        ax.set_xlabel("Generation", fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if log_scale:
            ax.set_yscale('log')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")

        return fig

    def plot_parallel_coordinates(
        self,
        front: np.ndarray,
        obj_names: list[str] | None = None,
        title: str = "Parallel Coordinates",
        save_path: Path | None = None,
        highlight_best: bool = True
    ) -> plt.Figure:
        """Plot parallel coordinates for multi-objective solutions.

        Args:
            front: Objective values array
            obj_names: Names of objectives
            title: Plot title
            save_path: Path to save figure
            highlight_best: Whether to highlight best solutions

        Returns:
            Matplotlib figure
        """
        n_obj = front.shape[1]
        if obj_names is None:
            obj_names = [f"Obj {i+1}" for i in range(n_obj)]

        # Normalize objectives to [0, 1]
        front_norm = (front - front.min(axis=0)) / (front.max(axis=0) - front.min(axis=0) + 1e-10)

        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot each solution
        for i, solution in enumerate(front_norm):
            color = plt.cm.viridis(i / len(front_norm))
            alpha = 0.3

            # Highlight some solutions
            if highlight_best and i < 5:  # Highlight first 5
                alpha = 0.8
                linewidth = 2
            else:
                linewidth = 1

            ax.plot(range(n_obj), solution, color=color, alpha=alpha, linewidth=linewidth)

        ax.set_xticks(range(n_obj))
        ax.set_xticklabels(obj_names, rotation=45, ha='right')
        ax.set_ylabel("Normalized Value", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")

        return fig

    def plot_box_comparison(
        self,
        results: dict[str, list[float]],
        metric_name: str = "Hypervolume",
        title: str = "Algorithm Comparison",
        save_path: Path | None = None,
        show_points: bool = True
    ) -> plt.Figure:
        """Create box plot comparing algorithms.

        Args:
            results: Dictionary mapping algorithm names to metric values
            metric_name: Name of the metric
            title: Plot title
            save_path: Path to save figure
            show_points: Whether to show individual points

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Prepare data for plotting
        data = []
        labels = []
        for name, values in results.items():
            data.append(values)
            labels.append(name)

        # Create box plot
        bp = ax.boxplot(data, labels=labels, patch_artist=True)

        # Customize box plot
        colors = sns.color_palette("husl", len(data))
        for patch, color in zip(bp['boxes'], colors, strict=False):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Add individual points if requested
        if show_points:
            for i, (d, color) in enumerate(zip(data, colors, strict=False)):
                x = np.random.normal(i + 1, 0.04, size=len(d))
                ax.scatter(x, d, color=color, alpha=0.5, s=30)

        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Rotate x labels if many algorithms
        if len(labels) > 5:
            plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")

        return fig

    def plot_heatmap_comparison(
        self,
        comparison_matrix: pd.DataFrame,
        title: str = "Pairwise Comparison",
        save_path: Path | None = None,
        cmap: str = "RdBu_r",
        annot: bool = True
    ) -> plt.Figure:
        """Plot heatmap of pairwise comparisons.

        Args:
            comparison_matrix: DataFrame with comparison values
            title: Plot title
            save_path: Path to save figure
            cmap: Colormap
            annot: Whether to annotate cells with values

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Create heatmap
        sns.heatmap(
            comparison_matrix,
            annot=annot,
            fmt='.3f',
            cmap=cmap,
            center=0.5,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )

        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")

        return fig

    def plot_runtime_comparison(
        self,
        runtimes: dict[str, dict[str, float]],
        title: str = "Runtime Comparison",
        save_path: Path | None = None
    ) -> plt.Figure:
        """Plot runtime comparison with breakdown.

        Args:
            runtimes: Nested dict with algorithm -> component -> time
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figsize[0] * 1.5, self.figsize[1]))

        # Total runtime comparison
        algorithms = list(runtimes.keys())
        total_times = [sum(times.values()) for times in runtimes.values()]

        bars1 = ax1.bar(algorithms, total_times)
        ax1.set_ylabel("Total Runtime (seconds)", fontsize=12)
        ax1.set_title("Total Runtime", fontsize=12)
        ax1.grid(True, alpha=0.3, axis='y')

        # Color bars
        colors = sns.color_palette("husl", len(algorithms))
        for bar, color in zip(bars1, colors, strict=False):
            bar.set_color(color)

        # Runtime breakdown
        components = set()
        for times in runtimes.values():
            components.update(times.keys())
        components = sorted(components)

        # Prepare data for stacked bar chart
        data = {comp: [] for comp in components}
        for algo in algorithms:
            for comp in components:
                data[comp].append(runtimes[algo].get(comp, 0))

        # Create stacked bar chart
        bottom = np.zeros(len(algorithms))
        for _i, comp in enumerate(components):
            ax2.bar(algorithms, data[comp], bottom=bottom,
                           label=comp, alpha=0.8)
            bottom += data[comp]

        ax2.set_ylabel("Runtime (seconds)", fontsize=12)
        ax2.set_title("Runtime Breakdown", fontsize=12)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3, axis='y')

        # Rotate x labels if needed
        if len(algorithms) > 5:
            for ax in [ax1, ax2]:
                ax.set_xticklabels(algorithms, rotation=45, ha='right')

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")

        return fig

    def create_summary_report(
        self,
        results: dict[str, Any],
        output_dir: Path,
        format: str = "png"
    ) -> None:
        """Create comprehensive visual report.

        Args:
            results: Complete results dictionary
            output_dir: Directory to save plots
            format: Image format
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract data for visualization
        fronts = {}
        convergence_histories = {}
        metric_values = {}
        runtimes = {}

        for result in results:
            name = result.get('algorithm', 'Unknown')

            # Pareto fronts
            if 'pareto_front' in result and 'F' in result['pareto_front']:
                fronts[name] = result['pareto_front']['F']

            # Convergence history
            if 'convergence_history' in result:
                convergence_histories[name] = pd.DataFrame(result['convergence_history'])

            # Metrics
            if 'final_metrics' in result:
                for metric, value in result['final_metrics'].items():
                    if metric not in metric_values:
                        metric_values[metric] = {}
                    metric_values[metric][name] = value

            # Runtime
            if 'timing_stats' in result:
                runtimes[name] = {k: v['total'] for k, v in result['timing_stats'].items()}

        # Generate plots
        plots_created = []

        # 1. Pareto front comparison
        if fronts:
            n_obj = next(iter(fronts.values())).shape[1]

            if n_obj == 2:
                fig = self.plot_pareto_front_2d(
                    fronts,
                    title="Pareto Front Comparison",
                    save_path=output_dir / f"pareto_front_2d.{format}"
                )
                plots_created.append("pareto_front_2d")
                plt.close(fig)

            elif n_obj >= 3:
                fig = self.plot_pareto_front_3d(
                    fronts,
                    title="3D Pareto Front Comparison",
                    save_path=output_dir / f"pareto_front_3d.{format}"
                )
                plots_created.append("pareto_front_3d")
                plt.close(fig)

                # Also create parallel coordinates for first algorithm
                first_algo = next(iter(fronts.keys()))
                fig = self.plot_parallel_coordinates(
                    fronts[first_algo],
                    title=f"Parallel Coordinates - {first_algo}",
                    save_path=output_dir / f"parallel_coords_{first_algo}.{format}"
                )
                plots_created.append(f"parallel_coords_{first_algo}")
                plt.close(fig)

        # 2. Convergence curves
        if convergence_histories:
            for metric in ['hypervolume', 'igd_plus', 'spacing']:
                fig = self.plot_convergence_curves(
                    convergence_histories,
                    metric=metric,
                    title=f"{metric.replace('_', ' ').title()} Convergence",
                    save_path=output_dir / f"convergence_{metric}.{format}"
                )
                plots_created.append(f"convergence_{metric}")
                plt.close(fig)

        # 3. Metric comparisons
        for metric, values in metric_values.items():
            if isinstance(next(iter(values.values())), list | np.ndarray):
                fig = self.plot_box_comparison(
                    values,
                    metric_name=metric.replace('_', ' ').title(),
                    title=f"{metric.replace('_', ' ').title()} Comparison",
                    save_path=output_dir / f"boxplot_{metric}.{format}"
                )
                plots_created.append(f"boxplot_{metric}")
                plt.close(fig)

        # 4. Runtime comparison
        if runtimes:
            fig = self.plot_runtime_comparison(
                runtimes,
                title="Runtime Analysis",
                save_path=output_dir / f"runtime_comparison.{format}"
            )
            plots_created.append("runtime_comparison")
            plt.close(fig)

        logger.info(f"Created {len(plots_created)} plots in {output_dir}")

        # Create index file
        with open(output_dir / "index.md", 'w') as f:
            f.write("# MOEA Optimization Results\n\n")
            f.write("## Generated Plots\n\n")
            for plot in plots_created:
                f.write(f"- [{plot}]({plot}.{format})\n")


# Utility functions for quick plotting
def plot_convergence(history_file: Path, metric: str = "hypervolume",
                    save_path: Path | None = None) -> plt.Figure:
    """Quick function to plot convergence from CSV file."""
    df = pd.read_csv(history_file)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['generation'], df[metric], linewidth=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f"{metric.replace('_', ' ').title()} Convergence")
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def compare_fronts_2d(front_files: dict[str, Path],
                     save_path: Path | None = None) -> plt.Figure:
    """Quick function to compare 2D Pareto fronts from numpy files."""
    fronts = {}
    for name, file in front_files.items():
        fronts[name] = np.load(file)

    visualizer = MOEAVisualizer()
    return visualizer.plot_pareto_front_2d(fronts, save_path=save_path)


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)

    # Sample Pareto fronts
    fronts = {
        "NSGA-III (CPU)": np.random.rand(50, 3) * np.array([1, 2, 3]),
        "TensorNSGA3 (GPU)": np.random.rand(60, 3) * np.array([1.1, 1.9, 2.8])
    }

    # Sample convergence history
    generations = np.arange(0, 200, 10)
    histories = {
        "NSGA-III (CPU)": pd.DataFrame({
            'generation': generations,
            'hypervolume': 100 - 80 * np.exp(-generations / 50) + np.random.randn(len(generations)) * 2
        }),
        "TensorNSGA3 (GPU)": pd.DataFrame({
            'generation': generations,
            'hypervolume': 100 - 70 * np.exp(-generations / 40) + np.random.randn(len(generations)) * 2
        })
    }

    # Create visualizer
    viz = MOEAVisualizer()

    # Test plots
    output_dir = Path("test_plots")
    output_dir.mkdir(exist_ok=True)

    # 2D Pareto front
    viz.plot_pareto_front_2d(
        {k: v[:, :2] for k, v in fronts.items()},
        save_path=output_dir / "test_pareto_2d.png"
    )

    # 3D Pareto front
    viz.plot_pareto_front_3d(
        fronts,
        save_path=output_dir / "test_pareto_3d.png"
    )

    # Convergence curves
    viz.plot_convergence_curves(
        histories,
        save_path=output_dir / "test_convergence.png"
    )

    print(f"Test plots saved to {output_dir}")
