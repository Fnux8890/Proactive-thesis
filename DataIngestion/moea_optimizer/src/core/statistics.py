"""Statistical analysis for multi-objective optimization results.

This module provides statistical tests and effect size measures for
comparing the performance of different optimization algorithms, including:
- Wilcoxon signed-rank test
- Mann-Whitney U test
- Vargha-Delaney A12 effect size
- Kruskal-Wallis test
- Post-hoc analysis with multiple comparison correction
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy import stats
from scipy.stats import friedmanchisquare, kruskal, mannwhitneyu, wilcoxon

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """Statistical analysis tools for MOEA comparison."""

    def __init__(self, alpha: float = 0.05):
        """Initialize statistical analyzer.

        Args:
            alpha: Significance level for hypothesis tests
        """
        self.alpha = alpha

    def wilcoxon_test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        alternative: str = 'two-sided'
    ) -> dict[str, float]:
        """Perform Wilcoxon signed-rank test for paired samples.

        Args:
            x: First sample
            y: Second sample (must be paired with x)
            alternative: 'two-sided', 'less', or 'greater'

        Returns:
            Dictionary with test statistic, p-value, and effect size
        """
        if len(x) != len(y):
            raise ValueError("Samples must have the same length for paired test")

        # Remove pairs where both values are identical
        differences = x - y
        mask = differences != 0
        if not np.any(mask):
            logger.warning("All pairs are identical, cannot perform Wilcoxon test")
            return {
                'statistic': np.nan,
                'p_value': 1.0,
                'effect_size': 0.5,
                'significant': False
            }

        # Perform test
        statistic, p_value = wilcoxon(x[mask], y[mask], alternative=alternative)

        # Calculate effect size (r = Z / sqrt(N))
        n = np.sum(mask)
        z_score = stats.norm.ppf(1 - p_value / 2)  # Two-tailed z-score
        effect_size = abs(z_score) / np.sqrt(n)

        return {
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': effect_size,
            'significant': p_value < self.alpha,
            'n_pairs': n
        }

    def mann_whitney_test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        alternative: str = 'two-sided'
    ) -> dict[str, float]:
        """Perform Mann-Whitney U test for independent samples.

        Args:
            x: First sample
            y: Second sample
            alternative: 'two-sided', 'less', or 'greater'

        Returns:
            Dictionary with test statistic, p-value, and effect size
        """
        # Perform test
        statistic, p_value = mannwhitneyu(x, y, alternative=alternative)

        # Calculate effect size (r = Z / sqrt(N))
        n = len(x) + len(y)
        z_score = stats.norm.ppf(1 - p_value / 2)
        effect_size = abs(z_score) / np.sqrt(n)

        # Also calculate A12 effect size
        a12 = self.vargha_delaney_a12(x, y)

        return {
            'statistic': statistic,
            'p_value': p_value,
            'effect_size_r': effect_size,
            'effect_size_a12': a12,
            'significant': p_value < self.alpha,
            'n_x': len(x),
            'n_y': len(y)
        }

    def vargha_delaney_a12(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate Vargha-Delaney A12 effect size.

        A12 measures the probability that a randomly selected value from x
        is larger than a randomly selected value from y.

        Args:
            x: First sample
            y: Second sample

        Returns:
            A12 effect size (0.5 = no effect, >0.5 = x tends to be larger)
        """
        m, n = len(x), len(y)

        # Calculate rank sum
        r1 = 0
        for xi in x:
            r1 += np.sum(xi > y) + 0.5 * np.sum(xi == y)

        # Calculate A12
        a12 = r1 / (m * n)

        return a12

    def interpret_a12(self, a12: float) -> str:
        """Interpret A12 effect size magnitude.

        Args:
            a12: A12 effect size value

        Returns:
            String interpretation
        """
        if a12 >= 0.71:
            return "large"
        elif a12 >= 0.64:
            return "medium"
        elif a12 >= 0.56:
            return "small"
        elif a12 >= 0.44 and a12 <= 0.56:
            return "negligible"
        elif a12 >= 0.36:
            return "small (reversed)"
        elif a12 >= 0.29:
            return "medium (reversed)"
        else:
            return "large (reversed)"

    def kruskal_wallis_test(
        self,
        *samples: np.ndarray
    ) -> dict[str, float | bool]:
        """Perform Kruskal-Wallis test for multiple independent samples.

        Args:
            *samples: Variable number of sample arrays

        Returns:
            Dictionary with test statistic, p-value, and significance
        """
        if len(samples) < 3:
            raise ValueError("Need at least 3 samples for Kruskal-Wallis test")

        statistic, p_value = kruskal(*samples)

        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'n_groups': len(samples),
            'n_total': sum(len(s) for s in samples)
        }

    def friedman_test(
        self,
        *samples: np.ndarray
    ) -> dict[str, float | bool]:
        """Perform Friedman test for multiple paired samples.

        Args:
            *samples: Variable number of sample arrays (must be same length)

        Returns:
            Dictionary with test statistic, p-value, and significance
        """
        if len(samples) < 3:
            raise ValueError("Need at least 3 samples for Friedman test")

        # Check all samples have same length
        lengths = [len(s) for s in samples]
        if len(set(lengths)) > 1:
            raise ValueError("All samples must have the same length for Friedman test")

        statistic, p_value = friedmanchisquare(*samples)

        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'n_groups': len(samples),
            'n_observations': lengths[0]
        }

    def posthoc_nemenyi(
        self,
        data: pd.DataFrame,
        val_col: str = 'value',
        group_col: str = 'group',
        block_col: str | None = None
    ) -> pd.DataFrame:
        """Perform Nemenyi post-hoc test after Friedman test.

        Args:
            data: DataFrame with values and groups
            val_col: Column name for values
            group_col: Column name for groups
            block_col: Column name for blocks (if applicable)

        Returns:
            Pairwise p-value matrix
        """
        if block_col:
            # Reshape data for Nemenyi test
            pivot_data = data.pivot(
                index=block_col,
                columns=group_col,
                values=val_col
            )
            return sp.posthoc_nemenyi_friedman(pivot_data.values.T)
        else:
            return sp.posthoc_nemenyi(
                data,
                val_col=val_col,
                group_col=group_col
            )

    def posthoc_dunn(
        self,
        data: pd.DataFrame,
        val_col: str = 'value',
        group_col: str = 'group',
        p_adjust: str = 'bonferroni'
    ) -> pd.DataFrame:
        """Perform Dunn's post-hoc test after Kruskal-Wallis.

        Args:
            data: DataFrame with values and groups
            val_col: Column name for values
            group_col: Column name for groups
            p_adjust: Method for p-value adjustment

        Returns:
            Pairwise p-value matrix
        """
        return sp.posthoc_dunn(
            data,
            val_col=val_col,
            group_col=group_col,
            p_adjust=p_adjust
        )

    def compare_algorithms(
        self,
        results: dict[str, list[float]],
        test: str = 'kruskal',
        posthoc: str = 'dunn',
        paired: bool = False
    ) -> dict[str, Any]:
        """Compare multiple algorithms with appropriate statistical tests.

        Args:
            results: Dictionary mapping algorithm names to performance values
            test: Main test ('kruskal' or 'friedman')
            posthoc: Post-hoc test ('dunn' or 'nemenyi')
            paired: Whether samples are paired

        Returns:
            Dictionary with test results and pairwise comparisons
        """
        # Prepare data
        algorithms = list(results.keys())
        samples = [np.array(results[alg]) for alg in algorithms]

        # Check if we have enough algorithms
        if len(algorithms) < 2:
            raise ValueError("Need at least 2 algorithms to compare")

        output = {
            'algorithms': algorithms,
            'n_algorithms': len(algorithms),
            'sample_sizes': {alg: len(results[alg]) for alg in algorithms}
        }

        # Perform main test
        if len(algorithms) == 2:
            # Use pairwise test
            if paired:
                output['test'] = 'wilcoxon'
                output['results'] = self.wilcoxon_test(samples[0], samples[1])
            else:
                output['test'] = 'mann_whitney'
                output['results'] = self.mann_whitney_test(samples[0], samples[1])
        else:
            # Use multi-group test
            if paired and test == 'friedman':
                output['test'] = 'friedman'
                output['results'] = self.friedman_test(*samples)
            else:
                output['test'] = 'kruskal_wallis'
                output['results'] = self.kruskal_wallis_test(*samples)

            # Perform post-hoc if significant
            if output['results']['significant']:
                # Create DataFrame for post-hoc tests
                data_list = []
                for alg, values in results.items():
                    for i, val in enumerate(values):
                        data_list.append({
                            'algorithm': alg,
                            'value': val,
                            'run': i
                        })
                df = pd.DataFrame(data_list)

                if posthoc == 'nemenyi' and paired:
                    output['posthoc'] = self.posthoc_nemenyi(
                        df,
                        val_col='value',
                        group_col='algorithm',
                        block_col='run'
                    )
                else:
                    output['posthoc'] = self.posthoc_dunn(
                        df,
                        val_col='value',
                        group_col='algorithm'
                    )

        # Calculate pairwise effect sizes
        output['effect_sizes'] = {}
        for i, alg1 in enumerate(algorithms):
            for j, alg2 in enumerate(algorithms):
                if i < j:
                    a12 = self.vargha_delaney_a12(
                        np.array(results[alg1]),
                        np.array(results[alg2])
                    )
                    output['effect_sizes'][f"{alg1}_vs_{alg2}"] = {
                        'a12': a12,
                        'interpretation': self.interpret_a12(a12)
                    }

        return output

    def generate_comparison_table(
        self,
        comparison_results: dict[str, Any]
    ) -> pd.DataFrame:
        """Generate a formatted comparison table.

        Args:
            comparison_results: Results from compare_algorithms

        Returns:
            DataFrame with comparison summary
        """
        rows = []

        # Main test result
        test_name = comparison_results['test']
        test_results = comparison_results['results']

        if test_name in ['wilcoxon', 'mann_whitney']:
            # Pairwise comparison
            alg1, alg2 = comparison_results['algorithms']
            rows.append({
                'Comparison': f"{alg1} vs {alg2}",
                'Test': test_name.replace('_', '-').title(),
                'p-value': f"{test_results['p_value']:.4f}",
                'Significant': '✓' if test_results['significant'] else '✗',
                'Effect Size': f"{test_results.get('effect_size_a12', test_results.get('effect_size', 'N/A')):.3f}"
            })
        else:
            # Multi-group comparison
            rows.append({
                'Comparison': 'All algorithms',
                'Test': test_name.replace('_', '-').title(),
                'p-value': f"{test_results['p_value']:.4f}",
                'Significant': '✓' if test_results['significant'] else '✗',
                'Effect Size': 'See pairwise'
            })

        # Effect sizes
        if 'effect_sizes' in comparison_results:
            for comparison, effect in comparison_results['effect_sizes'].items():
                alg1, alg2 = comparison.replace('_vs_', ' vs ').split(' vs ')
                rows.append({
                    'Comparison': f"{alg1} vs {alg2}",
                    'Test': 'A12 Effect Size',
                    'p-value': '-',
                    'Significant': '-',
                    'Effect Size': f"{effect['a12']:.3f} ({effect['interpretation']})"
                })

        return pd.DataFrame(rows)

    def summarize_results(
        self,
        all_results: dict[str, dict[str, list[float]]],
        metrics: list[str] | None = None
    ) -> pd.DataFrame:
        """Create summary statistics table for all algorithms and metrics.

        Args:
            all_results: Nested dict: algorithm -> metric -> values
            metrics: List of metrics to include (None = all)

        Returns:
            Summary DataFrame
        """
        rows = []

        for algorithm, metric_results in all_results.items():
            for metric, values in metric_results.items():
                if metrics is None or metric in metrics:
                    values_array = np.array(values)
                    rows.append({
                        'Algorithm': algorithm,
                        'Metric': metric,
                        'Mean': np.mean(values_array),
                        'Std': np.std(values_array),
                        'Median': np.median(values_array),
                        'IQR': np.percentile(values_array, 75) - np.percentile(values_array, 25),
                        'Min': np.min(values_array),
                        'Max': np.max(values_array),
                        'N': len(values_array)
                    })

        df = pd.DataFrame(rows)

        # Round numeric columns
        numeric_cols = ['Mean', 'Std', 'Median', 'IQR', 'Min', 'Max']
        df[numeric_cols] = df[numeric_cols].round(4)

        return df


def create_statistical_report(
    results: dict[str, dict[str, list[float]]],
    output_path: Path,
    alpha: float = 0.05
) -> None:
    """Create comprehensive statistical report.

    Args:
        results: Nested dict: algorithm -> metric -> values
        output_path: Path to save report
        alpha: Significance level
    """
    analyzer = StatisticalAnalyzer(alpha=alpha)

    report_lines = [
        "# Statistical Analysis Report",
        f"\nSignificance level: alpha = {alpha}",
        "\n## Summary Statistics\n"
    ]

    # Summary statistics
    summary_df = analyzer.summarize_results(results)
    report_lines.append(summary_df.to_markdown(index=False))

    # Statistical comparisons for each metric
    report_lines.append("\n## Statistical Comparisons\n")

    all_metrics = set()
    for metric_results in results.values():
        all_metrics.update(metric_results.keys())

    for metric in sorted(all_metrics):
        report_lines.append(f"\n### {metric.replace('_', ' ').title()}\n")

        # Get data for this metric
        metric_data = {}
        for alg, alg_results in results.items():
            if metric in alg_results:
                metric_data[alg] = alg_results[metric]

        if len(metric_data) < 2:
            report_lines.append("*Insufficient data for comparison*\n")
            continue

        # Perform comparison
        comparison = analyzer.compare_algorithms(metric_data)

        # Add comparison table
        comparison_table = analyzer.generate_comparison_table(comparison)
        report_lines.append(comparison_table.to_markdown(index=False))

        # Add post-hoc results if available
        if 'posthoc' in comparison:
            report_lines.append("\n**Post-hoc pairwise p-values:**")
            posthoc_df = comparison['posthoc']
            report_lines.append(f"```\n{posthoc_df}\n```")

    # Save report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))

    logger.info(f"Statistical report saved to {output_path}")


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)

    results = {
        "NSGA-III (CPU)": {
            "hypervolume": list(np.random.normal(100, 5, 30)),
            "igd_plus": list(np.random.normal(0.05, 0.01, 30)),
            "runtime": list(np.random.normal(120, 10, 30))
        },
        "TensorNSGA3 (GPU)": {
            "hypervolume": list(np.random.normal(98, 4, 30)),
            "igd_plus": list(np.random.normal(0.06, 0.015, 30)),
            "runtime": list(np.random.normal(25, 3, 30))
        },
        "Reference": {
            "hypervolume": list(np.random.normal(95, 6, 30)),
            "igd_plus": list(np.random.normal(0.08, 0.02, 30)),
            "runtime": list(np.random.normal(150, 15, 30))
        }
    }

    # Create analyzer
    analyzer = StatisticalAnalyzer()

    # Test pairwise comparison
    print("Pairwise comparison (CPU vs GPU on hypervolume):")
    cpu_hv = results["NSGA-III (CPU)"]["hypervolume"]
    gpu_hv = results["TensorNSGA3 (GPU)"]["hypervolume"]

    mw_result = analyzer.mann_whitney_test(cpu_hv, gpu_hv)
    print(f"Mann-Whitney U test: p-value = {mw_result['p_value']:.4f}")
    print(f"A12 effect size: {mw_result['effect_size_a12']:.3f}")
    print(f"Interpretation: {analyzer.interpret_a12(mw_result['effect_size_a12'])}")

    # Test multi-group comparison
    print("\n\nMulti-group comparison (all algorithms on hypervolume):")
    hv_data = {alg: res["hypervolume"] for alg, res in results.items()}
    comparison = analyzer.compare_algorithms(hv_data)

    print(f"Kruskal-Wallis test: p-value = {comparison['results']['p_value']:.4f}")

    # Create report
    create_statistical_report(results, Path("test_statistical_report.md"))
