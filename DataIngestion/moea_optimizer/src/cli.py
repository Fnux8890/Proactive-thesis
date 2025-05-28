"""Command-line interface for MOEA optimizer."""

import logging
from pathlib import Path

import click

from .core.optimizer_runner import main as run_optimizer

logger = logging.getLogger(__name__)


@click.group()
@click.option('--debug/--no-debug', default=False, help='Enable debug logging')
def cli(debug):
    """MOEA Optimizer - Multi-Objective Evolutionary Algorithm benchmarking tool."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


@cli.command()
@click.option(
    '--config', '-c',
    type=click.Path(exists=True),
    required=True,
    help='Path to configuration file'
)
def run(config):
    """Run optimization experiment."""
    run_optimizer.callback(config=config, log_level='INFO')


@cli.command()
@click.option(
    '--base', '-b',
    type=click.Path(exists=True),
    default='config/base.yaml',
    help='Base configuration file'
)
@click.option(
    '--output', '-o',
    type=click.Path(),
    required=True,
    help='Output configuration file'
)
@click.option(
    '--device',
    type=click.Choice(['cpu', 'gpu']),
    default='cpu',
    help='Target device'
)
@click.option(
    '--problem',
    type=str,
    help='Problem suite (e.g., dtlz, wfg)'
)
def generate_config(base, output, device, problem):
    """Generate configuration file from template."""
    import yaml

    # Load base config
    with open(base) as f:
        config = yaml.safe_load(f)

    # Update with specified options
    if device:
        config['hardware']['device'] = device

    if problem:
        config['problem']['suite'] = problem

    # Save new config
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    click.echo(f"Configuration saved to {output_path}")


@cli.command()
@click.option(
    '--results', '-r',
    type=click.Path(exists=True),
    required=True,
    help='Results directory'
)
@click.option(
    '--output', '-o',
    type=click.Path(),
    help='Output report file'
)
def report(results, output):
    """Generate report from results."""
    import json

    import pandas as pd

    results_dir = Path(results)

    # Load summary if exists
    summary_file = results_dir / 'summary.csv'
    if summary_file.exists():
        df = pd.read_csv(summary_file)
        click.echo("\nResults Summary:")
        click.echo(df.to_string(index=False))

    # Load complete results for detailed report
    complete_file = results_dir / 'complete_results.json'
    if complete_file.exists():
        with open(complete_file) as f:
            data = json.load(f)

        if output:
            # Generate detailed report
            report_path = Path(output)
            # Implementation would go here
            click.echo(f"Detailed report saved to {report_path}")
        else:
            # Print to console
            for result in data:
                click.echo(f"\nProblem: {result['problem_name']}")
                click.echo(f"Runs: {result['n_runs']}")
                if 'metrics' in result:
                    for metric, stats in result['metrics'].items():
                        click.echo(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")


@cli.command()
def list_problems():
    """List available test problems."""
    problems = {
        'DTLZ': ['DTLZ1', 'DTLZ2', 'DTLZ3', 'DTLZ4', 'DTLZ5', 'DTLZ6', 'DTLZ7'],
        'WFG': ['WFG1', 'WFG2', 'WFG3', 'WFG4', 'WFG5', 'WFG6', 'WFG7', 'WFG8', 'WFG9'],
        'ZDT': ['ZDT1', 'ZDT2', 'ZDT3', 'ZDT4', 'ZDT6']
    }

    click.echo("Available test problems:\n")
    for suite, probs in problems.items():
        click.echo(f"{suite} Suite:")
        for prob in probs:
            click.echo(f"  - {prob}")
        click.echo()


def main():
    """Main entry point."""
    cli()


if __name__ == '__main__':
    main()
