#!/usr/bin/env python3
"""
Experiment Tracker and Analysis System

This module provides comprehensive experiment tracking, analysis, and report generation
for MOEA greenhouse optimization experiments. It tracks multiple runs, accumulates
results in database, and generates statistical analysis reports.

Usage:
    python experiment_tracker.py run --experiments 5 --algorithm cpu
    python experiment_tracker.py analyze --experiment-id exp_2025_06_01_001
    python experiment_tracker.py report --output experiment_report.md
"""

import json
import uuid
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import logging
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import subprocess
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment_tracker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run"""
    experiment_id: str
    algorithm_type: str  # 'cpu' or 'gpu'
    moea_algorithm: str  # 'NSGA-II' or 'NSGA-III'
    population_size: int
    n_generations: int
    feature_table: str
    dataset_period: str
    timestamp: datetime
    
@dataclass
class ExperimentResult:
    """Results from a single experiment run"""
    experiment_id: str
    run_number: int
    runtime_seconds: float
    hypervolume: float
    spacing: float
    n_solutions: int
    pareto_front_path: str
    decision_variables_path: str
    timestamp: datetime

class ExperimentTracker:
    """Main class for tracking and analyzing MOEA experiments"""
    
    def __init__(self, db_url: str = None):
        self.db_url = db_url or "postgresql://postgres:postgres@localhost:5432/postgres"
        self.experiments_dir = Path("experiments")
        self.results_dir = self.experiments_dir / "results"
        self.reports_dir = self.experiments_dir / "reports"
        
        # Create directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize experiment tracking tables in database"""
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    # Create experiments table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS experiments (
                            id SERIAL PRIMARY KEY,
                            experiment_id VARCHAR(255) UNIQUE NOT NULL,
                            algorithm_type VARCHAR(50) NOT NULL,
                            moea_algorithm VARCHAR(50) NOT NULL,
                            population_size INTEGER NOT NULL,
                            n_generations INTEGER NOT NULL,
                            feature_table VARCHAR(255) NOT NULL,
                            dataset_period VARCHAR(100) NOT NULL,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                            metadata JSONB
                        );
                    """)
                    
                    # Create experiment_results table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS experiment_results (
                            id SERIAL PRIMARY KEY,
                            experiment_id VARCHAR(255) NOT NULL REFERENCES experiments(experiment_id),
                            run_number INTEGER NOT NULL,
                            runtime_seconds FLOAT NOT NULL,
                            hypervolume FLOAT NOT NULL,
                            spacing FLOAT NOT NULL,
                            n_solutions INTEGER NOT NULL,
                            pareto_front_path TEXT,
                            decision_variables_path TEXT,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                            metrics JSONB,
                            UNIQUE(experiment_id, run_number)
                        );
                    """)
                    
                    # Create indexes
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_experiment_id ON experiment_results(experiment_id);")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_algorithm_type ON experiments(algorithm_type);")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON experiment_results(created_at);")
                    
                    conn.commit()
                    logger.info("Database schema initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def create_experiment(self, algorithm_type: str, moea_algorithm: str = None, 
                         population_size: int = 100, n_generations: int = 500,
                         feature_table: str = "enhanced_sparse_features_full",
                         dataset_period: str = "2013-2016") -> str:
        """Create a new experiment configuration"""
        
        experiment_id = f"exp_{datetime.now().strftime('%Y_%m_%d_%H%M%S')}_{algorithm_type}"
        
        # Default MOEA algorithm based on hardware
        if moea_algorithm is None:
            moea_algorithm = "NSGA-III" if algorithm_type == "cpu" else "NSGA-II"
        
        config = ExperimentConfig(
            experiment_id=experiment_id,
            algorithm_type=algorithm_type,
            moea_algorithm=moea_algorithm,
            population_size=population_size,
            n_generations=n_generations,
            feature_table=feature_table,
            dataset_period=dataset_period,
            timestamp=datetime.now(timezone.utc)
        )
        
        # Save to database
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO experiments 
                        (experiment_id, algorithm_type, moea_algorithm, population_size, 
                         n_generations, feature_table, dataset_period, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        config.experiment_id,
                        config.algorithm_type,
                        config.moea_algorithm,
                        config.population_size,
                        config.n_generations,
                        config.feature_table,
                        config.dataset_period,
                        json.dumps(asdict(config), default=str)
                    ))
                    conn.commit()
                    logger.info(f"Created experiment: {experiment_id}")
        except Exception as e:
            logger.error(f"Failed to create experiment: {e}")
            raise
        
        return experiment_id
    
    def run_experiment(self, experiment_id: str, n_runs: int = 1) -> List[ExperimentResult]:
        """Execute multiple runs of a single experiment"""
        
        # Get experiment config
        config = self._get_experiment_config(experiment_id)
        if not config:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        results = []
        
        for run_num in range(1, n_runs + 1):
            logger.info(f"Starting run {run_num}/{n_runs} for experiment {experiment_id}")
            
            try:
                # Run the MOEA optimization
                result = self._execute_moea_run(config, run_num)
                
                # Save result to database
                self._save_experiment_result(result)
                results.append(result)
                
                logger.info(f"Completed run {run_num}: Runtime={result.runtime_seconds:.2f}s, "
                           f"Hypervolume={result.hypervolume:.4f}, Solutions={result.n_solutions}")
                
            except Exception as e:
                logger.error(f"Failed run {run_num} for experiment {experiment_id}: {e}")
                continue
        
        return results
    
    def _execute_moea_run(self, config: ExperimentConfig, run_number: int) -> ExperimentResult:
        """Execute a single MOEA run"""
        
        # Create unique run directory
        run_dir = self.results_dir / config.experiment_id / f"run_{run_number:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Update MOEA config files for this run
        self._update_moea_config(config, run_dir)
        
        # Execute MOEA container
        start_time = datetime.now()
        
        if config.algorithm_type == "cpu":
            container_name = "moea_optimizer_cpu"
        else:
            container_name = "moea_optimizer_gpu"
        
        try:
            # Run MOEA container
            cmd = [
                "docker", "compose", "-f", "docker-compose.full-comparison.yml",
                "run", "--rm", "-v", f"{run_dir.absolute()}:/app/experiments/output",
                container_name
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                raise RuntimeError(f"MOEA execution failed: {result.stderr}")
            
            runtime = (datetime.now() - start_time).total_seconds()
            
            # Parse results
            results_file = run_dir / "experiment" / "complete_results.json"
            if not results_file.exists():
                raise FileNotFoundError(f"Results file not found: {results_file}")
            
            with open(results_file) as f:
                moea_results = json.load(f)[0]  # First run results
            
            metrics = moea_results["metrics"]
            
            # Create result object
            experiment_result = ExperimentResult(
                experiment_id=config.experiment_id,
                run_number=run_number,
                runtime_seconds=runtime,
                hypervolume=metrics["hypervolume"]["mean"],
                spacing=metrics["spacing"]["mean"],
                n_solutions=metrics["n_solutions"]["mean"],
                pareto_front_path=str(run_dir / "experiment" / "GreenhouseOptimization" / "pareto_F.npy"),
                decision_variables_path=str(run_dir / "experiment" / "GreenhouseOptimization" / "pareto_X.npy"),
                timestamp=datetime.now(timezone.utc)
            )
            
            return experiment_result
            
        except Exception as e:
            logger.error(f"MOEA execution failed: {e}")
            raise
    
    def _update_moea_config(self, config: ExperimentConfig, run_dir: Path):
        """Update MOEA configuration files for specific run"""
        
        moea_config_file = f"moea_optimizer/config/moea_config_{config.algorithm_type}_full.toml"
        
        # Read current config
        with open(moea_config_file) as f:
            config_content = f.read()
        
        # Update output directory
        updated_config = config_content.replace(
            'output_dir = "experiments"',
            f'output_dir = "{run_dir / "experiment"}"'
        )
        
        # Write temporary config
        temp_config = run_dir / "moea_config.toml"
        with open(temp_config, 'w') as f:
            f.write(updated_config)
    
    def _get_experiment_config(self, experiment_id: str) -> Optional[ExperimentConfig]:
        """Retrieve experiment configuration from database"""
        
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT experiment_id, algorithm_type, moea_algorithm, 
                               population_size, n_generations, feature_table, 
                               dataset_period, created_at
                        FROM experiments 
                        WHERE experiment_id = %s
                    """, (experiment_id,))
                    
                    row = cur.fetchone()
                    if not row:
                        return None
                    
                    return ExperimentConfig(
                        experiment_id=row[0],
                        algorithm_type=row[1],
                        moea_algorithm=row[2],
                        population_size=row[3],
                        n_generations=row[4],
                        feature_table=row[5],
                        dataset_period=row[6],
                        timestamp=row[7]
                    )
        except Exception as e:
            logger.error(f"Failed to get experiment config: {e}")
            return None
    
    def _save_experiment_result(self, result: ExperimentResult):
        """Save experiment result to database"""
        
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO experiment_results 
                        (experiment_id, run_number, runtime_seconds, hypervolume, 
                         spacing, n_solutions, pareto_front_path, decision_variables_path, metrics)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (experiment_id, run_number) 
                        DO UPDATE SET
                            runtime_seconds = EXCLUDED.runtime_seconds,
                            hypervolume = EXCLUDED.hypervolume,
                            spacing = EXCLUDED.spacing,
                            n_solutions = EXCLUDED.n_solutions,
                            pareto_front_path = EXCLUDED.pareto_front_path,
                            decision_variables_path = EXCLUDED.decision_variables_path,
                            metrics = EXCLUDED.metrics
                    """, (
                        result.experiment_id,
                        result.run_number,
                        result.runtime_seconds,
                        result.hypervolume,
                        result.spacing,
                        result.n_solutions,
                        result.pareto_front_path,
                        result.decision_variables_path,
                        json.dumps(asdict(result), default=str)
                    ))
                    conn.commit()
                    logger.info(f"Saved result for {result.experiment_id} run {result.run_number}")
        except Exception as e:
            logger.error(f"Failed to save experiment result: {e}")
            raise
    
    def get_experiment_results(self, experiment_id: str = None) -> pd.DataFrame:
        """Retrieve experiment results as DataFrame"""
        
        try:
            with psycopg2.connect(self.db_url) as conn:
                query = """
                    SELECT 
                        e.experiment_id,
                        e.algorithm_type,
                        e.moea_algorithm,
                        e.population_size,
                        e.n_generations,
                        e.feature_table,
                        e.dataset_period,
                        r.run_number,
                        r.runtime_seconds,
                        r.hypervolume,
                        r.spacing,
                        r.n_solutions,
                        r.created_at as result_timestamp
                    FROM experiments e
                    LEFT JOIN experiment_results r ON e.experiment_id = r.experiment_id
                """
                
                params = []
                if experiment_id:
                    query += " WHERE e.experiment_id = %s"
                    params.append(experiment_id)
                
                query += " ORDER BY e.created_at, r.run_number"
                
                df = pd.read_sql_query(query, conn, params=params)
                return df
        except Exception as e:
            logger.error(f"Failed to get experiment results: {e}")
            return pd.DataFrame()
    
    def analyze_experiments(self, experiment_ids: List[str] = None) -> Dict:
        """Perform statistical analysis on experiment results"""
        
        df = self.get_experiment_results()
        if experiment_ids:
            df = df[df['experiment_id'].isin(experiment_ids)]
        
        if df.empty:
            return {"error": "No experiment data found"}
        
        # Group by algorithm type
        analysis = {}
        
        for algo_type in df['algorithm_type'].unique():
            algo_df = df[df['algorithm_type'] == algo_type]
            
            if algo_df.empty or algo_df['runtime_seconds'].isna().all():
                continue
            
            analysis[algo_type] = {
                "runtime_stats": {
                    "mean": float(algo_df['runtime_seconds'].mean()),
                    "std": float(algo_df['runtime_seconds'].std()),
                    "min": float(algo_df['runtime_seconds'].min()),
                    "max": float(algo_df['runtime_seconds'].max()),
                    "median": float(algo_df['runtime_seconds'].median())
                },
                "hypervolume_stats": {
                    "mean": float(algo_df['hypervolume'].mean()),
                    "std": float(algo_df['hypervolume'].std()),
                    "min": float(algo_df['hypervolume'].min()),
                    "max": float(algo_df['hypervolume'].max())
                },
                "solution_count_stats": {
                    "mean": float(algo_df['n_solutions'].mean()),
                    "std": float(algo_df['n_solutions'].std()),
                    "min": int(algo_df['n_solutions'].min()),
                    "max": int(algo_df['n_solutions'].max())
                },
                "total_runs": len(algo_df),
                "unique_experiments": algo_df['experiment_id'].nunique()
            }
        
        # Comparative analysis
        if len(analysis) > 1:
            cpu_data = df[df['algorithm_type'] == 'cpu']['runtime_seconds'].dropna()
            gpu_data = df[df['algorithm_type'] == 'gpu']['runtime_seconds'].dropna()
            
            if len(cpu_data) > 0 and len(gpu_data) > 0:
                speedup = cpu_data.mean() / gpu_data.mean()
                t_stat, p_value = stats.ttest_ind(cpu_data, gpu_data)
                
                analysis['comparison'] = {
                    "speedup_factor": float(speedup),
                    "statistical_significance": {
                        "t_statistic": float(t_stat),
                        "p_value": float(p_value),
                        "significant": p_value < 0.05
                    }
                }
        
        return analysis
    
    def generate_report(self, output_file: str, experiment_ids: List[str] = None):
        """Generate comprehensive markdown report"""
        
        df = self.get_experiment_results()
        if experiment_ids:
            df = df[df['experiment_id'].isin(experiment_ids)]
        
        analysis = self.analyze_experiments(experiment_ids)
        
        report_content = self._create_report_content(df, analysis)
        
        # Write report
        report_path = self.reports_dir / output_file
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Report generated: {report_path}")
        return str(report_path)
    
    def _create_report_content(self, df: pd.DataFrame, analysis: Dict) -> str:
        """Create markdown report content"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# MOEA Greenhouse Optimization: Experiment Analysis Report

*Generated: {timestamp}*

## Executive Summary

This report analyzes {len(df)} experimental runs across {df['experiment_id'].nunique()} unique experiments, comparing CPU and GPU-accelerated Multi-Objective Evolutionary Algorithms (MOEA) for greenhouse climate optimization.

### Key Findings

"""
        
        # Add comparison results if available
        if 'comparison' in analysis:
            speedup = analysis['comparison']['speedup_factor']
            significance = analysis['comparison']['statistical_significance']
            
            report += f"""
- **GPU Speedup**: {speedup:.1f}x faster than CPU implementation
- **Statistical Significance**: {"✅ Significant" if significance['significant'] else "❌ Not significant"} (p={significance['p_value']:.4f})
"""
        
        # Algorithm performance
        for algo_type, stats in analysis.items():
            if algo_type == 'comparison':
                continue
                
            report += f"""
### {algo_type.upper()} Algorithm Performance

- **Average Runtime**: {stats['runtime_stats']['mean']:.2f} ± {stats['runtime_stats']['std']:.2f} seconds
- **Hypervolume**: {stats['hypervolume_stats']['mean']:.4f} ± {stats['hypervolume_stats']['std']:.4f}
- **Solutions Found**: {stats['solution_count_stats']['mean']:.1f} ± {stats['solution_count_stats']['std']:.1f}
- **Total Runs**: {stats['total_runs']}
"""
        
        # Detailed results table
        if not df.empty:
            report += "\n## Detailed Results\n\n"
            report += "| Experiment ID | Algorithm | Runs | Avg Runtime (s) | Avg Hypervolume | Avg Solutions |\n"
            report += "|---------------|-----------|------|----------------|-----------------|---------------|\n"
            
            for exp_id in df['experiment_id'].unique():
                exp_df = df[df['experiment_id'] == exp_id]
                if exp_df.empty:
                    continue
                    
                algo_type = exp_df['algorithm_type'].iloc[0]
                avg_runtime = exp_df['runtime_seconds'].mean()
                avg_hypervolume = exp_df['hypervolume'].mean()
                avg_solutions = exp_df['n_solutions'].mean()
                run_count = len(exp_df)
                
                report += f"| {exp_id} | {algo_type.upper()} | {run_count} | {avg_runtime:.2f} | {avg_hypervolume:.4f} | {avg_solutions:.0f} |\n"
        
        # Technical specifications
        report += f"""

## Technical Specifications

### Experiment Configuration
- **Feature Set**: Enhanced sparse features (78 dimensions)
- **Training Data**: 223,825 samples from 2013-2016 greenhouse data
- **Population Size**: {df['population_size'].iloc[0] if not df.empty else 'N/A'}
- **Generations**: {df['n_generations'].iloc[0] if not df.empty else 'N/A'}

### Data Sources
- **Greenhouse Sensors**: Temperature, CO2, humidity, lighting
- **External Data**: Weather conditions, energy prices
- **Plant Phenotype**: Kalanchoe blossfeldiana growth parameters

### Performance Metrics
- **Hypervolume**: Measures solution quality and diversity
- **Spacing**: Measures solution distribution uniformity
- **Runtime**: Total optimization execution time

## Database Access

All experimental results are stored in the database and can be accessed via:

```sql
-- View all experiments
SELECT * FROM experiments ORDER BY created_at DESC;

-- View results for specific experiment
SELECT * FROM experiment_results 
WHERE experiment_id = 'your_experiment_id' 
ORDER BY run_number;

-- Compare algorithm performance
SELECT 
    e.algorithm_type,
    AVG(r.runtime_seconds) as avg_runtime,
    AVG(r.hypervolume) as avg_hypervolume,
    COUNT(*) as total_runs
FROM experiments e
JOIN experiment_results r ON e.experiment_id = r.experiment_id
GROUP BY e.algorithm_type;
```

## Reproducibility

To reproduce these experiments:

```bash
# Run CPU experiments (5 runs)
python experiment_tracker.py run --algorithm cpu --runs 5

# Run GPU experiments (5 runs)  
python experiment_tracker.py run --algorithm gpu --runs 5

# Generate analysis report
python experiment_tracker.py report --output comparison_report.md
```

---

*This report was generated by the automated experiment tracking system.*
*For questions or additional analysis, refer to the experiment tracker documentation.*
"""
        
        return report

def main():
    """Command-line interface for experiment tracker"""
    
    parser = argparse.ArgumentParser(description="MOEA Experiment Tracker")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run experiments
    run_parser = subparsers.add_parser('run', help='Run experiments')
    run_parser.add_argument('--algorithm', choices=['cpu', 'gpu'], required=True,
                           help='Algorithm type to run')
    run_parser.add_argument('--runs', type=int, default=1,
                           help='Number of runs to execute')
    run_parser.add_argument('--generations', type=int, default=500,
                           help='Number of MOEA generations')
    run_parser.add_argument('--population', type=int, default=100,
                           help='Population size')
    
    # Analyze experiments
    analyze_parser = subparsers.add_parser('analyze', help='Analyze experiment results')
    analyze_parser.add_argument('--experiment-id', nargs='+',
                               help='Specific experiment IDs to analyze')
    
    # Generate report
    report_parser = subparsers.add_parser('report', help='Generate experiment report')
    report_parser.add_argument('--output', default='experiment_report.md',
                              help='Output report filename')
    report_parser.add_argument('--experiment-id', nargs='+',
                              help='Specific experiment IDs to include')
    
    # List experiments
    list_parser = subparsers.add_parser('list', help='List all experiments')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize tracker
    tracker = ExperimentTracker()
    
    if args.command == 'run':
        # Create and run experiment
        experiment_id = tracker.create_experiment(
            algorithm_type=args.algorithm,
            n_generations=args.generations,
            population_size=args.population
        )
        
        print(f"Created experiment: {experiment_id}")
        
        results = tracker.run_experiment(experiment_id, args.runs)
        
        print(f"\nCompleted {len(results)} runs:")
        for result in results:
            print(f"  Run {result.run_number}: {result.runtime_seconds:.2f}s, "
                  f"HV={result.hypervolume:.4f}, Solutions={result.n_solutions}")
    
    elif args.command == 'analyze':
        analysis = tracker.analyze_experiments(args.experiment_id)
        print(json.dumps(analysis, indent=2))
    
    elif args.command == 'report':
        report_path = tracker.generate_report(args.output, args.experiment_id)
        print(f"Report generated: {report_path}")
    
    elif args.command == 'list':
        df = tracker.get_experiment_results()
        if df.empty:
            print("No experiments found")
        else:
            print("\nExperiments:")
            print(df.groupby(['experiment_id', 'algorithm_type']).size().reset_index(name='runs'))

if __name__ == "__main__":
    main()