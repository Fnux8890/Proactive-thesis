# MOEA Experiment System: Complete Guide

## Overview

The experiment system provides comprehensive tracking, execution, and analysis of Multi-Objective Evolutionary Algorithm (MOEA) experiments for greenhouse optimization. It automatically tracks multiple runs, stores results in database, and generates statistical analysis reports.

## Key Features

- **Automated Experiment Tracking**: Database storage of experiment configurations and results
- **Multiple Run Management**: Execute multiple runs for statistical significance
- **CPU vs GPU Comparison**: Fair algorithmic comparison with performance metrics
- **Statistical Analysis**: Automated significance testing and confidence intervals
- **Report Generation**: Comprehensive markdown reports with visualizations
- **Database Integration**: Persistent storage with SQL query access

## Quick Start

### 1. Run Your First Experiment

```bash
# Quick test with 2 runs each algorithm
./run_multiple_experiments.sh --quick-test

# Production runs with statistical significance
./run_multiple_experiments.sh --cpu-runs 5 --gpu-runs 5

# CPU-only experiment with 10 runs
./run_multiple_experiments.sh --algorithm cpu --runs 10
```

### 2. Generate Analysis Report

```bash
# Generate comprehensive report
python3 experiments/experiment_tracker.py report --output my_experiment_report.md

# Analyze specific experiments
python3 experiments/experiment_tracker.py analyze --experiment-id exp_2025_06_01_001
```

### 3. Access Database Results

```sql
-- View all experiments
SELECT * FROM experiments ORDER BY created_at DESC;

-- View results with performance metrics
SELECT 
    e.experiment_id,
    e.algorithm_type,
    AVG(r.runtime_seconds) as avg_runtime,
    AVG(r.hypervolume) as avg_hypervolume,
    COUNT(*) as total_runs
FROM experiments e
JOIN experiment_results r ON e.experiment_id = r.experiment_id
GROUP BY e.experiment_id, e.algorithm_type
ORDER BY avg_runtime;
```

## System Architecture

### Database Schema

The experiment system uses two main tables:

**experiments** table:
- `experiment_id`: Unique identifier (e.g., "exp_2025_06_01_143022_cpu")
- `algorithm_type`: "cpu" or "gpu"
- `moea_algorithm`: "NSGA-II" or "NSGA-III"
- `population_size`, `n_generations`: MOEA parameters
- `feature_table`: Data source for optimization
- `metadata`: JSON configuration details

**experiment_results** table:
- `experiment_id`: References experiments table
- `run_number`: Sequential run number (1, 2, 3...)
- `runtime_seconds`: Total execution time
- `hypervolume`: Solution quality metric
- `spacing`: Solution distribution metric
- `n_solutions`: Number of Pareto-optimal solutions found
- `pareto_front_path`, `decision_variables_path`: File locations

### File Structure

```
experiments/
├── experiment_tracker.py         # Main tracking system
├── requirements.txt              # Python dependencies
├── results/                      # Individual run results
│   └── exp_2025_06_01_001/      # Experiment-specific directory
│       ├── run_001/             # Individual run data
│       │   └── experiment/      # MOEA output files
│       └── run_002/
└── reports/                     # Generated analysis reports
    └── experiment_report_*.md   # Timestamped reports
```

## Command Reference

### Experiment Tracker CLI

```bash
# Run experiments
python3 experiments/experiment_tracker.py run --algorithm cpu --runs 5
python3 experiments/experiment_tracker.py run --algorithm gpu --runs 3

# Analyze results
python3 experiments/experiment_tracker.py analyze
python3 experiments/experiment_tracker.py analyze --experiment-id exp_001 exp_002

# Generate reports
python3 experiments/experiment_tracker.py report --output comparison.md
python3 experiments/experiment_tracker.py report --experiment-id exp_001

# List experiments
python3 experiments/experiment_tracker.py list
```

### Multi-Run Script

```bash
# Basic usage
./run_multiple_experiments.sh --cpu-runs 5 --gpu-runs 5

# Algorithm-specific
./run_multiple_experiments.sh --algorithm cpu --runs 10

# Custom parameters
./run_multiple_experiments.sh --generations 1000 --population 200

# Quick testing
./run_multiple_experiments.sh --quick-test

# Help
./run_multiple_experiments.sh --help
```

## Performance Metrics Explained

### Hypervolume
- Measures quality and diversity of Pareto front
- Higher values indicate better optimization performance
- Considers both convergence and spread of solutions

### Spacing
- Measures uniformity of solution distribution
- Lower values indicate more evenly distributed solutions
- Important for diverse trade-off options

### Runtime
- Total optimization execution time (seconds)
- Includes model evaluation and algorithm overhead
- Primary metric for speedup comparison

### Solution Count
- Number of non-dominated solutions found
- More solutions provide more trade-off options
- Quality depends on hypervolume coverage

## Experimental Design Best Practices

### Statistical Significance

For reliable results, run at least 5-10 repetitions per configuration:

```bash
# Minimum for significance testing
./run_multiple_experiments.sh --cpu-runs 5 --gpu-runs 5

# Better statistical power
./run_multiple_experiments.sh --cpu-runs 10 --gpu-runs 10
```

### Parameter Sensitivity

Test different MOEA parameters:

```bash
# Test population sizes
./run_multiple_experiments.sh --population 50 --runs 3
./run_multiple_experiments.sh --population 100 --runs 3
./run_multiple_experiments.sh --population 200 --runs 3

# Test generation counts
./run_multiple_experiments.sh --generations 250 --runs 3
./run_multiple_experiments.sh --generations 500 --runs 3
./run_multiple_experiments.sh --generations 1000 --runs 3
```

### Feature Set Comparison

Compare different feature sets by modifying config files:

1. Edit `moea_optimizer/config/moea_config_*_full.toml`
2. Change `feature_table` parameter
3. Run experiments with consistent parameters

## Database Queries for Analysis

### Basic Performance Comparison

```sql
SELECT 
    algorithm_type,
    COUNT(*) as total_runs,
    AVG(runtime_seconds) as avg_runtime,
    STDDEV(runtime_seconds) as std_runtime,
    AVG(hypervolume) as avg_hypervolume,
    AVG(n_solutions) as avg_solutions
FROM experiments e
JOIN experiment_results r ON e.experiment_id = r.experiment_id
GROUP BY algorithm_type;
```

### Recent Experiments

```sql
SELECT 
    e.experiment_id,
    e.algorithm_type,
    e.created_at,
    COUNT(r.run_number) as completed_runs,
    AVG(r.runtime_seconds) as avg_runtime
FROM experiments e
LEFT JOIN experiment_results r ON e.experiment_id = r.experiment_id
WHERE e.created_at > NOW() - INTERVAL '24 hours'
GROUP BY e.experiment_id, e.algorithm_type, e.created_at
ORDER BY e.created_at DESC;
```

### Statistical Analysis

```sql
-- CPU vs GPU performance comparison
WITH stats AS (
    SELECT 
        e.algorithm_type,
        AVG(r.runtime_seconds) as mean_runtime,
        STDDEV(r.runtime_seconds) as std_runtime,
        COUNT(*) as n_samples
    FROM experiments e
    JOIN experiment_results r ON e.experiment_id = r.experiment_id
    GROUP BY e.algorithm_type
)
SELECT 
    *,
    (SELECT mean_runtime FROM stats WHERE algorithm_type = 'cpu') / 
    (SELECT mean_runtime FROM stats WHERE algorithm_type = 'gpu') as speedup_factor
FROM stats;
```

## Troubleshooting

### Common Issues

**1. Database Connection Failed**
```bash
# Check PostgreSQL is running
docker ps | grep postgres

# Test connection
psql -h localhost -U postgres -d postgres -c "SELECT 1;"
```

**2. MOEA Container Fails**
```bash
# Check container logs
docker compose -f docker-compose.full-comparison.yml logs moea_optimizer_cpu

# Verify CUDA for GPU
nvidia-smi
```

**3. Python Dependencies Missing**
```bash
# Install requirements
pip3 install -r experiments/requirements.txt

# Or use conda
conda install psycopg2 pandas numpy scipy matplotlib seaborn
```

### Performance Issues

**1. Slow Database Queries**
```sql
-- Add indexes for better performance
CREATE INDEX IF NOT EXISTS idx_experiment_runtime ON experiment_results(runtime_seconds);
CREATE INDEX IF NOT EXISTS idx_experiment_algorithm ON experiments(algorithm_type);
```

**2. Large Result Files**
- Results stored in `experiments/results/` can grow large
- Clean periodically: `rm -rf experiments/results/old_experiment_*`
- Archive important results before cleanup

## Advanced Usage

### Custom Experiment Configuration

Create specialized experiment configurations:

```python
# Custom experiment with specific parameters
tracker = ExperimentTracker()

experiment_id = tracker.create_experiment(
    algorithm_type="gpu",
    moea_algorithm="NSGA-III",
    population_size=150,
    n_generations=750,
    feature_table="custom_features",
    dataset_period="2014-2015"
)

results = tracker.run_experiment(experiment_id, n_runs=5)
```

### Automated Experiment Scheduling

```bash
#!/bin/bash
# Daily experiment schedule

# Monday: CPU baseline
if [ $(date +%u) -eq 1 ]; then
    ./run_multiple_experiments.sh --algorithm cpu --runs 5
fi

# Wednesday: GPU comparison
if [ $(date +%u) -eq 3 ]; then
    ./run_multiple_experiments.sh --algorithm gpu --runs 5
fi

# Friday: Full comparison
if [ $(date +%u) -eq 5 ]; then
    ./run_multiple_experiments.sh --cpu-runs 3 --gpu-runs 3
    python3 experiments/experiment_tracker.py report --output weekly_report.md
fi
```

### Integration with External Tools

**MLflow Integration** (future enhancement):
```python
import mlflow

# Log experiment to MLflow
with mlflow.start_run():
    mlflow.log_param("algorithm_type", "gpu")
    mlflow.log_param("population_size", 100)
    mlflow.log_metric("hypervolume", result.hypervolume)
    mlflow.log_metric("runtime", result.runtime_seconds)
```

## Results Interpretation

### Performance Metrics Meaning

**Hypervolume > 0**: Good optimization performance
- Values 0-1: Basic optimization
- Values 1-5: Good performance
- Values > 5: Excellent performance

**Spacing < 1**: Well-distributed solutions
- Values 0-0.1: Excellent distribution
- Values 0.1-1: Good distribution
- Values > 1: Poor distribution

**Runtime Comparison**:
- GPU typically 20-30x faster than CPU
- Varies with problem complexity and hardware

### Statistical Significance

The system automatically calculates statistical significance using t-tests:
- **p < 0.05**: Statistically significant difference
- **p ≥ 0.05**: No significant difference
- **Multiple runs**: Required for meaningful statistics

## Future Enhancements

### Planned Features

1. **Real-time Monitoring**: Web dashboard for experiment progress
2. **Automated Parameter Tuning**: Hyperparameter optimization
3. **Cloud Integration**: Distributed experiment execution
4. **Advanced Visualizations**: Interactive plots and dashboards
5. **Model Versioning**: Track feature engineering changes

### Contributing

To add new features or metrics:

1. Extend `ExperimentResult` dataclass for new metrics
2. Update database schema with migrations
3. Modify `_execute_moea_run()` for data collection
4. Update report generation templates
5. Add appropriate SQL queries for analysis

## Support and Documentation

- **System Logs**: `experiment_tracker.log`
- **Database Schema**: Check `_init_database()` method
- **Example Queries**: See SQL sections above
- **Performance Tuning**: Monitor with `htop` and `nvidia-smi`

The experiment system is designed to be both simple for basic use and powerful for advanced analysis. Start with the quick start guide and gradually explore the advanced features as your experimental needs grow.