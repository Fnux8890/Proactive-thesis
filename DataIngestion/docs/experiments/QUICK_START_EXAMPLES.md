# Experiment System: Quick Start Examples

## Basic Usage Examples

### Example 1: First-Time Setup and Test

```bash
# 1. Quick test to verify everything works
./run_multiple_experiments.sh --quick-test

# Expected output:
# === MOEA Multi-Run Experiment Suite ===
# Configuration:
#   CPU Runs: 2
#   GPU Runs: 2
#   Generations: 100
#   Population: 50
#   Quick Mode: true
# 
# ✓ Prerequisites check passed
# Starting cpu experiments (2 runs)...
# [cpu] Run 1/2
# Created experiment: exp_2025_06_01_143022_cpu
# ...
# ✓ All experiments completed successfully!
```

### Example 2: Production Statistical Analysis

```bash
# 2. Run statistically significant experiments
./run_multiple_experiments.sh --cpu-runs 5 --gpu-runs 5

# This will:
# - Create 2 experiments (1 CPU, 1 GPU)
# - Run 5 repetitions of each
# - Generate statistical comparison report
# - Store all results in database
```

### Example 3: Single Algorithm Deep Analysis

```bash
# 3. Focus on GPU performance with many runs
./run_multiple_experiments.sh --algorithm gpu --runs 10

# This will:
# - Create 1 GPU experiment
# - Run 10 repetitions for statistical confidence
# - Enable detailed performance analysis
```

## Database Usage Examples

### Example 4: Accessing Your Results

```sql
-- Connect to database
psql -h localhost -U postgres -d postgres

-- See all your experiments
SELECT 
    experiment_id,
    algorithm_type,
    created_at,
    population_size,
    n_generations
FROM experiments 
ORDER BY created_at DESC;

-- Example output:
--     experiment_id         | algorithm_type |      created_at       | population_size | n_generations 
-- -------------------------+----------------+----------------------+-----------------+---------------
--  exp_2025_06_01_143022_gpu |     gpu        | 2025-06-01 14:30:22  |      100        |      500
--  exp_2025_06_01_143015_cpu |     cpu        | 2025-06-01 14:30:15  |      100        |      500
```

### Example 5: Performance Comparison Query

```sql
-- Compare CPU vs GPU performance
WITH performance_summary AS (
    SELECT 
        e.algorithm_type,
        AVG(r.runtime_seconds) as avg_runtime,
        AVG(r.hypervolume) as avg_hypervolume,
        AVG(r.n_solutions) as avg_solutions,
        COUNT(*) as total_runs,
        STDDEV(r.runtime_seconds) as std_runtime
    FROM experiments e
    JOIN experiment_results r ON e.experiment_id = r.experiment_id
    GROUP BY e.algorithm_type
)
SELECT 
    *,
    ROUND(
        (SELECT avg_runtime FROM performance_summary WHERE algorithm_type = 'cpu') / 
        NULLIF((SELECT avg_runtime FROM performance_summary WHERE algorithm_type = 'gpu'), 0),
        2
    ) as speedup_factor
FROM performance_summary;

-- Example output:
-- algorithm_type | avg_runtime | avg_hypervolume | avg_solutions | total_runs | std_runtime | speedup_factor
-- ---------------+-------------+-----------------+---------------+------------+-------------+----------------
--      cpu       |    5.44     |      0.00       |      12       |     5      |    0.12     |      27.20
--      gpu       |    0.20     |      3.59       |      26       |     5      |    0.02     |      27.20
```

## Report Generation Examples

### Example 6: Generate Analysis Report

```bash
# Generate comprehensive report
python3 experiments/experiment_tracker.py report --output my_analysis.md

# Generated report will include:
# - Executive summary with key findings
# - Statistical significance testing
# - Performance comparison tables
# - Database access instructions
# - Reproducibility commands
```

### Example 7: View Report Content

```bash
# Check generated report
cat experiments/reports/my_analysis.md

# Example output:
# # MOEA Greenhouse Optimization: Experiment Analysis Report
# 
# *Generated: 2025-06-01 14:35:22*
# 
# ## Executive Summary
# 
# This report analyzes 10 experimental runs across 2 unique experiments, 
# comparing CPU and GPU-accelerated Multi-Objective Evolutionary Algorithms...
# 
# ### Key Findings
# 
# - **GPU Speedup**: 27.2x faster than CPU implementation
# - **Statistical Significance**: ✅ Significant (p=0.0001)
```

## Advanced Examples

### Example 8: Custom Parameter Testing

```bash
# Test different population sizes
./run_multiple_experiments.sh --population 50 --cpu-runs 3 --gpu-runs 3
./run_multiple_experiments.sh --population 200 --cpu-runs 3 --gpu-runs 3

# Test different generation counts
./run_multiple_experiments.sh --generations 250 --cpu-runs 3 --gpu-runs 3
./run_multiple_experiments.sh --generations 1000 --cpu-runs 3 --gpu-runs 3
```

### Example 9: Analyzing Specific Experiments

```bash
# List all experiments to get IDs
python3 experiments/experiment_tracker.py list

# Example output:
# Experiments:
# experiment_id               algorithm_type  runs
# exp_2025_06_01_143022_gpu   gpu            5
# exp_2025_06_01_143015_cpu   cpu            5

# Analyze specific experiments
python3 experiments/experiment_tracker.py analyze --experiment-id exp_2025_06_01_143022_gpu exp_2025_06_01_143015_cpu

# Generate report for specific experiments only
python3 experiments/experiment_tracker.py report --experiment-id exp_2025_06_01_143022_gpu --output gpu_only_report.md
```

### Example 10: Extract Raw Data for External Analysis

```sql
-- Export data for R/Python analysis
COPY (
    SELECT 
        e.experiment_id,
        e.algorithm_type,
        e.moea_algorithm,
        e.population_size,
        e.n_generations,
        r.run_number,
        r.runtime_seconds,
        r.hypervolume,
        r.spacing,
        r.n_solutions,
        r.created_at
    FROM experiments e
    JOIN experiment_results r ON e.experiment_id = r.experiment_id
    ORDER BY e.algorithm_type, r.run_number
) TO '/tmp/experiment_results.csv' WITH CSV HEADER;
```

### Example 11: Monitoring Experiment Progress

```bash
# Monitor database in real-time
watch -n 10 'psql -h localhost -U postgres -d postgres -c "
SELECT 
    e.algorithm_type,
    COUNT(r.run_number) as completed_runs,
    AVG(r.runtime_seconds) as avg_runtime
FROM experiments e
LEFT JOIN experiment_results r ON e.experiment_id = r.experiment_id
WHERE e.created_at > NOW() - INTERVAL '\''1 hour'\''
GROUP BY e.algorithm_type;
"'

# Monitor GPU usage during experiments
watch -n 5 nvidia-smi

# Monitor system resources
htop
```

## Practical Workflows

### Workflow 1: Daily Performance Testing

```bash
#!/bin/bash
# daily_experiments.sh

DATE=$(date +%Y%m%d)
echo "Starting daily experiments for $DATE"

# Quick performance check
./run_multiple_experiments.sh --quick-test

# Generate daily report
python3 experiments/experiment_tracker.py report --output "daily_report_$DATE.md"

echo "Daily experiments completed. Report: experiments/reports/daily_report_$DATE.md"
```

### Workflow 2: Algorithm Comparison Study

```bash
#!/bin/bash
# algorithm_comparison.sh

echo "=== Algorithm Comparison Study ==="

# Run comprehensive comparison
./run_multiple_experiments.sh --cpu-runs 10 --gpu-runs 10 --generations 500

# Generate detailed analysis
python3 experiments/experiment_tracker.py report --output algorithm_comparison_study.md

# Extract statistical data
psql -h localhost -U postgres -d postgres -c "
SELECT 
    algorithm_type,
    AVG(runtime_seconds) as mean_runtime,
    STDDEV(runtime_seconds) as std_runtime,
    MIN(runtime_seconds) as min_runtime,
    MAX(runtime_seconds) as max_runtime,
    AVG(hypervolume) as mean_hypervolume,
    AVG(n_solutions) as mean_solutions
FROM experiments e
JOIN experiment_results r ON e.experiment_id = r.experiment_id
GROUP BY algorithm_type;
" > algorithm_stats.txt

echo "Algorithm comparison completed!"
echo "- Report: experiments/reports/algorithm_comparison_study.md"
echo "- Statistics: algorithm_stats.txt"
```

### Workflow 3: Performance Scaling Analysis

```bash
#!/bin/bash
# scaling_analysis.sh

echo "=== Performance Scaling Analysis ==="

# Test different population sizes
for pop_size in 50 100 200 400; do
    echo "Testing population size: $pop_size"
    ./run_multiple_experiments.sh --population $pop_size --cpu-runs 3 --gpu-runs 3 --generations 250
    sleep 10
done

# Generate scaling report
python3 experiments/experiment_tracker.py report --output scaling_analysis.md

echo "Scaling analysis completed!"
```

## Expected Results

### Typical Performance Numbers

Based on the current system, you can expect:

**CPU Performance (NSGA-III)**:
- Runtime: 5-6 seconds per optimization
- Hypervolume: 0.0-0.1 (variable)
- Solutions: 10-15 Pareto-optimal points
- Consistency: ±10% variation between runs

**GPU Performance (NSGA-II)**:
- Runtime: 0.2-0.3 seconds per optimization
- Hypervolume: 3-4 (typically better coverage)
- Solutions: 20-30 Pareto-optimal points
- Consistency: ±5% variation between runs

**Speedup Factor**:
- Typical range: 20-30x faster
- Depends on problem complexity
- More pronounced with larger populations

### Statistical Significance

With 5+ runs per algorithm:
- p-values typically < 0.001 (highly significant)
- 95% confidence intervals clearly separated
- Effect size (Cohen's d) typically > 2.0 (large effect)

## Troubleshooting Examples

### Problem: "Database connection failed"

```bash
# Check if PostgreSQL is running
docker ps | grep postgres

# If not running, start the database
docker compose up -d db

# Test connection manually
psql -h localhost -U postgres -d postgres -c "SELECT 1;"
```

### Problem: "MOEA container fails"

```bash
# Check container logs
docker compose -f docker-compose.full-comparison.yml logs moea_optimizer_cpu
docker compose -f docker-compose.full-comparison.yml logs moea_optimizer_gpu

# For GPU issues, check CUDA
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi
```

### Problem: "No results generated"

```bash
# Check experiment directory
ls -la experiments/results/

# Check database content
psql -h localhost -U postgres -d postgres -c "SELECT COUNT(*) FROM experiments;"
psql -h localhost -U postgres -d postgres -c "SELECT COUNT(*) FROM experiment_results;"

# Check for error logs
tail -f experiment_tracker.log
```

These examples provide a complete foundation for using the experiment system effectively. Start with the basic examples and gradually move to more advanced workflows as you become comfortable with the system.