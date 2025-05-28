# MOEA Optimizer

Multi-Objective Evolutionary Algorithm (MOEA) benchmarking framework for comparing CPU and GPU implementations.

## Implementation Status

### ✅ Completed

#### Epic 1: Environment & CI/CD
- [x] Created `pyproject.toml` with pinned dependencies
- [x] Poetry configuration for reproducible installs
- [x] Project structure following the software plan

#### Epic 2: CPU Baseline (pymoo)
- [x] NSGA-III wrapper implementation (`nsga3_pymoo.py`)
- [x] Configuration loading from YAML files
- [x] Progress logging and callbacks
- [x] Results serialization (numpy arrays, CSV metrics)

#### Epic 3: Core Infrastructure
- [x] Configuration loader with YAML inheritance
- [x] Random seed management for reproducibility
- [x] Timer utilities with CUDA synchronization support
- [x] CLI interface with Click

#### Epic 5: Evaluation & Stats
- [x] Performance metrics (HV, IGD+, epsilon, spacing)
- [x] Convergence tracking
- [x] Results aggregation across multiple runs

### 🚧 In Progress

#### Epic 3: GPU Tensor (EvoX)
- [ ] TensorNSGA-III implementation wrapper
- [ ] Batch evaluation on GPU
- [ ] Memory monitoring

#### Epic 4: Benchmark Suite
- [x] DTLZ wrapper for pymoo
- [ ] WFG suite wrapper
- [ ] Real-world problem interfaces

### 📋 TODO

#### Epic 6: Automation & Reporting
- [ ] Visualization module (convergence plots, Pareto fronts)
- [ ] Statistical comparison (Wilcoxon, A12 effect size)
- [ ] HTML report generation
- [ ] GitHub Actions workflow

## Installation

```bash
# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Install with GPU support
poetry install -E gpu
```

## Usage

### Running Experiments

```bash
# Run CPU benchmark
poetry run moea-optimizer run --config config/cpu_dtlz.yaml

# Run GPU benchmark (when implemented)
poetry run moea-optimizer run --config config/gpu_dtlz.yaml

# Generate custom configuration
poetry run moea-optimizer generate-config \
  --base config/base.yaml \
  --output config/custom.yaml \
  --device gpu \
  --problem dtlz
```

### CLI Commands

```bash
# List available test problems
poetry run moea-optimizer list-problems

# Generate report from results
poetry run moea-optimizer report --results results/cpu_dtlz

# Run with debug logging
poetry run moea-optimizer --debug run --config config/cpu_dtlz.yaml
```

## Configuration

Configuration files use YAML format with inheritance from `base.yaml`:

```yaml
# config/base.yaml
seeds:
  numpy: 42
  torch: 42
  replications: 5

algorithm:
  population_size: 100
  n_generations: 200
  
# config/cpu_dtlz.yaml (inherits from base.yaml)
hardware:
  device: "cpu"
  
problem:
  suite: "dtlz"
  problems:
    - name: "DTLZ1"
      n_var: 7
      n_obj: 3
```

## Project Structure

```
moea_optimizer/
├── config/
│   ├── base.yaml           # Base configuration
│   ├── cpu_dtlz.yaml      # CPU DTLZ benchmark
│   └── gpu_dtlz.yaml      # GPU DTLZ benchmark
├── src/
│   ├── algorithms/
│   │   ├── cpu/
│   │   │   └── nsga3_pymoo.py
│   │   └── gpu/
│   │       └── nsga3_tensor.py (TODO)
│   ├── core/
│   │   ├── config_loader.py
│   │   ├── optimizer_runner.py
│   │   └── evaluation.py
│   ├── utils/
│   │   ├── seed.py
│   │   └── timer.py
│   └── cli.py
├── tests/
├── pyproject.toml
└── README.md
```

## Results Format

Results are saved in the following structure:

```
results/
└── experiment_name/
    ├── config.yaml           # Configuration used
    ├── summary.csv          # Aggregated metrics
    ├── complete_results.json # Detailed results
    ├── report.md            # Markdown report
    └── problem_name/
        └── run_0/
            ├── pareto_F.npy     # Objective values
            ├── pareto_X.npy     # Decision variables
            ├── metrics.json     # Performance metrics
            └── convergence.csv  # Convergence history
```

## Development

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src --cov-report=html

# Run specific test
poetry run pytest tests/test_evaluation.py
```

### Code Quality

```bash
# Format code
poetry run black src/

# Sort imports
poetry run isort src/

# Lint code
poetry run flake8 src/

# Type checking
poetry run mypy src/
```

## Next Steps

1. **Implement GPU wrapper**: Create `nsga3_tensor.py` using EvoX
2. **Add visualization**: Implement plotting functions in `visualiser.py`
3. **Statistical tests**: Add Wilcoxon and effect size calculations
4. **CI/CD**: Set up GitHub Actions workflow
5. **Docker support**: Add GPU-enabled Dockerfile

## Contributing

1. Follow the established code structure
2. Add tests for new functionality
3. Update documentation
4. Run code quality checks before committing

## License

[Your license here]