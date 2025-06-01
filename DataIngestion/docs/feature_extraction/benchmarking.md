# Benchmarks

This directory contains performance benchmarking tools and results for the feature extraction pipeline.

## Structure

```
benchmarks/
├── src/                    # Benchmark source code
│   └── benchmark_db_utils.py
├── docker/                 # Docker configuration
│   └── benchmark.dockerfile
├── results/                # Benchmark results
│   ├── benchmark_results_20250525_131859.json
│   └── benchmark_results_20250525_141649.json
├── benchmark_requirements.txt
└── README.md
```

## Running Benchmarks

### Local Execution

```bash
# Install dependencies
pip install -r benchmark_requirements.txt

# Run benchmark
python src/benchmark_db_utils.py
```

### Docker Execution

```bash
# Build benchmark image
docker build -f docker/benchmark.dockerfile -t feature-extraction-benchmark .

# Run benchmark container
docker run --network=dataingestion_default \
  -v $(pwd)/results:/app/results \
  feature-extraction-benchmark
```

## Benchmark Metrics

The benchmarks measure:
- Database connection pool performance
- Query execution times
- Data retrieval rates
- Memory usage patterns
- Connection thread safety

## Results Format

Results are saved as JSON files with timestamps:
- Timestamp of execution
- Performance metrics
- System information
- Configuration parameters

## Analysis

To analyze benchmark results:
```python
import json
import pandas as pd

# Load results
with open('results/benchmark_results_TIMESTAMP.json', 'r') as f:
    data = json.load(f)

# Convert to DataFrame for analysis
df = pd.DataFrame(data['metrics'])
print(df.describe())
```