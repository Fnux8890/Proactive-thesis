# Sparse Pipeline Experiments

This directory contains performance experiments and benchmarks for the sparse pipeline architecture.

## Experiment Scripts

### 1. Basic Performance Measurement
```bash
# Run 3 iterations of the pipeline and measure timing
./run_pipeline_experiment.sh
```

This creates a timestamped JSON file with:
- Execution times for each stage
- Data metrics (records processed, features extracted)
- Statistical summary (mean, std dev)

### 2. GPU vs CPU Comparison
```bash
# Run both CPU and GPU experiments and generate comparison report
./run_comparison_experiment.sh
```

This runs the pipeline with:
- `DISABLE_GPU=true` for CPU-only baseline
- `DISABLE_GPU=false` for GPU acceleration
- Generates a markdown report with speedup analysis

## Configuration

Experiments can be configured via environment variables:

```bash
# Number of runs per experiment (default: 3)
NUM_RUNS=5 ./run_pipeline_experiment.sh

# Date range for processing
DATE_RANGE_START="2014-01-01"
DATE_RANGE_END="2014-12-31"

# Force CPU-only mode
DISABLE_GPU=true ./run_pipeline_experiment.sh
```

## Results Structure

```
experiments/
├── results/
│   ├── sparse_pipeline_baseline_YYYYMMDD_HHMMSS.json
│   ├── sparse_pipeline_cpu_YYYYMMDD_HHMMSS.json
│   ├── sparse_pipeline_gpu_YYYYMMDD_HHMMSS.json
│   └── comparison_summary_YYYYMMDD_HHMMSS.json
└── performance_report_YYYYMMDD_HHMMSS.md
```

## Interpreting Results

### Key Metrics

1. **Ingestion Time**: Time to load raw data into PostgreSQL
   - I/O bound, no GPU benefit expected
   - Depends on disk speed and database performance

2. **Sparse Pipeline Time**: Time for stages 2-4
   - Stage 1: Hourly aggregation (SQL query)
   - Stage 2: Gap filling (CPU-bound)
   - Stage 3: Feature extraction (GPU-accelerated)
   - Stage 4: Era creation (CPU-bound)

3. **Feature Extraction Rate**: Features per second
   - Primary GPU performance indicator
   - Currently limited by partial GPU implementation

### Expected Performance

Based on architecture analysis:
- **Current GPU speedup**: ~2-3x (only basic statistics on GPU)
- **Potential GPU speedup**: 15-20x (with full GPU implementation)

### Statistical Validation

Running multiple iterations helps account for:
- System load variations
- Database caching effects
- GPU warm-up time
- Network latency variations

Standard deviation should be <10% of mean for reliable results.

## Future Experiments

Planned experiments to add:

1. **Scaling Test**: Vary date ranges to test scalability
2. **Memory Profile**: Track GPU/CPU memory usage
3. **Batch Size Optimization**: Find optimal window batch size
4. **Multi-GPU**: Test with multiple GPUs (when available)
5. **Algorithm Comparison**: Benchmark individual feature algorithms

## Visualization

Results can be visualized using the JSON data:

```python
import json
import matplotlib.pyplot as plt

with open('results/comparison_summary_*.json', 'r') as f:
    data = json.load(f)

# Create speedup chart
metrics = data['metrics']
speedups = [1, metrics['speedup']['overall']]
labels = ['CPU Baseline', 'GPU Accelerated']

plt.bar(labels, speedups)
plt.ylabel('Relative Performance')
plt.title('GPU Acceleration Speedup')
plt.show()
```