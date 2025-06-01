# Enhanced Sparse Pipeline - CPU vs GPU Benchmark Experiments

This directory contains comprehensive benchmarking scripts and results for comparing CPU vs GPU performance of the Enhanced Sparse Pipeline.

## 📋 Experiment Overview

The benchmarks compare:
- **Basic vs Enhanced** pipeline modes
- **CPU vs GPU** performance
- **Different batch sizes** (12, 24, 48, 96)
- **Various dataset sizes** (1, 3, 6, 12 months)

### Key Metrics Measured
- Processing time (seconds)
- Features extracted per second
- Total feature count
- GPU utilization (%)
- Memory usage (MB)
- Speedup ratios

## 🚀 Quick Start

### Option 1: Quick Benchmark (5-10 minutes)

```bash
# Run quick benchmark with subset of tests
./quick_benchmark.sh
```

This runs 7 key tests to validate:
- Basic vs Enhanced mode
- CPU vs GPU performance
- Different batch sizes
- GPU utilization

### Option 2: Docker-based Benchmark (10-20 minutes)

```bash
# Run benchmarks in Docker container
./docker_benchmark.sh
```

This runs 9 comprehensive tests entirely within Docker containers, ensuring consistency.

### Option 3: Full Benchmark Suite (30-60 minutes)

```bash
# Run complete benchmark suite
./run_cpu_gpu_benchmarks.sh
```

This runs all combinations of:
- 4 batch sizes × 4 date ranges × 2 modes × 2 devices = up to 64 tests

## 📊 Expected Results

### Performance Targets

| Metric | Basic Pipeline | Enhanced Pipeline | Improvement |
|--------|----------------|-------------------|-------------|
| **GPU Utilization** | 65-75% | 85-95% | +20-30% |
| **Feature Count** | ~350 | ~1,200+ | +3.4x |
| **Processing Speed** | 77 feat/s | 150+ feat/s | +2x |
| **GPU Speedup** | 5-10x | 10-20x | Over CPU |

### Sample Results

```
Enhanced GPU (1 month): 1,200 features in 8s = 150 feat/s (GPU: 87%)
Basic GPU (1 month): 350 features in 4.5s = 77 feat/s (GPU: 68%)
Enhanced CPU (1 month): 1,200 features in 95s = 12.6 feat/s
Basic CPU (1 month): 350 features in 35s = 10 feat/s
```

## 📁 Directory Structure

```
enhanced_sparse_pipeline/
├── README.md                    # This file
├── quick_benchmark.sh          # Quick validation script
├── docker_benchmark.sh         # Docker-based benchmarks
├── run_cpu_gpu_benchmarks.sh   # Full benchmark suite
├── scripts/
│   └── analyze_benchmark_results.py  # Result analysis
├── results/                    # Benchmark results by timestamp
│   └── YYYYMMDD_HHMMSS/
│       ├── benchmark_results.csv
│       ├── benchmark_summary.json
│       ├── performance_report.md
│       ├── performance_comparison.png
│       ├── detailed_analysis.png
│       └── scaling_analysis.png
└── logs/                       # Detailed execution logs
    └── YYYYMMDD_HHMMSS/
        └── *.log
```

## 🔍 Understanding Results

### Result Files

1. **benchmark_results.csv**: Raw data with all metrics
2. **performance_report.md**: Comprehensive analysis report
3. **performance_comparison.png**: Visual comparison charts
4. **scaling_analysis.png**: Dataset size scaling graphs

### Key Metrics Explained

- **features_per_second**: Primary performance metric
- **gpu_util_avg**: Average GPU utilization (target: 85-95%)
- **feature_count**: Total features extracted (enhanced: ~1,200)
- **memory_peak_mb**: Maximum GPU memory used
- **speedup**: GPU performance vs CPU baseline

## 🐛 Troubleshooting

### Low GPU Utilization (<85%)
- Increase batch size: `--batch-size 48` or `--batch-size 96`
- Check GPU availability: `nvidia-smi`
- Verify enhanced mode: `ENHANCED_MODE=true`

### Out of Memory Errors
- Reduce batch size: `--batch-size 12`
- Check available GPU memory: `nvidia-smi`
- Monitor during execution: `watch -n 1 nvidia-smi`

### Slow Performance
- Ensure GPU is enabled: `DISABLE_GPU=false`
- Check Docker GPU runtime: `docker run --gpus all nvidia/cuda:12.4.1-base nvidia-smi`
- Verify data is loaded: Check database has sensor data

## 📈 Interpreting Performance

### Good Results
- Enhanced GPU: 150+ features/second
- GPU utilization: 85-95%
- GPU speedup: 10-20x over CPU
- Feature count: 1,200+ (enhanced mode)

### Optimization Tips
1. **Batch Size**: 24-48 typically optimal
2. **GPU Memory**: Keep under 80% of available
3. **Data Range**: Larger datasets show better GPU efficiency
4. **Resolution**: Multi-resolution adds 5x features

## 🔬 Running Custom Experiments

### Modify Test Configurations

Edit `BENCHMARKS` array in scripts:
```bash
declare -a BATCH_SIZES=(12 24 48 96)
declare -a DATE_RANGES=(
    "2014-01-01:2014-01-31"    # 1 month
    "2014-01-01:2014-06-30"    # 6 months
)
```

### Add New Tests

In `docker_benchmark.sh`, add to `BENCHMARKS`:
```python
{"name": "custom_test", "mode": "enhanced", "gpu": True, 
 "batch": 32, "start": "2014-01-01", "end": "2014-12-31"}
```

## 📊 Visualizing Results

After running benchmarks:
```bash
# View performance report
cat results/*/performance_report.md

# Open visualization plots
xdg-open results/*/performance_comparison.png

# Quick CSV analysis
column -t -s',' results/*/benchmark_results.csv | less
```

## 🎯 Success Criteria

The enhanced sparse pipeline is considered successful if:
- ✅ GPU utilization reaches 85-95% (vs 65-75% basic)
- ✅ Feature count is 1,200+ (vs 350 basic)
- ✅ Processing speed is 150+ feat/s (vs 77 basic)
- ✅ GPU provides 10-20x speedup over CPU
- ✅ Handles 1+ year of data efficiently

## 📝 Notes

- First run may be slower due to Docker image building
- GPU tests require NVIDIA GPU with CUDA support
- CPU tests are included for baseline comparison
- Enhanced CPU mode is slow for large datasets (expected)

---

Ready to benchmark? Start with `./quick_benchmark.sh` for rapid validation!