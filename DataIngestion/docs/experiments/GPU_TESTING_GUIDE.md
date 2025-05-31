# GPU Testing Guide for Sparse Pipeline

## Overview

This guide provides step-by-step instructions for testing GPU acceleration in the sparse pipeline after fixing the GPU detection bug. The testing framework includes multiple scripts for validation, performance comparison, and monitoring.

## Prerequisites

### 1. System Requirements
- NVIDIA GPU with CUDA support
- NVIDIA drivers installed (version 525+ recommended)
- Docker with nvidia-container-toolkit
- At least 4GB GPU memory
- Linux or WSL2 with GPU support

### 2. Verify GPU Setup
```bash
# Check NVIDIA drivers
nvidia-smi

# Test Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## Testing Scripts

### 1. GPU Validation Script
**Purpose**: Quick validation of GPU configuration before running tests

```bash
./validate_gpu_setup.sh
```

**What it checks**:
- NVIDIA driver installation
- Docker GPU runtime
- Environment configuration
- Container build success
- GPU initialization in sparse pipeline

**Expected Output**:
```
✓ nvidia-smi found
✓ Docker can access GPU
✓ DISABLE_GPU=false in .env.sparse
✓ GPU capabilities configured in docker-compose.sparse.yml
✓ Sparse pipeline container built successfully
✓ GPU successfully initialized in sparse pipeline
```

### 2. Quick GPU Test
**Purpose**: Fast single-month comparison between CPU and GPU

```bash
./quick_gpu_test.sh
```

**Features**:
- Tests May 2014 data (1 month)
- Single run for each mode
- ~2-3 minutes total execution
- Immediate feedback on GPU functionality

**Expected Results**:
```
CPU Time: 10.87s
GPU Time: 4.15s
Speedup: 2.6x
GPU Working: Yes
```

### 3. Comprehensive CPU vs GPU Comparison
**Purpose**: Statistical comparison with multiple runs

```bash
./run_cpu_vs_gpu_comparison.sh
```

**Features**:
- 5 runs each for CPU and GPU
- 6 months of data (Jan-Jun 2014)
- Statistical analysis (mean, std dev)
- Detailed performance report
- ~30-45 minutes total execution

**Expected Results**:
- CPU Mean: 12.65s ± 0.05s
- GPU Mean: 4.85s ± 0.10s
- Speedup: 2.6x
- Feature Rate Improvement: 2.6x

### 4. GPU Monitoring During Tests
**Purpose**: Real-time GPU utilization tracking

```bash
./monitor_gpu_during_test.sh
```

**Features**:
- Second-by-second GPU monitoring
- Utilization, memory, temperature tracking
- Generates usage plots
- Interactive menu for test selection

**Monitoring Metrics**:
- GPU Utilization (%)
- Memory Usage (MB)
- Temperature (°C)
- Power Consumption (W)

## Step-by-Step Testing Procedure

### Phase 1: Initial Validation (5 minutes)

1. **Set up environment**:
   ```bash
   cd /mnt/c/Users/fhj88/Documents/Github/Proactive-thesis/DataIngestion
   cp .env.sparse .env
   ```

2. **Rebuild container with GPU fix**:
   ```bash
   docker compose -f docker-compose.sparse.yml build --no-cache sparse_pipeline
   ```

3. **Run validation**:
   ```bash
   ./validate_gpu_setup.sh
   ```

4. **Check for issues**:
   - If any checks fail, resolve before proceeding
   - Common fixes in troubleshooting section below

### Phase 2: Quick Functionality Test (5 minutes)

1. **Run quick test**:
   ```bash
   ./quick_gpu_test.sh
   ```

2. **Verify GPU activation**:
   - Check for "CUDA context initialized" in logs
   - Confirm speedup > 1.5x
   - Review generated report

3. **Monitor GPU during test** (optional):
   ```bash
   # In a separate terminal
   watch -n 1 nvidia-smi
   ```

### Phase 3: Comprehensive Performance Analysis (45 minutes)

1. **Run full comparison**:
   ```bash
   ./run_cpu_vs_gpu_comparison.sh
   ```

2. **During execution**:
   - Monitor progress in terminal
   - Check for any failed runs
   - Observe consistency across runs

3. **Review results**:
   - Open generated report: `docs/experiments/results/cpu_gpu_comparison/comparison_report_*.md`
   - Verify GPU was active in all GPU runs
   - Check statistical significance

### Phase 4: GPU Utilization Analysis (10 minutes)

1. **Run monitored test**:
   ```bash
   ./monitor_gpu_during_test.sh
   # Select option 1 for quick GPU test
   ```

2. **Analyze usage patterns**:
   - Peak GPU utilization (expect 60-80%)
   - Memory usage (expect 1-2GB)
   - Temperature stability

3. **Review plots**:
   - GPU usage timeline
   - Memory consumption pattern
   - Identify optimization opportunities

## Expected Performance Improvements

### Conservative Estimates (Partially Optimized)
| Metric | CPU | GPU | Improvement |
|--------|-----|-----|-------------|
| 1 Month | 10.9s | 4.2s | 2.6x |
| 6 Months | 12.7s | 4.9s | 2.6x |
| Feature Rate | 29 feat/s | 75 feat/s | 2.6x |

### Optimistic Estimates (Fully Optimized)
| Metric | CPU | GPU | Improvement |
|--------|-----|-----|-------------|
| 1 Month | 10.9s | 1.4s | 7.8x |
| 6 Months | 12.7s | 1.6s | 7.9x |
| Feature Rate | 29 feat/s | 230 feat/s | 7.9x |

## Interpreting Results

### GPU Successfully Activated
- Log shows: "CUDA context initialized for sparse pipeline"
- Speedup > 1.5x observed
- GPU utilization > 10% during feature extraction
- Consistent performance across runs

### GPU Not Working
- Log shows: "GPU disabled by environment variable"
- No speedup or slower than CPU
- GPU utilization remains at 0%
- High variance in results

## Troubleshooting

### Issue: "nvidia-smi not found"
**Solution**:
```bash
# Install NVIDIA drivers
sudo apt update
sudo apt install nvidia-driver-525
sudo reboot
```

### Issue: "Docker cannot access GPU"
**Solution**:
```bash
# Install nvidia-container-toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update
sudo apt install nvidia-container-toolkit
sudo systemctl restart docker
```

### Issue: "GPU disabled by environment variable"
**Solution**:
1. Check `.env` file has `DISABLE_GPU=false`
2. Rebuild container: `docker compose -f docker-compose.sparse.yml build --no-cache sparse_pipeline`
3. Ensure no override in docker-compose command

### Issue: "CUDA out of memory"
**Solution**:
1. Reduce batch size: Set `SPARSE_BATCH_SIZE=12` in `.env`
2. Close other GPU applications
3. Monitor GPU memory with `nvidia-smi`

### Issue: Low GPU utilization (<20%)
**Possible Causes**:
1. CPU bottleneck in data loading
2. Small batch size
3. Insufficient parallelization

**Solutions**:
1. Increase batch size if memory allows
2. Profile code to identify bottlenecks
3. Consider implementing more GPU kernels

## Performance Optimization Tips

1. **Batch Size Tuning**:
   - Start with 24 (default)
   - Increase to 48 or 96 if GPU memory allows
   - Monitor memory usage to avoid OOM

2. **Window Size Optimization**:
   - Default: 12-hour windows
   - Try 24-hour windows for better GPU utilization
   - Balance accuracy vs performance

3. **GPU Memory Management**:
   - Use `CUDA_VISIBLE_DEVICES` to select specific GPU
   - Set memory growth for TensorFlow/PyTorch if used
   - Clear GPU cache between runs

## Reporting Results

When reporting test results, include:

1. **System Information**:
   - GPU model (from `nvidia-smi`)
   - CUDA version
   - Docker version
   - OS and kernel version

2. **Test Configuration**:
   - Date range tested
   - Number of runs
   - Batch size and window configuration

3. **Performance Metrics**:
   - Mean execution times (CPU vs GPU)
   - Standard deviation
   - Speedup factor
   - Feature extraction rate

4. **GPU Utilization**:
   - Average/peak utilization
   - Memory usage
   - Any errors or warnings

## Next Steps After Testing

1. **If GPU working correctly**:
   - Run full year test (2014 complete)
   - Test MOEA integration
   - Deploy to production

2. **If GPU not working**:
   - Review troubleshooting section
   - Check system logs: `dmesg | grep -i nvidia`
   - File issue with detailed logs

3. **For further optimization**:
   - Profile GPU kernels
   - Implement additional GPU algorithms
   - Test multi-GPU scaling

## Appendix: Log Examples

### Successful GPU Initialization
```
[INFO] gpu_feature_extraction: Starting GPU sparse pipeline mode
[INFO] gpu_feature_extraction: CUDA context initialized for sparse pipeline
[INFO] Sparse pipeline initialized with GPU acceleration
```

### Failed GPU Initialization
```
[INFO] gpu_feature_extraction: Starting GPU sparse pipeline mode
[INFO] gpu_feature_extraction: GPU disabled by environment variable (DISABLE_GPU=false)
[WARN] GPU initialization failed: CUDA error. Using CPU fallback.
```

### Healthy GPU Utilization
```
GPU: 65% | Mem: 1823/4096 MB | Temp: 72°C | Power: 45W
```