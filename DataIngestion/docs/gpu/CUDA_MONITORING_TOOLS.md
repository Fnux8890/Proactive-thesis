# CUDA Monitoring and Profiling Tools Guide

## Real-time Monitoring

### 1. **nvidia-smi** (System Monitoring Interface)
```bash
# Basic GPU status
nvidia-smi

# Continuous monitoring (refresh every 1 second)
nvidia-smi -l 1

# Detailed view with processes
nvidia-smi -q

# Monitor specific metrics
nvidia-smi --query-gpu=gpu_name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free --format=csv -l 1

# Docker container GPU monitoring
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### 2. **nvtop** (Interactive GPU Process Viewer)
```bash
# Install nvtop
sudo apt install nvtop

# Run interactive monitor
nvtop

# Features:
# - Real-time GPU usage graphs
# - Process list with GPU memory usage
# - Temperature and power monitoring
# - Kill processes directly from interface
```

### 3. **gpustat** (Simple GPU Status)
```bash
# Install
pip install gpustat

# Show GPU status
gpustat -cp

# Watch mode
watch -n 1 gpustat -cp
```

## Profiling Tools

### 1. **Nsight Compute** (Kernel-level Profiling)
```bash
# Profile all kernels
ncu -o profile_report ./gpu_feature_extraction

# Profile specific kernel
ncu --kernel-name compute_statistics -o stats_kernel ./gpu_feature_extraction

# Generate roofline analysis
ncu --set roofline -o roofline_report ./gpu_feature_extraction

# Detailed analysis with source correlation
ncu --set full --source-folders /path/to/source -o detailed_report ./gpu_feature_extraction
```

### 2. **Nsight Systems** (System-wide Profiling)
```bash
# Basic profiling
nsys profile -o timeline_report ./gpu_feature_extraction

# With CUDA API trace
nsys profile -t cuda,nvtx,osrt -o full_report ./gpu_feature_extraction

# Generate statistics
nsys stats timeline_report.nsys-rep
```

### 3. **CUDA Built-in Profiling**
```cuda
// Add to your CUDA code for custom metrics
#include <cuda_profiler_api.h>

cudaProfilerStart();
// Your kernel launches
cudaProfilerStop();
```

## Docker Integration

### Add CUDA profiling to Docker container:
```dockerfile
# In your Dockerfile
RUN apt-get update && apt-get install -y \
    cuda-nsight-compute-11-8 \
    cuda-nsight-systems-11-8 \
    nvtop

# Enable profiling permissions
ENV CUDA_INJECTION64_PATH=""
```

### Run container with profiling:
```bash
# With Nsight Compute access
docker run --gpus all --cap-add=SYS_ADMIN --security-opt seccomp=unconfined \
    -v $(pwd)/profiles:/profiles \
    gpu_feature_extraction \
    ncu -o /profiles/kernel_analysis ./app

# With nvidia-smi monitoring
docker exec -it gpu_feature_extraction_level_a nvidia-smi
```

## Debugging CUDA Errors

### 1. **Enable Synchronous Execution**
```bash
export CUDA_LAUNCH_BLOCKING=1
```

### 2. **Memory Checking**
```bash
# Run with cuda-memcheck
cuda-memcheck ./gpu_feature_extraction

# Check for race conditions
cuda-memcheck --tool racecheck ./gpu_feature_extraction

# Initialize uninitialized memory
cuda-memcheck --tool initcheck ./gpu_feature_extraction
```

### 3. **GDB CUDA Debugging**
```bash
# Compile with debug info
nvcc -g -G your_kernel.cu

# Debug with cuda-gdb
cuda-gdb ./gpu_feature_extraction
```

## Performance Metrics to Monitor

### Key Metrics:
1. **Occupancy**: Ratio of active warps to maximum warps
2. **SM Efficiency**: How well the SMs are utilized
3. **Memory Throughput**: GB/s achieved vs theoretical
4. **Warp Execution Efficiency**: % of active threads in executed warps
5. **Shared Memory Bank Conflicts**: Should be minimized
6. **Register Spilling**: Indicates if using too many registers

### Quick Performance Check:
```bash
# In your container
echo "=== GPU Status ==="
nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,utilization.memory,memory.used --format=csv

echo -e "\n=== CUDA Compilation Info ==="
nvcc --version

echo -e "\n=== Available CUDA Tools ==="
which ncu nsys cuda-memcheck cuda-gdb
```

## Integration with Our Pipeline

### 1. Add monitoring to docker-compose.yml:
```yaml
gpu_feature_extraction_level_a:
  environment:
    CUDA_LAUNCH_BLOCKING: ${CUDA_DEBUG:-0}
    CUDA_VISIBLE_DEVICES: 0
  # Add profiling capabilities
  cap_add:
    - SYS_ADMIN
  security_opt:
    - seccomp:unconfined
```

### 2. Create profiling script:
```bash
#!/bin/bash
# profile_gpu_extraction.sh

# Monitor GPU in background
nvidia-smi -l 1 > gpu_usage.log &
NVIDIA_PID=$!

# Run with profiling
docker compose run --rm \
    -e CUDA_LAUNCH_BLOCKING=1 \
    gpu_feature_extraction_level_a \
    ncu --set roofline -o /app/profiles/level_a_profile

# Stop monitoring
kill $NVIDIA_PID

# Analyze results
nsys stats /app/profiles/level_a_profile.nsys-rep
```

### 3. Add to Rust code for better profiling:
```rust
// In main.rs
#[cfg(feature = "profile")]
{
    println!("Profiling enabled - synchronizing CUDA operations");
    std::env::set_var("CUDA_LAUNCH_BLOCKING", "1");
}
```