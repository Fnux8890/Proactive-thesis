# MOEA CPU vs GPU Performance Comparison Report

## Executive Summary

This report documents the performance comparison between CPU and GPU implementations of the Multi-Objective Evolutionary Algorithm (MOEA) optimizer in the DataIngestion pipeline. The tests were conducted on a system with 32GB RAM and an NVIDIA GeForce RTX 4070 GPU.

## Test Configuration

### Hardware Specifications
- **CPU**: (System CPU - likely Intel/AMD multi-core)
- **GPU**: NVIDIA GeForce RTX 4070 (12GB VRAM)
- **System RAM**: 32GB
- **Shared Memory**: 8GB allocated for Docker containers

### Software Stack
- **CPU Implementation**: pymoo 0.6.* with NumPy backend
- **GPU Implementation**: PyTorch 2.5.0 with CUDA support, using custom NSGA-III implementation
- **Container Base Images**:
  - CPU: `python:3.11-slim`
  - GPU: `nvcr.io/nvidia/pytorch:24.10-py3`

### MOEA Configuration
- **Algorithm**: NSGA-III (Non-dominated Sorting Genetic Algorithm III)
- **Population Size**: 100 individuals
- **Generations**: 200
- **Total Evaluations**: 20,000 (100 × 200)
- **Objectives**: 4 (multi-objective greenhouse optimization)
- **Decision Variables**: 6

## Performance Results

### Execution Time Comparison

| Implementation | Total Time | Evaluations/sec | Speed Factor |
|----------------|------------|-----------------|--------------|
| CPU (pymoo)    | 2.17s      | ~9,217          | 1.0x (baseline) |
| GPU (PyTorch)  | 0.11s      | ~181,818        | **19.73x faster** |

### Key Performance Metrics

1. **Absolute Performance**:
   - CPU processes approximately 9,217 evaluations per second
   - GPU processes approximately 181,818 evaluations per second

2. **Speedup Factor**: The GPU implementation is **19.73x faster** than the CPU implementation

3. **Efficiency Gains**:
   - Time saved per run: 2.06 seconds (94.9% reduction)
   - For 1,000 optimization runs: ~34 minutes saved
   - For 10,000 optimization runs: ~5.7 hours saved

## Memory Configuration

### Shared Memory Settings
The GPU implementation uses optimized memory settings to prevent PyTorch warnings and improve performance:

```yaml
moea_optimizer_gpu:
  ipc: host
  ulimits:
    memlock:
      soft: -1
      hard: -1
    stack:
      soft: 67108864
      hard: 67108864
  shm_size: '8gb'
```

### Memory Type Clarification
- **Shared Memory (`/dev/shm`)**: System RAM used for inter-process communication
- **VRAM**: GPU's dedicated memory (12GB on RTX 4070) for tensor operations
- The 8GB shared memory allocation does not affect VRAM usage

## Implementation Details

### CPU Implementation (pymoo)
- Uses NumPy for numerical operations
- Single-threaded evaluation by default
- Memory-efficient but computationally intensive
- Well-established library with proven algorithms

### GPU Implementation (PyTorch)
- Tensor-based operations on CUDA cores
- Parallel evaluation of entire population
- Custom NSGA-III implementation with fallback
- Leverages GPU's parallel processing capabilities

## Scalability Considerations

### When to Use CPU
- Small population sizes (< 50 individuals)
- Limited GPU memory
- When reproducibility with established libraries is critical
- Development and debugging

### When to Use GPU
- Large population sizes (100+ individuals)
- Many generations required
- Real-time optimization needs
- Batch processing multiple optimization runs

## Future Optimization Opportunities

1. **Enhanced GPU Utilization**:
   - Implement proper evox NSGA3 integration
   - Use mixed precision (FP16) for larger populations
   - Batch multiple optimization runs on GPU

2. **Memory Optimization**:
   - Dynamic shared memory allocation based on population size
   - GPU memory pooling for multiple concurrent runs

3. **Algorithm Enhancements**:
   - GPU-accelerated fitness evaluation
   - Parallel constraint handling
   - Multi-GPU support for extremely large populations

## Conclusion

The GPU implementation provides a dramatic **19.73x speedup** over the CPU implementation for MOEA optimization. This performance improvement makes it feasible to:
- Run more comprehensive optimization studies
- Use larger population sizes for better solution diversity
- Perform real-time optimization in production environments
- Significantly reduce computational time for research iterations

The investment in GPU acceleration for evolutionary algorithms demonstrates clear benefits, particularly for computationally intensive multi-objective optimization problems in the greenhouse climate control domain.

## Appendix: Docker Configuration

The optimized Docker Compose configuration for GPU support:

```yaml
moea_optimizer_gpu:
  ipc: host
  ulimits:
    memlock:
      soft: -1
      hard: -1
    stack:
      soft: 67108864
      hard: 67108864
  shm_size: '8gb'
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

This configuration ensures optimal memory allocation and eliminates PyTorch shared memory warnings while maximizing GPU performance.