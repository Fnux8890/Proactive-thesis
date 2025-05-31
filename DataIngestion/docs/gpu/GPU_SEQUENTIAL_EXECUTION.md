# GPU Sequential Execution Strategy

## Overview

Based on research and testing, we've implemented **sequential execution** for GPU feature extraction to prevent memory conflicts and segmentation faults when processing large datasets.

## Why Sequential Instead of Parallel?

### Research Findings:
1. **Memory Conflicts**: Each CUDA container uses ~600MB base GPU memory, making parallel execution risky
2. **No Native GPU Sharing**: Docker doesn't natively support GPU sharing like CPU/memory resources
3. **MPS Limitations**: NVIDIA MPS has compatibility issues with Docker and pre-Volta GPUs

### Our Dataset Scale:
- **Level A**: 198,109 eras
- **Level B**: 278,472 eras (but only ~15 with MIN_ERA_ROWS=500)
- **Level C**: 850,710 eras (!)

## Implementation

### Sequential Dependencies in docker-compose.yml:

```yaml
gpu_feature_extraction_level_b:
  depends_on:
    gpu_feature_extraction_level_a:
      condition: service_completed_successfully

gpu_feature_extraction_level_c:
  depends_on:
    gpu_feature_extraction_level_b:
      condition: service_completed_successfully
```

### Key Optimizations:

1. **Filtered Processing**: 
   - MIN_ERA_ROWS filters out small/invalid eras
   - Level C: 5000 rows minimum (reduces 850K to manageable number)

2. **Batch Size Tuning**:
   - Level A: 500 (larger eras, smaller batches)
   - Level B: 1000 (medium eras)
   - Level C: 100 (many small eras, tiny batches)

3. **Safety Features**:
   - Grid dimension capping (max 65535)
   - Shared memory limits (48KB)
   - Empty array checks

## Usage

### Local Development:
```bash
docker compose up gpu_feature_extraction_level_a
# This automatically triggers B after A, and C after B
```

### Production:
```bash
docker compose -f docker-compose.yml -f docker-compose.production.yml up gpu_feature_extraction_level_a
```

### Environment Variables:
```env
# Adjust these based on your GPU memory
MIN_ERA_ROWS_A=1000
MIN_ERA_ROWS_B=500
MIN_ERA_ROWS_C=5000  # Critical for Level C!

BATCH_SIZE_A=500
BATCH_SIZE_B=1000
BATCH_SIZE_C=100
```

## Performance

Sequential execution provides:
- **Predictable memory usage**: Each level gets full GPU access
- **No segmentation faults**: Prevents exit code 139 errors
- **Better throughput**: ~1000+ eras/second when not competing

## Alternative Approaches (Not Recommended)

1. **MPS (Multi-Process Service)**: Requires host-level configuration, security risks
2. **GPU Partitioning**: Complex setup, not supported in Docker Compose
3. **Parallel with Memory Limits**: Still causes conflicts due to CUDA context

## Monitoring

Watch GPU usage during execution:
```bash
nvidia-smi -l 1  # Updates every second
```

Check container logs:
```bash
docker compose logs -f gpu_feature_extraction_level_a
```

## Troubleshooting

If you still get segmentation faults:
1. Increase MIN_ERA_ROWS to filter more
2. Decrease BATCH_SIZE
3. Check GPU memory with `nvidia-smi`
4. Ensure only one container uses GPU at a time