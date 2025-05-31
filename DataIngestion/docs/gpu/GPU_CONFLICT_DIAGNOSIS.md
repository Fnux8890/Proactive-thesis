# GPU Conflict Diagnosis

## Root Causes Identified

### 1. **All 3 containers using same GPU device**
```yaml
# All three services have:
NVIDIA_VISIBLE_DEVICES: all
CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-0}  # Defaults to device 0
```

This causes:
- Memory allocation conflicts
- CUDA context conflicts  
- Segmentation faults (exit 139)

### 2. **Level C has 850,710 eras!**
- That's 4x more than Level A (198,109)
- 43x more than Level B (19,847 that B actually processes)
- Massive memory requirements

### 3. **SQLx slow queries (>1 second)**
- Fetching 1.4M+ rows per query
- No pagination or chunking

## Why Level B Works
- Level B only processes 15 eras (with MIN_ERA_ROWS=500)
- Much smaller memory footprint
- Completes before conflicts arise

## Solutions

### Option 1: Run services sequentially (Quick fix)
Instead of running all three at once, run them one at a time.

### Option 2: GPU isolation (Better)
Assign different GPU indices if you have multiple GPUs, or use MPS for sharing.

### Option 3: Reduce Level C processing (Best for now)
Level C's 850K eras is unrealistic - likely needs filtering.

### Option 4: Implement proper batching
Process eras in smaller chunks to avoid memory exhaustion.