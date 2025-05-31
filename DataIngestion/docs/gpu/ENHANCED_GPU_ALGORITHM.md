# Enhanced GPU Algorithm with Extended Shared Memory Usage

## Current vs Enhanced Algorithm

### Current Implementation (96 bytes)
```cuda
// Only computing basic statistics per warp
__shared__ float shared[num_warps * 3];  // sum, min, max
```

### Enhanced Implementation (up to 48KB)
We can extend the algorithm to compute more statistics in parallel:

```cuda
// Extended statistics per warp
struct WarpStats {
    float sum;           // 4 bytes
    float sum_sq;        // 4 bytes (for variance)
    float sum_cube;      // 4 bytes (for skewness)
    float sum_quad;      // 4 bytes (for kurtosis)
    float min;           // 4 bytes
    float max;           // 4 bytes
    float count;         // 4 bytes
    int min_idx;         // 4 bytes (track position)
    int max_idx;         // 4 bytes
    float percentile_25; // 4 bytes
    float percentile_75; // 4 bytes
    float median;        // 4 bytes
};  // Total: 48 bytes per warp

// For 256 threads (8 warps): 8 * 48 = 384 bytes
// Still only 0.4% of available shared memory on RTX 4070!
```

## Benefits of Extended Algorithm

1. **Single-Pass Computation**: Calculate variance, skewness, and kurtosis in one pass
2. **Percentile Tracking**: Approximate percentiles using shared memory buffers
3. **Position Tracking**: Know where min/max values occur in the time series
4. **Better Memory Coalescing**: Process multiple statistics simultaneously

## Implementation Changes

### 1. Update features.rs
```rust
// Calculate extended shared memory size
let warp_stats_size = 48; // bytes per warp for extended stats
let num_warps = (block_size + 31) / 32;
let shared_mem_bytes = (num_warps * warp_stats_size) as u32; // 384 bytes for 256 threads

// Optional: Add histogram bins for better percentile estimation
let histogram_bins = 256;
let histogram_size = histogram_bins * std::mem::size_of::<u32>() as u32; // 1KB
let total_shared_mem = shared_mem_bytes + histogram_size; // 1.384 KB total
```

### 2. Enhanced Kernel Features
```cuda
__global__ void compute_statistics_extended(
    const float* __restrict__ input,
    StatisticalFeaturesExtended* __restrict__ output,
    const unsigned int n
) {
    extern __shared__ char shared_mem[];
    WarpStats* warp_stats = (WarpStats*)shared_mem;
    unsigned int* histogram = (unsigned int*)&shared_mem[num_warps * sizeof(WarpStats)];
    
    // Initialize histogram in shared memory
    if (threadIdx.x < 256) {
        histogram[threadIdx.x] = 0;
    }
    __syncthreads();
    
    // Compute extended statistics with histogram
    // ... implementation ...
}
```

## Memory Usage Comparison

| GPU | Max Shared Memory/Block | Current Usage | Enhanced Usage | Utilization |
|-----|------------------------|---------------|----------------|-------------|
| RTX 4070 | 99 KB | 96 B (0.1%) | 1.4 KB (1.4%) | Still minimal |
| A100 | 163 KB | 96 B (0.06%) | 1.4 KB (0.9%) | Plenty of headroom |

## Advanced Features We Could Add

### 1. Sliding Window Cache (10-20 KB)
Cache recent values for temporal features:
- Autocorrelation computation
- Trend detection
- Change point detection

### 2. FFT Workspace (32-48 KB)
Shared memory for frequency domain analysis:
- Power spectral density
- Dominant frequencies
- Spectral entropy

### 3. Multi-Resolution Statistics (5-10 KB)
Compute statistics at multiple time scales:
- Hourly aggregates
- Daily patterns
- Weekly trends

## Era Detection Issue

The current era detection created too many eras:
- Level A: 437,254 eras (avg 9,107 hours but many are minutes)
- Level B: 278,472 eras
- Level C: 1,168,688 eras

This suggests the parameters need adjustment:
```yaml
# Recommended new parameters
"--pelt-min-size", "1440",     # 5 days minimum (1440 * 5min samples)
"--bocpd-lambda", "500.0",      # Much less sensitive
"--hmm-states", "3"             # Fewer states
```