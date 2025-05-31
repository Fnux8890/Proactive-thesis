# cudarc 0.16.4 API Fix Guide

## The Problem
The code is trying to use methods that don't exist on `Arc<CudaContext>`. The API has changed significantly from older versions.

## Correct API Usage in cudarc 0.16.4

### 1. Context and Stream Creation
```rust
use cudarc::driver::{CudaContext, CudaStream, CudaSlice, LaunchConfig};

// Create context (returns Arc<CudaContext>)
let ctx = CudaContext::new(0)?;  // 0 is the device ordinal

// Create stream
let stream = ctx.new_stream()?;  // Returns Arc<CudaStream>
```

### 2. Memory Allocation
There's no `alloc_zeros` method. You need to:
```rust
// Option 1: Allocate uninitialized memory and then fill with zeros
let mut d_data: CudaSlice<f32> = unsafe { stream.alloc_async(size)? };

// Option 2: Allocate and copy from a zero-filled host vector
let zeros = vec![0.0f32; size];
let d_data = stream.htod_copy(zeros)?;  // This transfers ownership

// Option 3: For sync allocation
let mut d_data: CudaSlice<f32> = ctx.alloc_zeros::<f32>(size)?; // If this method exists
```

### 3. Memory Transfer Operations

#### Host to Device (htod)
```rust
// Synchronous copy (doesn't transfer ownership)
let host_data = vec![1.0, 2.0, 3.0, 4.0];
let mut d_data: CudaSlice<f32> = unsafe { stream.alloc_async(host_data.len())? };
stream.htod_sync(&d_data, &host_data)?;

// Or using htod_copy (transfers ownership)
let d_data = stream.htod_copy(host_data)?;
```

#### Device to Host (dtoh)
```rust
// Synchronous copy
let mut host_result = vec![0.0f32; size];
stream.dtoh_sync(&mut host_result, &d_data)?;
```

### 4. Key Differences from Old API
- Memory operations are on `CudaStream`, not `CudaContext`
- No direct `alloc_zeros` method on streams
- Parameter order might be (destination, source) for some operations
- Most operations return `Arc<T>` wrapped types

### 5. Fixing the GpuFeatureExtractor

The main issues to fix:
1. Change from `ctx.alloc_zeros` to proper allocation method
2. Change from `ctx.htod_sync_copy_into` to `stream.htod_sync`
3. Change from `ctx.dtoh_sync_copy_into` to `stream.dtoh_sync`
4. Ensure proper parameter ordering

### Example Fix
```rust
// Old (incorrect)
let mut d_input = self.ctx.alloc_zeros::<f32>(values.len())?;
self.ctx.htod_sync_copy_into(values, &mut d_input)?;

// New (correct)
let zeros = vec![0.0f32; values.len()];
let mut d_input = self.stream.htod_copy(zeros)?;
self.stream.htod_sync(&d_input, values)?;

// Or simpler:
let d_input = self.stream.htod_copy(values.to_vec())?;
```

## Alternative Approach
If the context does have `alloc_zeros`, it might be used like:
```rust
// The context might have these methods, not the stream
let mut d_data = self.ctx.alloc_zeros::<f32>(size)?;
self.ctx.htod_sync_copy_into(&host_data, &mut d_data)?;
let mut result = vec![0.0f32; size];
self.ctx.dtoh_sync_copy_into(&d_data, &mut result)?;
```

## To Verify the Correct API
1. Check `cargo doc --open` for cudarc
2. Look at working examples in the cudarc repository
3. Use rust-analyzer to see available methods on each type