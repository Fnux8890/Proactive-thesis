# cudarc 0.16.4 API Research Needed

## The Problem

We need to understand how to do basic GPU memory operations in cudarc 0.16.4. The API has changed significantly from older versions, and the methods we're trying to use don't exist on `Arc<CudaContext>`.

## What We're Trying to Do

Here are the specific operations we need to perform, with the code that's currently failing:

### 1. Allocate GPU Memory with Zeros

**What we're trying (FAILS):**
```rust
let mut d_input = self.ctx.alloc_zeros::<f32>(values.len())?;
```

**Error:**
```
error[E0599]: no method named `alloc_zeros` found for struct `Arc<CudaContext>`
```

**What we need to know:**
- How do we allocate GPU memory filled with zeros in cudarc 0.16.4?
- Is there an `alloc_zeros` method somewhere else?
- Do we need to allocate uninitialized memory and then fill it?

### 2. Copy Data from Host to Device

**What we're trying (FAILS):**
```rust
let data = vec![1.0f32, 2.0, 3.0, 4.0];
let mut d_buffer = self.ctx.alloc_zeros::<f32>(data.len())?;
self.ctx.htod_sync_copy_into(&data, &mut d_buffer)?;
```

**Error:**
```
error[E0599]: no method named `htod_sync_copy_into` found for struct `Arc<CudaContext>`
```

**What we need to know:**
- How do we copy a Vec<f32> from host memory to GPU memory?
- Is the method on a different object (stream, device)?
- What's the correct method name and signature?

### 3. Copy Data from Device to Host

**What we're trying (FAILS):**
```rust
let mut results = vec![0.0f32; output_size];
self.ctx.dtoh_sync_copy_into(&d_output, &mut results)?;
```

**Error:**
```
error[E0599]: no method named `dtoh_sync_copy_into` found for struct `Arc<CudaContext>`
```

**What we need to know:**
- How do we copy data from GPU back to a host Vec?
- What's the correct method for device-to-host transfer?

## Complete Example of What We Need

Here's a minimal example of what we're trying to accomplish:

```rust
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

fn gpu_computation_example() -> Result<(), Box<dyn std::error::Error>> {
    // Step 1: Initialize CUDA
    let ctx = CudaContext::new(0)?;  // This works
    let stream = ctx.default_stream();  // This works
    
    // Step 2: Prepare host data
    let host_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let n = host_data.len();
    
    // Step 3: Allocate GPU memory
    // QUESTION: How do we do this in cudarc 0.16.4?
    // let mut d_input = ?????;  // Need to allocate n floats on GPU
    
    // Step 4: Copy data to GPU
    // QUESTION: How do we do this in cudarc 0.16.4?
    // ?????(host_data, d_input)?;  // Need to copy host_data to d_input
    
    // Step 5: Allocate output buffer on GPU
    // QUESTION: How do we allocate output buffer?
    // let mut d_output = ?????;  // Need to allocate n floats for output
    
    // Step 6: Launch kernel (we know how to do this part)
    // ... kernel launch code ...
    
    // Step 7: Copy results back to host
    // QUESTION: How do we copy back?
    // let mut results = vec![0.0f32; n];
    // ?????(d_output, results)?;  // Need to copy d_output to results
    
    Ok(())
}
```

## What to Look For

When researching cudarc 0.16.4, please look for:

1. **Memory Allocation Methods**
   - Where is `alloc` or `alloc_zeros` defined?
   - Is it on `CudaContext`, `CudaStream`, or some other type?
   - What's the return type? (CudaSlice<T>, CudaBuffer<T>, etc.)

2. **Memory Transfer Methods**
   - What are the methods for host-to-device transfer?
   - What are the methods for device-to-host transfer?
   - Are they synchronous or asynchronous?
   - Are they on the context, stream, or the buffer itself?

3. **Memory Management Pattern**
   - Do we need to explicitly free memory?
   - Are buffers automatically freed when dropped?
   - Is there a special pattern for memory lifetime?

## Where to Look

1. **cudarc GitHub Repository**
   - Look for examples in the repository
   - Check the tests directory for usage patterns
   - Look at the CHANGELOG for API changes

2. **cudarc Documentation**
   - Check docs.rs for cudarc 0.16.4
   - Look for migration guides from older versions

3. **Example Code**
   - Search for projects using cudarc 0.16.4
   - Look for memory allocation and transfer examples

## Specific Questions

1. In cudarc 0.16.4, what type has the `alloc_zeros` method?
2. What's the correct way to copy a `Vec<f32>` to GPU memory?
3. What's the correct way to copy GPU memory back to a `Vec<f32>`?
4. Are memory operations on `Arc<CudaContext>`, `Arc<CudaStream>`, or some other type?
5. What's the type of GPU memory buffers? (CudaSlice, CudaBuffer, etc.)

## Example Search Queries

- "cudarc 0.16.4 memory allocation"
- "cudarc htod_sync_copy_into replacement"
- "cudarc 0.16 breaking changes"
- "cudarc CudaContext alloc_zeros"
- "cudarc memory transfer example"

Once we understand these basic operations, we can connect all the GPU kernels that are already written and get full GPU acceleration working.