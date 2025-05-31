# GPU Feature Extraction Refactoring for cudarc 0.16.4

## Overview

This document outlines the refactoring needed to update the GPU feature extraction code from cudarc 0.10 to 0.16.4.

## Key API Changes in cudarc 0.16

Based on the compilation errors and cudarc changelog:

1. **Device Management**
   - `CudaDevice` is now in the crate root: `use cudarc::CudaDevice;`
   - Device creation might use a builder pattern

2. **Memory Management**
   - `DevicePtr<T>` → `CudaSlice<T>` for device memory
   - New memory allocation APIs

3. **Kernel Launch**
   - `LaunchAsync` trait removed
   - New kernel launch mechanism
   - `LaunchConfig` might have different fields

4. **Module Loading**
   - PTX compilation API updated
   - Module loading simplified

## Refactoring Steps

### Step 1: Update Imports

```rust
// Old (0.10)
use cudarc::driver::{CudaDevice, CudaStream, DevicePtr, LaunchAsync, LaunchConfig};

// New (0.16)
use cudarc::{CudaDevice, CudaStream};
use cudarc::driver::{CudaSlice, LaunchConfig};
```

### Step 2: Update Device Creation

```rust
// Old (0.10)
let device = Arc::new(CudaDevice::new(0)?);

// New (0.16) - might need adjustment
let device = Arc::new(CudaDevice::new(0)?);
```

### Step 3: Update Memory Types

```rust
// Old (0.10)
fn launch_kernel(input: &DevicePtr<f32>, output: &DevicePtr<f32>)

// New (0.16)
fn launch_kernel(input: &CudaSlice<f32>, output: &CudaSlice<f32>)
```

### Step 4: Update Kernel Launch

The kernel launch mechanism has changed significantly. We need to:
1. Remove `LaunchAsync` trait usage
2. Update launch syntax
3. Adjust parameter passing

### Step 5: Fix Compile Options

```rust
// Old (0.10)
CompileOptions {
    arch: Some("sm_70"),
    include_paths: vec![],
    definitions: vec![], // This field no longer exists
    ..Default::default()
}

// New (0.16)
CompileOptions {
    arch: Some("sm_70"),
    include_paths: vec![],
    ..Default::default()
}
```

## Implementation Plan

Since this is a significant refactoring effort and the Docker build is the primary deployment method, we recommend:

1. **Short-term**: Use Docker build which handles all dependencies
2. **Medium-term**: Create a feature branch for the cudarc 0.16 migration
3. **Long-term**: Fully migrate to cudarc 0.16 with proper testing

## Docker Build (Immediate Solution)

```bash
cd DataIngestion
docker compose build gpu_feature_extraction
docker compose up gpu_feature_extraction
```

## Next Steps

1. Study cudarc 0.16 examples and documentation
2. Create a minimal working example with 0.16
3. Incrementally update each module
4. Test thoroughly with GPU hardware

## Resources

- cudarc GitHub: https://github.com/coreylowman/cudarc
- cudarc docs: https://docs.rs/cudarc/0.16.4
- Migration examples: Check the cudarc repository for examples/