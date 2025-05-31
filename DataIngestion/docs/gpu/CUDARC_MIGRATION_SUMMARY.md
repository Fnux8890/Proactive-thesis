# CUDARC 0.16.4 Migration Summary

This document summarizes the changes made to migrate the GPU feature extraction code from an older cudarc version to 0.16.4.

## Key API Changes

### 1. Import Changes
- `CudaDevice` is now accessed through `cudarc::driver::safe::CudaContext`
- Removed `DevicePtr`, replaced with `CudaSlice<T>`
- Removed `LaunchAsync` trait
- Added `PushKernelArg` trait for kernel argument handling

### 2. Context and Device Management
```rust
// Old
let device = CudaDevice::new(0)?;
let stream = device.fork_default_stream()?;

// New
let ctx = CudaContext::new(0)?;  // Returns Arc<CudaContext>
let stream = ctx.default_stream();
```

### 3. Memory Operations
Memory operations are now performed on streams, not contexts or devices:

```rust
// Old
let d_data = device.htod_sync_copy(data)?;
let d_output = device.alloc_zeros::<T>(size)?;
device.dtoh_sync_copy(&d_data)?;

// New
let d_data = stream.memcpy_stod(data)?;
let d_output = stream.alloc_zeros::<T>(size)?;
let result = stream.memcpy_dtov(&d_data)?;
```

### 4. PTX Module Loading
```rust
// Old
let module = device.load_ptx(ptx, "module_name", &["func1", "func2"])?;

// New
let module = ctx.load_module(ptx)?;  // Returns Arc<CudaModule>
```

### 5. Kernel Function Loading and Launching
```rust
// Old
let func = module.get_function("kernel_name")?;
func.launch(config, (arg1, arg2, arg3))?;

// New
let func = module.load_function("kernel_name")?;
let mut builder = stream.launch_builder(&func);
builder.arg(&arg1);
builder.arg(&arg2);
builder.arg(&arg3);
unsafe { builder.launch(config)? };
```

### 6. Type Safety Improvements
- Added `ValidAsZeroBits` trait requirement for types used with `alloc_zeros`
- `DeviceRepr` trait is now in `cudarc::driver::safe` module

### 7. Configuration Changes
- `CompileOptions` no longer has a `definitions` field
- Clap 4.0 requires the "env" feature for environment variable support

## Code Structure Changes

### Features.rs
- Updated to use stream-based memory operations
- Changed from context/device methods to stream methods
- Fixed shared memory size to be u32 instead of usize

### Kernels.rs
- Updated module storage to use `Arc<CudaModule>`
- Changed kernel launching to use launch builders
- Added `PushKernelArg` import for argument handling

### Main.rs
- Removed double Arc wrapping of CudaContext
- Updated to use new context/stream initialization

### Database Types
- Changed `era_level` from `char` to `String` for sqlx compatibility
- Added `sqlx::FromRow` derive for database structs

## Compilation Notes
- All GPU operations are now more type-safe
- Memory operations are stream-centric
- Kernel launches require explicit unsafe blocks
- The API is more consistent with modern Rust patterns