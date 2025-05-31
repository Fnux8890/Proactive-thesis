// Test cudarc 0.16.4 API
use cudarc::driver::{CudaContext, CudaDevice, CudaSlice, CudaStream, LaunchArgs, LaunchConfig};
use cudarc::nvrtc::{compile_ptx};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA
    cudarc::driver::init()?;
    
    // Get device
    let dev = CudaDevice::new(0)?;
    println!("Device: {:?}", dev);
    
    // Create context
    let ctx = CudaContext::new(dev)?;
    
    // Create stream
    let stream = ctx.stream()?;
    
    // Test memory allocation
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let mut device_data = ctx.alloc_zeros::<f32>(data.len())?;
    ctx.htod_sync_copy_into(&data, &mut device_data)?;
    
    // Test kernel compilation
    let kernel = r#"
extern "C" __global__ void test_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}
"#;
    
    let ptx = compile_ptx(kernel)?;
    let module = ctx.load_ptx(ptx, "test", &["test_kernel"])?;
    let func = module.get_fn("test_kernel")?;
    
    // Test launch config
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    
    // Launch kernel
    let args = LaunchArgs::builder()
        .push(&device_data)
        .push(4i32)
        .config(cfg)
        .build();
    
    stream.launch(&func, args)?;
    stream.synchronize()?;
    
    // Copy back
    let mut result = vec![0.0f32; 4];
    ctx.dtoh_sync_copy_into(&device_data, &mut result)?;
    
    println!("Result: {:?}", result);
    println!("Test successful!");
    Ok(())
}