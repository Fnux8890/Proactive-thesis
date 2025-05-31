use cudarc::driver::safe::{CudaContext, CudaStream};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create context
    let ctx = CudaContext::new(0)?;
    
    // Get device name
    println!("Device: {:?}", ctx.device_name()?);
    
    // Create stream
    let stream = ctx.new_stream()?;
    
    // Test memory allocation
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    
    // Check methods available on context
    println!("Testing memory operations...");
    
    // The API might be:
    // let device_buffer = ctx.alloc(data.len())?;
    // ctx.htod_copy(&data, &device_buffer)?;
    
    Ok(())
}