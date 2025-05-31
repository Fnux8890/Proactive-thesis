use cudarc::driver::safe::{CudaContext};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create context
    let ctx = CudaContext::new(0)?;
    
    // Test data
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let size = data.len();
    
    // Try different allocation methods
    // Option 1: Direct allocation with type
    // let d_buffer = ctx.alloc::<f32>(size)?;
    
    // Option 2: Using unsafe allocation  
    // let d_buffer = unsafe { ctx.alloc::<f32>(size)? };
    
    // Option 3: Allocation with zeros (if method exists)
    // let d_buffer = ctx.alloc_zeros::<f32>(size)?;
    
    // Option 4: Transfer and allocate at once
    // let d_buffer = ctx.htod_copy(&data)?;
    
    // Option 5: Check if there's a Device type
    // let device = ctx.device();
    
    // Print to see what compiles
    println!("Testing cudarc 0.16.4 API");
    
    Ok(())
}