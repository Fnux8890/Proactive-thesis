use cudarc::driver::safe::{CudaContext, CudaDevice};
use cudarc::driver::CudaSlice;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create device first
    let device = CudaDevice::new(0)?;
    
    // Test data
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    
    // Try device-based allocation
    let d_buffer = device.alloc_zeros::<f32>(data.len())?;
    
    // Copy data
    device.htod_sync_copy_into(&data, &mut d_buffer)?;
    
    // Copy back
    let mut result = vec![0.0f32; data.len()];
    device.dtoh_sync_copy_into(&d_buffer, &mut result)?;
    
    println!("Result: {:?}", result);
    
    Ok(())
}