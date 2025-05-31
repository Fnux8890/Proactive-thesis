// Test program to find correct cudarc 0.16.4 imports
use cudarc::driver::safe::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing cudarc 0.16.4 imports");
    
    // Try to create a device
    let device = CudaDevice::new(0)?;
    println!("Device created successfully");
    
    Ok(())
}