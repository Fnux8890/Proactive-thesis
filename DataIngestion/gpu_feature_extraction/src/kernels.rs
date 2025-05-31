use anyhow::Result;
use cudarc::driver::safe::{CudaContext, CudaModule, CudaStream, CudaSlice, LaunchConfig};
use cudarc::driver::PushKernelArg;
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::collections::HashMap;
use std::sync::Arc;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct StatisticalFeatures {
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
    pub skewness: f32,
    pub kurtosis: f32,
}

// Ensure StatisticalFeatures is safe for GPU operations
unsafe impl cudarc::driver::safe::DeviceRepr for StatisticalFeatures {}
unsafe impl cudarc::driver::ValidAsZeroBits for StatisticalFeatures {}

// Compile-time check: ensure struct size matches between host and device
const _: () = assert!(std::mem::size_of::<StatisticalFeatures>() == 24, "StatisticalFeatures size mismatch");

mod thermal_time;
mod psychrometric;
mod actuator_dynamics;
mod frequency_domain;
mod temporal_dependencies;
mod entropy_complexity;
mod wavelet_features;
mod economic_features;
mod environment_coupling;
mod rolling_statistics_extended;
mod stress_counters;

pub struct KernelManager {
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    modules: HashMap<String, Arc<CudaModule>>,
}

impl KernelManager {
    pub fn new(ctx: Arc<CudaContext>) -> Result<Self> {
        let mut modules = HashMap::new();
        
        // Compile and load statistical kernel
        let statistical_ptx = compile_statistical_kernel()?;
        let statistical_module = ctx.load_module(statistical_ptx)?;
        modules.insert("statistical".to_string(), statistical_module);
        
        // Compile and load thermal time kernels
        let thermal_ptx = compile_kernel_source(thermal_time::thermal_time_kernels())?;
        let thermal_module = ctx.load_module(thermal_ptx)?;
        modules.insert("thermal".to_string(), thermal_module);
        
        // Compile and load psychrometric kernels
        let psychro_ptx = compile_kernel_source(psychrometric::psychrometric_kernels())?;
        let psychro_module = ctx.load_module(psychro_ptx)?;
        modules.insert("psychrometric".to_string(), psychro_module);
        
        // Compile and load actuator dynamics kernels
        let actuator_ptx = compile_kernel_source(actuator_dynamics::actuator_dynamics_kernels())?;
        let actuator_module = ctx.load_module(actuator_ptx)?;
        modules.insert("actuator".to_string(), actuator_module);
        
        // Compile and load frequency domain kernels
        // TODO: Fix FFT kernel compilation - cufft.h not available in NVRTC
        // let freq_ptx = compile_kernel_source(frequency_domain::frequency_domain_kernels())?;
        // let freq_module = ctx.load_module(freq_ptx)?;
        // modules.insert("frequency".to_string(), freq_module);
        
        // Compile and load temporal dependency kernels
        let temporal_ptx = compile_kernel_source(temporal_dependencies::temporal_dependency_kernels())?;
        let temporal_module = ctx.load_module(temporal_ptx)?;
        modules.insert("temporal".to_string(), temporal_module);
        
        // Compile and load entropy/complexity kernels
        let entropy_ptx = compile_kernel_source(entropy_complexity::entropy_complexity_kernels())?;
        let entropy_module = ctx.load_module(entropy_ptx)?;
        modules.insert("entropy".to_string(), entropy_module);
        
        // Compile and load wavelet kernels
        let wavelet_ptx = compile_kernel_source(wavelet_features::wavelet_kernels())?;
        let wavelet_module = ctx.load_module(wavelet_ptx)?;
        modules.insert("wavelet".to_string(), wavelet_module);
        
        // Compile and load economic kernels
        let economic_ptx = compile_kernel_source(economic_features::economic_kernels())?;
        let economic_module = ctx.load_module(economic_ptx)?;
        modules.insert("economic".to_string(), economic_module);
        
        // Compile and load environment coupling kernels
        let env_coupling_ptx = compile_kernel_source(environment_coupling::environment_coupling_kernels())?;
        let env_coupling_module = ctx.load_module(env_coupling_ptx)?;
        modules.insert("environment_coupling".to_string(), env_coupling_module);
        
        // Compile and load extended rolling statistics kernels
        let rolling_ext_ptx = compile_kernel_source(rolling_statistics_extended::rolling_statistics_extended_kernels())?;
        let rolling_ext_module = ctx.load_module(rolling_ext_ptx)?;
        modules.insert("rolling_extended".to_string(), rolling_ext_module);
        
        // Compile and load stress counter kernels
        let stress_ptx = compile_kernel_source(stress_counters::stress_counter_kernels())?;
        let stress_module = ctx.load_module(stress_ptx)?;
        modules.insert("stress_counters".to_string(), stress_module);
        
        Ok(Self { ctx, modules })
    }
    
    pub fn launch_statistical_kernel(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        input: &CudaSlice<f32>,
        output: &CudaSlice<StatisticalFeatures>,
        n: u32,
    ) -> Result<()> {
        let module = self.modules.get("statistical")
            .ok_or_else(|| anyhow::anyhow!("Statistical module not loaded"))?;
        
        let func = module.load_function("compute_statistics")?;
        
        let mut builder = stream.launch_builder(&func);
        builder.arg(input);
        builder.arg(output);
        builder.arg(&n);
        
        unsafe { builder.launch(config)? };
        stream.synchronize()?;
        Ok(())
    }
    
    pub fn launch_rolling_stats_kernel(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        input: &CudaSlice<f32>,
        means: &CudaSlice<f32>,
        stds: &CudaSlice<f32>,
        n: u32,
        window_size: u32,
    ) -> Result<()> {
        let module = self.modules.get("statistical")
            .ok_or_else(|| anyhow::anyhow!("Statistical module not loaded"))?;
        
        let func = module.load_function("rolling_stats")?;
        
        let mut builder = stream.launch_builder(&func);
        builder.arg(input);
        builder.arg(means);
        builder.arg(stds);
        builder.arg(&n);
        builder.arg(&window_size);
        
        unsafe { builder.launch(config)? };
        stream.synchronize()?;
        Ok(())
    }
    
    pub fn launch_vpd_kernel(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        temp: &CudaSlice<f32>,
        rh: &CudaSlice<f32>,
        vpd: &CudaSlice<f32>,
        n: u32,
    ) -> Result<()> {
        let module = self.modules.get("statistical")
            .ok_or_else(|| anyhow::anyhow!("Statistical module not loaded"))?;
        
        let func = module.load_function("compute_vpd")?;
        
        let mut builder = stream.launch_builder(&func);
        builder.arg(temp);
        builder.arg(rh);
        builder.arg(vpd);
        builder.arg(&n);
        
        unsafe { builder.launch(config)? };
        stream.synchronize()?;
        Ok(())
    }
    
    // Extended rolling statistics kernels
    pub fn launch_rolling_percentiles_kernel(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        input: &CudaSlice<f32>,
        percentiles: &CudaSlice<f32>,
        output: &CudaSlice<f32>,
        n: u32,
        window_size: u32,
        num_percentiles: u32,
    ) -> Result<()> {
        let module = self.modules.get("rolling_extended")
            .ok_or_else(|| anyhow::anyhow!("Rolling extended module not loaded"))?;
        
        let func = module.load_function("rolling_percentiles")?;
        
        let mut builder = stream.launch_builder(&func);
        builder.arg(input);
        builder.arg(percentiles);
        builder.arg(output);
        builder.arg(&n);
        builder.arg(&window_size);
        builder.arg(&num_percentiles);
        
        unsafe { builder.launch(config)? };
        stream.synchronize()?;
        Ok(())
    }
    
    // Temporal dependency kernels
    pub fn launch_acf_kernel(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        input: &CudaSlice<f32>,
        output: &CudaSlice<f32>,
        n: u32,
        max_lag: u32,
    ) -> Result<()> {
        let module = self.modules.get("temporal")
            .ok_or_else(|| anyhow::anyhow!("Temporal module not loaded"))?;
        
        let func = module.load_function("compute_acf")?;
        
        let mut builder = stream.launch_builder(&func);
        builder.arg(input);
        builder.arg(output);
        builder.arg(&n);
        builder.arg(&max_lag);
        
        unsafe { builder.launch(config)? };
        stream.synchronize()?;
        Ok(())
    }
    
    // Entropy kernels
    pub fn launch_shannon_entropy_kernel(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        input: &CudaSlice<f32>,
        output: &CudaSlice<f32>,
        n: u32,
        num_bins: u32,
    ) -> Result<()> {
        let module = self.modules.get("entropy")
            .ok_or_else(|| anyhow::anyhow!("Entropy module not loaded"))?;
        
        let func = module.load_function("shannon_entropy")?;
        
        let mut builder = stream.launch_builder(&func);
        builder.arg(input);
        builder.arg(output);
        builder.arg(&n);
        builder.arg(&num_bins);
        
        unsafe { builder.launch(config)? };
        stream.synchronize()?;
        Ok(())
    }
    
    // Wavelet kernels
    pub fn launch_dwt_decomposition_kernel(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        input: &CudaSlice<f32>,
        approx: &CudaSlice<f32>,
        detail: &CudaSlice<f32>,
        n: u32,
    ) -> Result<()> {
        let module = self.modules.get("wavelet")
            .ok_or_else(|| anyhow::anyhow!("Wavelet module not loaded"))?;
        
        let func = module.load_function("dwt_decomposition_db4")?;
        
        let mut builder = stream.launch_builder(&func);
        builder.arg(input);
        builder.arg(approx);
        builder.arg(detail);
        builder.arg(&n);
        
        unsafe { builder.launch(config)? };
        stream.synchronize()?;
        Ok(())
    }
    
    // Thermal time kernels
    pub fn launch_gdd_kernel(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        temp: &CudaSlice<f32>,
        output: &CudaSlice<f32>,
        n: u32,
        base_temp: f32,
        upper_temp: f32,
    ) -> Result<()> {
        let module = self.modules.get("thermal")
            .ok_or_else(|| anyhow::anyhow!("Thermal module not loaded"))?;
        
        let func = module.load_function("compute_gdd")?;
        
        let mut builder = stream.launch_builder(&func);
        builder.arg(temp);
        builder.arg(output);
        builder.arg(&n);
        builder.arg(&base_temp);
        builder.arg(&upper_temp);
        
        unsafe { builder.launch(config)? };
        stream.synchronize()?;
        Ok(())
    }
    
    // Environment coupling kernels
    pub fn launch_thermal_coupling_kernel(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        temp_in: &CudaSlice<f32>,
        temp_out: &CudaSlice<f32>,
        radiation: &CudaSlice<f32>,
        output: &CudaSlice<f32>,
        n: u32,
    ) -> Result<()> {
        let module = self.modules.get("environment_coupling")
            .ok_or_else(|| anyhow::anyhow!("Environment coupling module not loaded"))?;
        
        let func = module.load_function("thermal_coupling_slope")?;
        
        let mut builder = stream.launch_builder(&func);
        builder.arg(temp_in);
        builder.arg(temp_out);
        builder.arg(radiation);
        builder.arg(output);
        builder.arg(&n);
        
        unsafe { builder.launch(config)? };
        stream.synchronize()?;
        Ok(())
    }
    
    // Actuator dynamics kernels
    pub fn launch_actuator_response_kernel(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        signal: &CudaSlice<f32>,
        edge_count: &CudaSlice<f32>,
        duty_cycle: &CudaSlice<f32>,
        n: u32,
    ) -> Result<()> {
        let module = self.modules.get("actuator")
            .ok_or_else(|| anyhow::anyhow!("Actuator module not loaded"))?;
        
        let func = module.load_function("count_edges")?;
        
        let mut builder = stream.launch_builder(&func);
        builder.arg(signal);
        builder.arg(edge_count);
        builder.arg(&n);
        builder.arg(&0.5f32); // threshold
        
        unsafe { builder.launch(config)? };
        
        // Also compute duty cycle
        let func = module.load_function("duty_cycle")?;
        let mut builder = stream.launch_builder(&func);
        builder.arg(signal);
        builder.arg(duty_cycle);
        builder.arg(&n);
        builder.arg(&0.5f32); // threshold
        
        unsafe { builder.launch(config)? };
        stream.synchronize()?;
        Ok(())
    }
    
    // Stress counter kernels  
    pub fn launch_stress_count_kernel(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        values: &CudaSlice<f32>,
        count: &CudaSlice<f32>,
        integral: &CudaSlice<f32>,
        n: u32,
        low_threshold: f32,
        high_threshold: f32,
    ) -> Result<()> {
        let module = self.modules.get("stress_counters")
            .ok_or_else(|| anyhow::anyhow!("Stress counters module not loaded"))?;
        
        let func = module.load_function("stress_count")?;
        
        let mut builder = stream.launch_builder(&func);
        builder.arg(values);
        builder.arg(count);
        builder.arg(integral);
        builder.arg(&n);
        builder.arg(&low_threshold);
        builder.arg(&high_threshold);
        
        unsafe { builder.launch(config)? };
        stream.synchronize()?;
        Ok(())
    }
    
    // Economic features kernels
    pub fn launch_energy_cost_kernel(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        consumption: &CudaSlice<f32>,
        prices: &CudaSlice<f32>,
        output: &CudaSlice<f32>,
        n: u32,
    ) -> Result<()> {
        let module = self.modules.get("economic")
            .ok_or_else(|| anyhow::anyhow!("Economic module not loaded"))?;
        
        let func = module.load_function("weighted_energy_consumption")?;
        
        let mut builder = stream.launch_builder(&func);
        builder.arg(consumption);
        builder.arg(prices);
        builder.arg(output);
        builder.arg(&n);
        
        unsafe { builder.launch(config)? };
        stream.synchronize()?;
        Ok(())
    }
}

fn compile_kernel_source(kernel_code: &str) -> Result<cudarc::nvrtc::Ptx> {
    // Add CUDA math constants to the beginning of the source
    let source_with_constants = format!(
        r#"
// Define CUDA math constants
#ifndef CUDART_INF_F
#define CUDART_INF_F __int_as_float(0x7f800000)
#endif

{}
"#,
        kernel_code
    );
    
    let opts = CompileOptions {
        arch: Some("sm_70"),  // Adjust based on your GPU
        include_paths: vec![],
        ..Default::default()
    };
    
    compile_ptx_with_opts(&source_with_constants, opts).map_err(Into::into)
}

fn compile_statistical_kernel() -> Result<cudarc::nvrtc::Ptx> {
    let kernel_code = r#"
// Define CUDA math constants
#define CUDART_INF_F __int_as_float(0x7f800000)

extern "C" {

// Warp reduction utilities
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ float warp_reduce_min(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Statistical features structure
struct StatisticalFeatures {
    float mean;
    float std;
    float min;
    float max;
    float skewness;
    float kurtosis;
};

// Main statistical computation kernel
__global__ void compute_statistics(
    const float* __restrict__ input,
    StatisticalFeatures* __restrict__ output,
    const unsigned int n
) {
    extern __shared__ float shared[];
    
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    const int lane = tid % 32;
    const int warp_id = tid / 32;
    const int num_warps = (blockDim.x + 31) / 32;
    
    // Initialize accumulators
    float sum = 0.0f;
    float min_val = CUDART_INF_F;
    float max_val = -CUDART_INF_F;
    int count = 0;
    
    // First pass: compute sum, min, max
    for (int i = gid; i < n; i += gridDim.x * blockDim.x) {
        float val = input[i];
        sum += val;
        min_val = fminf(min_val, val);
        max_val = fmaxf(max_val, val);
        count++;
    }
    
    // Warp-level reduction
    sum = warp_reduce_sum(sum);
    min_val = warp_reduce_min(min_val);
    max_val = warp_reduce_max(max_val);
    
    // Store warp results in shared memory
    if (lane == 0) {
        shared[warp_id] = sum;
        shared[num_warps + warp_id] = min_val;
        shared[2 * num_warps + warp_id] = max_val;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (warp_id == 0) {
        float warp_sum = (lane < num_warps) ? shared[lane] : 0.0f;
        float warp_min = (lane < num_warps) ? shared[num_warps + lane] : CUDART_INF_F;
        float warp_max = (lane < num_warps) ? shared[2 * num_warps + lane] : -CUDART_INF_F;
        
        warp_sum = warp_reduce_sum(warp_sum);
        warp_min = warp_reduce_min(warp_min);
        warp_max = warp_reduce_max(warp_max);
        
        if (lane == 0) {
            float mean = warp_sum / n;
            
            // For now, simplified statistics (full implementation would need multiple passes)
            output->mean = mean;
            output->std = 0.0f;  // Placeholder
            output->min = warp_min;
            output->max = warp_max;
            output->skewness = 0.0f;  // Placeholder
            output->kurtosis = 0.0f;  // Placeholder
        }
    }
}

// Rolling statistics kernel
__global__ void rolling_stats(
    const float* __restrict__ input,
    float* __restrict__ means,
    float* __restrict__ stds,
    const unsigned int n,
    const unsigned int window_size
) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int num_windows = n - window_size + 1;
    
    if (tid < num_windows) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        
        // Compute statistics for this window
        for (int i = 0; i < window_size; i++) {
            float val = input[tid + i];
            sum += val;
            sum_sq += val * val;
        }
        
        float mean = sum / window_size;
        float variance = (sum_sq / window_size) - (mean * mean);
        
        means[tid] = mean;
        stds[tid] = sqrtf(fmaxf(variance, 0.0f));
    }
}

// VPD computation kernel
__global__ void compute_vpd(
    const float* __restrict__ temp,
    const float* __restrict__ rh,
    float* __restrict__ vpd,
    const unsigned int n
) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < n) {
        float t = temp[tid];
        float h = rh[tid];
        
        // Saturation vapor pressure (Magnus formula)
        float es = 0.6108f * expf((17.27f * t) / (t + 237.3f));
        
        // Actual vapor pressure
        float ea = es * (h / 100.0f);
        
        // VPD in kPa
        vpd[tid] = es - ea;
    }
}

} // extern "C"
"#;

    let opts = CompileOptions {
        arch: Some("sm_70"),  // Adjust based on your GPU
        include_paths: vec![],
        ..Default::default()
    };
    
    compile_ptx_with_opts(kernel_code, opts).map_err(Into::into)
}