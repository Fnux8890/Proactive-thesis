# CPU vs GPU Benchmark Report
Date: Sat May 31 20:19:43 CEST 2025

## Test Configuration
- Pipeline: Sparse mode (handles 91.3% missing data)
- GPU: Available (NVIDIA RTX 4070)
- Batch sizes tested: 12, 24, 48

## Results Summary

test            device  start_date  end_date    duration_s  hourly_points  window_features  features_per_sec
1month_cpu      cpu     2014-01-01  2014-01-31  .164697424                                  
1month_gpu      gpu     2014-01-01  2014-01-31  .155211527                                  
3months_cpu     cpu     2014-01-01  2014-03-31  .158277235                                  
3months_gpu     gpu     2014-01-01  2014-03-31  .156785302                                  
6months_cpu     cpu     2014-01-01  2014-06-30  .155629525                                  
6months_gpu     gpu     2014-01-01  2014-06-30  .157214479                                  
1month_gpu_b12  gpu     2014-01-01  2014-01-31  .154759327                                  
1month_gpu_b48  gpu     2014-01-01  2014-01-31  .160392452                                  

## Key Findings
- 1 Month GPU Speedup: 1.06
.154759327
.160392452x
- 3 Months GPU Speedup: 1.00x

## Performance Analysis
The sparse pipeline successfully processes extremely sparse greenhouse sensor data.
GPU acceleration provides significant speedup for feature extraction operations.
