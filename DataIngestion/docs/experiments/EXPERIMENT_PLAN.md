# Sparse Pipeline Performance Experiment Plan

## Objective

Measure and compare the performance gains of the sparse pipeline architecture with GPU acceleration versus CPU-only processing.

## Experimental Design

### Variables

**Independent Variable**: Processing mode (CPU vs GPU)
- CPU: `DISABLE_GPU=true`
- GPU: `DISABLE_GPU=false`

**Dependent Variables**:
1. Total execution time (seconds)
2. Feature extraction rate (features/second)
3. Memory usage (checkpoint size)
4. Stage-wise timing breakdown

**Controlled Variables**:
- Date range: 2014-01-01 to 2014-07-01 (6 months)
- Window size: 24 hours
- Minimum coverage: 10%
- Gap filling: Max 2 hours
- Number of runs: 3 per condition

### Hypotheses

1. **H1**: GPU acceleration provides significant speedup for feature extraction
   - Expected: 2-3x speedup with current partial GPU implementation
   - Potential: 15-20x with full GPU implementation

2. **H2**: Overall pipeline speedup is limited by non-GPU stages
   - Ingestion (I/O bound): No speedup
   - Aggregation (SQL): No speedup
   - Gap filling (CPU): No speedup
   - Feature extraction (GPU): Significant speedup
   - Era creation (CPU): No speedup

3. **H3**: Performance variability is low (<10% std dev)
   - System is deterministic
   - Database caching stabilizes after first run

## Methodology

### Setup
1. Clean environment (remove volumes, clear checkpoints)
2. Fresh database instance
3. Identical hardware/software configuration

### Procedure
1. Run 3 iterations with CPU-only mode
2. Allow 5-second cooldown
3. Run 3 iterations with GPU-enabled mode
4. Generate comparison report

### Measurements

For each run, capture:
- Stage timing (ingestion, sparse pipeline)
- Data metrics (records, features, eras)
- System metrics (if available)

### Statistical Analysis
- Calculate mean and standard deviation
- Compute speedup ratios
- Identify bottlenecks

## Expected Results

Based on architecture analysis:

| Stage | CPU Time | GPU Time | Speedup | Notes |
|-------|----------|----------|---------|-------|
| Ingestion | ~15s | ~15s | 1.0x | I/O bound |
| Aggregation | ~1s | ~1s | 1.0x | SQL query |
| Gap Filling | ~0.5s | ~0.5s | 1.0x | Simple CPU operation |
| Feature Extraction | ~10s | ~3-4s | 2.5-3x | Partial GPU impl |
| Era Creation | ~0.1s | ~0.1s | 1.0x | Aggregation only |
| **Total** | ~27s | ~20s | **1.35x** | Limited by CPU stages |

## Significance

This experiment will:
1. Validate the sparse pipeline architecture
2. Quantify current GPU benefits
3. Identify optimization opportunities
4. Provide baseline for future improvements

## Next Steps

After baseline measurement:
1. Profile GPU utilization
2. Identify CPU bottlenecks
3. Port additional algorithms to GPU
4. Re-run experiments to measure improvements