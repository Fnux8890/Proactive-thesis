# GPU Sparse Data Implementation Plan

## Executive Summary

We flip the pipeline: **GPU Feature Extraction FIRST → Smart Filling → Simple Eras**

This approach maximizes data usage while acknowledging the sparse reality.

## Why This Works

1. **GPU handles sparse data efficiently** - We just pass a validity mask
2. **Features guide interpolation** - Not blind mathematical filling
3. **Eras emerge from data patterns** - Not forced by algorithms

## Implementation Roadmap

### Week 1: GPU Kernel Modifications

#### 1.1 Add Sparse Data Support
```cuda
// Modified kernel with NULL handling
__global__ void compute_features_sparse(
    const float* data,
    const bool* valid_mask,  // Which values are not NULL
    Features* output,
    int n
) {
    // Process only valid data points
    int valid_count = 0;
    float sum = 0.0f;
    
    for (int i = tid; i < n; i += stride) {
        if (valid_mask[i]) {
            sum += data[i];
            valid_count++;
        }
    }
    
    // Output includes coverage metric
    output->coverage = (float)valid_count / n;
}
```

#### 1.2 Sliding Window Processing
```rust
// In gpu_feature_extraction/src/main.rs
pub async fn process_sliding_windows(
    data: &DataFrame,
    window_hours: usize,
    slide_hours: usize,
) -> Result<Vec<WindowFeatures>> {
    let mut features = Vec::new();
    
    for window in data.windows(window_hours * 60).step(slide_hours * 60) {
        let coverage = calculate_coverage(&window);
        
        if coverage > 0.1 {  // At least 10% data
            let gpu_features = extract_gpu_features(&window).await?;
            features.push(WindowFeatures {
                start: window.start_time(),
                end: window.end_time(),
                coverage,
                features: gpu_features,
            });
        }
    }
    
    Ok(features)
}
```

### Week 2: Smart Gap Filling

#### 2.1 Pattern-Based Interpolation
```python
class SmartGapFiller:
    def __init__(self, features_df):
        self.features_df = features_df
        self.fill_methods = {}
        
    def fill_gaps(self, df):
        # Phase 1: Linear interpolation for short gaps
        df_filled = self._fill_short_gaps(df, max_gap_hours=1)
        
        # Phase 2: Pattern matching for medium gaps
        df_filled = self._fill_by_patterns(df_filled, max_gap_hours=6)
        
        # Phase 3: Domain constraints
        df_filled = self._apply_physics(df_filled)
        
        return df_filled
```

#### 2.2 Domain-Specific Rules
```python
GREENHOUSE_CONSTRAINTS = {
    'air_temp_c': {
        'min': 10, 'max': 40,
        'max_change_per_minute': 0.5,
        'typical_night_drop': 5
    },
    'co2_measured_ppm': {
        'min': 300, 'max': 1500,
        'day_range': (400, 800),
        'night_range': (600, 1000)
    },
    'relative_humidity_percent': {
        'min': 30, 'max': 95,
        'typical_range': (60, 85)
    }
}
```

### Week 3: Feature-Based Era Creation

#### 3.1 Clustering Approach
```python
def create_eras_from_features(features_df):
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(features_df[FEATURE_COLS])
    
    # DBSCAN clustering
    clusters = DBSCAN(eps=0.3, min_samples=4).fit_predict(X)
    
    # Merge adjacent windows in same cluster
    eras = merge_adjacent_clusters(features_df, clusters)
    
    return eras
```

#### 3.2 Quality Tracking
```python
class EraQuality:
    def __init__(self, era):
        self.era_id = era['era_id']
        self.coverage = era['avg_coverage']
        self.consistency = self._calculate_consistency()
        self.confidence = self._calculate_confidence()
```

### Week 4: Pipeline Integration

#### 4.1 Docker Compose Updates
```yaml
gpu_sparse_processor:
  build:
    context: ./gpu_feature_extraction
    dockerfile: Dockerfile
  environment:
    PROCESSING_MODE: "sparse_sliding_window"
    WINDOW_HOURS: 24
    SLIDE_HOURS: 6
    MIN_COVERAGE: 0.1
  command: [
    "--mode", "sparse",
    "--fill-gaps", "true",
    "--create-eras", "true"
  ]
```

#### 4.2 Database Schema
```sql
-- Sliding window features
CREATE TABLE gpu_window_features (
    window_id SERIAL PRIMARY KEY,
    start_time TIMESTAMP WITH TIME ZONE,
    end_time TIMESTAMP WITH TIME ZONE,
    coverage FLOAT,
    features JSONB,
    quality_score FLOAT
);

-- Smart filled data with provenance
CREATE TABLE smart_filled_data (
    time TIMESTAMP WITH TIME ZONE PRIMARY KEY,
    sensor_values JSONB,  -- All sensors
    fill_metadata JSONB   -- How each value was filled
);

-- Feature-based eras
CREATE TABLE feature_eras (
    era_id INTEGER PRIMARY KEY,
    start_time TIMESTAMP WITH TIME ZONE,
    end_time TIMESTAMP WITH TIME ZONE,
    cluster_id INTEGER,
    quality_metrics JSONB
);
```

## Validation Plan

### Test on June 2014 (Best Data)
1. Original coverage: ~10%
2. Target after filling: 40%+
3. Expected eras: 20-30
4. Quality threshold: 0.7

### Success Metrics
```python
def validate_results(original, filled, eras):
    metrics = {
        'coverage_gain': (filled.notna().mean() - original.notna().mean()).mean(),
        'realistic_values': check_physical_constraints(filled),
        'era_quality': eras['quality_score'].mean(),
        'processing_time': measure_gpu_time()
    }
    
    assert metrics['coverage_gain'] > 0.3  # 30% improvement
    assert metrics['realistic_values'] > 0.95  # 95% pass physics checks
    assert metrics['era_quality'] > 0.7  # High confidence eras
    assert metrics['processing_time'] < 300  # Under 5 minutes
```

## Risk Mitigation

### Risk 1: Over-interpolation
- **Mitigation**: Track fill confidence, mark synthetic data

### Risk 2: GPU memory with sparse data
- **Mitigation**: Process in chunks, use streaming

### Risk 3: Poor era quality
- **Mitigation**: Fallback to fixed time windows

## Immediate Next Steps

1. **Today**: Run `implement_gpu_sparse_pipeline.py` on June 2014
2. **Tomorrow**: Modify GPU kernels for sparse support
3. **This Week**: Implement smart filling algorithm
4. **Next Week**: Integrate into main pipeline

## Code to Run Now

```bash
# Test the new approach
cd DataIngestion
python implement_gpu_sparse_pipeline.py

# This will:
# 1. Extract features using sliding windows
# 2. Fill gaps intelligently
# 3. Create eras from feature similarity
# 4. Generate visualization
```

## Expected Output

```
Original coverage: 8.7%
Filled coverage: 42.3%
Coverage improvement: 33.6%
Windows extracted: 112 → 156 (after filling)
Eras created: 28
Era duration: 24.0 - 336.0 hours
```

This approach works WITH the sparse data rather than fighting against it!