# GPU-First Pipeline Redesign: Working with Sparse Data

## Core Insight
Instead of Era Detection → Feature Extraction, we flip it:
**GPU Feature Extraction → Smart Aggregation → Simplified Eras**

## Why This Works Better

1. **GPU can handle sparse data efficiently** - just skip NULL values
2. **We can compute features on whatever data exists** - no need for continuous signals
3. **Aggregation happens AFTER feature extraction** - preserving all available information
4. **Smart filling becomes possible** - using computed features to guide interpolation

## New Pipeline Architecture

```
Raw Sparse Data 
    ↓
GPU Feature Extraction (on raw data chunks)
    ↓
Smart Gap Filling (using extracted features)
    ↓
Feature Aggregation (hourly/daily)
    ↓
Simple Era Definition (based on feature similarity)
    ↓
Model Training
```

## Phase 1: GPU Feature Extraction on Raw Data

### 1.1 Modify GPU Kernel for Sparse Data
```rust
// In gpu_feature_extraction/src/kernels/statistics.cu
__global__ void compute_statistics_sparse(
    const float* __restrict__ input,
    const bool* __restrict__ valid_mask,  // NEW: mask for non-NULL values
    StatisticalFeatures* __restrict__ output,
    const unsigned int n
) {
    // Only process valid data points
    float sum = 0.0f;
    int valid_count = 0;
    
    for (int i = gid; i < n; i += gridDim.x * blockDim.x) {
        if (valid_mask[i]) {  // Skip NULL values
            float val = input[i];
            sum += val;
            valid_count++;
        }
    }
    
    // Compute statistics only on valid data
    if (valid_count > 0) {
        output->mean = sum / valid_count;
        output->coverage = (float)valid_count / n;  // Track data density
    }
}
```

### 1.2 Process Data in Smart Chunks
```python
# Instead of eras, use sliding windows
def extract_features_sliding_window(df, window_hours=24, slide_hours=6):
    features = []
    
    for start in range(0, len(df), slide_hours * 60):
        window = df.iloc[start:start + window_hours * 60]
        
        # Only process if window has minimum data
        coverage = window.notna().mean()
        if coverage.mean() > 0.1:  # At least 10% data
            gpu_features = gpu_extract(window)
            features.append({
                'start_time': window.index[0],
                'end_time': window.index[-1],
                'coverage': coverage.mean(),
                'features': gpu_features
            })
    
    return features
```

## Phase 2: Smart Gap Filling

### 2.1 Pattern-Based Interpolation
```python
def smart_fill_gaps(df, features_df):
    """Use extracted features to guide interpolation"""
    
    # 1. Identify similar time periods (same hour of day, similar season)
    def find_similar_periods(timestamp):
        hour = timestamp.hour
        month = timestamp.month
        
        similar = features_df[
            (features_df['hour'] == hour) & 
            (features_df['month'] == month) &
            (features_df['coverage'] > 0.5)
        ]
        return similar
    
    # 2. Fill using patterns from similar periods
    for col in df.columns:
        if col in ['air_temp_c', 'co2_measured_ppm']:  # Fillable sensors
            # Short gaps: linear interpolation
            df[col] = df[col].interpolate(method='linear', limit=60)  # Max 1 hour
            
            # Medium gaps: use similar periods
            mask = df[col].isna()
            for idx in df[mask].index:
                similar = find_similar_periods(idx)
                if len(similar) > 0:
                    df.loc[idx, col] = similar[col].median()
    
    return df
```

### 2.2 Domain-Aware Filling
```python
def apply_domain_constraints(df):
    """Apply greenhouse physics constraints"""
    
    # Temperature continuity
    df['air_temp_c'] = df['air_temp_c'].clip(10, 40)  # Reasonable greenhouse range
    
    # CO2 follows day/night patterns
    df['is_day'] = (df.index.hour >= 6) & (df.index.hour <= 20)
    
    # Fill CO2 using typical patterns
    day_co2 = df[df['is_day']]['co2_measured_ppm'].median()
    night_co2 = df[~df['is_day']]['co2_measured_ppm'].median()
    
    df.loc[df['is_day'] & df['co2_measured_ppm'].isna(), 'co2_measured_ppm'] = day_co2
    df.loc[~df['is_day'] & df['co2_measured_ppm'].isna(), 'co2_measured_ppm'] = night_co2
    
    return df
```

## Phase 3: Feature-Based Era Creation

### 3.1 Cluster Similar Periods
```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def create_eras_from_features(features_df):
    """Create eras by clustering similar feature periods"""
    
    # Use key features for clustering
    feature_cols = ['mean_temp', 'mean_co2', 'temp_stability', 'photoperiod']
    X = features_df[feature_cols].fillna(0)
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Cluster similar periods
    clusterer = DBSCAN(eps=0.5, min_samples=4)  # 4 * 6 hours = 24 hours minimum
    features_df['era_id'] = clusterer.fit_predict(X_scaled)
    
    # Merge adjacent same-cluster periods
    eras = []
    for era_id in features_df['era_id'].unique():
        if era_id >= 0:  # Skip noise points
            era_periods = features_df[features_df['era_id'] == era_id]
            eras.append({
                'era_id': era_id,
                'start_time': era_periods['start_time'].min(),
                'end_time': era_periods['end_time'].max(),
                'num_windows': len(era_periods),
                'avg_coverage': era_periods['coverage'].mean()
            })
    
    return pd.DataFrame(eras)
```

## Phase 4: GPU-Accelerated Processing Pipeline

### 4.1 Batch Processing with Quality Tracking
```rust
// In gpu_feature_extraction/src/main.rs
pub struct QualityAwareFeatures {
    pub features: StatisticalFeatures,
    pub quality_metrics: DataQuality,
}

pub struct DataQuality {
    pub coverage: f32,          // Percentage of non-NULL values
    pub gap_ratio: f32,         // Ratio of gaps to data
    pub max_gap_minutes: u32,   // Longest continuous gap
    pub confidence: f32,        // Overall confidence score
}

async fn process_with_quality(data: &DataFrame) -> Result<Vec<QualityAwareFeatures>> {
    let mut results = Vec::new();
    
    // Process in overlapping windows for better coverage
    for window in data.windows(WINDOW_SIZE).step_by(STEP_SIZE) {
        let quality = assess_data_quality(window);
        
        // Only process if minimum quality met
        if quality.coverage > 0.1 {
            let features = gpu_extract_features(window).await?;
            results.push(QualityAwareFeatures { features, quality });
        }
    }
    
    Ok(results)
}
```

## Implementation Plan

### Week 1: GPU Sparse Data Support
1. Modify GPU kernels to handle NULL masks
2. Add data quality metrics to feature output
3. Test on June 2014 data (best coverage)

### Week 2: Smart Gap Filling
1. Implement pattern-based interpolation
2. Add domain constraints (temperature ranges, CO2 patterns)
3. Validate filled data maintains realistic patterns

### Week 3: Feature-Based Eras
1. Extract features using GPU on sliding windows
2. Cluster similar periods
3. Create era labels from clusters

### Week 4: Pipeline Integration
1. Connect all components
2. Add quality tracking throughout
3. Validate on multiple months

## Key Advantages

1. **Uses ALL available data** - doesn't require continuous signals
2. **GPU efficiency** - parallel processing of sparse windows
3. **Smart filling** - physics-based, not just mathematical
4. **Quality awareness** - tracks confidence throughout
5. **Flexible eras** - based on actual patterns, not algorithms

## Validation Strategy

```python
def validate_pipeline_output(original_df, processed_df, eras_df):
    metrics = {
        'coverage_improvement': (
            processed_df.notna().mean() - original_df.notna().mean()
        ).mean(),
        'era_count': len(eras_df),
        'avg_era_duration': eras_df['duration'].mean(),
        'data_quality_score': calculate_quality_score(processed_df),
        'feature_consistency': check_feature_consistency(processed_df, eras_df)
    }
    
    assert metrics['coverage_improvement'] > 0.2  # 20% more usable data
    assert 20 <= metrics['era_count'] <= 200     # Reasonable era count
    assert metrics['data_quality_score'] > 0.7    # Good quality output
    
    return metrics
```

## SQL Schema Updates

```sql
-- New tables for GPU-first approach
CREATE TABLE gpu_extracted_windows (
    window_id SERIAL PRIMARY KEY,
    start_time TIMESTAMP WITH TIME ZONE,
    end_time TIMESTAMP WITH TIME ZONE,
    coverage FLOAT,
    features JSONB,
    quality_metrics JSONB
);

CREATE TABLE smart_filled_data (
    time TIMESTAMP WITH TIME ZONE PRIMARY KEY,
    -- All sensor columns
    air_temp_c REAL,
    co2_measured_ppm REAL,
    -- Quality tracking
    fill_method VARCHAR(50),  -- 'original', 'interpolated', 'pattern', 'domain'
    confidence FLOAT
);

CREATE TABLE feature_based_eras (
    era_id INTEGER PRIMARY KEY,
    start_time TIMESTAMP WITH TIME ZONE,
    end_time TIMESTAMP WITH TIME ZONE,
    window_count INTEGER,
    avg_coverage FLOAT,
    cluster_features JSONB
);
```

## Success Metrics

1. **Data Coverage**: Increase from 8% to 40%+ through smart filling
2. **Era Quality**: 50-100 meaningful eras (not 1.8M noise)
3. **Processing Speed**: <5 minutes for full pipeline on GPU
4. **Model Viability**: Sufficient features for basic modeling

This approach maximizes data usage while being honest about quality!