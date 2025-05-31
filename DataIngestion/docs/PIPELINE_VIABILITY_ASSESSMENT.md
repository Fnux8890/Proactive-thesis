# Pipeline Viability Assessment

## Critical Discovery: Data Sparsity Crisis

After thorough investigation of `sensor_data_merged`, we've discovered that the fundamental assumption of the pipeline - continuous time-series data - is false.

### The Numbers Don't Lie

| Metric | Value | Impact |
|--------|-------|---------|
| **Empty rows** | 91.3% (1.3M/1.45M) | Most timestamps have NO data |
| **Light intensity NULLs** | 99.9% | Critical sensor essentially missing |
| **Temperature NULLs** | 98.7% | Basic climate data sparse |
| **Best daily coverage** | 10% | Even best days are 90% empty |
| **Typical coverage** | <2% | Data appears in tiny "islands" |

### What This Means

```
Expected: ████████████████████ (continuous data)
Reality:  █░░░█░░░░░█░░░░░█░░ (sparse islands)
```

## Current Pipeline Components Assessment

### 1. **Preprocessing** ⚠️ PARTIALLY VIABLE
- **Current**: Expects to interpolate small gaps
- **Reality**: Cannot interpolate 10-hour gaps
- **Verdict**: Needs complete redesign for sparse data

### 2. **Era Detection (PELT/BOCPD/HMM)** ❌ NOT VIABLE
- **Current**: Detects changepoints in continuous signals
- **Reality**: 95% gaps mean no meaningful signal to analyze
- **Verdict**: Will detect noise as changepoints
- **Evidence**: Created 1.88M "eras" from sparse noise

### 3. **Feature Extraction (tsfresh)** ❌ NOT VIABLE AS-IS
- **Current**: Calculates rolling statistics, FFT, autocorrelation
- **Reality**: Cannot compute time-series features on 95% NULL data
- **Verdict**: Must switch to aggregated features only

### 4. **GPU Feature Extraction** ⚠️ NEEDS RETHINKING
- **Current**: Optimized for dense data processing
- **Reality**: Will waste GPU resources on empty data
- **Verdict**: Need to pre-filter viable periods first

### 5. **Model Training** ⚠️ LIMITED VIABILITY
- **Current**: Assumes rich feature sets
- **Reality**: Very limited features possible
- **Verdict**: Simple models only, limited objectives

## Recommended Path Forward

### Option A: Radical Simplification (Recommended)

1. **Accept Reality**: Work with sparse data as-is
2. **Fixed Time Windows**: Monthly/weekly eras instead of detection
3. **Aggregate Only**: Daily minimum, likely weekly
4. **Limited Scope**: Focus on what's possible, not ideal

```python
# New simplified pipeline
def process_sparse_greenhouse_data():
    # 1. Filter to viable periods only
    viable_data = filter_days_with_coverage(min_coverage=0.3)
    
    # 2. Aggregate to daily
    daily_data = aggregate_to_daily(viable_data)
    
    # 3. Fixed monthly eras
    monthly_eras = create_fixed_monthly_windows(daily_data)
    
    # 4. Simple features only
    features = calculate_basic_stats(monthly_eras)
    
    return features
```

### Option B: Abandon Current Data

1. **Find Better Data Source**: This data may be unusable
2. **Synthetic Data**: Generate realistic greenhouse data
3. **Simulation Only**: Focus on forward simulation, not historical

### Option C: Hybrid Approach

1. **Best Periods Only**: June 2014, Feb-Mar 2014
2. **Dense Processing**: Full pipeline on good months
3. **Document Limitations**: Clear about what's not possible

## Specific Recommendations by Component

### 1. **Preprocessing Redesign**
```python
# OLD: Assumes continuous data
df = df.interpolate(method='time', limit=3)

# NEW: Handle sparse reality
def preprocess_sparse(df):
    # Only keep rows with actual data
    df = df.dropna(how='all')
    
    # Aggregate to reduce sparsity
    df = df.resample('1H').mean()
    
    # Mark coverage quality
    df['data_quality'] = df.notna().sum(axis=1) / len(df.columns)
    
    return df[df['data_quality'] > 0.3]
```

### 2. **Era "Detection"**
```sql
-- Forget algorithms, use domain knowledge
CREATE VIEW era_labels_monthly AS
SELECT 
    ROW_NUMBER() OVER (ORDER BY month) as era_id,
    month as start_time,
    month + INTERVAL '1 month' as end_time,
    'month' as era_type
FROM (
    SELECT DISTINCT DATE_TRUNC('month', time) as month
    FROM sensor_data_merged
    WHERE air_temp_c IS NOT NULL
) t;
```

### 3. **Feature Extraction**
```python
# Focus on what's possible with sparse data
def extract_sparse_features(daily_data):
    features = {
        'mean_temp': daily_data['temp'].mean(),
        'temp_range': daily_data['temp'].max() - daily_data['temp'].min(),
        'days_with_data': daily_data['temp'].notna().sum(),
        'coverage_quality': daily_data['temp'].notna().mean()
    }
    # Skip: FFT, autocorrelation, rolling stats
    return features
```

## Impact on Research Objectives

### What's Still Possible ✓
- Basic climate analysis (monthly/seasonal)
- Energy usage patterns (very coarse)
- Operational period identification (manual)

### What's No Longer Possible ✗
- Detailed plant growth modeling
- High-resolution optimization
- Minute-by-minute control strategies
- Complex time-series analysis
- Reliable changepoint detection

## Action Items

### Immediate (This Week)
1. Run `find_viable_data_periods.py` to identify usable periods
2. Create filtered dataset with only viable data
3. Test simplified pipeline on June 2014

### Short Term (Next 2 Weeks)
1. Redesign preprocessing for sparse data
2. Implement fixed-window era generation
3. Create aggregated feature extraction

### Long Term (Month)
1. Document all limitations clearly
2. Adjust research objectives to match data reality
3. Consider alternative data sources

## Conclusion

**The current pipeline architecture assumes continuous time-series data that doesn't exist.**

We must either:
1. Radically simplify to work with sparse data
2. Find better data
3. Abandon historical analysis for pure simulation

The sparse, fragmented nature of the data is not a problem to solve - it's the fundamental constraint we must design around.

## Script to Run

```bash
cd DataIngestion
python find_viable_data_periods.py
```

This will identify exactly which periods are viable for different types of analysis.