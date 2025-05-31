# 🚨 CRITICAL: Pipeline Redesign Required

## Executive Summary

**The current pipeline architecture is fundamentally incompatible with the actual data characteristics.**

We have discovered that the sensor_data_merged table contains:
- **91.3% completely empty rows** (1.3M out of 1.45M)
- **99.9% NULL values** for light intensity
- **95-99% NULL values** for most sensors
- **No continuous data periods** - only sparse "islands" of measurements

## The Core Problem

### What We Assumed
```
Continuous Time Series → Era Detection → Feature Extraction → Model Training
     (Dense data)         (Changepoints)    (Time features)    (ML models)
```

### What We Actually Have
```
Sparse Event Data → ? → ? → ?
  (91% empty)     
```

## Data Reality Check

### 1. **Islands of Data**
```sql
-- Example: A typical day's data pattern
Time        | Temp | CO2  | Light | Status
00:00:00   | NULL | NULL | NULL  | Empty
00:01:00   | NULL | NULL | NULL  | Empty
...
10:15:00   | 22.5 | 450  | NULL  | Partial data
10:16:00   | NULL | NULL | NULL  | Empty
...
(1380 empty minutes out of 1440 in a day)
```

### 2. **Sensor Availability Timeline**
```
2013-2014: Some temp/humidity data (Aarslev)
2014-2015: Sporadic CO2 data (KnudJepsen)
2015-2016: Almost no environmental data
Light data: Only 1,520 readings in 3 years!
```

### 3. **Source File Chaos**
Different CSV files provide completely different sensor subsets:
- `celle5.csv`: temp, humidity, CO2
- `NO3_LAMPEGRP_1.csv`: CO2, radiation, pipe_temp (no air temp!)
- `MortenSDUData.csv`: Only light intensity (corrupted values)

## Why Current Pipeline Fails

### 1. **Era Detection (PELT/BOCPD/HMM)**
- **Requires**: Continuous signal to detect changepoints
- **Reality**: 95% gaps make changepoint detection meaningless
- **Result**: Algorithms detect noise in sparse data as "changes"

### 2. **Feature Extraction (tsfresh)**
- **Requires**: Time series with regular sampling
- **Reality**: Can't calculate rolling statistics on 95% NULL data
- **Result**: Features are mostly NULL or meaningless

### 3. **Preprocessing**
- **Current**: Expects to interpolate small gaps
- **Reality**: Can't interpolate 10-hour gaps between single data points
- **Result**: preprocessed_features still has 93% NULL values

## Proposed Solution: Complete Architectural Pivot

### Option 1: Event-Based Architecture
```python
# Treat data as discrete events, not continuous series
Event(
    timestamp="2014-06-15 10:30:00",
    sensor="air_temp_c", 
    value=22.5,
    source="aarslev_celle5.csv"
)
```

**Benefits**:
- Natural fit for sparse data
- No interpolation of massive gaps
- Can still aggregate to meaningful periods

### Option 2: Data Density Filtering
```python
# Only process time periods with sufficient data
def find_viable_periods(min_coverage=0.5):
    # Find hours/days with >50% sensor coverage
    viable_periods = []
    for period in time_chunks:
        if period.data_coverage > min_coverage:
            viable_periods.append(period)
    return viable_periods
```

**Benefits**:
- Work with dense data islands only
- Traditional methods work on filtered data
- Much smaller but higher quality dataset

### Option 3: Aggregated Analysis Only
```sql
-- Work only at daily/weekly aggregation levels
CREATE MATERIALIZED VIEW daily_greenhouse_data AS
SELECT 
    DATE_TRUNC('day', time) as day,
    AVG(air_temp_c) as avg_temp,
    COUNT(air_temp_c) as temp_readings,  -- Track data density
    AVG(co2_measured_ppm) as avg_co2,
    COUNT(co2_measured_ppm) as co2_readings
FROM sensor_data_merged
WHERE air_temp_c IS NOT NULL 
   OR co2_measured_ppm IS NOT NULL
GROUP BY 1
HAVING COUNT(*) > 100;  -- At least 100 readings in the day
```

**Benefits**:
- Smooths over gaps
- More stable for era detection
- Still captures operational patterns

## Immediate Actions Required

### 1. **Stop Current Pipeline**
The current approach will not work with this data structure.

### 2. **Data Audit**
```python
# Find best continuous data periods
def audit_data_continuity():
    results = {}
    for month in all_months:
        results[month] = {
            'coverage': calculate_coverage(month),
            'sensors': list_available_sensors(month),
            'max_gap': find_largest_gap(month)
        }
    return results
```

### 3. **Prototype New Approach**
Pick the best month (June 2014) and test:
- Event-based processing
- Daily aggregation only
- Simple rule-based "eras" (e.g., monthly)

## Recommended New Pipeline Architecture

```
1. Raw Data Ingestion
   ↓
2. Event Store (sparse data as-is)
   ↓
3. Data Density Analysis
   ↓
4. Viable Period Extraction
   ↓
5. Aggregation (hourly/daily only)
   ↓
6. Simple Era Definition (fixed windows or rules)
   ↓
7. Feature Engineering (on aggregated data)
   ↓
8. Model Training (on complete periods only)
```

## Impact on Project Goals

### What We Can Still Do
- Energy usage patterns (using sparse CO2/temp data)
- Seasonal analysis (monthly aggregations)
- Basic climate control assessment

### What We Cannot Do
- High-resolution time series analysis
- Detailed plant growth modeling (no light data)
- Minute-by-minute optimization
- Traditional changepoint detection

## Decision Points

1. **Continue with sparse data** → Requires complete pipeline redesign
2. **Find better data source** → Delay project but maintain architecture
3. **Hybrid approach** → Use current pipeline only on best data periods

## Recommendation

**Implement Option 2 (Data Density Filtering) + Option 3 (Aggregated Analysis)**

1. Filter to periods with >50% data coverage
2. Aggregate to daily resolution
3. Use fixed monthly eras instead of detection
4. Focus on June 2014 for proof of concept
5. Document data limitations clearly

This approach:
- Salvages existing pipeline investment
- Works with data reality
- Still provides valuable insights
- Can be implemented quickly

## Next Steps

1. Create data density report by month/sensor
2. Implement filtering to extract viable periods
3. Test simplified pipeline on June 2014 data
4. Document which analyses are possible vs impossible
5. Set realistic expectations with stakeholders

**The data sparsity is not a bug to fix - it's the fundamental characteristic we must design around.**