# Era Enhancement Quick Start Guide

## The Problem
Era detection is creating 1.88 million segments because greenhouse data is too stable for traditional changepoint algorithms designed for detecting equipment failures or market crashes.

## The Solution
Transform the data to emphasize meaningful operational patterns before applying era detection.

## What You Can Do Right Now

### 1. Run Domain Feature Analysis (5 minutes)
```bash
cd DataIngestion
python create_domain_features.py
```

This will:
- Calculate photoperiod from lamp status (we DO have this data!)
- Create heating demand metrics
- Identify VPD stress periods
- Generate control strategy indicators

### 2. Key Features We Can Create Today

| Feature | Formula | What It Captures |
|---------|---------|------------------|
| **photoperiod_hours** | `sum(any_lamp_on) / 60` over 24h | Daily light cycle (major driver) |
| **heating_demand_c** | `max(0, setpoint - actual_temp)` | Energy usage patterns |
| **night_temp_drop** | `temp_day_avg - temp_night_avg` | Diurnal management strategy |
| **vpd_stress_hours** | Hours outside 8-12 hPa range | Plant stress periods |
| **operational_state** | Composite of multiple features | Distinct operating modes |

### 3. Immediate Next Steps

#### Step 1: Analyze Current Data (Today)
```sql
-- Check daily patterns
SELECT 
    DATE_TRUNC('day', time) as day,
    AVG(CASE WHEN lamp_grp1_no3_status THEN 1 ELSE 0 END) as lamp_usage,
    AVG(air_temp_c) as avg_temp,
    STDDEV(air_temp_c) as temp_variability
FROM sensor_data_merged
GROUP BY 1
ORDER BY 1;
```

#### Step 2: Create Aggregated Views (Tomorrow)
```sql
-- Daily view for Level A
CREATE MATERIALIZED VIEW sensor_data_daily AS
SELECT 
    DATE_TRUNC('day', time) as day,
    AVG(dli_sum) as dli_daily,
    SUM(CASE WHEN lamp_grp1_no3_status THEN 1 ELSE 0 END) / 60.0 as photoperiod_hours,
    AVG(air_temp_c) as temp_avg,
    STDDEV(air_temp_c) as temp_std
FROM sensor_data_merged
GROUP BY 1;
```

#### Step 3: Test Era Detection on Aggregated Data
```bash
docker compose run era_detector \
    --db-table sensor_data_daily \
    --pelt-min-size 7 \
    --resample-every 1d
```

## Why This Will Work

1. **Photoperiod Changes**: Major operational shifts (winter→summer lighting)
2. **Temperature Strategies**: Heating patterns change seasonally
3. **Control Modes**: Different strategies for different growth phases

## Expected Results After Enhancement

| Level | Current | Target | Time Scale | Key Features |
|-------|---------|--------|------------|--------------|
| A | 437K eras | 30-50 | Monthly | Photoperiod, Season |
| B | 278K eras | 200-500 | Weekly | Temperature strategy, VPD |
| C | 1.1M eras | 2-5K | Daily | Actuator patterns |

## Validation Check
```python
# After running enhanced era detection
df = pd.read_sql("""
    SELECT COUNT(*) as era_count, 
           AVG(EXTRACT(EPOCH FROM (end_time - start_time))/86400) as avg_days
    FROM era_labels_level_a
""", engine)

assert 20 <= df['era_count'] <= 100, "Level A should have 20-100 eras"
assert df['avg_days'] >= 7, "Level A eras should be at least a week"
```

## Full Epic Implementation
See `ERA_DETECTION_ENHANCEMENT_EPICS.md` for the complete 7-epic plan.

## Questions?
The data shows clear patterns - we just need to help the algorithms see them!