# Era Detection Analysis

## Key Observations

### Level A (PELT - Large Structural Changes)
- 12 eras total
- All start at different times but end at 2016-09-08
- This suggests: **The greenhouse had major operational changes that lasted until the end**

### Level B (BOCPD - Medium Operational Changes)  
- 15 eras total
- Mix of short (47-72 hours) and long (years) eras
- Examples:
  - Era 1561: 72 hours (Dec 6-9, 2013)
  - Era 2444: 47 hours (Dec 12-14, 2013)
  - Era 180: 1000+ days (spans to end)

### What This Means

The era detection found:
1. **Changepoints** where greenhouse operation changed
2. **Long stable periods** that continued until dataset end
3. **Short periods** of different operation (maintenance? experiments?)

## The "Island" Pattern

If your data has gaps (collection stopped/started), the eras might represent:
- **Continuous operation periods** between gaps
- **Different greenhouse configurations** that persisted

## Are We Using Era Tables?

YES! The GPU feature extraction:
1. Queries `era_labels_level_a/b/c` tables
2. For each era, fetches ALL sensor data in that time range
3. Computes features for each era

## The Real Issue

The problem isn't that eras span years - it's that we're fetching ALL data points in those years:
- Era 144: 291,314 rows (limited to 100K by our LIMIT)
- That's ~100 rows/hour for 3 years of continuous data

## Possible Solutions

### 1. If Eras Are Correct (Long Stable Periods)
```sql
-- Sample the data instead of using all points
SELECT * FROM sensor_data_merged 
WHERE time >= $1 AND time < $2
AND EXTRACT(MINUTE FROM time) = 0  -- One sample per hour
ORDER BY time;
```

### 2. If You Want Smaller Eras
Re-run era detection with more sensitive parameters:
```yaml
# In docker-compose.yml
command: [
  "--pelt-min-size", "288",     # 1 hour minimum era
  "--pelt-penalty", "10",        # Lower = more changepoints
  "--bocpd-lambda", "10.0",      # Lower = more sensitive
  "--hmm-states", "10"           # More states = more eras
]
```

### 3. Create Fixed-Size Eras
```sql
-- Create daily eras instead
INSERT INTO era_labels_level_d
SELECT 
  ROW_NUMBER() OVER (ORDER BY day) as era_id,
  'level_d' as era_level,
  day as start_time,
  day + INTERVAL '1 day' as end_time,
  COUNT(*) as rows
FROM (
  SELECT DATE_TRUNC('day', time) as day
  FROM sensor_data_merged
  GROUP BY 1
) t;
```

## UPDATED: Fixed Era Detection Parameters

The era detection parameters have been updated in docker-compose.yml:

| Parameter | Old Value | New Value | Rationale |
|-----------|-----------|-----------|-----------|
| `pelt-min-size` | 48 (4 hours) | **288 (24 hours)** | Prevents micro-eras, ensures meaningful operational periods |
| `bocpd-lambda` | 200.0 | **50.0** | More sensitive to operational changes |
| `hmm-states` | 5 | **10** | Captures more nuanced operational states |

### Expected Results After Re-running

- **Level A (PELT)**: 10-50 eras representing major structural changes (seasonal, equipment changes)
- **Level B (BOCPD)**: 50-200 eras representing operational shifts (week-to-week variations)
- **Level C (HMM)**: 200-1000 eras representing daily operational patterns

### Data Sampling Already Implemented

The db.rs file has been updated with intelligent sampling:
```sql
-- For eras > 7 days, sample one reading every 5 minutes
AND (($2 - $1 < interval '7 days') OR EXTRACT(EPOCH FROM time) % 300 = 0)
LIMIT 100000  -- Safety limit
```

## Recommendation

1. First, check if data has gaps:
```sql
-- Find gaps > 1 hour
WITH time_diffs AS (
  SELECT 
    time,
    LAG(time) OVER (ORDER BY time) as prev_time,
    time - LAG(time) OVER (ORDER BY time) as gap
  FROM sensor_data_merged
)
SELECT 
  prev_time as gap_start,
  time as gap_end,
  gap
FROM time_diffs
WHERE gap > INTERVAL '1 hour'
ORDER BY gap DESC
LIMIT 20;
```

2. **Clean and re-run era detection** with the new parameters
3. **Test GPU feature extraction** with the fixed shared memory configuration
4. **Monitor performance** - should be much faster with reasonable era sizes

See `GPU_TESTING_COMMANDS.md` for detailed testing instructions.