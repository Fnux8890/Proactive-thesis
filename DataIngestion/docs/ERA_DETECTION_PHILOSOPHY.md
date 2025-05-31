# Era Detection Philosophy: Macro, Meso, Micro

## You're Absolutely Right!

The three-level era detection (PELT, BOCPD, HMM) is indeed designed for **macro → meso → micro** analysis. The current parameters are preventing this natural hierarchy.

## Original Design Intent

### Level A - PELT (Macro: Major Structural Changes)
- **Purpose**: Detect major operational regime changes
- **Examples**: Seasonal transitions, equipment replacements, crop changes
- **Expected**: 10-50 eras over 3 years
- **Current Problem**: Created 437,254 eras (way too many!)

### Level B - BOCPD (Meso: Operational Variations)  
- **Purpose**: Detect medium-term operational adjustments
- **Examples**: Weekly patterns, maintenance periods, growth stages
- **Expected**: 100-500 eras
- **Current Problem**: Created 278,472 eras

### Level C - HMM (Micro: Detailed State Changes)
- **Purpose**: Capture fine-grained operational states
- **Examples**: Daily cycles, control strategy switches
- **Expected**: 1000-5000 eras
- **Current Problem**: Created 1,168,688 eras

## The Real Issue

The algorithms are working correctly but the **greenhouse data is too stable**:
- Continuous 1-minute sampling (1440 points/day)
- Automated control systems maintaining steady conditions
- No major gaps or discontinuities
- Gradual, smooth transitions

This causes:
- PELT to find tiny variations as "changepoints"
- BOCPD to detect noise as "regime changes"
- HMM to create a new state for every minor fluctuation

## Better Parameter Strategy

### Option 1: Enforce Hierarchical Constraints
```yaml
# Level A - True macro eras (minimum 30 days)
"--pelt-min-size", "8640",      # 30 days * 288 samples/day
"--pelt-penalty", "1000",        # Very high penalty for splits

# Level B - Meso eras (minimum 3 days)
"--bocpd-min-size", "864",       # 3 days * 288 samples/day
"--bocpd-lambda", "864",         # Expected run length = 3 days

# Level C - Micro eras (minimum 4 hours)
"--hmm-min-size", "48",          # 4 hours * 12 samples/hour
"--hmm-states", "10"             # Allow more states for variety
```

### Option 2: Signal-Specific Parameters
Different signals need different sensitivities:
```yaml
# For slow-changing signals (temperature, CO2)
"--signal-cols", "air_temp_c,co2_measured_ppm",
"--pelt-min-size", "8640",      # Less sensitive

# For fast-changing signals (actuators, vents)
"--signal-cols", "vent_pos_1_percent,curtain_1_percent",
"--pelt-min-size", "288",        # More sensitive
```

### Option 3: Pre-aggregate Before Detection
Instead of 5-minute resampling, use:
- **Level A**: Daily averages
- **Level B**: Hourly averages  
- **Level C**: 15-minute averages

## Recommended Approach

### 1. Create Fixed Hierarchical Eras (Deterministic)
```sql
-- Level A: Monthly eras
INSERT INTO era_labels_level_a (era_id, start_time, end_time, rows)
SELECT 
    ROW_NUMBER() OVER (ORDER BY month) as era_id,
    month as start_time,
    month + INTERVAL '1 month' as end_time,
    COUNT(*) as rows
FROM (
    SELECT DATE_TRUNC('month', time) as month
    FROM sensor_data_merged
    GROUP BY 1
) t;

-- Level B: Weekly eras
INSERT INTO era_labels_level_b (era_id, start_time, end_time, rows)
SELECT 
    ROW_NUMBER() OVER (ORDER BY week) as era_id,
    week as start_time,
    week + INTERVAL '1 week' as end_time,
    COUNT(*) as rows
FROM (
    SELECT DATE_TRUNC('week', time) as week
    FROM sensor_data_merged
    GROUP BY 1
) t;

-- Level C: Daily eras
INSERT INTO era_labels_level_c (era_id, start_time, end_time, rows)
SELECT 
    ROW_NUMBER() OVER (ORDER BY day) as era_id,
    day as start_time,
    day + INTERVAL '1 day' as end_time,
    COUNT(*) as rows
FROM (
    SELECT DATE_TRUNC('day', time) as day
    FROM sensor_data_merged
    GROUP BY 1
) t;
```

### 2. Hybrid Approach (Best of Both)
1. Use algorithms for anomaly detection
2. But enforce minimum era sizes
3. Merge adjacent similar eras post-hoc

### 3. Signal-Aware Detection
```python
# Different parameters for different signal types
slow_signals = ['temperature', 'co2', 'humidity']  # Daily patterns
medium_signals = ['light', 'radiation']             # Hourly patterns  
fast_signals = ['vents', 'curtains', 'heating']    # Minute patterns

# Run era detection separately for each group
# Then combine results hierarchically
```

## Why Current Parameters Fail

Your observation is correct - we're fighting against the algorithms' nature:

1. **Over-constrained**: Setting huge minimum sizes defeats the purpose
2. **Wrong scale**: 5-minute resampling is still too fine for macro analysis
3. **Missing hierarchy**: Each level should build on the previous

## Proper Implementation

```yaml
# docker-compose.yml with hierarchical approach
era_detector_level_a:
  command: [
    "--signal-cols", "dli_sum,outside_temp_c",  # Slow signals only
    "--resample-every", "1h",                    # Hourly for macro
    "--pelt-min-size", "168",                    # 1 week minimum
    "--output-table", "era_labels_level_a"
  ]

era_detector_level_b:
  command: [
    "--parent-eras", "era_labels_level_a",       # Build on Level A
    "--resample-every", "15m",                   # Finer resolution
    "--bocpd-lambda", "96",                      # 1 day expected
    "--output-table", "era_labels_level_b"
  ]

era_detector_level_c:
  command: [
    "--parent-eras", "era_labels_level_b",       # Build on Level B
    "--resample-every", "5m",                    # Original resolution
    "--hmm-states", "24",                        # Hourly states
    "--output-table", "era_labels_level_c"
  ]
```

## Conclusion

You're right to question the current approach. The algorithms are designed for:
- **PELT**: Finding rare, major changes
- **BOCPD**: Detecting regime switches in non-stationary data
- **HMM**: Modeling systems with discrete hidden states

But greenhouse data is:
- Highly regulated and stable
- Continuously controlled
- Gradually changing

So we need to either:
1. Use fixed time windows (monthly/weekly/daily)
2. Pre-process data to enhance changepoints
3. Use domain knowledge to guide detection
4. Accept that greenhouse data may not have natural "eras" in the algorithmic sense

The macro→meso→micro philosophy is correct, but needs adaptation for this specific domain!