# Era Detection Improvement Plan

## Current Issues

### 1. Data Characteristics
- **Sampling Rate**: 1 reading/minute (1440/day) - extremely dense
- **Stability**: Greenhouse environment is highly controlled, minimal abrupt changes
- **No Gaps**: Continuous data collection from 2013-12-01 to 2016-09-08

### 2. Algorithm Mismatch
Current algorithms detect too many "changepoints" because:
- **PELT**: Designed for detecting rare, significant changes (e.g., equipment failure)
- **BOCPD**: Expects non-stationary data with regime changes
- **HMM**: Creates new states for minor fluctuations

Result: 1.88M total eras instead of expected ~5,500

### 3. Parameter Issues
Even with adjusted parameters:
- Setting huge minimum sizes defeats algorithm purpose
- Algorithms fight against stable greenhouse data
- No hierarchical relationship between levels

## Proposed Solutions

### Solution 1: Pre-aggregation Before Era Detection

Modify the preprocessing pipeline to aggregate data appropriately for each level:

```python
# In pre_process/core/processing_steps.py
class HierarchicalAggregator:
    """Aggregate data at different time scales for era detection."""
    
    def aggregate_for_level(self, df: pd.DataFrame, level: str) -> pd.DataFrame:
        if level == 'A':
            # Daily aggregates for macro analysis
            return df.resample('1D').agg({
                'air_temp_c': ['mean', 'std'],
                'co2_measured_ppm': ['mean', 'std'],
                'dli_sum': 'sum',
                'total_lamps_on': 'mean',
                'heating_setpoint_c': 'mean'
            })
        elif level == 'B':
            # Hourly aggregates for meso analysis
            return df.resample('1H').agg({
                'air_temp_c': 'mean',
                'co2_measured_ppm': 'mean',
                'light_intensity_umol': 'mean',
                'vpd_hpa': 'mean',
                'vent_pos_1_percent': 'mean'
            })
        else:  # Level C
            # 15-minute aggregates for micro analysis
            return df.resample('15T').mean()
```

### Solution 2: Signal-Specific Era Detection

Modify era_detection_rust to group signals by behavior:

```rust
// In era_detection_rust/src/optimal_signals.rs
impl OptimalSignals {
    pub fn get_signals_for_level(&self, level: &str) -> Vec<String> {
        match level {
            "A" => vec![
                // Slow-changing environmental signals
                "dli_sum".to_string(),
                "outside_temp_c".to_string(),
                "heating_degree_days".to_string(),  // New derived feature
            ],
            "B" => vec![
                // Medium-speed control signals
                "air_temp_c".to_string(),
                "co2_measured_ppm".to_string(),
                "relative_humidity_percent".to_string(),
                "vpd_hpa".to_string(),
            ],
            "C" => vec![
                // Fast-changing actuator signals
                "vent_pos_1_percent".to_string(),
                "curtain_1_percent".to_string(),
                "lamp_grp1_no3_status".to_string(),
                "co2_dosing_status".to_string(),
            ],
            _ => self.get_primary_signals(),
        }
    }
}
```

### Solution 3: Hierarchical Era Detection

Implement parent-child relationships between era levels:

```rust
// In era_detection_rust/src/main.rs
#[derive(Parser, Debug)]
struct Cli {
    // ... existing fields ...
    
    /// Parent era table to constrain detection
    #[clap(long)]
    parent_era_table: Option<String>,
    
    /// Ensure child eras don't cross parent boundaries
    #[clap(long, default_value_t = true)]
    respect_parent_boundaries: bool,
}

// In the main processing logic
async fn detect_eras_hierarchical(
    df: DataFrame,
    parent_eras: Option<Vec<Era>>,
    args: &Cli,
) -> Result<Vec<Era>> {
    if let Some(parents) = parent_eras {
        // Process each parent era independently
        let mut all_child_eras = Vec::new();
        
        for parent in parents {
            let subset = df.filter(
                col("time").gt_eq(lit(parent.start_time))
                .and(col("time").lt(lit(parent.end_time)))
            )?;
            
            let child_eras = detect_eras_for_subset(subset, args)?;
            all_child_eras.extend(child_eras);
        }
        
        Ok(all_child_eras)
    } else {
        // No parent constraints, process normally
        detect_eras_standard(df, args)
    }
}
```

### Solution 4: Domain-Aware Features

Add greenhouse-specific derived features to preprocessed_features:

```sql
-- Add to preprocessed_features table
ALTER TABLE preprocessed_features ADD COLUMN IF NOT EXISTS heating_degree_hours REAL;
ALTER TABLE preprocessed_features ADD COLUMN IF NOT EXISTS photoperiod_hours REAL;
ALTER TABLE preprocessed_features ADD COLUMN IF NOT EXISTS night_temperature_drop REAL;
ALTER TABLE preprocessed_features ADD COLUMN IF NOT EXISTS morning_ramp_rate REAL;

-- Update extended_features JSONB with domain-specific metrics
UPDATE preprocessed_features
SET extended_features = extended_features || 
    jsonb_build_object(
        'heating_degree_hours', GREATEST(0, 20 - air_temp_c),
        'photoperiod_hours', CASE 
            WHEN total_lamps_on > 0 THEN 1 
            ELSE 0 
        END,
        'night_temp_drop', CASE 
            WHEN EXTRACT(HOUR FROM time) BETWEEN 0 AND 6 
            THEN heating_setpoint_c - air_temp_c 
            ELSE 0 
        END
    );
```

### Solution 5: Fixed-Window Fallback

Create deterministic eras when algorithms fail:

```sql
-- Create fixed monthly eras for Level A
CREATE OR REPLACE FUNCTION create_fixed_eras_level_a() RETURNS void AS $$
BEGIN
    DELETE FROM era_labels_level_a;
    
    INSERT INTO era_labels_level_a (era_id, start_time, end_time, rows)
    SELECT 
        ROW_NUMBER() OVER (ORDER BY month) as era_id,
        month as start_time,
        (month + INTERVAL '1 month')::timestamp as end_time,
        COUNT(*) as rows
    FROM (
        SELECT DATE_TRUNC('month', time) as month
        FROM preprocessed_features
        GROUP BY 1
    ) t;
END;
$$ LANGUAGE plpgsql;

-- Similar for weekly (Level B) and daily (Level C)
```

## Implementation Steps

### 1. Update Preprocessing Pipeline

```yaml
# In data_processing_config.json
{
  "aggregation_levels": {
    "level_a": {
      "interval": "1D",
      "method": "mean",
      "signals": ["temperature", "co2", "dli"]
    },
    "level_b": {
      "interval": "1H",
      "method": "mean",
      "signals": ["all_environmental"]
    },
    "level_c": {
      "interval": "15T",
      "method": "mean",
      "signals": ["all"]
    }
  }
}
```

### 2. Modify Docker Compose

```yaml
# Three separate era detection services
era_detector_level_a:
  extends: era_detector
  command: [
    "--db-table", "preprocessed_features_daily",
    "--pelt-min-size", "7",      # 7 days minimum
    "--output-table", "era_labels_level_a",
    "--signal-cols", "dli_sum,heating_degree_days",
    "--algorithm", "pelt"
  ]

era_detector_level_b:
  extends: era_detector
  depends_on:
    era_detector_level_a:
      condition: service_completed_successfully
  command: [
    "--db-table", "preprocessed_features_hourly",
    "--parent-era-table", "era_labels_level_a",
    "--bocpd-lambda", "168",     # 1 week expected
    "--output-table", "era_labels_level_b",
    "--algorithm", "bocpd"
  ]

era_detector_level_c:
  extends: era_detector
  depends_on:
    era_detector_level_b:
      condition: service_completed_successfully
  command: [
    "--db-table", "preprocessed_features_15min",
    "--parent-era-table", "era_labels_level_b",
    "--hmm-states", "24",        # Hourly patterns
    "--output-table", "era_labels_level_c",
    "--algorithm", "hmm"
  ]
```

### 3. Update Data Source Integration Documentation

The current DATA_SOURCE_INTEGRATION.md is mostly accurate but needs updates:

1. **Era Detection Philosophy**: Add section explaining macro/meso/micro hierarchy
2. **Preprocessed Features**: Document new domain-specific features
3. **Aggregation Tables**: Document new aggregated tables for each level
4. **Era Relationships**: Explain parent-child era constraints

## Expected Outcomes

With these improvements:
- **Level A**: 30-40 monthly/seasonal eras
- **Level B**: 150-200 weekly operational patterns  
- **Level C**: 1000-1500 daily cycles

This aligns with the intended macro→meso→micro philosophy while respecting greenhouse data characteristics.

## CUDA Memcheck Integration

To ensure GPU feature extraction works correctly with new era structures:

```bash
# Add to docker-compose GPU services
environment:
  CUDA_MEMCHECK: ${CUDA_MEMCHECK:-0}
  
# Run with memory checking
CUDA_MEMCHECK=1 docker compose up gpu_feature_extraction_level_a
```

The test_cuda_memcheck.sh script can validate:
- No memory leaks with varying era sizes
- No race conditions in parallel processing
- Proper initialization of all GPU memory
- No out-of-bounds access with different data volumes