# Era Detection Enhancement - Epic Plan

## Overview
This plan addresses the core issue: era detection algorithms are creating 1.88M segments instead of ~5,000 because greenhouse data is too stable for traditional changepoint detection. We need to adapt the pipeline to work with the characteristics of controlled environment agriculture.

## Epic 1: Data Characterization & Analysis
**Goal**: Understand the true nature of greenhouse operational patterns

### User Stories
1. **Analyze Signal Variability**
   - Calculate coefficient of variation for each sensor over different time windows
   - Identify which signals actually have meaningful changes vs noise
   - Output: Signal stability report ranking sensors by variability

2. **Identify Natural Operational Cycles**
   - Detect daily patterns in lamp usage (photoperiod calculation using lamp_grp*_status)
   - Find weekly patterns in control strategies
   - Identify seasonal trends in heating/cooling
   - Output: Operational pattern catalog

3. **Data Quality Assessment**
   - Map missing data patterns
   - Identify sensor malfunctions or maintenance periods
   - Find true data gaps that might indicate operational changes
   - Output: Data quality report with gap analysis

### Deliverables
- SQL queries and Python scripts for pattern analysis
- Jupyter notebook with visualizations
- Recommendation report for signal selection per era level

## Epic 2: Domain-Specific Feature Engineering
**Goal**: Create greenhouse-specific features that capture meaningful operational changes

### User Stories
1. **Calculate Photoperiod Features**
   ```python
   # We CAN calculate this from lamp status!
   photoperiod_hours = (
       df['lamp_grp1_no3_status'] | 
       df['lamp_grp1_no4_status'] | 
       df['lamp_grp2_no3_status'] | 
       df['lamp_grp2_no4_status']
   ).rolling('24H').sum() / 60  # minutes to hours
   ```

2. **Create Thermal Features**
   - Heating degree minutes: `max(0, setpoint - actual_temp)`
   - Night temperature drop: Temperature difference between day/night
   - Temperature ramp rates: Rate of change during transitions

3. **Derive Control Strategy Indicators**
   - Ventilation strategy: Combined vent positions + outside temp
   - Light supplementation strategy: Lamp usage vs outside light
   - Climate integration: How actuators respond to sensors

4. **Calculate Energy Intensity Metrics**
   - Lighting energy proxy: `sum(lamp_status) * time * power_per_lamp`
   - Heating intensity: Pipe temperature × flow (if available)
   - Total energy index: Normalized energy use per DLI achieved

### Deliverables
- Feature engineering functions in `pre_process/core/feature_engineering.py`
- Updated preprocessed_features table with new columns
- Feature importance analysis for era detection

## Epic 3: Hierarchical Data Aggregation Pipeline
**Goal**: Pre-aggregate data at appropriate time scales for each era level

### User Stories
1. **Implement Multi-Resolution Aggregation**
   ```python
   class HierarchicalAggregator:
       def create_aggregated_views(self):
           # Level A: Daily aggregates
           daily_df = self.aggregate_daily(
               focus_on=['dli_sum', 'heating_degree_minutes', 'photoperiod']
           )
           
           # Level B: Hourly aggregates  
           hourly_df = self.aggregate_hourly(
               focus_on=['temp_stability', 'co2_control', 'vpd_management']
           )
           
           # Level C: 15-minute aggregates
           quarter_hourly_df = self.aggregate_quarter_hourly(
               focus_on=['actuator_changes', 'immediate_responses']
           )
   ```

2. **Create Materialized Views**
   - `preprocessed_daily_view` for Level A
   - `preprocessed_hourly_view` for Level B
   - `preprocessed_quarter_hourly_view` for Level C

3. **Add Change Detection Features**
   - Rolling statistics over multiple windows
   - Rate of change indicators
   - Anomaly scores using isolation forest

### Deliverables
- Aggregation module in preprocessing pipeline
- TimescaleDB continuous aggregates
- Performance benchmarks

## Epic 4: Adaptive Era Detection Algorithms
**Goal**: Modify era detection to work with stable greenhouse data

### User Stories
1. **Implement Sensitivity Scaling**
   ```rust
   // In era_detection_rust
   fn calculate_adaptive_penalty(signal_variance: f64, level: &str) -> f64 {
       match level {
           "A" => base_penalty * (1.0 / signal_variance).min(10.0),
           "B" => base_penalty * (1.0 / signal_variance).min(5.0),
           "C" => base_penalty * (1.0 / signal_variance).min(2.0),
           _ => base_penalty
       }
   }
   ```

2. **Add Domain Constraints**
   - Minimum era duration based on biological processes
   - Respect control system cycle times
   - Align with maintenance schedules

3. **Implement Ensemble Detection**
   - Run multiple algorithms with different parameters
   - Vote on changepoints
   - Merge consensus boundaries

### Deliverables
- Enhanced era_detection_rust with adaptive parameters
- Validation scripts comparing detected eras with known events
- Configuration templates for different greenhouse types

## Epic 5: Hierarchical Era Relationships
**Goal**: Ensure era levels follow macro→meso→micro hierarchy

### User Stories
1. **Implement Parent-Child Constraints**
   ```sql
   -- Ensure Level B eras don't cross Level A boundaries
   CREATE OR REPLACE FUNCTION validate_era_hierarchy() 
   RETURNS TRIGGER AS $$
   BEGIN
       IF EXISTS (
           SELECT 1 FROM era_labels_level_a a
           WHERE NEW.start_time < a.end_time 
           AND NEW.end_time > a.start_time
           GROUP BY a.era_id
           HAVING COUNT(*) > 1
       ) THEN
           RAISE EXCEPTION 'Era crosses parent boundary';
       END IF;
       RETURN NEW;
   END;
   $$ LANGUAGE plpgsql;
   ```

2. **Create Era Lineage Tracking**
   - Add parent_era_id to child tables
   - Build era family trees
   - Enable drill-down analysis

3. **Implement Cascading Detection**
   - Level A defines macro boundaries
   - Level B detects within Level A eras
   - Level C detects within Level B eras

### Deliverables
- Database schema updates with foreign keys
- Hierarchical detection workflow
- Era relationship visualization tools

## Epic 6: Fallback & Hybrid Strategies
**Goal**: Provide reliable era detection when algorithms fail

### User Stories
1. **Implement Fixed-Window Fallback**
   ```python
   def create_fallback_eras(level: str, data_range: tuple) -> pd.DataFrame:
       if level == 'A':
           return create_monthly_eras(data_range)
       elif level == 'B':
           return create_weekly_eras(data_range)
       else:
           return create_daily_eras(data_range)
   ```

2. **Create Hybrid Detection**
   - Use algorithms for anomaly/event detection
   - Overlay with fixed windows
   - Merge boundaries where they align

3. **Add Manual Override Capability**
   - Allow marking known events (maintenance, crop changes)
   - Incorporate external knowledge
   - Adjust detection around fixed points

### Deliverables
- Fallback era generation functions
- Manual event registration interface
- Hybrid detection pipeline

## Epic 7: Validation & Monitoring
**Goal**: Ensure era detection quality and provide ongoing monitoring

### User Stories
1. **Create Era Quality Metrics**
   - Variance within eras (should be low)
   - Difference between eras (should be significant)
   - Era size distribution (should follow expected pattern)

2. **Implement Validation Pipeline**
   ```python
   class EraValidator:
       def validate_detection_quality(self, eras: pd.DataFrame) -> dict:
           return {
               'total_eras': len(eras),
               'avg_duration': eras['duration'].mean(),
               'size_distribution': eras['duration'].describe(),
               'stability_score': self.calculate_within_era_variance(),
               'separation_score': self.calculate_between_era_distance()
           }
   ```

3. **Add Continuous Monitoring**
   - Alert on excessive era creation
   - Track detection performance over time
   - Compare with expected patterns

### Deliverables
- Validation test suite
- Monitoring dashboard
- Alert configuration

## Implementation Priority & Timeline

### Phase 1 (Weeks 1-2): Foundation
- Epic 1: Data Characterization
- Epic 2: Domain-Specific Features (start)

### Phase 2 (Weeks 3-4): Core Enhancement  
- Epic 2: Complete feature engineering
- Epic 3: Hierarchical Aggregation
- Epic 4: Adaptive Algorithms (start)

### Phase 3 (Weeks 5-6): Advanced Features
- Epic 4: Complete adaptive algorithms
- Epic 5: Hierarchical Relationships
- Epic 6: Fallback Strategies

### Phase 4 (Week 7-8): Quality & Deployment
- Epic 7: Validation & Monitoring
- Integration testing
- Documentation & deployment

## Success Metrics
- Level A: 30-50 eras (monthly/seasonal patterns)
- Level B: 200-500 eras (weekly/operational patterns)
- Level C: 2000-5000 eras (daily/control patterns)
- Processing time: < 10 minutes for full pipeline
- GPU feature extraction: < 1 minute per era level

## Technical Debt to Address
1. Update preprocessed_features to properly use era_identifier
2. Fix table naming inconsistency (era_label_level_* vs era_labels_level_*)
3. Optimize database indexes for era-based queries
4. Add comprehensive logging to era detection

## Risk Mitigation
- Keep existing pipeline operational during enhancement
- Test each epic independently before integration
- Maintain backwards compatibility with existing features
- Document all changes thoroughly