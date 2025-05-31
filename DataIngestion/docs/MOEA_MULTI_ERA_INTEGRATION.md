# MOEA Multi-Era Integration Guide

## Overview

This guide explains how to integrate multi-level era detection results into the MOEA optimization process for improved performance and context-aware optimization.

## Current State

The MOEA optimizers (CPU and GPU) currently:
- Use pre-trained surrogate models (LightGBM)
- Evaluate objectives based on decision variables
- Do NOT use era labels or temporal segmentation
- Treat all time periods uniformly

## Benefits of Multi-Era Integration

### 1. Temporal Context Awareness
- **Level A (PELT)**: Seasonal strategies (weeks/months)
- **Level B (BOCPD)**: Operational patterns (days/weeks)  
- **Level C (HMM)**: Fine control (hours/days)

### 2. Adaptive Optimization
- Adjust control strategies based on era characteristics
- Tighten bounds during stable periods
- Increase responsiveness during transitions

### 3. Hierarchical Objectives
- Energy efficiency optimized at macro scale (Level A)
- Plant growth at operational scale (Level B)
- Quality control at micro scale (Level C)

## Implementation Steps

### Step 1: Train Models with Multi-Level Features

Enable multi-level features when training surrogate models:

```bash
# Set environment variables
export USE_MULTI_LEVEL_FEATURES=true
export FEATURE_TABLES=tsfresh_features_level_a,tsfresh_features_level_b,tsfresh_features_level_c

# Train models
docker compose up model_builder
```

### Step 2: Add Era Context Provider

Create an era context provider to supply temporal information:

```python
from moea_optimizer.src.core.era_context_provider import EraContextProvider

# Initialize with database connection
db_url = "postgresql://user:pass@host:5432/greenhouse"
era_provider = EraContextProvider(db_url)

# Get context for current time
context = era_provider.get_context()
```

### Step 3: Modify MOEA Configuration

Update the MOEA configuration to use era context:

```toml
# moea_config.toml
[context]
use_era_context = true
era_levels = ["a", "b", "c"]

[optimization.adaptive]
adjust_bounds_by_era = true
adjust_weights_by_era = true
```

### Step 4: Update Problem Definition

Modify the MOEA problem to incorporate era context:

```python
# In MOEAProblem.__init__
if config.context.use_era_context:
    from ..core.era_context_provider import EraContextProvider
    self.era_provider = EraContextProvider(db_url)
    self.era_strategy = MultiEraOptimizationStrategy(self.era_provider)

# In MOEAProblem.evaluate
if self.era_provider:
    era_context = self.era_provider.get_context()
    
    # Adjust bounds based on era
    if self.config.optimization.adaptive.adjust_bounds_by_era:
        adjusted_bounds = self.era_strategy.adjust_bounds_for_era(
            self.bounds, era_context
        )
    
    # Pass era context to surrogate models
    context['era_info'] = era_context
```

## Usage Examples

### Example 1: Era-Aware Optimization Run

```bash
# Run MOEA with era context
docker compose run moea_optimizer_gpu python -m src.cli run \
  --config /app/config/moea_config_era_aware.toml \
  --use-era-context \
  --era-levels a,b,c
```

### Example 2: Analyzing Era Impact

```python
# Compare optimization with and without era context
results_without_era = run_optimization(use_era_context=False)
results_with_era = run_optimization(use_era_context=True)

# Analyze improvements
print(f"Energy savings: {calculate_improvement(results_with_era, results_without_era, 'energy'):.1%}")
print(f"Growth improvement: {calculate_improvement(results_with_era, results_without_era, 'growth'):.1%}")
```

### Example 3: Era-Specific Strategies

```python
# Define strategies for different era types
strategies = {
    'stable_long': {  # Level A stable era
        'update_frequency': 'daily',
        'control_precision': 'coarse',
        'objective_weights': {'energy': 0.4, 'growth': 0.3, 'quality': 0.3}
    },
    'transition_fast': {  # Level C transition
        'update_frequency': 'hourly',
        'control_precision': 'fine',
        'objective_weights': {'energy': 0.2, 'growth': 0.5, 'quality': 0.3}
    }
}
```

## Configuration Options

### Era Context Configuration

```toml
[era_context]
# Which era levels to use
use_levels = ["a", "b", "c"]

# How to combine era information
combination_method = "hierarchical"  # or "weighted", "voting"

# Era feature importance
level_weights = {a = 0.3, b = 0.5, c = 0.2}

# Minimum era duration to consider (hours)
min_era_duration = {a = 168, b = 24, c = 1}
```

### Adaptive Optimization

```toml
[adaptive_optimization]
# Bound adjustment
bound_adjustment = {
    stable = 0.8,      # Tighten bounds by 20% in stable periods
    transition = 1.2   # Loosen bounds by 20% in transitions
}

# Weight adjustment factors
weight_factors = {
    energy = {stable = 1.2, transition = 0.8},
    growth = {stable = 0.9, transition = 1.3},
    quality = {stable = 1.0, transition = 1.1}
}
```

## Performance Monitoring

### Metrics to Track

1. **Optimization Quality**
   - Hypervolume improvement with era context
   - Convergence speed comparison
   - Solution diversity

2. **Computational Efficiency**
   - Context loading time
   - Memory usage with multi-level features
   - Cache hit rates

3. **Practical Benefits**
   - Energy savings by era type
   - Growth consistency across eras
   - Reduced control oscillations

### Logging Era Usage

```python
# Enable detailed era logging
import logging
logging.getLogger('era_context').setLevel(logging.DEBUG)

# Log era transitions
logger.info(f"Era transition detected: {old_era} -> {new_era}")
logger.info(f"Adjusting strategy: {old_strategy} -> {new_strategy}")
```

## Troubleshooting

### Common Issues

1. **Missing Era Data**
   - Ensure era detection has been run for all levels
   - Check database tables: `era_labels_level_a/b/c`

2. **Performance Degradation**
   - Reduce era levels used (e.g., only B and C)
   - Enable era caching in configuration
   - Use simpler combination methods

3. **Inconsistent Results**
   - Verify era alignment across levels
   - Check for gaps in era coverage
   - Validate era duration thresholds

### Debugging Commands

```bash
# Check era data availability
docker compose exec db psql -U postgres -c "
  SELECT level, COUNT(*) as era_count, 
         MIN(start_time) as earliest,
         MAX(end_time) as latest
  FROM (
    SELECT 'A' as level, * FROM era_labels_level_a
    UNION ALL
    SELECT 'B', * FROM era_labels_level_b
    UNION ALL  
    SELECT 'C', * FROM era_labels_level_c
  ) combined
  GROUP BY level;
"

# Test era context provider
docker compose run moea_optimizer_gpu python -c "
from src.core.era_context_provider import EraContextProvider
provider = EraContextProvider('$DATABASE_URL')
print(provider.get_context())
"
```

## Future Enhancements

1. **Online Era Detection**
   - Real-time era detection during optimization
   - Dynamic strategy switching

2. **Era Prediction**
   - Forecast upcoming era transitions
   - Preemptive strategy adjustments

3. **Multi-Facility Optimization**
   - Share era patterns across greenhouses
   - Transfer learning for new facilities

## Conclusion

Integrating multi-level era detection into MOEA optimization provides:
- Context-aware control strategies
- Improved optimization performance
- Better adaptation to changing conditions

The hierarchical nature of era detection (macro/meso/micro scales) aligns well with multi-objective optimization, enabling different objectives to be prioritized at appropriate time scales.