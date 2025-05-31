# DataFrame Fragmentation Fix in Model Builder

## Problem
When running the model_builder in Docker, multiple PerformanceWarnings were generated:
```
PerformanceWarning: DataFrame is highly fragmented. This is usually the result of 
calling `frame.insert` many times, which has poor performance. Consider joining all 
columns at once using pd.concat(axis=1) instead.
```

## Location
- **File**: `model_builder/src/utils/multi_level_data_loader.py`
- **Function**: `_create_cross_level_features()`
- **Lines**: 232-233

## Root Cause
The code was adding multiple columns to a DataFrame one by one in a loop:
```python
for name, values in new_features.items():
    df[name] = values  # This causes fragmentation
```

Each assignment causes pandas to:
1. Allocate new memory for the entire DataFrame
2. Copy all existing data
3. Add the new column
4. Potentially leave fragmented memory

## Solution Implemented
Replace the loop with a single `pd.concat()` operation:
```python
# Add new features to dataframe using concat for better performance
if new_features:
    new_features_df = pd.DataFrame(new_features, index=df.index)
    df = pd.concat([df, new_features_df], axis=1)
```

## Performance Benefits
1. **Single Memory Allocation**: All columns added at once
2. **No Fragmentation**: Contiguous memory layout
3. **Faster Execution**: Especially noticeable with many features
4. **No Warnings**: Clean output logs

## Verification
Run the model builder to verify the warnings are gone:
```bash
docker compose run model_builder
```

## Best Practices
When adding multiple columns to a DataFrame:
- ❌ Don't: Add columns one by one in a loop
- ✅ Do: Collect all new columns and add them with `pd.concat()`
- ✅ Do: Use `pd.DataFrame.assign()` for a few columns
- ✅ Do: Build the complete DataFrame structure upfront when possible

## Other Considerations
The fix maintains 100% compatibility - no functional changes, only performance improvements.