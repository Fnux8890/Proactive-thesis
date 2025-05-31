# DataFrame Fragmentation Fix

## Issue
The multi_level_data_loader.py was generating PerformanceWarnings due to DataFrame fragmentation when adding multiple columns one by one in a loop.

## Root Cause
The original code at line 232-233 was:
```python
for name, values in new_features.items():
    df[name] = values
```

This pattern causes pandas to repeatedly reallocate memory for the DataFrame, leading to:
- Memory fragmentation
- Poor performance
- Multiple warning messages

## Solution
Replaced the loop with a single `pd.concat()` operation:
```python
if new_features:
    new_features_df = pd.DataFrame(new_features, index=df.index)
    df = pd.concat([df, new_features_df], axis=1)
```

## Benefits
1. **Performance**: Single memory allocation instead of multiple
2. **No Warnings**: Eliminates the fragmentation warnings
3. **Cleaner Code**: More idiomatic pandas usage
4. **Better Memory Usage**: Reduced memory fragmentation

## Other Locations Checked
- Line 134: `base_df['base_level'] = base_level` - This is fine as it's a single column assignment, not in a loop
- Line 244: `df_copy['source_level'] = level.upper()` - Also fine, single assignment

## Performance Impact
This change can significantly improve performance when adding many features, especially with large datasets. The improvement is proportional to the number of features being added.

## Testing
After this fix, the model_builder should run without fragmentation warnings:
```bash
docker compose run model_builder
```

No functional changes were made - the output remains identical, just more efficient.