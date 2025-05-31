# Era ID Column Fix

## Problem

The model training pipeline was failing with the error:
```
column "era_id" does not exist
```

This occurred because the feature extraction pipeline created tables with an 'index' column instead of 'era_id', but the model training code expected 'era_id'.

## Root Cause

The `tsfresh` library's `extract_features` function returns a DataFrame with the ID column as the index. When this was saved to the database:
1. The index was saved as a column named 'index'
2. The model training code expected this column to be named 'era_id'

## Solution

Modified both `PostgreSQLDataLoader` and `MultiLevelDataLoader` to handle tables with either column name:

1. **Column Detection**: Added SQL queries to check which columns exist in the table
2. **Dynamic Query Generation**: 
   - If 'era_id' exists, use it directly
   - If only 'index' exists, rename it to 'era_id' in the SELECT query
3. **Column Cleanup**: Remove duplicate 'index' column if both exist after renaming

### Code Changes

#### PostgreSQLDataLoader (train_lightgbm_surrogate.py)
```python
# Check which columns exist
col_check_query = f"""
SELECT column_name
FROM information_schema.columns
WHERE table_name = '{self.config.features_table}'
AND column_name IN ('era_id', 'index')
"""

# Use appropriate column
if has_era_id:
    query = f"SELECT * FROM {table} ORDER BY era_id"
elif has_index:
    query = f'SELECT "index" AS era_id, * FROM {table} ORDER BY "index"'
```

#### MultiLevelDataLoader (multi_level_data_loader.py)
Same approach applied for each feature table level (A, B, C).

## Testing

Use the test script to verify the fix:
```bash
cd DataIngestion
python test_era_id_fix.py
```

## Impact

- No changes needed to feature extraction pipeline
- Model training can now handle both column naming conventions
- Backward compatible with existing tables that have 'era_id'

## Future Considerations

Consider standardizing the column naming in the feature extraction pipeline to always use 'era_id' to avoid this ambiguity.