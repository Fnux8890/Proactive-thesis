# DB Utils Migration Guide

## Overview
This guide helps you migrate from the current db_utils implementation to the optimized version with minimal disruption.

## Key Improvements in Optimized Version

1. **Connection Pooling**: Configurable pool size, overflow, timeout, and recycling
2. **Performance Monitoring**: Built-in timing and statistics tracking
3. **Optimized Writes**: Uses `method='multi'` and chunking for efficient batch inserts
4. **Better Error Handling**: More robust connection management with context managers
5. **SQLAlchemy 2.0 Ready**: Uses future flag for forward compatibility

## Migration Steps

### Step 1: Backup Current Implementation
```bash
# Backup existing files
cp feature/db_utils.py feature/db_utils_backup.py
cp feature-gpu/db_utils.py feature-gpu/db_utils_backup.py
cp pre_process/db_utils.py pre_process/db_utils_backup.py
```

### Step 2: Test Optimized Version
1. Copy `db_utils_optimized.py` to a test location
2. Run the built-in test: `python db_utils_optimized.py`
3. Verify connection and basic operations work

### Step 3: Gradual Migration

#### Option A: Drop-in Replacement (Minimal Changes)
The optimized version is backwards compatible. Simply replace the existing files:

```python
# No code changes needed in consuming modules
# The API remains the same:
connector = SQLAlchemyPostgresConnector(
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
    db_name=DB_NAME
)
```

#### Option B: Enable Optimizations (Recommended)
Update consuming code to use new features:

```python
# Use optimized connection pooling
connector = SQLAlchemyPostgresConnector(
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
    db_name=DB_NAME,
    pool_size=20,  # Increased from default 5
    max_overflow=40,  # Increased from default 10
    enable_performance_logging=True
)

# Use optimized write method
connector.write_dataframe(
    df,
    table_name="features",
    chunksize=10000,  # Larger chunks for better performance
    method='multi'  # Batch inserts
)
```

### Step 4: Update Import Statements
Since the optimized version includes logging, update modules that need it:

```python
# Add to modules using db_utils
import logging
logging.basicConfig(level=logging.INFO)
```

### Step 5: Consolidate Duplicate Files
1. Create a shared module location:
   ```bash
   mkdir -p feature_extraction/shared
   cp db_utils_optimized.py feature_extraction/shared/db_utils.py
   ```

2. Update imports in all modules:
   ```python
   # Old import
   from db_utils import SQLAlchemyPostgresConnector
   
   # New import
   from ..shared.db_utils import SQLAlchemyPostgresConnector
   ```

## Performance Testing

### Before Migration
```python
import time
from db_utils import SQLAlchemyPostgresConnector

# Test current performance
connector = SQLAlchemyPostgresConnector(...)
start = time.time()
df = pd.DataFrame({'data': range(100000)})
connector.write_dataframe(df, 'test_table')
print(f"Current version: {time.time() - start:.2f} seconds")
```

### After Migration
```python
# Test optimized performance
from db_utils_optimized import SQLAlchemyPostgresConnector

connector = SQLAlchemyPostgresConnector(..., enable_performance_logging=True)
# Same test...
# Check logs for detailed metrics
stats = connector.get_performance_stats()
print(stats)
```

## Rollback Plan

If issues occur:
1. Restore backup files: `cp feature/db_utils_backup.py feature/db_utils.py`
2. Restart services
3. Report issues with performance stats

## Expected Performance Gains

Based on the optimizations:
- **Connection time**: 20-50% faster with pooling
- **Write operations**: 3-10x faster with batch inserts
- **Large datasets**: Significantly better with chunking
- **Concurrent operations**: Much better with larger pool size

## Monitoring After Migration

1. Check logs for performance metrics
2. Monitor database connections: 
   ```sql
   SELECT count(*) FROM pg_stat_activity WHERE datname = 'your_db';
   ```
3. Use performance stats API:
   ```python
   stats = connector.get_performance_stats()
   ```

## Common Issues and Solutions

### Issue: Too many connections
**Solution**: Reduce pool_size and max_overflow parameters

### Issue: Connection timeout
**Solution**: Increase pool_timeout or add more connections

### Issue: Memory usage with large DataFrames
**Solution**: Reduce chunksize parameter in write_dataframe

## Next Steps

After successful migration:
1. Remove backup files
2. Update documentation
3. Consider Phase 2 optimizations (psycopg3) if needed
4. Set up performance monitoring dashboards