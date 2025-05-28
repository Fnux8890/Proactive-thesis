# DB Utils Optimization Report

## Current State Analysis

### 1. SQLAlchemyPostgresConnector Audit

**Strengths:**
- Clean abstraction with BaseDBConnector interface
- Supports dependency injection (can accept external engine)
- Basic error handling in place
- Connection testing on initialization

**Weaknesses Identified:**
1. **No connection pooling configuration** - Using default SQLAlchemy settings
2. **Inefficient write_dataframe implementation** - Uses basic `to_sql()` without optimization
3. **No performance monitoring/timing**
4. **Multiple duplicate files** across directories (feature/, feature-gpu/, pre_process/)
5. **No batch size control** for large dataframes
6. **No method parameter in to_sql()** call

### 2. Connection Pooling Analysis
The current implementation creates engine with: `create_engine(db_url)` with no pooling parameters.

SQLAlchemy defaults:
- pool_size: 5
- max_overflow: 10
- No pool recycling configured

### 3. Write Performance Issues
The `write_dataframe` method uses basic `df.to_sql()` without:
- `method='multi'` for batch inserts
- `chunksize` parameter for large dataframes
- Transaction management
- Progress tracking

## Recommended Optimizations

### Phase 1: Immediate Improvements

1. **Add connection pooling configuration**
```python
self.engine = create_engine(
    db_url,
    pool_size=20,
    max_overflow=40,
    pool_timeout=30,
    pool_recycle=1800,  # Recycle connections after 30 minutes
    pool_pre_ping=True  # Test connections before using
)
```

2. **Optimize write_dataframe method**
```python
def write_dataframe(self, df: pd.DataFrame, table_name: str,
                   if_exists: str = "append", index: bool = False,
                   index_label: str | None = None, chunksize: int = 5000) -> None:
    # Add timing
    start_time = time.time()
    
    # Use method='multi' for PostgreSQL
    df.to_sql(
        table_name, 
        self.engine, 
        if_exists=if_exists,
        index=index, 
        index_label=index_label,
        method='multi',
        chunksize=chunksize
    )
    
    elapsed = time.time() - start_time
    logging.info(f"Wrote {len(df)} rows to {table_name} in {elapsed:.2f} seconds")
```

3. **Add performance monitoring**
```python
import time
import logging

def _log_performance(operation: str, start_time: float, rows: int = None):
    elapsed = time.time() - start_time
    msg = f"Operation '{operation}' completed in {elapsed:.2f} seconds"
    if rows:
        msg += f" ({rows} rows, {rows/elapsed:.0f} rows/sec)"
    logging.info(msg)
```

4. **Consolidate duplicate files**
- Create a single shared db_utils module
- Update imports across all modules

### Phase 2: Advanced Optimizations (If Needed)

1. **Implement psycopg3 connector for better performance**
2. **Add support for COPY operations for bulk inserts**
3. **Implement connection context managers**
4. **Add query result caching for repeated queries**
5. **Support for async operations**

## Performance Benchmarks Needed

Before proceeding with Phase 2, we should:
1. Measure current write performance with large datasets (100K+ rows)
2. Test with optimized settings
3. Compare with psycopg3 direct implementation
4. Profile memory usage during operations

## Next Steps

1. Implement Phase 1 optimizations
2. Add comprehensive logging
3. Run performance benchmarks
4. Consolidate duplicate db_utils files
5. Consider Phase 2 if performance is still inadequate