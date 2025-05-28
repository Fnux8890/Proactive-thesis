# Storage Alternatives for Preprocessed Features

## Current Approach: JSONB Only
```sql
CREATE TABLE preprocessed_features (
    time TIMESTAMPTZ NOT NULL,
    era_identifier TEXT NOT NULL,
    features JSONB NOT NULL
);
```

### Pros:
- Maximum flexibility for varying schemas
- Easy to add new features without schema changes
- Handles sparse data well

### Cons:
- 30-50% query performance overhead
- 2-3x storage overhead (field names repeated)
- Type casting required for every value
- Limited indexing capabilities

## Option 1: Hybrid Approach (Recommended)
Core columns as native PostgreSQL columns + JSONB for extended features.

### Pros:
- 80% faster queries on core columns
- Type safety for critical fields
- Better compression with TimescaleDB
- Maintains flexibility for new features
- Easy migration path

### Cons:
- Schema changes needed for new core columns
- Slightly more complex insertion logic

### Implementation:
```sql
-- See create_preprocessed_hypertable_hybrid.sql
```

## Option 2: Pure Columnar (Traditional)
All features as individual columns.

### Pros:
- Best query performance
- Native PostgreSQL indexing
- Strong type safety
- Optimal storage with compression

### Cons:
- Schema changes for every new feature
- Difficult to handle sparse data
- Complex migrations
- Limited to ~1600 columns

### Example:
```sql
CREATE TABLE preprocessed_features_columnar (
    time TIMESTAMPTZ NOT NULL,
    era_identifier TEXT NOT NULL,
    air_temp_c REAL,
    relative_humidity_percent REAL,
    -- ... 50+ more columns
    PRIMARY KEY (time, era_identifier)
);
```

## Option 3: Parquet Files
Store preprocessed data as Parquet files.

### Pros:
- Excellent compression (10-20x)
- Columnar storage format
- Fast analytical queries
- Easy archival

### Cons:
- No real-time updates
- Requires external query engine
- Complex integration with era detection
- File management overhead

### Implementation:
```python
# Save to Parquet
df.to_parquet(
    f"preprocessed_era_{era_identifier}.parquet",
    engine='pyarrow',
    compression='snappy',
    index=True
)

# Read from Parquet
df = pd.read_parquet(
    "preprocessed_era_*.parquet",
    engine='pyarrow'
)
```

## Option 4: Hybrid with Parquet Archive
Use hybrid table for recent data, archive to Parquet.

### Workflow:
1. Recent data (< 30 days): Hybrid table
2. Historical data: Compressed Parquet files
3. Query router to combine sources

### Pros:
- Optimal performance for recent data
- Minimal storage for archives
- Scales to years of data

### Cons:
- Complex query routing
- Requires maintenance jobs

## Performance Comparison

| Approach | Query Speed | Storage Size | Flexibility | Complexity |
|----------|-------------|--------------|-------------|------------|
| JSONB Only | Slow (1x) | Large (3x) | Excellent | Low |
| Hybrid | Fast (5x) | Medium (1.5x) | Good | Medium |
| Pure Columnar | Fastest (10x) | Small (1x) | Poor | High |
| Parquet Files | Fast (8x) | Smallest (0.3x) | Good | High |

## Recommendation

**Short-term**: Implement the hybrid approach
- Minimal code changes
- Immediate performance benefits
- Maintains backward compatibility

**Long-term**: Add Parquet archival
- Archive data older than 90 days
- Keep recent data in hybrid table
- Use Apache Arrow for fast queries

## Migration Steps

1. Create hybrid table (schema provided)
2. Run migration script (provided)
3. Update `save_to_timescaledb()` to use hybrid approach
4. Update era detection to query native columns
5. Monitor performance improvements
6. Gradually migrate historical data

## Quick Wins (No Code Changes)

1. Add indexes on JSONB fields:
```sql
CREATE INDEX idx_features_air_temp ON preprocessed_features ((features->>'air_temp_c'));
CREATE INDEX idx_features_gin ON preprocessed_features USING GIN (features);
```

2. Create materialized view for common queries:
```sql
CREATE MATERIALIZED VIEW preprocessed_core_features AS
SELECT 
    time,
    era_identifier,
    (features->>'air_temp_c')::REAL as air_temp_c,
    (features->>'relative_humidity_percent')::REAL as relative_humidity_percent,
    (features->>'co2_measured_ppm')::REAL as co2_measured_ppm,
    (features->>'total_lamps_on')::REAL as total_lamps_on
FROM preprocessed_features;

CREATE INDEX idx_mv_time ON preprocessed_core_features (time);
```

3. Enable TimescaleDB compression:
```sql
ALTER TABLE preprocessed_features SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'era_identifier'
);
SELECT add_compression_policy('preprocessed_features', INTERVAL '7 days');
```