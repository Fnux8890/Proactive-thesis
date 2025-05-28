# Complete Hybrid Storage Migration Guide

## Current Status
- **Preprocessing**: Still using JSONB-only (`preprocessed_features` table)
- **Era Detection**: Updated to support BOTH JSONB and hybrid tables

## Migration Path

### Phase 1: Parallel Testing (Recommended)
Run both storage approaches in parallel to validate results:

1. **Keep existing JSONB pipeline running**
2. **Add hybrid table and test**:
```bash
# Create hybrid table
docker exec -it dataingest-db-1 psql -U postgres -d postgres <<EOF
$(cat feature_extraction/pre_process/create_preprocessed_hypertable_hybrid.sql)
EOF

# Run preprocessing with hybrid storage (one-time test)
# Modify preprocess.py temporarily to use database_operations_hybrid
```

3. **Test era detection with both**:
```bash
# JSONB table
./era_detector --db-table preprocessed_features

# Hybrid table  
./era_detector --db-table preprocessed_features_hybrid
```

### Phase 2: Full Migration

#### Step 1: Update Preprocessing
```python
# In preprocess.py, change line 22:
from database_operations_hybrid import save_to_timescaledb_hybrid as save_to_timescaledb

# Update the save call (around line 400+):
db_config = {
    'host': os.getenv('DB_HOST', 'db'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'postgres'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres')
}
save_to_timescaledb(df, era_identifier, db_config, table_name='preprocessed_features_hybrid')
```

#### Step 2: Migrate Historical Data
```python
# One-time migration script
from database_operations_hybrid import migrate_jsonb_to_hybrid

db_config = {
    'host': 'db',
    'port': '5432',
    'database': 'postgres',
    'user': 'postgres',
    'password': 'postgres'
}

migrate_jsonb_to_hybrid(
    db_config,
    source_table='preprocessed_features',
    target_table='preprocessed_features_hybrid'
)
```

#### Step 3: Update Era Detection Default
```rust
// In era_detection_rust/src/main.rs, update default table:
#[clap(long, default_value = "preprocessed_features_hybrid")]
db_table: String,
```

#### Step 4: Update Docker Compose
```yaml
era_detector:
  environment:
    - ERA_TABLE=preprocessed_features_hybrid  # Add this
```

### Phase 3: Cleanup (After Validation)

1. **Verify data integrity**:
```sql
-- Compare row counts
SELECT 
    'JSONB' as storage,
    COUNT(*) as rows,
    MIN(time) as earliest,
    MAX(time) as latest
FROM preprocessed_features
UNION ALL
SELECT 
    'Hybrid' as storage,
    COUNT(*) as rows,
    MIN(time) as earliest,
    MAX(time) as latest
FROM preprocessed_features_hybrid;
```

2. **Archive JSONB table** (don't delete immediately):
```sql
ALTER TABLE preprocessed_features RENAME TO preprocessed_features_jsonb_archive;
```

3. **Rename hybrid to standard name** (optional):
```sql
ALTER TABLE preprocessed_features_hybrid RENAME TO preprocessed_features;
-- Update era detection default back to 'preprocessed_features'
```

## Performance Gains

### Query Performance
```sql
-- JSONB approach (slow)
EXPLAIN ANALYZE
SELECT time, (features->>'air_temp_c')::float 
FROM preprocessed_features 
WHERE time >= '2014-01-01' AND time < '2014-01-02';

-- Hybrid approach (fast)
EXPLAIN ANALYZE
SELECT time, air_temp_c 
FROM preprocessed_features_hybrid 
WHERE time >= '2014-01-01' AND time < '2014-01-02';
```

Expected improvements:
- **Query speed**: 5-10x faster
- **Storage size**: 50% reduction
- **Index efficiency**: Native B-tree indexes
- **Era detection**: 3-5x faster processing

## Rollback Plan

If issues arise:
1. Change preprocessing import back to original
2. Update era detection table parameter
3. All data remains in original JSONB table

## Monitoring

Track performance improvements:
```sql
-- Create monitoring view
CREATE VIEW storage_performance AS
SELECT 
    pg_size_pretty(pg_total_relation_size('preprocessed_features')) as jsonb_size,
    pg_size_pretty(pg_total_relation_size('preprocessed_features_hybrid')) as hybrid_size,
    (SELECT COUNT(*) FROM preprocessed_features) as jsonb_rows,
    (SELECT COUNT(*) FROM preprocessed_features_hybrid) as hybrid_rows;
```