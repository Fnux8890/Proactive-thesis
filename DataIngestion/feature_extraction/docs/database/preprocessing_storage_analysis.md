# Preprocessing Data Storage Analysis: JSONB vs Alternatives

## Current Approach: JSONB Storage

### Table Schema
```sql
CREATE TABLE preprocessed_features (
    time TIMESTAMPTZ NOT NULL,
    era_identifier TEXT NOT NULL,
    features JSONB NOT NULL
);
```

### How Data is Stored
The preprocessing pipeline stores all sensor readings and calculated features as key-value pairs in a single JSONB column. Example:
```json
{
    "air_temp_c": 21.5,
    "relative_humidity_percent": 55.2,
    "co2_measured_ppm": 850,
    "total_lamps_on": 4,
    "dli_sum": 12.5,
    // ... many more fields
}
```

## Analysis of Current JSONB Approach

### Pros
1. **Schema Flexibility**: Can add new features without ALTER TABLE operations
2. **Simplicity**: Single table structure, easy to understand
3. **Development Speed**: No need to manage migrations for new features
4. **Variable Feature Sets**: Different eras can have different features without NULL columns
5. **TimescaleDB Compatible**: Works well with hypertables for time-series optimization

### Cons
1. **Query Performance**: 
   - JSONB field extraction is slower than native columns
   - Cannot use column-based compression effectively
   - Index creation on JSONB fields is limited and less efficient
   
2. **Type Safety**: 
   - No database-level type validation
   - Casting required on every query (e.g., `(features->>'air_temp_c')::float`)
   - Runtime errors possible from type mismatches

3. **Storage Overhead**: 
   - JSONB stores field names with every row (redundant storage)
   - Larger storage footprint compared to native columns
   - Text representation of numbers takes more space

4. **Query Complexity**: 
   - Complex queries become verbose with JSONB extraction
   - Aggregations require casting and extraction in every query
   - JOIN operations on JSONB fields are inefficient

5. **Tool Compatibility**: 
   - Many BI tools and query builders don't handle JSONB well
   - Standard SQL analytics become more complex

## Alternative Approaches

### 1. Traditional Relational Schema (Raw Columns)

#### Implementation
```sql
CREATE TABLE preprocessed_features_relational (
    time TIMESTAMPTZ NOT NULL,
    era_identifier TEXT NOT NULL,
    air_temp_c FLOAT,
    relative_humidity_percent FLOAT,
    co2_measured_ppm FLOAT,
    radiation_w_m2 FLOAT,
    light_intensity_umol FLOAT,
    heating_setpoint_c FLOAT,
    pipe_temp_1_c FLOAT,
    pipe_temp_2_c FLOAT,
    flow_temp_1_c FLOAT,
    flow_temp_2_c FLOAT,
    vent_lee_afd3_percent FLOAT,
    vent_wind_afd3_percent FLOAT,
    total_lamps_on FLOAT,
    dli_sum FLOAT,
    vpd_hpa FLOAT,
    humidity_deficit_g_m3 FLOAT,
    -- ... more columns as needed
    PRIMARY KEY (time, era_identifier)
);
```

#### Pros
- **Best Query Performance**: Direct column access, no extraction overhead
- **Type Safety**: Database enforces data types
- **Compression**: TimescaleDB can compress columns efficiently
- **Indexing**: Can create efficient indexes on any column
- **Standard SQL**: Works with all tools and frameworks

#### Cons
- **Schema Rigidity**: Adding columns requires migrations
- **NULL Storage**: Many NULL values if features vary by era
- **Development Overhead**: Schema changes need coordination
- **Column Limit**: PostgreSQL has a limit of ~1600 columns per table

### 2. Parquet File Storage

#### Implementation
```python
# Save each era/segment as Parquet file
df_processed.to_parquet(f"era_{era_id}_segment_{segment_num}.parquet")

# Store metadata in PostgreSQL
CREATE TABLE preprocessed_metadata (
    time_start TIMESTAMPTZ NOT NULL,
    time_end TIMESTAMPTZ NOT NULL,
    era_identifier TEXT NOT NULL,
    file_path TEXT NOT NULL,
    row_count INTEGER,
    features TEXT[],  -- List of available features
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

#### Pros
- **Excellent Compression**: Columnar storage with efficient compression
- **Fast Analytics**: Optimized for analytical queries
- **Schema Evolution**: Each file can have different schemas
- **Parallel Processing**: Easy to process files in parallel
- **Cost Effective**: Cheaper storage for large datasets

#### Cons
- **No Real-time Queries**: Must load files to query
- **File Management**: Need to manage file lifecycle
- **Transaction Support**: No ACID guarantees across files
- **Join Complexity**: Difficult to join across time periods

### 3. Hybrid Approach (Recommended)

#### Implementation
Combine the best of both worlds:

```sql
-- Core sensor data in columns for fast queries
CREATE TABLE preprocessed_core_features (
    time TIMESTAMPTZ NOT NULL,
    era_identifier TEXT NOT NULL,
    -- Most commonly queried features as columns
    air_temp_c FLOAT,
    relative_humidity_percent FLOAT,
    co2_measured_ppm FLOAT,
    light_intensity_umol FLOAT,
    total_lamps_on INTEGER,
    -- Extended features in JSONB for flexibility
    extended_features JSONB,
    PRIMARY KEY (time, era_identifier)
);

-- Create indexes on common query patterns
CREATE INDEX idx_air_temp ON preprocessed_core_features(air_temp_c);
CREATE INDEX idx_time_era ON preprocessed_core_features(time, era_identifier);
CREATE INDEX idx_extended_features ON preprocessed_core_features USING GIN(extended_features);
```

#### Migration Code Example
```python
def save_to_hybrid_table(df: pd.DataFrame, era_identifier: str, engine):
    # Define core columns
    core_columns = [
        'air_temp_c', 'relative_humidity_percent', 'co2_measured_ppm',
        'light_intensity_umol', 'total_lamps_on'
    ]
    
    # Prepare core data
    core_df = pd.DataFrame()
    core_df['time'] = df.index
    core_df['era_identifier'] = era_identifier
    
    # Add core columns
    for col in core_columns:
        if col in df.columns:
            core_df[col] = df[col]
        else:
            core_df[col] = None
    
    # Extended features as JSONB
    extended_cols = [col for col in df.columns if col not in core_columns]
    extended_features = []
    for _, row in df.iterrows():
        ext_feat = {col: row[col] for col in extended_cols if pd.notna(row[col])}
        extended_features.append(json.dumps(ext_feat))
    
    core_df['extended_features'] = extended_features
    
    # Write to database
    core_df.to_sql('preprocessed_core_features', engine, 
                   if_exists='append', index=False, method='multi')
```

## Recommendations for Migration

### 1. Immediate Steps (Minimal Disruption)
1. **Create Materialized Views** for common query patterns:
```sql
CREATE MATERIALIZED VIEW preprocessed_features_expanded AS
SELECT 
    time,
    era_identifier,
    (features->>'air_temp_c')::float AS air_temp_c,
    (features->>'relative_humidity_percent')::float AS relative_humidity_percent,
    -- ... other common fields
FROM preprocessed_features;

CREATE INDEX ON preprocessed_features_expanded(time, era_identifier);
```

2. **Add GIN Indexes** for JSONB queries:
```sql
CREATE INDEX idx_features_gin ON preprocessed_features USING GIN(features);
```

### 2. Short-term Migration (Hybrid Approach)
1. Implement the hybrid schema alongside existing JSONB table
2. Modify `save_to_timescaledb()` to write to both tables
3. Update downstream queries to use new table
4. Gradually phase out JSONB-only table

### 3. Long-term Considerations
1. **Analyze Feature Usage**: Track which features are queried most
2. **Partition by Era**: Use PostgreSQL partitioning for large datasets
3. **Archive Old Data**: Move old eras to Parquet for long-term storage
4. **Monitor Performance**: Track query performance improvements

## Impact on Downstream Stages

### Era Detection (Rust)
- Currently extracts specific JSONB fields in SQL
- Would benefit significantly from columnar storage
- Recommended: Use hybrid approach with core features as columns

### Feature Extraction
- Reads all data into DataFrame anyway
- Performance impact minimal for batch processing
- Could benefit from Parquet for large historical analyses

## Conclusion

The current JSONB approach prioritizes flexibility over performance. For a production system processing time-series sensor data, I recommend:

1. **Immediate**: Add indexes and materialized views
2. **Short-term**: Migrate to hybrid approach (core columns + JSONB for extended)
3. **Long-term**: Consider Parquet for historical data archival

The hybrid approach balances performance needs with development flexibility while maintaining backward compatibility.