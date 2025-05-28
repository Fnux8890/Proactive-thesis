# Era Detector Complete Fix Guide

## Initial Problem
The `era_detector` container was exiting immediately with code 0 without any output.

## Root Causes Identified & Fixed

### 1. Missing Command Arguments
**Problem**: The Rust-based era_detector requires command-line arguments but the Dockerfile only specified an ENTRYPOINT without a CMD.
**Solution**: Modified Dockerfile to use environment variables and default table name.

### 2. Logger Initialization Order
**Problem**: Logger was initialized after argument parsing, hiding early error messages.
**Solution**: Moved `env_logger::init()` to the very beginning of `main()` function.

### 3. JSONB Data Structure Mismatch (Main Issue)
**Problem**: The era_detector was failing with "No signal columns selected for resampling aggregation" because the `preprocessed_features` table stores all sensor data in a JSONB column rather than individual columns.

**Details**:
- The `preprocessed_features` table has only 3 columns: `time`, `era_identifier`, and `features` (JSONB)
- All sensor data is stored as key-value pairs inside the `features` JSONB column
- The era_detector was expecting traditional columnar data format

## Solution
Modified `feature_extraction/era_detection_rust/src/db.rs` to detect when loading from `preprocessed_features` table and dynamically extract JSONB fields as columns.

### Changes Made

1. **Updated Cargo.toml** to include `serde_json` dependency:
```toml
postgres = { version = "0.19", features = ["with-chrono-0_4", "with-serde_json-1"] }
serde_json = "1.0"
```

2. **Modified `load_feature_df` function** in `db.rs`:
   - Added special handling for `preprocessed_features` table
   - Uses PostgreSQL's JSONB arrow operator (`->>`) to extract fields
   - Casts extracted values to float for numeric processing
   - Includes comprehensive list of common sensor fields

### SQL Query Structure
The modified query extracts JSONB fields like this:
```sql
SELECT 
    time,
    (features->>'air_temp_c')::float AS air_temp_c,
    (features->>'relative_humidity_percent')::float AS relative_humidity_percent,
    -- ... more fields
FROM "preprocessed_features"
ORDER BY time
```

### Fields Extracted
- Environmental: air_temp_c, relative_humidity_percent, co2_measured_ppm, radiation_w_m2, light_intensity_umol
- Heating: heating_setpoint_c, pipe_temp_1_c, pipe_temp_2_c, flow_temp_1_c, flow_temp_2_c
- Ventilation: vent_lee_afd3_percent, vent_wind_afd3_percent, vent_pos_1_percent, vent_pos_2_percent
- Curtains/Windows: curtain_1-4_percent, window_1-2_percent
- Lighting: lamp_grp1-4_no3-4_status, total_lamps_on, dli_sum
- Other: vpd_hpa, humidity_deficit values, co2 status fields, external conditions

## Testing
After these changes, rebuild and run:
```bash
cd feature_extraction/era_detection_rust
docker build -f dockerfile -t era_detector .
docker compose up --build era_detector
```

## Complete Usage Guide

### Running Era Detector
```bash
# Standard run (uses environment variables from docker-compose.yml)
docker compose up --build era_detector

# Manual run with explicit parameters
docker compose run --rm era_detector \
  --db-dsn "postgresql://postgres:postgres@db:5432/postgres?sslmode=prefer" \
  --db-table "preprocessed_features" \
  --resample-every "5m" \
  --min-coverage "0.9"
```

### Parameters Reference
```
--db-dsn <string>           # Database connection string
--db-table <string>         # Table name (default: "preprocessed_features")
--resample-every <string>   # Resampling interval (default: "5m")
--min-coverage <float>      # Minimum data coverage (default: 0.9)
--signal-cols <col1,col2>   # Specific columns to analyze
--pelt-min-size <int>       # PELT algorithm minimum segment size (default: 48)
--bocpd-lambda <float>      # BOCPD expected run length (default: 200.0)
--hmm-states <int>          # HMM number of states (default: 5)
--hmm-iterations <int>      # HMM training iterations (default: 20)
```

### Debugging Steps

1. **Check if preprocessed data exists:**
   ```bash
   docker compose exec db psql -U postgres -c "\dt"
   docker compose exec db psql -U postgres -c "SELECT COUNT(*) FROM preprocessed_features"
   ```

2. **View detailed logs:**
   ```bash
   docker compose run --rm -e RUST_LOG=debug era_detector
   ```

3. **Check available JSONB fields:**
   ```bash
   docker compose exec db psql -U postgres -c "SELECT jsonb_object_keys(features) FROM preprocessed_features LIMIT 1"
   ```

### Expected Output
When running correctly, you should see:
```
[INFO] Era Detector starting up...
[INFO] Database connection pool initialized successfully
[INFO] Loading preprocessed_features table with JSONB expansion
[INFO] Loaded DataFrame. Shape: (X, Y), Load time: Z.XXs
[INFO] Selected N columns for era detection
[INFO] Running Level A (PELT) detection...
[INFO] Running Level B (BOCPD) detection...
[INFO] Running Level C (HMM) detection...
[INFO] Persisting results to database...
[INFO] Era detection complete!
```

## Alternative: Python Era Detection
If the Rust version has issues, use the Python implementation:
```bash
docker compose up era_detection  # Python version
```

## Future Improvements
1. Consider creating a materialized view that expands JSONB to columns for better performance
2. Make the field list configurable rather than hard-coded
3. Add field type detection to handle non-numeric fields appropriately