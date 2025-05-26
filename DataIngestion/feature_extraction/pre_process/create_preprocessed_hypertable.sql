-- CREATE TABLE AND HYPERTABLE FOR PREPROCESSED DATA
-- This script should be run before the preprocess pipeline executes.
-- It creates a table for storing preprocessed data from any era, with a flexible JSONB column for features.

BEGIN;

-- 1. Drop the table if it exists (for idempotency in dev/test)
DROP TABLE IF EXISTS preprocessed_features;

-- 2. Create the new table with a hybrid schema
CREATE TABLE preprocessed_features (
    time TIMESTAMPTZ NOT NULL,
    era_identifier TEXT NOT NULL,
    air_temp_c REAL,
    relative_humidity_percent REAL,
    co2_measured_ppm REAL,
    light_intensity_umol REAL,
    radiation_w_m2 REAL,
    total_lamps_on REAL, -- Assuming REAL, adjust if INTEGER
    dli_sum REAL,
    vpd_hpa REAL,
    heating_setpoint_c REAL,
    co2_status INTEGER, -- Based on cast in database_operations_hybrid.py
    source_file TEXT,
    format_type TEXT,
    extended_features JSONB, -- For any other features
    PRIMARY KEY (time, era_identifier) -- Ensures uniqueness for ON CONFLICT
);

-- 3. Convert to TimescaleDB hypertable
-- (Requires TimescaleDB extension to be installed)
SELECT create_hypertable('preprocessed_features', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '7 days');

COMMIT;

-- Usage:
-- When writing from Python, insert each row as (time, era_identifier, features_dict_as_jsonb)
-- Example insert:
-- INSERT INTO preprocessed_features (time, era_identifier, features) VALUES ('2024-01-01T00:00:00Z', 'Era1', '{"temp": 21.5, "humidity": 55.2}'); 