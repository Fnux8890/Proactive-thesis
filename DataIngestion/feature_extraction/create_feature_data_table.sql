-- CREATE TABLE FOR GPU-EXTRACTED FEATURES
-- This script creates a table for storing features extracted by the GPU feature extraction pipeline.
-- Features are stored as JSONB for flexibility, with metadata about era and computation time.

BEGIN;

-- 1. Drop the table if it exists (for idempotency in dev/test)
DROP TABLE IF EXISTS feature_data CASCADE;

-- 2. Create the feature_data table
CREATE TABLE feature_data (
    -- Primary identification
    id BIGSERIAL PRIMARY KEY,
    
    -- Era identification
    era_id TEXT NOT NULL,              -- Unique identifier for the era (e.g., "2014-01-15_level_A_era_1")
    era_level TEXT NOT NULL,           -- Era detection level: 'A', 'B', or 'C'
    
    -- Time boundaries for this era
    era_start_time TIMESTAMPTZ NOT NULL,
    era_end_time TIMESTAMPTZ NOT NULL,
    
    -- Feature data
    features JSONB NOT NULL,           -- GPU-extracted features stored as JSONB
    feature_set TEXT NOT NULL,         -- Identifies the feature extraction configuration used
    
    -- Metadata
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),  -- When the features were computed
    compute_duration_ms INTEGER,       -- How long the computation took in milliseconds
    gpu_device TEXT,                   -- Which GPU was used (if applicable)
    
    -- Data quality metrics
    sample_count INTEGER,              -- Number of samples in this era
    missing_data_percentage REAL,      -- Percentage of missing data in the era
    
    -- Versioning and tracking
    pipeline_version TEXT,             -- Version of the feature extraction pipeline
    config_hash TEXT,                  -- Hash of the configuration used
    
    -- Ensure unique features per era and feature set
    CONSTRAINT unique_era_features UNIQUE (era_id, feature_set)
);

-- 3. Create indexes for efficient querying

-- Index for querying by era_id
CREATE INDEX idx_feature_data_era_id ON feature_data (era_id);

-- Index for querying by era_level
CREATE INDEX idx_feature_data_era_level ON feature_data (era_level);

-- Index for time-based queries
CREATE INDEX idx_feature_data_era_times ON feature_data (era_start_time, era_end_time);

-- Index for computation time tracking
CREATE INDEX idx_feature_data_computed_at ON feature_data (computed_at DESC);

-- Index for feature set queries
CREATE INDEX idx_feature_data_feature_set ON feature_data (feature_set);

-- GIN index for JSONB queries on features
CREATE INDEX idx_feature_data_features_gin ON feature_data USING gin (features);

-- 4. Add comments for documentation

COMMENT ON TABLE feature_data IS 'Stores GPU-extracted features for each detected era in the greenhouse data';
COMMENT ON COLUMN feature_data.era_id IS 'Unique identifier for the era, typically includes date and level information';
COMMENT ON COLUMN feature_data.era_level IS 'Era detection level: A (structural changes), B (operational changes), or C (fine-grained changes)';
COMMENT ON COLUMN feature_data.features IS 'JSONB containing all extracted features for this era';
COMMENT ON COLUMN feature_data.feature_set IS 'Identifies which feature extraction configuration was used (e.g., "tsfresh_efficient", "custom_gpu_v1")';
COMMENT ON COLUMN feature_data.computed_at IS 'Timestamp when these features were computed';
COMMENT ON COLUMN feature_data.compute_duration_ms IS 'Time taken to compute features in milliseconds';
COMMENT ON COLUMN feature_data.gpu_device IS 'GPU device identifier used for computation (e.g., "cuda:0", "GeForce RTX 3090")';
COMMENT ON COLUMN feature_data.sample_count IS 'Number of data points in this era';
COMMENT ON COLUMN feature_data.missing_data_percentage IS 'Percentage of missing values in the raw data for this era';
COMMENT ON COLUMN feature_data.pipeline_version IS 'Version of the feature extraction pipeline for reproducibility';
COMMENT ON COLUMN feature_data.config_hash IS 'Hash of the configuration used for feature extraction';

-- 5. Create a view for easy access to latest features per era
CREATE OR REPLACE VIEW latest_features AS
SELECT DISTINCT ON (era_id, feature_set)
    era_id,
    era_level,
    era_start_time,
    era_end_time,
    features,
    feature_set,
    computed_at,
    compute_duration_ms,
    gpu_device,
    sample_count,
    missing_data_percentage
FROM feature_data
ORDER BY era_id, feature_set, computed_at DESC;

COMMENT ON VIEW latest_features IS 'Shows the most recently computed features for each era and feature set combination';

COMMIT;

-- Usage examples:
-- 
-- Insert features:
-- INSERT INTO feature_data (era_id, era_level, era_start_time, era_end_time, features, feature_set)
-- VALUES (
--     '2014-01-15_level_A_era_1',
--     'A',
--     '2014-01-15 00:00:00+00',
--     '2014-01-15 23:59:59+00',
--     '{"mean_temp": 21.5, "std_temp": 2.3, "max_light": 450.2, "total_energy": 1234.5}'::jsonb,
--     'tsfresh_efficient'
-- );
--
-- Query features for a specific era:
-- SELECT * FROM feature_data WHERE era_id = '2014-01-15_level_A_era_1';
--
-- Query all features computed in the last 24 hours:
-- SELECT * FROM feature_data WHERE computed_at > NOW() - INTERVAL '24 hours';
--
-- Get specific feature values using JSONB operators:
-- SELECT era_id, features->>'mean_temp' AS mean_temperature 
-- FROM feature_data 
-- WHERE feature_set = 'tsfresh_efficient';