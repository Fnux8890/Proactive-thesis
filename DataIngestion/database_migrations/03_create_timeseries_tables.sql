-- Migration: 03_create_timeseries_tables.sql
-- Description: Creates time series tables for processed data using TimescaleDB hypertables
-- Version: 1.0.0
-- Date: 2024-02-25

-- Aarslev time series table
CREATE TABLE IF NOT EXISTS timeseries.aarslev_data (
    id BIGSERIAL,
    time TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Core measurements from Aarslev greenhouse data
    -- Based on the data structure in the DataOutline.md
    relative_humidity DOUBLE PRECISION,
    co2_level DOUBLE PRECISION,
    co2_status INTEGER,
    air_temperature DOUBLE PRECISION,
    flow_temperature DOUBLE PRECISION,
    
    -- Additional derived or calculated fields
    vapor_pressure_deficit DOUBLE PRECISION,
    dew_point DOUBLE PRECISION,
    heat_index DOUBLE PRECISION,
    
    -- External factors
    solar_radiation DOUBLE PRECISION,
    outside_temperature DOUBLE PRECISION,
    
    -- Metadata and tracking fields
    source_file TEXT NOT NULL,
    original_record_id BIGINT, -- Reference to staging record
    
    -- Processing info
    _processed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    _feature_extracted BOOLEAN DEFAULT FALSE,
    _feature_extracted_at TIMESTAMP WITH TIME ZONE,
    _data_quality_score DOUBLE PRECISION,
    _outlier_score DOUBLE PRECISION,
    
    -- Primary key consisting of id and time
    PRIMARY KEY (id, time)
);

-- Create TimescaleDB hypertable for Aarslev data
SELECT create_hypertable('timeseries.aarslev_data', 'time', if_not_exists => TRUE);

-- Knudjepsen time series table
CREATE TABLE IF NOT EXISTS timeseries.knudjepsen_data (
    id BIGSERIAL,
    time TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Core measurements - adjust as needed based on actual data structure
    temperature DOUBLE PRECISION,
    humidity DOUBLE PRECISION,
    co2_level DOUBLE PRECISION,
    light_intensity DOUBLE PRECISION,
    irrigation_flow DOUBLE PRECISION,
    
    -- Additional derived fields
    vapor_pressure_deficit DOUBLE PRECISION,
    daily_light_integral DOUBLE PRECISION,
    
    -- Metadata and tracking fields
    source_file TEXT NOT NULL,
    original_record_id BIGINT,
    
    -- Processing info
    _processed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    _feature_extracted BOOLEAN DEFAULT FALSE,
    _feature_extracted_at TIMESTAMP WITH TIME ZONE,
    _data_quality_score DOUBLE PRECISION,
    _outlier_score DOUBLE PRECISION,
    
    PRIMARY KEY (id, time)
);

-- Create TimescaleDB hypertable for Knudjepsen data
SELECT create_hypertable('timeseries.knudjepsen_data', 'time', if_not_exists => TRUE);

-- Generic time series table for other data sources
CREATE TABLE IF NOT EXISTS timeseries.generic_data (
    id BIGSERIAL,
    time TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Source identification
    source TEXT NOT NULL, -- e.g., 'weather_service', 'external_sensor'
    source_id TEXT, -- Optional source-specific identifier
    
    -- Flexible data storage as JSONB, but with common fields extracted
    -- for direct querying
    measurement_type TEXT NOT NULL, -- e.g., 'temperature', 'humidity', 'co2'
    value DOUBLE PRECISION NOT NULL,
    unit TEXT,
    
    -- Additional data in flexible format
    metadata JSONB,
    
    -- Tracking fields
    source_file TEXT,
    original_record_id BIGINT,
    
    -- Processing info
    _processed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    _feature_extracted BOOLEAN DEFAULT FALSE,
    _feature_extracted_at TIMESTAMP WITH TIME ZONE,
    _data_quality_score DOUBLE PRECISION,
    
    PRIMARY KEY (id, time)
);

-- Create TimescaleDB hypertable for generic data
SELECT create_hypertable('timeseries.generic_data', 'time', if_not_exists => TRUE);

-- Create a view for easily joining Aarslev and weather data
CREATE OR REPLACE VIEW timeseries.aarslev_with_weather AS
SELECT 
    a.time,
    a.relative_humidity,
    a.co2_level,
    a.co2_status,
    a.air_temperature,
    a.flow_temperature,
    a.vapor_pressure_deficit,
    a.dew_point,
    a.heat_index,
    w.value as outside_temperature
FROM 
    timeseries.aarslev_data a
LEFT JOIN 
    timeseries.generic_data w 
    ON date_trunc('hour', a.time) = date_trunc('hour', w.time)
    AND w.source = 'weather_service'
    AND w.measurement_type = 'temperature';

-- Create table for ingestion results
CREATE TABLE IF NOT EXISTS timeseries.ingestion_results (
    id BIGSERIAL PRIMARY KEY,
    file_path TEXT NOT NULL,
    processed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    success BOOLEAN NOT NULL,
    error_message TEXT,
    source TEXT NOT NULL, -- 'aarslev', 'knudjepsen', etc.
    file_type TEXT NOT NULL, -- 'csv', 'json', 'excel'
    record_count INTEGER,
    processing_time_ms INTEGER,
    data_start_time TIMESTAMP WITH TIME ZONE,
    data_end_time TIMESTAMP WITH TIME ZONE,
    metadata JSONB
);

-- Create indexes for better query performance
CREATE INDEX idx_aarslev_data_time ON timeseries.aarslev_data (time DESC);
CREATE INDEX idx_aarslev_data_feature_extraction ON timeseries.aarslev_data (_feature_extracted);
CREATE INDEX idx_aarslev_data_source_file ON timeseries.aarslev_data (source_file);

CREATE INDEX idx_knudjepsen_data_time ON timeseries.knudjepsen_data (time DESC);
CREATE INDEX idx_knudjepsen_data_feature_extraction ON timeseries.knudjepsen_data (_feature_extracted);
CREATE INDEX idx_knudjepsen_data_source_file ON timeseries.knudjepsen_data (source_file);

CREATE INDEX idx_generic_data_time ON timeseries.generic_data (time DESC);
CREATE INDEX idx_generic_data_source ON timeseries.generic_data (source);
CREATE INDEX idx_generic_data_measurement_type ON timeseries.generic_data (measurement_type);
CREATE INDEX idx_generic_data_source_measurement ON timeseries.generic_data (source, measurement_type);
CREATE INDEX idx_generic_data_feature_extraction ON timeseries.generic_data (_feature_extracted);

CREATE INDEX idx_ingestion_results_file_path ON timeseries.ingestion_results (file_path);
CREATE INDEX idx_ingestion_results_source ON timeseries.ingestion_results (source);
CREATE INDEX idx_ingestion_results_processed_at ON timeseries.ingestion_results (processed_at DESC);

-- Time-based retention policy function
CREATE OR REPLACE FUNCTION timeseries.create_retention_policy(
    p_months INTEGER DEFAULT 60 -- 5 years default retention
)
RETURNS VOID AS $$
BEGIN
    -- Create a policy that drops chunks older than the specified months
    PERFORM add_drop_chunks_policy('timeseries.aarslev_data', 
                                  INTERVAL '1 month' * p_months);
    PERFORM add_drop_chunks_policy('timeseries.knudjepsen_data', 
                                  INTERVAL '1 month' * p_months);
    PERFORM add_drop_chunks_policy('timeseries.generic_data', 
                                  INTERVAL '1 month' * p_months);
END;
$$ LANGUAGE plpgsql;

-- Create compression policies
CREATE OR REPLACE FUNCTION timeseries.create_compression_policy()
RETURNS VOID AS $$
BEGIN
    -- Compress chunks that are older than 7 days, check daily
    PERFORM add_compression_policy('timeseries.aarslev_data', 
                                  INTERVAL '7 days',
                                  if_not_exists => TRUE);
    PERFORM add_compression_policy('timeseries.knudjepsen_data', 
                                  INTERVAL '7 days',
                                  if_not_exists => TRUE);
    PERFORM add_compression_policy('timeseries.generic_data', 
                                  INTERVAL '7 days',
                                  if_not_exists => TRUE);
END;
$$ LANGUAGE plpgsql;

-- Apply retention and compression policies
-- Note: Uncommenting these will actually apply the policies
-- SELECT timeseries.create_retention_policy(60); -- 5 years
-- SELECT timeseries.create_compression_policy();

-- Store information about this migration in the registry
SELECT metadata.log_migration(
    '03_create_timeseries_tables',
    'Created time series tables with TimescaleDB hypertables',
    TRUE
); 