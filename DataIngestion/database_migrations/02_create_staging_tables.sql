-- Migration: 02_create_staging_tables.sql
-- Description: Creates staging tables for raw data ingestion
-- Version: 1.0.0
-- Date: 2024-02-25

-- Common column definitions for raw data tables
-- These tables will hold data in a flexible structure before processing

-- Aarslev staging table for raw data
CREATE TABLE IF NOT EXISTS staging.raw_aarslev (
    id BIGSERIAL PRIMARY KEY,
    source_file TEXT NOT NULL,
    record_number INTEGER NOT NULL,
    raw_data JSONB NOT NULL,  -- Store raw data in flexible JSONB format
    
    -- Metadata columns
    _ingested_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    _processed BOOLEAN DEFAULT FALSE,
    _processed_at TIMESTAMP WITH TIME ZONE,
    _status TEXT DEFAULT 'new', -- 'new', 'processed', 'error'
    _error_message TEXT,
    _hash TEXT, -- Hash of raw data for deduplication
    
    -- Allow filtering by date range even with JSONB data
    measurement_time TIMESTAMP WITH TIME ZONE,
    
    CONSTRAINT unique_aarslev_record UNIQUE (source_file, record_number)
);

-- Knudjepsen staging table for raw data
CREATE TABLE IF NOT EXISTS staging.raw_knudjepsen (
    id BIGSERIAL PRIMARY KEY,
    source_file TEXT NOT NULL,
    record_number INTEGER NOT NULL,
    raw_data JSONB NOT NULL,  -- Store raw data in flexible JSONB format
    
    -- Metadata columns
    _ingested_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    _processed BOOLEAN DEFAULT FALSE,
    _processed_at TIMESTAMP WITH TIME ZONE,
    _status TEXT DEFAULT 'new', -- 'new', 'processed', 'error'
    _error_message TEXT,
    _hash TEXT, -- Hash of raw data for deduplication
    
    -- Allow filtering by date range even with JSONB data
    measurement_time TIMESTAMP WITH TIME ZONE,
    
    CONSTRAINT unique_knudjepsen_record UNIQUE (source_file, record_number)
);

-- Create a generic staging table for other data sources
CREATE TABLE IF NOT EXISTS staging.raw_generic (
    id BIGSERIAL PRIMARY KEY,
    source TEXT NOT NULL, -- Source identifier (e.g., 'weather_service', 'external_sensor')
    source_file TEXT,
    record_number INTEGER,
    raw_data JSONB NOT NULL,
    
    -- Metadata columns
    _ingested_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    _processed BOOLEAN DEFAULT FALSE,
    _processed_at TIMESTAMP WITH TIME ZONE,
    _status TEXT DEFAULT 'new',
    _error_message TEXT,
    _hash TEXT,
    
    measurement_time TIMESTAMP WITH TIME ZONE,
    
    CONSTRAINT unique_generic_record UNIQUE (source, source_file, record_number)
);

-- Create table to track file ingestion status
CREATE TABLE IF NOT EXISTS staging.ingestion_files (
    id BIGSERIAL PRIMARY KEY,
    source TEXT NOT NULL, -- 'aarslev', 'knudjepsen', etc.
    file_path TEXT NOT NULL,
    file_size BIGINT,
    file_hash TEXT, -- File hash for integrity check
    record_count INTEGER,
    
    -- Status tracking
    status TEXT DEFAULT 'pending', -- 'pending', 'processing', 'completed', 'error'
    error_message TEXT,
    
    -- Timestamps
    discovered_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ingestion_started_at TIMESTAMP WITH TIME ZONE,
    ingestion_completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Processing stats
    processing_time_ms INTEGER,
    
    CONSTRAINT unique_ingestion_file UNIQUE (source, file_path)
);

-- Create indexes for better query performance
CREATE INDEX idx_raw_aarslev_status ON staging.raw_aarslev (_status);
CREATE INDEX idx_raw_aarslev_measurement_time ON staging.raw_aarslev (measurement_time);
CREATE INDEX idx_raw_aarslev_processed ON staging.raw_aarslev (_processed);

CREATE INDEX idx_raw_knudjepsen_status ON staging.raw_knudjepsen (_status);
CREATE INDEX idx_raw_knudjepsen_measurement_time ON staging.raw_knudjepsen (measurement_time);
CREATE INDEX idx_raw_knudjepsen_processed ON staging.raw_knudjepsen (_processed);

CREATE INDEX idx_raw_generic_status ON staging.raw_generic (_status);
CREATE INDEX idx_raw_generic_source ON staging.raw_generic (source);
CREATE INDEX idx_raw_generic_measurement_time ON staging.raw_generic (measurement_time);

CREATE INDEX idx_ingestion_files_status ON staging.ingestion_files (status);
CREATE INDEX idx_ingestion_files_source ON staging.ingestion_files (source);

-- Create JSONB indexes for common query patterns
CREATE INDEX idx_raw_aarslev_jsonb_path ON staging.raw_aarslev USING gin (raw_data jsonb_path_ops);
CREATE INDEX idx_raw_knudjepsen_jsonb_path ON staging.raw_knudjepsen USING gin (raw_data jsonb_path_ops);
CREATE INDEX idx_raw_generic_jsonb_path ON staging.raw_generic USING gin (raw_data jsonb_path_ops);

-- Store information about this migration in the registry
SELECT metadata.log_migration(
    '02_create_staging_tables',
    'Created staging tables for raw data ingestion',
    TRUE
); 