-- Migration: 01_create_schemas.sql
-- Description: Creates the base schema structure for the data ingestion pipeline
-- Version: 1.0.0
-- Date: 2024-02-25

-- Create schemas if they don't exist
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS timeseries;
CREATE SCHEMA IF NOT EXISTS features;
CREATE SCHEMA IF NOT EXISTS metadata;

-- Create extensions for TimescaleDB and other functionality
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create a table to track migrations
CREATE TABLE IF NOT EXISTS metadata.schema_migrations (
    id SERIAL PRIMARY KEY,
    version VARCHAR(255) NOT NULL,
    description TEXT,
    applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    success BOOLEAN NOT NULL,
    error_message TEXT
);

-- Insert record for this migration
INSERT INTO metadata.schema_migrations 
    (version, description, success) 
VALUES 
    ('01_create_schemas', 'Initial schema setup', TRUE);

-- Create a function to log migration execution
CREATE OR REPLACE FUNCTION metadata.log_migration(
    p_version VARCHAR(255),
    p_description TEXT,
    p_success BOOLEAN,
    p_error_message TEXT DEFAULT NULL
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO metadata.schema_migrations
        (version, description, success, error_message)
    VALUES
        (p_version, p_description, p_success, p_error_message);
END;
$$ LANGUAGE plpgsql;

-- Create a table to track data lineage
CREATE TABLE IF NOT EXISTS metadata.data_lineage (
    id SERIAL PRIMARY KEY,
    source_entity VARCHAR(255) NOT NULL,
    target_entity VARCHAR(255) NOT NULL,
    process_name VARCHAR(255) NOT NULL,
    process_version VARCHAR(50),
    transformation_details JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(255) DEFAULT CURRENT_USER
);

-- Create a table to track schema changes
CREATE TABLE IF NOT EXISTS metadata.schema_registry (
    id SERIAL PRIMARY KEY,
    entity_name VARCHAR(255) NOT NULL,
    schema_definition JSONB NOT NULL,
    version INTEGER NOT NULL,
    valid_from TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    valid_to TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(255) DEFAULT CURRENT_USER,
    CONSTRAINT unique_entity_version UNIQUE (entity_name, version)
);

-- Create indexes for better performance
CREATE INDEX idx_schema_migrations_version ON metadata.schema_migrations (version);
CREATE INDEX idx_data_lineage_source ON metadata.data_lineage (source_entity);
CREATE INDEX idx_data_lineage_target ON metadata.data_lineage (target_entity);
CREATE INDEX idx_schema_registry_entity ON metadata.schema_registry (entity_name);
CREATE INDEX idx_schema_registry_valid_period ON metadata.schema_registry (valid_from, valid_to);

COMMENT ON SCHEMA staging IS 'Raw data before processing';
COMMENT ON SCHEMA timeseries IS 'Processed time series data';
COMMENT ON SCHEMA features IS 'Extracted features from time series data';
COMMENT ON SCHEMA metadata IS 'Metadata about schemas, processes, and lineage'; 