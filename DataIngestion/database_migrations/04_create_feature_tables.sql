-- Migration: 04_create_feature_tables.sql
-- Description: Creates tables for storing extracted features from time series data
-- Version: 1.0.0
-- Date: 2024-02-25

-- Table for extracted features
CREATE TABLE IF NOT EXISTS features.extracted_features (
    id BIGSERIAL PRIMARY KEY,
    
    -- Feature identification
    feature_name TEXT NOT NULL,
    source TEXT NOT NULL, -- 'aarslev', 'knudjepsen', 'weather', etc.
    type TEXT NOT NULL, -- 'statistical', 'temporal', 'derived', etc.
    
    -- Time information
    calculation_start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    calculation_end_time TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Feature value and context
    value DOUBLE PRECISION,
    vector_value DOUBLE PRECISION[], -- For vector-valued features
    text_value TEXT, -- For categorical or text features
    
    -- Metadata
    algorithm TEXT, -- Algorithm used for extraction
    parameters JSONB, -- Parameters used for extraction
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Data quality and evaluation
    quality_score DOUBLE PRECISION,
    reliability_score DOUBLE PRECISION,
    sample_size INTEGER, -- Number of data points used
    
    -- Links to source data
    source_query TEXT, -- Query used to extract the feature
    
    CONSTRAINT unique_feature_extraction UNIQUE (feature_name, source, calculation_start_time, calculation_end_time)
);

-- Table for feature importance
CREATE TABLE IF NOT EXISTS features.feature_importance (
    id BIGSERIAL PRIMARY KEY,
    feature_name TEXT NOT NULL,
    source TEXT NOT NULL,
    
    -- Importance metrics
    importance_score DOUBLE PRECISION NOT NULL,
    rank INTEGER,
    
    -- Context
    target_variable TEXT, -- What objective this feature is important for
    algorithm TEXT, -- Algorithm used to calculate importance
    
    -- Tracking
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT unique_feature_importance UNIQUE (feature_name, source, target_variable)
);

-- Table for feature sets used in models
CREATE TABLE IF NOT EXISTS features.feature_sets (
    id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    
    -- Set properties
    source TEXT NOT NULL, -- 'aarslev', 'knudjepsen', 'combined', etc.
    feature_count INTEGER NOT NULL,
    features TEXT[] NOT NULL, -- Array of feature names
    
    -- Selection information
    selection_method TEXT, -- 'mrmr', 'rfe', 'manual', etc.
    selection_parameters JSONB,
    
    -- Tracking
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by TEXT DEFAULT CURRENT_USER,
    active BOOLEAN DEFAULT TRUE,
    
    CONSTRAINT unique_feature_set_name UNIQUE (name, source)
);

-- Feature metadata for documentation and discovery
CREATE TABLE IF NOT EXISTS features.feature_metadata (
    id BIGSERIAL PRIMARY KEY,
    feature_name TEXT NOT NULL,
    
    -- Documentation
    display_name TEXT,
    description TEXT,
    units TEXT,
    
    -- Categorization
    category TEXT, -- 'climate', 'energy', 'plant_health', etc.
    tags TEXT[],
    
    -- Technical details
    data_type TEXT, -- 'numeric', 'categorical', 'vector', etc.
    calculation_method TEXT,
    formula TEXT,
    source_fields TEXT[],
    
    -- Statistical properties
    typical_range JSONB, -- {"min": 0, "max": 100}
    distribution TEXT, -- 'normal', 'log-normal', etc.
    
    -- Reference information
    reference_url TEXT,
    reference_paper TEXT,
    
    -- Tracking
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT unique_feature_metadata_name UNIQUE (feature_name)
);

-- Feature extraction runs for tracking
CREATE TABLE IF NOT EXISTS features.extraction_runs (
    id BIGSERIAL PRIMARY KEY,
    
    -- Run information
    run_id TEXT NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE,
    status TEXT DEFAULT 'running', -- 'running', 'completed', 'failed'
    
    -- Scope
    data_sources TEXT[],
    time_range_start TIMESTAMP WITH TIME ZONE,
    time_range_end TIMESTAMP WITH TIME ZONE,
    
    -- Results
    features_count INTEGER,
    error_message TEXT,
    
    -- Performance tracking
    processing_time_ms INTEGER,
    cpu_usage DOUBLE PRECISION,
    memory_usage_mb INTEGER,
    
    -- Metadata
    configuration JSONB,
    
    CONSTRAINT unique_extraction_run UNIQUE (run_id)
);

-- Feature time series for tracking changes over time
CREATE TABLE IF NOT EXISTS features.feature_timeseries (
    id BIGSERIAL,
    time TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Feature identification
    feature_name TEXT NOT NULL,
    source TEXT NOT NULL,
    
    -- Value over time
    value DOUBLE PRECISION NOT NULL,
    
    -- Context
    extraction_run_id TEXT,
    window_size INTERVAL,
    
    PRIMARY KEY (id, time)
);

-- Create TimescaleDB hypertable for feature time series
SELECT create_hypertable('features.feature_timeseries', 'time', if_not_exists => TRUE);

-- Create indexes for better query performance
CREATE INDEX idx_extracted_features_name ON features.extracted_features (feature_name);
CREATE INDEX idx_extracted_features_source ON features.extracted_features (source);
CREATE INDEX idx_extracted_features_time ON features.extracted_features (calculation_start_time, calculation_end_time);
CREATE INDEX idx_extracted_features_type ON features.extracted_features (type);

CREATE INDEX idx_feature_importance_name ON features.feature_importance (feature_name);
CREATE INDEX idx_feature_importance_score ON features.feature_importance (importance_score DESC);
CREATE INDEX idx_feature_importance_target ON features.feature_importance (target_variable);

CREATE INDEX idx_feature_sets_name ON features.feature_sets (name);
CREATE INDEX idx_feature_sets_source ON features.feature_sets (source);
CREATE INDEX idx_feature_sets_features ON features.feature_sets USING gin (features);
CREATE INDEX idx_feature_sets_active ON features.feature_sets (active);

CREATE INDEX idx_feature_metadata_name ON features.feature_metadata (feature_name);
CREATE INDEX idx_feature_metadata_category ON features.feature_metadata (category);
CREATE INDEX idx_feature_metadata_tags ON features.feature_metadata USING gin (tags);

CREATE INDEX idx_extraction_runs_status ON features.extraction_runs (status);
CREATE INDEX idx_extraction_runs_time ON features.extraction_runs (start_time DESC);

CREATE INDEX idx_feature_timeseries_name ON features.feature_timeseries (feature_name, source);
CREATE INDEX idx_feature_timeseries_run ON features.feature_timeseries (extraction_run_id);

-- Create a view for top features by importance
CREATE OR REPLACE VIEW features.top_features AS
SELECT 
    fi.feature_name,
    fi.source,
    fi.importance_score,
    fi.target_variable,
    fm.display_name,
    fm.description,
    fm.category,
    fm.units
FROM 
    features.feature_importance fi
JOIN 
    features.feature_metadata fm ON fi.feature_name = fm.feature_name
ORDER BY 
    fi.importance_score DESC;

-- Store information about this migration in the registry
SELECT metadata.log_migration(
    '04_create_feature_tables',
    'Created tables for storing extracted features from time series data',
    TRUE
); 