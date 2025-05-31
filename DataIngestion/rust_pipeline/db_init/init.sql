-- init.sql

-- Exit on error
\set ON_ERROR_STOP true

-- Create TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create the main table for sensor data
CREATE TABLE IF NOT EXISTS sensor_data (
    -- Core Identification & Timing
    time TIMESTAMPTZ NOT NULL,       -- from timestamp_utc (Must be NOT NULL for hypertable)
    source_system TEXT NULL,         -- from source_system
    source_file TEXT NULL,           -- from source_file
    format_type TEXT NULL,           -- from format_type
    uuid TEXT NULL,                  -- from uuid (used by JSON streams)
    lamp_group VARCHAR(100) NULL,    -- ADDED: from lamp_group_id

    -- Common Environmental Measurements
    air_temp_c DOUBLE PRECISION NULL,
    air_temp_middle_c DOUBLE PRECISION NULL,
    outside_temp_c DOUBLE PRECISION NULL,
    relative_humidity_percent DOUBLE PRECISION NULL,
    humidity_deficit_g_m3 DOUBLE PRECISION NULL,
    radiation_w_m2 DOUBLE PRECISION NULL,
    light_intensity_lux DOUBLE PRECISION NULL,
    light_intensity_umol DOUBLE PRECISION NULL,
    outside_light_w_m2 DOUBLE PRECISION NULL,
    co2_measured_ppm DOUBLE PRECISION NULL,
    co2_required_ppm DOUBLE PRECISION NULL,
    co2_dosing_status DOUBLE PRECISION NULL, -- Assuming float based on column_map
    co2_status DOUBLE PRECISION NULL,        -- Assuming float based on column_map
    rain_status BOOLEAN NULL,

    -- Control System State
    vent_pos_1_percent DOUBLE PRECISION NULL,
    vent_pos_2_percent DOUBLE PRECISION NULL,
    vent_lee_afd3_percent DOUBLE PRECISION NULL,
    vent_wind_afd3_percent DOUBLE PRECISION NULL,
    vent_lee_afd4_percent DOUBLE PRECISION NULL,
    vent_wind_afd4_percent DOUBLE PRECISION NULL,
    curtain_1_percent DOUBLE PRECISION NULL,
    curtain_2_percent DOUBLE PRECISION NULL,
    curtain_3_percent DOUBLE PRECISION NULL,
    curtain_4_percent DOUBLE PRECISION NULL,
    window_1_percent DOUBLE PRECISION NULL,
    window_2_percent DOUBLE PRECISION NULL,
    lamp_grp1_no3_status BOOLEAN NULL,
    lamp_grp2_no3_status BOOLEAN NULL,
    lamp_grp3_no3_status BOOLEAN NULL,
    lamp_grp4_no3_status BOOLEAN NULL,
    lamp_grp1_no4_status BOOLEAN NULL,
    lamp_grp2_no4_status BOOLEAN NULL,
    measured_status_bool BOOLEAN NULL,

    -- Heating & Flow
    heating_setpoint_c DOUBLE PRECISION NULL,
    pipe_temp_1_c DOUBLE PRECISION NULL,
    pipe_temp_2_c DOUBLE PRECISION NULL,
    flow_temp_1_c DOUBLE PRECISION NULL,
    flow_temp_2_c DOUBLE PRECISION NULL,

    -- Forecasts
    temperature_forecast_c DOUBLE PRECISION NULL,
    sun_radiation_forecast_w_m2 DOUBLE PRECISION NULL,
    temperature_actual_c DOUBLE PRECISION NULL,
    sun_radiation_actual_w_m2 DOUBLE PRECISION NULL,

    -- Knudjepsen Specific / Others
    vpd_hpa DOUBLE PRECISION NULL,
    humidity_deficit_afd3_g_m3 DOUBLE PRECISION NULL,
    relative_humidity_afd3_percent DOUBLE PRECISION NULL,
    humidity_deficit_afd4_g_m3 DOUBLE PRECISION NULL,
    relative_humidity_afd4_percent DOUBLE PRECISION NULL,
    behov INTEGER NULL,
    status_str TEXT NULL,
    timer_on INTEGER NULL,
    timer_off INTEGER NULL,
    dli_sum DOUBLE PRECISION NULL,
    oenske_ekstra_lys TEXT NULL,
    lampe_timer_on BIGINT NULL,
    lampe_timer_off BIGINT NULL,
    value DOUBLE PRECISION NULL -- Generic value from JSON streams
);

-- Convert the table to a TimescaleDB hypertable, partitioned by time
-- Use SELECT ... IF NOT EXISTS to prevent error if already a hypertable
SELECT create_hypertable('sensor_data', 'time', if_not_exists => TRUE);

-- Optional: Add indexes for common query patterns
-- CREATE INDEX IF NOT EXISTS idx_sensor_data_source_time ON sensor_data (source_system, time DESC);
-- CREATE INDEX IF NOT EXISTS idx_sensor_data_uuid_time ON sensor_data (uuid, time DESC);

-- Grant privileges (adjust user/db names as needed)
-- GRANT ALL PRIVILEGES ON TABLE sensor_data TO your_rust_app_user; 