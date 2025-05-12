-- MERGE SCRIPT - NO DOWNSAMPLING (USES LATEST RECORD PER MINUTE) AND COMPLETE TIME SERIES
-- REVISED BASED ON VERIFICATION SCRIPT OUTPUT AND USER FEEDBACK

-- Optional: Start a transaction for safety
BEGIN;

-- 0. Configuration
-- Target interval is '1 minute'.

-- 1. Create a new table with an explicit structure, EXCLUDING the dead columns
DROP TABLE IF EXISTS sensor_data_merged; 

CREATE TABLE sensor_data_merged (
    "time" timestamp with time zone NOT NULL,
    source_system text,
    source_file text,
    format_type text,
    uuid text,
    lamp_group character varying,
    air_temp_c double precision,
    air_temp_middle_c double precision,
    outside_temp_c double precision,
    relative_humidity_percent double precision,
    humidity_deficit_g_m3 double precision,
    radiation_w_m2 double precision,
    light_intensity_lux double precision,
    light_intensity_umol double precision,
    outside_light_w_m2 double precision,
    co2_measured_ppm double precision,
    co2_required_ppm double precision,
    co2_dosing_status double precision, 
    co2_status double precision,        
    rain_status boolean,                
    vent_pos_1_percent double precision,
    vent_pos_2_percent double precision,
    vent_lee_afd3_percent double precision,
    vent_wind_afd3_percent double precision,
    vent_lee_afd4_percent double precision,
    vent_wind_afd4_percent double precision,
    curtain_1_percent double precision,
    curtain_2_percent double precision,
    curtain_3_percent double precision,
    curtain_4_percent double precision,
    window_1_percent double precision,
    window_2_percent double precision,
    lamp_grp1_no3_status boolean,       
    lamp_grp2_no3_status boolean,       
    lamp_grp3_no3_status boolean,       
    lamp_grp4_no3_status boolean,       
    lamp_grp1_no4_status boolean,       
    lamp_grp2_no4_status boolean,       
    measured_status_bool boolean,       
    heating_setpoint_c double precision,
    pipe_temp_1_c double precision,
    pipe_temp_2_c double precision,
    flow_temp_1_c double precision,
    flow_temp_2_c double precision,
    temperature_forecast_c double precision,
    sun_radiation_forecast_w_m2 double precision,
    temperature_actual_c double precision,
    sun_radiation_actual_w_m2 double precision,
    vpd_hpa double precision,
    humidity_deficit_afd3_g_m3 double precision,
    relative_humidity_afd3_percent double precision,
    humidity_deficit_afd4_g_m3 double precision,
    relative_humidity_afd4_percent double precision,
    dli_sum double precision 
    -- Dead columns (behov, status_str, timer_on, timer_off, oenske_ekstra_lys, lampe_timer_on, lampe_timer_off, value) are omitted.
);

-- 2. Convert sensor_data_merged to a TimescaleDB hypertable FIRST
--    Ensure TimescaleDB is installed and enabled in your database.
SELECT create_hypertable('sensor_data_merged', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day');

-- 3. Determine the full time range from the original sensor_data
--    and generate a complete series of timestamps for the target interval.
WITH time_range AS (
    SELECT
        MIN(time) as min_t,
        MAX(time) as max_t
    FROM sensor_data -- Assuming 'sensor_data' is your source table
    WHERE time IS NOT NULL 
),
complete_time_series AS (
    SELECT generate_series(
        date_trunc('minute', (SELECT min_t FROM time_range)),
        (SELECT max_t FROM time_range),
        INTERVAL '1 minute' 
    ) AS minute_interval
    WHERE (SELECT min_t FROM time_range) IS NOT NULL AND (SELECT max_t FROM time_range) IS NOT NULL 
),
-- 4. Select the LATEST record from sensor_data for each 1-minute interval.
--    This avoids downsampling by aggregation if multiple records exist within a minute.
latest_sensor_data_per_minute AS (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY date_trunc('minute', time) ORDER BY time DESC) as rn
    FROM sensor_data
    WHERE time IS NOT NULL
)
-- 5. Insert the merged data, joining the latest record per minute with the complete time series.
INSERT INTO sensor_data_merged (
    "time", source_system, source_file, format_type, uuid, lamp_group, air_temp_c,
    air_temp_middle_c, outside_temp_c, relative_humidity_percent,
    humidity_deficit_g_m3, radiation_w_m2, light_intensity_lux,
    light_intensity_umol, outside_light_w_m2, co2_measured_ppm,
    co2_required_ppm, co2_dosing_status, co2_status, rain_status,
    vent_pos_1_percent, vent_pos_2_percent, vent_lee_afd3_percent,
    vent_wind_afd3_percent, vent_lee_afd4_percent, vent_wind_afd4_percent,
    curtain_1_percent, curtain_2_percent, curtain_3_percent,
    curtain_4_percent, window_1_percent, window_2_percent,
    lamp_grp1_no3_status, lamp_grp2_no3_status, lamp_grp3_no3_status,
    lamp_grp4_no3_status, lamp_grp1_no4_status, lamp_grp2_no4_status,
    measured_status_bool, heating_setpoint_c, pipe_temp_1_c,
    pipe_temp_2_c, flow_temp_1_c, flow_temp_2_c,
    temperature_forecast_c, sun_radiation_forecast_w_m2,
    temperature_actual_c, sun_radiation_actual_w_m2, vpd_hpa,
    humidity_deficit_afd3_g_m3, relative_humidity_afd3_percent,
    humidity_deficit_afd4_g_m3, relative_humidity_afd4_percent,
    dli_sum
)
SELECT
    cts.minute_interval, 
    lsd.source_system, lsd.source_file, lsd.format_type, lsd.uuid, lsd.lamp_group,
    -- Apply outlier filtering for specific columns if needed, directly on the selected record's value
    CASE WHEN lsd.air_temp_c < 60 AND lsd.air_temp_c > -40 THEN lsd.air_temp_c ELSE NULL END,
    lsd.air_temp_middle_c, lsd.outside_temp_c, lsd.relative_humidity_percent,
    lsd.humidity_deficit_g_m3, lsd.radiation_w_m2, lsd.light_intensity_lux,
    lsd.light_intensity_umol, 
    lsd.outside_light_w_m2, 
    CASE WHEN lsd.co2_measured_ppm >= 0 AND lsd.co2_measured_ppm < 5000 THEN lsd.co2_measured_ppm ELSE NULL END,
    lsd.co2_required_ppm, 
    -- Direct casting from source if types are compatible or source is text '0'/'1'/NULL for numeric/boolean
    CAST(lsd.co2_dosing_status AS DOUBLE PRECISION), 
    CAST(lsd.co2_status AS DOUBLE PRECISION),        
    CAST(lsd.rain_status AS BOOLEAN),                
    lsd.vent_pos_1_percent, lsd.vent_pos_2_percent, lsd.vent_lee_afd3_percent,
    lsd.vent_wind_afd3_percent, lsd.vent_lee_afd4_percent, lsd.vent_wind_afd4_percent,
    lsd.curtain_1_percent, lsd.curtain_2_percent, lsd.curtain_3_percent,
    lsd.curtain_4_percent, lsd.window_1_percent, lsd.window_2_percent,
    CAST(lsd.lamp_grp1_no3_status AS BOOLEAN),       
    CAST(lsd.lamp_grp2_no3_status AS BOOLEAN),       
    CAST(lsd.lamp_grp3_no3_status AS BOOLEAN),       
    CAST(lsd.lamp_grp4_no3_status AS BOOLEAN),       
    CAST(lsd.lamp_grp1_no4_status AS BOOLEAN),       
    CAST(lsd.lamp_grp2_no4_status AS BOOLEAN),       
    CAST(lsd.measured_status_bool AS BOOLEAN),       
    lsd.heating_setpoint_c, lsd.pipe_temp_1_c,
    lsd.pipe_temp_2_c, lsd.flow_temp_1_c, lsd.flow_temp_2_c,
    lsd.temperature_forecast_c, lsd.sun_radiation_forecast_w_m2,
    lsd.temperature_actual_c, lsd.sun_radiation_actual_w_m2, lsd.vpd_hpa,
    lsd.humidity_deficit_afd3_g_m3, lsd.relative_humidity_afd3_percent,
    lsd.humidity_deficit_afd4_g_m3, lsd.relative_humidity_afd4_percent,
    -- DLI Calculation based on the selected latest record for the minute
    SUM(
        CASE 
            WHEN lsd.light_intensity_umol = 'NaN'::double precision THEN 0.0 
            ELSE COALESCE(lsd.light_intensity_umol, 0.0) 
        END * 60.0 / 1000000.0
    ) OVER (PARTITION BY date_trunc('day', cts.minute_interval)) AS dli_sum
FROM
    complete_time_series cts
LEFT JOIN
    latest_sensor_data_per_minute lsd ON cts.minute_interval = date_trunc('minute', lsd.time) AND lsd.rn = 1
;

-- 6. **Optional but Recommended:** Verify the merged data
-- SELECT * FROM sensor_data_merged ORDER BY time LIMIT 100;
-- SELECT COUNT(*) FROM sensor_data_merged;

-- Optional: Commit the transaction
COMMIT;

