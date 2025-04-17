-- MERGE SCRIPT - RUN AFTER INGESTION AND OPTIONAL CHECKS

-- Optional: Start a transaction for safety
BEGIN;

-- 1. Create a new table with the same structure as the original
DROP TABLE IF EXISTS sensor_data_merged; -- Drop if it exists from a previous run
CREATE TABLE sensor_data_merged (LIKE sensor_data INCLUDING ALL);

-- 2. Insert the merged data using aggregation
--    Group by 'time' and apply an aggregate function (like MAX) to all other columns.
INSERT INTO sensor_data_merged (
    time, source_system, source_file, format_type, uuid, air_temp_c,
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
    behov, status_str, timer_on, timer_off, dli_sum,
    oenske_ekstra_lys, lampe_timer_on, lampe_timer_off, value
)
SELECT
    time,
    -- Apply MAX() or appropriate aggregate to non-grouping columns
    MAX(source_system), MAX(source_file), MAX(format_type), MAX(uuid), MAX(air_temp_c),
    MAX(air_temp_middle_c), MAX(outside_temp_c), MAX(relative_humidity_percent),
    MAX(humidity_deficit_g_m3), MAX(radiation_w_m2), MAX(light_intensity_lux),
    MAX(light_intensity_umol), MAX(outside_light_w_m2), MAX(co2_measured_ppm),
    MAX(co2_required_ppm), MAX(co2_dosing_status), MAX(co2_status), MAX(rain_status::int)::bool, -- MAX on bool needs cast
    MAX(vent_pos_1_percent), MAX(vent_pos_2_percent), MAX(vent_lee_afd3_percent),
    MAX(vent_wind_afd3_percent), MAX(vent_lee_afd4_percent), MAX(vent_wind_afd4_percent),
    MAX(curtain_1_percent), MAX(curtain_2_percent), MAX(curtain_3_percent),
    MAX(curtain_4_percent), MAX(window_1_percent), MAX(window_2_percent),
    MAX(lamp_grp1_no3_status::int)::bool, MAX(lamp_grp2_no3_status::int)::bool, MAX(lamp_grp3_no3_status::int)::bool,
    MAX(lamp_grp4_no3_status::int)::bool, MAX(lamp_grp1_no4_status::int)::bool, MAX(lamp_grp2_no4_status::int)::bool,
    MAX(measured_status_bool::int)::bool, MAX(heating_setpoint_c), MAX(pipe_temp_1_c),
    MAX(pipe_temp_2_c), MAX(flow_temp_1_c), MAX(flow_temp_2_c),
    MAX(temperature_forecast_c), MAX(sun_radiation_forecast_w_m2),
    MAX(temperature_actual_c), MAX(sun_radiation_actual_w_m2), MAX(vpd_hpa),
    MAX(humidity_deficit_afd3_g_m3), MAX(relative_humidity_afd3_percent),
    MAX(humidity_deficit_afd4_g_m3), MAX(relative_humidity_afd4_percent),
    MAX(behov), MAX(status_str), MAX(timer_on), MAX(timer_off), MAX(dli_sum),
    MAX(oenske_ekstra_lys), MAX(lampe_timer_on), MAX(lampe_timer_off), MAX(value)
FROM
    sensor_data -- Read from the original table with duplicates
GROUP BY
    time; -- Group rows with the same timestamp

-- 3. **Optional but Recommended:** Verify the merged data count
-- SELECT COUNT(*) FROM sensor_data_merged;
-- SELECT COUNT(DISTINCT time) FROM sensor_data; -- These counts should match

-- 4. **Optional:** Replace the original table with the merged one
--    **USE WITH CAUTION - THIS IS DESTRUCTIVE**
-- DROP TABLE sensor_data;
-- ALTER TABLE sensor_data_merged RENAME TO sensor_data;
-- You might need to re-apply constraints, indexes, or hypertable properties
-- Example for TimescaleDB:
-- SELECT create_hypertable('sensor_data', 'time');

-- Optional: Commit the transaction
COMMIT;