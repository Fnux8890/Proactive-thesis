    -- check_data_integrity.sql (CTE Version)
    -- This script should return rows ONLY if validation fails.

    WITH failed_checks AS (
        -- Check 1: Look for unreasonable air temperature values
        SELECT
            'Range Error' AS check_type,
            time,
            source_file,
            'air_temp_c' AS column_name,
            air_temp_c::text AS value
        FROM sensor_data
        WHERE air_temp_c < -50 OR air_temp_c > 60 -- Adjust ranges as needed
        LIMIT 1 -- Limit within the sub-select

        UNION ALL

        -- Check 2: Look for NULL timestamps
        SELECT
            'Null Timestamp' AS check_type,
            time, -- Will be NULL here
            source_file,
            'time' AS column_name,
            'NULL' AS value
        FROM sensor_data
        WHERE time IS NULL
        LIMIT 1

        UNION ALL

        -- Check 3: Look for unreasonable humidity values
        SELECT
            'Range Error' AS check_type,
            time,
            source_file,
            'relative_humidity_percent' AS column_name,
            relative_humidity_percent::text AS value
        FROM sensor_data
        WHERE relative_humidity_percent < 0 OR relative_humidity_percent > 105 -- Allow slightly over 100
        LIMIT 1

        -- Add more UNION ALL blocks for other checks here
    )
    -- Select the first row (if any) from all combined failed checks
    SELECT *
    FROM failed_checks
    LIMIT 1; -- Limit the final result to 0 or 1 row