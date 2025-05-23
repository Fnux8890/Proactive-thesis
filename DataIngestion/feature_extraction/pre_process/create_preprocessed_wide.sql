-- CREATE TABLE AND HYPERTABLE FOR PREPROCESSED DATA IN WIDE FORMAT
-- Mirrors the behaviour of the previous JSONB hypertable script
-- but with primitive columns for direct tsfresh querying.

BEGIN;

DROP TABLE IF EXISTS preprocessed_wide;

CREATE TABLE preprocessed_wide (
    time TIMESTAMPTZ NOT NULL,
    era_identifier TEXT NOT NULL,
    -- sensor & engineered columns (DOUBLE PRECISION / INTEGER / BOOLEAN)
    PRIMARY KEY (time, era_identifier)
);

SELECT create_hypertable(
    'preprocessed_wide',
    'time',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '7 days'
);

COMMIT;
