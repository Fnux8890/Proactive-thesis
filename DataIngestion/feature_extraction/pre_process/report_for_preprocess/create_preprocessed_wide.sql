-- CREATE TABLE AND HYPERTABLE FOR PREPROCESSED DATA IN WIDE FORMAT
-- The behaviour mirrors the script used by the pipeline itself
-- ensuring consistent table creation for local experimentation.

BEGIN;

DROP TABLE IF EXISTS preprocessed_wide;

CREATE TABLE preprocessed_wide (
    time TIMESTAMPTZ NOT NULL,
    era_identifier TEXT NOT NULL,
    -- sensor & engineered columns added by preprocess.py
    PRIMARY KEY (time, era_identifier)
);

SELECT create_hypertable(
    'preprocessed_wide',
    'time',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '7 days'
);

COMMIT;
