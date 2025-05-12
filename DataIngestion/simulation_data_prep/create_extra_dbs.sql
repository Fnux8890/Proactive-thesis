-- create_extra_dbs.sql
-- Creates databases needed by Prefect and MLflow.
-- These commands run directly, as CREATE DATABASE cannot be in a function/transaction
-- during initialization via /docker-entrypoint-initdb.d/

\set ON_ERROR_STOP true

-- Attempt to create databases. psql might throw an error if they already exist,
-- but the script should continue with the next command due to how the entrypoint handles it.
-- Alternatively, connect manually and create if needed.
CREATE DATABASE prefect;
CREATE DATABASE mlflow;
CREATE DATABASE greenhouse; -- Add database for Feast offline store

-- Optional: Grant privileges if needed
-- GRANT ALL PRIVILEGES ON DATABASE prefect TO postgres;
-- GRANT ALL PRIVILEGES ON DATABASE mlflow TO postgres;

\echo 'Finished creating prefect and mlflow databases (if they did not exist).' 