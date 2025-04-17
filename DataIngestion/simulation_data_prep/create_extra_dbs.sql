-- create_extra_dbs.sql
-- Creates databases needed by other services if they don't exist.

-- Function to create database if it doesn't exist
-- (PostgreSQL doesn't have explicit IF NOT EXISTS for CREATE DATABASE)
-- This is a common workaround.
CREATE OR REPLACE FUNCTION create_database_if_not_exists(dbname text) RETURNS void AS $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = dbname) THEN
        EXECUTE 'CREATE DATABASE ' || quote_ident(dbname);
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Create the databases
SELECT create_database_if_not_exists('prefect');
SELECT create_database_if_not_exists('mlflow');

-- Optional: Grant privileges if needed, though default user usually has rights
-- GRANT ALL PRIVILEGES ON DATABASE prefect TO postgres;
-- GRANT ALL PRIVILEGES ON DATABASE mlflow TO postgres; 