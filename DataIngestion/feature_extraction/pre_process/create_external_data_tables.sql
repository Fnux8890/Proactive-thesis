-- External data tables for Enhanced Sparse Pipeline
-- These tables store weather and energy data for feature enrichment

-- External weather data from Open-Meteo API
CREATE TABLE IF NOT EXISTS external_weather_aarhus (
    time TIMESTAMPTZ PRIMARY KEY,
    temperature_2m REAL,
    relative_humidity_2m REAL,
    precipitation REAL,
    rain REAL,
    snowfall REAL,
    weathercode INTEGER,
    surface_pressure REAL,
    cloudcover REAL,
    cloudcover_low REAL,
    cloudcover_mid REAL,
    cloudcover_high REAL,
    shortwave_radiation REAL,
    direct_radiation REAL,
    diffuse_radiation REAL,
    windspeed_10m REAL,
    winddirection_10m REAL,
    windgusts_10m REAL,
    pressure_msl REAL,
    vapour_pressure_deficit REAL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create hypertable for time-series optimization
SELECT create_hypertable('external_weather_aarhus', 'time', if_not_exists => TRUE);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_weather_temp ON external_weather_aarhus (temperature_2m);
CREATE INDEX IF NOT EXISTS idx_weather_radiation ON external_weather_aarhus (shortwave_radiation);

-- Danish energy spot prices
CREATE TABLE IF NOT EXISTS external_energy_prices_dk (
    "HourUTC" TIMESTAMPTZ NOT NULL,
    "PriceArea" VARCHAR(10) NOT NULL,
    "SpotPriceDKK" REAL,
    "SpotPriceEUR" REAL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY ("HourUTC", "PriceArea")
);

-- Create hypertable for time-series optimization
SELECT create_hypertable('external_energy_prices_dk', 'HourUTC', if_not_exists => TRUE);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_energy_price_area ON external_energy_prices_dk ("PriceArea");
CREATE INDEX IF NOT EXISTS idx_energy_spot_price ON external_energy_prices_dk ("SpotPriceDKK");

-- Grant permissions
GRANT SELECT, INSERT, UPDATE ON external_weather_aarhus TO postgres;
GRANT SELECT, INSERT, UPDATE ON external_energy_prices_dk TO postgres;