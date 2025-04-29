"""Feast feature definitions for daily greenhouse climate data.

Assumptions:
• TimescaleDB container is reachable at host name `db` (docker‑compose network).
• Offline store table is `features_daily` (see load_data ingestion).
• DataFrame ingested must expose columns:
    greenhouse_id (int64)
    event_timestamp (datetime)
    DLI_mol_m2_d, GDD_daily, DIF_daily, Lamp_kWh_daily …
  You can expand the schema list as more features stabilise.
"""
from datetime import timedelta

from feast import Entity, Field, FeatureView
from feast.types import Float32, Float64, Int64
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgresSource,
)

# -----------------------------------------------------------------------------
# Entity
# -----------------------------------------------------------------------------

greenhouse = Entity(
    name="greenhouse_id",
    description="Identifier for individual greenhouse / compartment",
    dtype=Int64,
)

# -----------------------------------------------------------------------------
# Data source – Timescale/Postgres table created by load_data ingestion
# -----------------------------------------------------------------------------

offline_source = PostgresSource(
    name="daily_features_source",
    table="features_daily",              # created automatically by Feast ingest
    timestamp_field="event_timestamp",   # must exist in ingested DataFrame
    # created_timestamp_column optional
)

# -----------------------------------------------------------------------------
# Feature View – daily aggregated climate + energy metrics
# -----------------------------------------------------------------------------

daily_climate_view = FeatureView(
    name="daily_climate_view",
    entities=[greenhouse],
    ttl=timedelta(days=365),
    online=True,
    schema=[
        Field(name="DLI_mol_m2_d", dtype=Float32),
        Field(name="GDD_daily", dtype=Float32),
        Field(name="DIF_daily", dtype=Float32),
        Field(name="Lamp_kWh_daily", dtype=Float32),
        Field(name="DLI_suppl_mol_m2_d", dtype=Float32),
    ],
    source=offline_source,
)

# -----------------------------------------------------------------------------
# Helper list for store.apply() convenience
# -----------------------------------------------------------------------------

entities = [greenhouse]
feature_views = [daily_climate_view]
