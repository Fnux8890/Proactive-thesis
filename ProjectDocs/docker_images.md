# Docker Images

Several directories provide Dockerfiles to build project images.

```mermaid
graph TD
    DI[DataIngestion]
    DI --> RP_Docker["rust_pipeline/Dockerfile"]
    DI --> SimDocker["simulation_data_prep/Dockerfile"]
    Scripts["scripts/Dockerfile"]
```

- `DataIngestion/rust_pipeline/Dockerfile` – builds the Rust-based ingestion pipeline.
- `DataIngestion/simulation_data_prep/Dockerfile` – environment for Prefect-based preparation.
- `scripts/Dockerfile` – minimal image for PDF and spreadsheet converters.

