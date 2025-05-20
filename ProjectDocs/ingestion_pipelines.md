# Data Ingestion Pipelines

This document outlines the main pipelines and supporting directories in the `DataIngestion` folder. The pipelines mix Rust and Python services orchestrated via Docker Compose.

```mermaid
graph TD
    subgraph DataIngestion
        RP["rust_pipeline"]
        FP["feature_extraction"]
        MB["model_builder"]
        SD["simulation_data_prep"]
        GP["Gan_producer"]
    end
    RP --> FP
    FP --> MB
    SD --> FP
    GP --> MB
```

- **rust_pipeline** – core data ingestion written in Rust.
- **feature_extraction** – Python utilities for preprocessing and feature engineering.
- **model_builder** – scripts for training models with MLflow integration.
- **simulation_data_prep** – containers for preparing simulated datasets with Prefect flows.
- **Gan_producer** – prototype GAN-based data generation.

