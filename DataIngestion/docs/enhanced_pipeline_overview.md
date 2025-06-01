# Overview of the Enhanced DataIngestion Pipeline

This document provides a summary of the "Enhanced Pipeline" as configured by `docker-compose.enhanced.yml` and its associated environment settings (e.g., `.env.enhanced`).

## Purpose

The Enhanced Pipeline is a specialized configuration of the DataIngestion system designed for comprehensive greenhouse climate control optimization. Its primary focus is to process time-series sensor data characterized by extreme sparsity (e.g., ~91.3% missing values). It generates a rich set of features by combining raw sensor data with external contextual information (weather, energy prices, plant phenotype data) and then uses these features to train surrogate models. These models, in turn, feed into a Multi-Objective Evolutionary Algorithm (MOEA) for optimizing greenhouse control strategies.

## Pipeline Stages

The pipeline is orchestrated via Docker Compose and consists of the following key stages:

1.  **Database Initialization (`db` service):**
    *   Uses PostgreSQL with TimescaleDB extension.
    *   Uses PostgreSQL with TimescaleDB extension.
    *   Initializes with schemas for raw sensor data. Separate tables, managed by the `feature_extraction/pre_process/` module, are used to store cleaned external data (weather, energy prices, plant phenotype data like Kalanchoe growth metrics from `phenotype.json`).

2.  **Rust Data Ingestion (`rust_pipeline` service):**
    *   Handles the initial, high-performance ingestion of raw sensor data (e.g., CSVs) into the database.
    *   Ensures data is correctly formatted and stored for subsequent processing.

3.  **Enhanced Sparse Feature Extraction (`enhanced_sparse_pipeline` service):**
    *   This is the core of the enhanced pipeline, implemented within the `gpu_feature_extraction` module. It has a sophisticated hybrid architecture:
    *   **Rust Component (`gpu_feature_extraction/src/`):**
        *   Orchestrates the feature extraction process (e.g., `enhanced_sparse_pipeline.rs`, `sparse_pipeline.rs`).
        *   Performs significant CPU-bound data manipulation using the Polars DataFrame library and Rayon for parallelism.
        *   Interacts directly with the database (`db.rs` using `sqlx`) to fetch raw data and potentially store intermediate results.
        *   Manages and executes custom CUDA C/C++ kernels (from `kernels/` subdirectory, compiled via `build.rs`) for highly performance-critical, low-level GPU tasks.
        *   Integrates pre-processed external data (weather, energy, growth) via `external_data.rs` by reading from tables prepared by `feature_extraction/pre_process/`.
        *   Communicates with Python scripts for other GPU tasks via a JSON-based bridge (`python_bridge.rs`, `sparse_hybrid_bridge.rs`).
    *   **Python Components (`gpu_feature_extraction/`):**
        *   **`sparse_gpu_features.py`:**
            *   Utilizes RAPIDS libraries (`cuDF` for GPU DataFrames, `CuPy` for GPU array computations).
            *   Focuses on DataFrame-centric operations: coverage analysis, detailed gap analysis (longest, mean, count), event detection, temporal pattern analysis, and sparse-aware multi-sensor correlations.
            *   Includes a CPU fallback mechanism using Pandas/NumPy if RAPIDS is unavailable.
        *   **`gpu_features_pytorch.py`:**
            *   Uses PyTorch for a distinct set of GPU-accelerated feature calculations, explicitly replacing some older custom CUDA kernel implementations.
            *   Computes extended univariate statistics (percentiles, skewness, kurtosis, MAD, IQR, Shannon entropy).
            *   Calculates weather coupling features (temperature differentials, solar efficiency, response lags using PyTorch convolutions for cross-correlation).
            *   Likely computes energy-related features and plant growth features (GDD, DLI from phenotype data) as suggested by its internal dataclasses (`EnergyFeatures`, `GrowthFeatures`).
    *   **Sparsity-Aware:** The entire module is designed to handle data with high levels of missing values, computing features on available data and providing coverage metrics.
    *   Operates in both "enhanced mode" and "sparse mode" as per its configuration, leveraging the appropriate feature sets.

4.  **Model Building (`model_builder` service):**
    *   Trains machine learning models (specifically surrogate models for the MOEA) using the rich feature set from the `enhanced_sparse_pipeline`.
    *   **Technology Stack:**
        *   Uses **PyTorch Lightning** for training deep learning models (e.g., LSTMs or other sequence models suitable for time-series data), ensuring structured and GPU-accelerated training.
        *   Uses **LightGBM** for training gradient boosted decision tree models, which are effective for tabular data.
    *   **Surrogate Models:** The primary goal is to build fast-evaluating surrogate models that predict various greenhouse and plant outcomes. These are then used by the MOEA.
        *   Key training scripts: `src/training/train_all_objectives.py` (orchestrator), `src/training/train_lightgbm_surrogate.py`, and `src/training/train_surrogate.py` (likely for PyTorch-based surrogates).
    *   **MLOps:** Employs **MLflow** (via `mlflow-skinny`) for experiment tracking, model versioning, and management.
    *   **Database Interaction:** Fetches features from the database (e.g., `enhanced_features` table) using SQLAlchemy and psycopg2.
    *   Configured to `USE_SPARSE_FEATURES`, ensuring it uses the output of the specialized feature extraction stage.

5.  **MOEA Optimization (`moea_optimizer` service):**
    *   Performs multi-objective optimization using the trained surrogate models from the `model_builder` stage.
    *   Aims to find Pareto-optimal solutions for greenhouse control strategies, balancing objectives such as plant growth, energy cost, and plant stress.
    *   Leverages GPU acceleration for the optimization algorithms.

## Key Configuration Aspects (from `docker-compose.enhanced.yml` & `.env.enhanced`)

*   **GPU Utilization:** The pipeline is heavily reliant on NVIDIA GPUs, with resources reserved for feature extraction, model building, and MOEA.
*   **Target Data:** Processes data within a defined date range (e.g., `2014-01-01` to `2014-07-01` as per `.env.enhanced`).
*   **Feature Output:** Stores the generated features in a dedicated database table (e.g., `enhanced_features`).
*   **Modular Design:** Each stage is containerized and depends on the successful completion of its predecessors.

## How it Works

1.  **External Data Preparation (`feature_extraction/pre_process/`):** Scripts (e.g., `preprocess.py`) fetch, clean, and store external data (weather, energy prices, plant phenotype data from `phenotype.json`) into dedicated database tables.
2.  **Raw Sensor Data Ingestion (`rust_pipeline` service):** Ingests raw sensor data into the database.
3.  **Enhanced Sparse Feature Extraction (`enhanced_sparse_pipeline` service / `gpu_feature_extraction` module):
    *   The Rust component reads raw sensor data and the pre-processed external data.
    *   It performs CPU-bound processing (Polars, Rayon) and executes custom CUDA kernels for specific low-level GPU tasks.
    *   It calls Python scripts (`sparse_gpu_features.py` with RAPIDS, `gpu_features_pytorch.py` with PyTorch) for higher-level GPU-accelerated feature calculations (coverage, gaps, events, patterns, extended stats, weather coupling, growth features, etc.).
    *   The resulting comprehensive feature set is stored (e.g., in `enhanced_features` table).
4.  **Surrogate Model Training (`model_builder` service):
    *   Fetches the rich features from the database.
    *   Trains various surrogate models using PyTorch Lightning (for DL models) and LightGBM (for tree-based models).
    *   Tracks experiments and manages models using MLflow.
5.  **Multi-Objective Optimization (`moea_optimizer` service):
    *   Uses the trained surrogate models to efficiently evaluate potential greenhouse control strategies.
    *   Applies GPU-accelerated evolutionary algorithms to find Pareto-optimal solutions balancing multiple objectives (e.g., yield, energy cost, plant stress).

This enhanced pipeline represents a sophisticated, multi-stage, and multi-technology approach. It effectively handles sparse sensor data, enriches it with diverse external sources, and employs a hybrid CPU/GPU strategy across Rust, Python (RAPIDS, PyTorch), and custom CUDA kernels to generate features. These features then fuel the training of diverse surrogate models, which are ultimately used by a GPU-accelerated MOEA for advanced greenhouse control optimization.
