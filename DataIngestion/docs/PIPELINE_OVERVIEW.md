# Pipeline Overview

## Complete Data Processing Pipeline

The greenhouse data processing pipeline consists of 6 stages that transform raw sensor data into optimized control strategies:

### Stage 1: Data Ingestion (Rust)
- **Service**: `rust_pipeline`
- **Input**: Raw CSV/JSON files from greenhouse sensors
- **Output**: `sensor_data` table in TimescaleDB
- **Features**: 
  - High-performance async ingestion
  - Data validation and null handling
  - Batch inserts for efficiency

### Stage 2: Data Preprocessing (Python)
- **Service**: `preprocessing`
- **Input**: Raw data from `sensor_data` table
- **Output**: `preprocessed_greenhouse_data` hypertable
- **Features**:
  - Time regularization (5-minute intervals)
  - Island detection and handling for sparse data
  - External data enrichment (weather, energy prices)
  - Missing value imputation

### Stage 3: Era Detection (Rust)
- **Service**: `era_detector`
- **Input**: Preprocessed data
- **Output**: Era labels identifying operational periods
- **Methods**:
  - PELT (Level A): Large structural changes
  - BOCPD (Level B): Medium operational changes
  - HMM (Level C): Small state transitions

### Stage 4: Feature Extraction (Python/GPU)
- **Services**: 
  - CPU: `feature_extraction_level_a/b/c`
  - GPU: `gpu_feature_extraction_level_a/b/c`
- **Input**: Preprocessed data + Era labels
- **Output**: Feature tables for each era level
- **Features**:
  - TSFresh feature calculation
  - GPU acceleration for large datasets
  - Multi-level feature extraction

### Stage 5: Model Building (Python)
- **Service**: `model_builder`
- **Input**: Multi-level features from all era levels
- **Output**: Trained surrogate models for MOEA objectives
- **Objectives**:
  - Energy consumption
  - Plant growth rate
  - Temperature stability
  - Humidity control
  - CO2 efficiency

### Stage 6: MOEA Optimization
- **Services**: `moea_optimizer_cpu` or `moea_optimizer_gpu`
- **Input**: Trained models + constraints
- **Output**: Pareto-optimal control strategies
- **Algorithm**: NSGA-III with GPU acceleration

## Running the Pipeline

### Local Development
```bash
# Full pipeline
docker compose up

# Specific stages
docker compose up rust_pipeline preprocessing era_detector
docker compose up gpu_feature_extraction_level_a gpu_feature_extraction_level_b gpu_feature_extraction_level_c
```

### Production Deployment
```bash
# With production optimizations
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## Data Flow

```
Raw CSV/JSON → rust_pipeline → sensor_data
                                    ↓
                              preprocessing → preprocessed_greenhouse_data
                                                        ↓
                                                  era_detector → era_labels
                                                        ↓
                                    ┌───────────────────┼───────────────────┐
                                    ↓                   ↓                   ↓
                          Level A Features     Level B Features     Level C Features
                                    └───────────────────┼───────────────────┘
                                                        ↓
                                                  model_builder → trained_models
                                                        ↓
                                                moea_optimizer → optimal_controls
```

## Key Considerations

1. **Sparse Data Handling**: The pipeline is designed to handle data with >90% missing values
2. **GPU Acceleration**: Critical for feature extraction and MOEA optimization at scale
3. **Multi-Level Features**: Different era sizes capture different control dynamics
4. **Island Detection**: Preprocessing identifies and handles disconnected data periods