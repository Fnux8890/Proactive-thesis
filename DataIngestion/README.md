# DataIngestion Pipeline

Comprehensive data pipeline for greenhouse climate control optimization, featuring parallel processing, GPU acceleration, and multi-environment support.

## Quick Start

### 🚀 Complete End-to-End Pipeline (Recommended)

For the full greenhouse optimization pipeline with CPU vs GPU MOEA comparison:

```bash
# Run complete pipeline with 2013-2016 dataset
./run_full_pipeline_experiment.sh

# Or run with specific parameters
START_DATE="2013-12-01" END_DATE="2016-09-08" ./run_full_pipeline_experiment.sh
```

**What this includes:**
- ✅ Full data ingestion (2013-2016 greenhouse data)
- ✅ Enhanced sparse feature extraction with GPU acceleration
- ✅ Comprehensive model training (LightGBM + LSTM)
- ✅ **CPU vs GPU MOEA optimization comparison**
- ✅ Real-world evaluation and performance analysis
- ✅ Automated results collection and reporting

### 🔬 Quick Experiments

```bash
# Multi-run statistical comparison (5 CPU + 5 GPU runs)
./run_multiple_experiments.sh --cpu-runs 5 --gpu-runs 5

# Quick performance test
./quick_performance_test.sh

# GPU-specific optimization test
./run_gpu_performance_test.sh
```

### 🏠 Local Development (CPU-only, minimal features)
```bash
# Automatically uses docker-compose.override.yml
docker compose up

# Optional: Include development tools
docker compose --profile dev-tools up
```

### 🏭 Production Deployment (High-Performance Hardware)
```bash
# Full parallel processing with monitoring
docker compose -f docker-compose.prod.yml up
```

## 📋 Requirements

### System Requirements

| Component | Minimum | Recommended | Production |
|-----------|---------|-------------|------------|
| **RAM** | 8GB | 16GB | 32GB+ |
| **CPU Cores** | 4 | 8 | 16+ |
| **Disk Space** | 20GB | 50GB | 100GB+ |
| **GPU** | None | GTX 1660+ | RTX 4070+ |
| **VRAM** | N/A | 6GB | 12GB+ |

### Software Requirements

```bash
# Essential
docker >= 20.10
docker compose >= 2.0

# For GPU acceleration (optional)
nvidia-docker2
CUDA Toolkit >= 11.8

# For data preparation
git
wget or curl
```

### GPU Setup (Optional, for TensorNSGA3)

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu20.04 nvidia-smi
```

## 🏗️ Pipeline Architecture & Usage

### Complete Pipeline Overview

The system consists of 6 integrated stages:

1. **🔧 Rust Data Ingestion** - High-performance CSV/JSON parsing with async I/O
2. **🧹 Python Preprocessing** - Data cleaning, enrichment, and time regularization
3. **📊 Era Detection** - Changepoint detection (PELT, BOCPD, HMM) in Rust
4. **⚡ Feature Extraction** - Hybrid GPU/CPU feature computation (600+ features)
5. **🤖 Model Building** - LightGBM surrogate + LSTM training with MLflow
6. **🎯 MOEA Optimization** - Multi-objective greenhouse control optimization

### Docker Compose Configurations

| File | Purpose | Use Case | Features |
|------|---------|----------|----------|
| `docker-compose.yml` | Base configuration | Development/Testing | Core services |
| `docker-compose.enhanced.yml` | Enhanced features | Research/Experiments | Full feature set |
| `docker-compose.full-comparison.yml` | **Complete pipeline** | **Production/Research** | **CPU vs GPU MOEA** |
| `docker-compose.sparse.yml` | Sparse features | Quick testing | Minimal features |
| `docker-compose.prod.yml` | Production deployment | Cloud deployment | Optimized performance |

### Usage Examples

#### 🔥 Most Common: Full Pipeline Experiment
```bash
# Complete greenhouse optimization pipeline (2-4 hours)
./run_full_pipeline_experiment.sh

# Custom date range
START_DATE="2014-01-01" END_DATE="2014-12-31" ./run_full_pipeline_experiment.sh
```

#### 🚀 Quick Start: Enhanced Pipeline
```bash
# Run enhanced pipeline with all features
docker compose -f docker-compose.enhanced.yml up

# Run specific stages
docker compose -f docker-compose.enhanced.yml up db rust_pipeline enhanced_sparse_pipeline
```

#### ⚡ Performance Testing
```bash
# Multi-run statistical analysis
./run_multiple_experiments.sh --cpu-runs 3 --gpu-runs 3

# Quick performance comparison
./quick_performance_test.sh

# GPU-only optimization test
docker compose -f docker-compose.full-comparison.yml up moea_optimizer_gpu
```

#### 🔬 Development & Testing
```bash
# Local development with minimal features
docker compose up

# Build specific service
docker compose build model_builder

# Run with custom configuration
CONFIG_PATH=./config/custom.toml docker compose up
```

See [docs/UNIFIED_PIPELINE_ARCHITECTURE.md](docs/UNIFIED_PIPELINE_ARCHITECTURE.md) for detailed flow.

## Feature Extraction Approach

For detailed documentation on the feature extraction components, see [docs/feature_extraction/README.md](./docs/feature_extraction/README.md).

Since tsfresh is CPU-only, we use a hybrid approach:

### GPU Acceleration
- **Data Loading**: cuDF for fast I/O
- **Preprocessing**: GPU-accelerated transformations
- **Rolling Features**: GPU-computed statistics
- **Feature Selection**: GPU correlation analysis

### CPU Processing  
- **tsfresh Features**: 600+ statistical features
- **Complex Calculations**: Entropy, peak detection
- **Binary Sensors**: Lamp status, rain sensors

### Parallel Processing
- **Smart Distribution**: Analyzes era characteristics
- **4 GPU Workers**: Large eras, continuous sensors
- **4-8 CPU Workers**: Small eras, categorical data

## Environment Configuration

### Local Development (`docker-compose.override.yml`)
- **Features**: Minimal set for fast iteration
- **Resources**: Limited CPU/memory
- **Debugging**: Verbose logging, code hot-reload
- **Database**: Local PostgreSQL with exposed ports

### Production (`docker-compose.production.yml`)
- **Features**: Comprehensive 600+ features
- **Parallel Processing**: 4 GPU workers + 6 CPU workers
- **Resources**: Optimized for A2 instance (4 GPUs, 48 vCPUs, 340GB RAM)
- **Monitoring**: Full Prometheus/Grafana stack
- **High Performance**: All stages run in parallel where possible

## Feature Sets

| Set | Features | Use Case | Processing Time |
|-----|----------|----------|-----------------|
| **Minimal** | Basic stats (mean, std, min, max) | Local testing | 5-10 min/era |
| **Efficient** | + Autocorrelation, spectral | Development | 10-20 min/era |
| **Comprehensive** | All 600+ tsfresh features | Production | 2-5 min/era |

## Running Specific Stages

```bash
# Run only preprocessing
docker compose up preprocessing

# Run up to era detection
docker compose up preprocessing era_detector

# Run with specific GPU workers (production)
docker compose -f docker-compose.yml -f docker-compose.production.yml up \
  feature-coordinator feature-gpu-worker-0 feature-gpu-worker-1
```

## Monitoring

### Local Development
- pgAdmin: http://localhost:5050 (admin@local.dev/admin)
- Redis Commander: http://localhost:8081

### Cloud Production
- Grafana: http://INSTANCE_IP:3000 (admin/admin)
- Prometheus: http://INSTANCE_IP:9090
- GPU Metrics: http://INSTANCE_IP:9400

## Project Structure

```
DataIngestion/
   docker-compose.yml           # Base configuration
   docker-compose.override.yml  # Local dev overrides
   docker-compose.cloud.yml     # Cloud production config
   docs/
        architecture/                       # System design docs
        deployment/                         # Deployment instructions
        feature_extraction/                 # Detailed docs for feature extraction
        operations/                         # Operational guides
        pipelines/                          # Guides for specific pipeline runs
        notes/                              # Developer notes and historical summaries
   rust_pipeline/             # Stage 1: Data ingestion
   feature_extraction/        # Stages 2-4: Processing
      pre_process/          # Stage 2: Preprocessing
      era_detection_rust/   # Stage 3: Era detection
      feature/              # Stage 4: CPU features
      feature-gpu/          # Stage 4: GPU features
      parallel/             # Parallel coordination
   model_builder/            # Stage 5: Model training
   moea_optimizer/          # Stage 6: Optimization
```

## 📊 Data Requirements

### Input Data Structure

The pipeline expects greenhouse sensor data in the following format:

```bash
# Required directory structure
Data/
├── aarslev/
│   ├── temperature_sunradiation_jan_feb_2014.json.csv
│   ├── weather_jan_feb_2014.csv
│   └── winter2014.csv
└── knudjepsen/
    └── [sensor_data_files].csv
```

### Data Format Requirements

**Sensor Data Columns:**
- `timestamp` - ISO format datetime
- `temperature` - Air temperature (°C)
- `humidity` - Relative humidity (%)
- `co2` - CO2 concentration (ppm)
- `light` - Light intensity (μmol/m²/s)
- `heating`, `ventilation`, `lamp` - Actuator states
- Additional sensor columns as needed

**Date Range:** 2013-12-01 to 2016-09-08 (full dataset)

### Expected Results

After running the complete pipeline, you'll get:

```bash
experiments/full_experiment/[timestamp]/
├── experiment_summary.json           # Experiment metadata
├── checkpoints/                      # Feature extraction results
│   ├── stage3_features.json         # 223,825 enhanced features
│   └── stage4_eras.json            # Era detection results  
├── models/                          # Trained models
│   ├── energy_consumption_model.pt  # Energy prediction model (R²≥0.85)
│   ├── plant_growth_model.pt       # Growth prediction model (R²≥0.80)
│   └── training_summary.json       # Model performance metrics
├── moea_cpu/                        # CPU optimization results
│   ├── pareto_X.npy                # Pareto-optimal control variables
│   ├── pareto_F.npy                # Objective values
│   └── experiment/complete_results.json
├── moea_gpu/                        # GPU optimization results (TensorNSGA3)
│   ├── pareto_X.npy                # Better Pareto front (26 vs 12 solutions)
│   ├── pareto_F.npy                # Superior objective values
│   └── experiment/complete_results.json
└── evaluation_results/              # Performance analysis
    ├── comprehensive_evaluation_report.json
    ├── cpu_vs_gpu_comparison.md
    └── economic_impact_analysis.json
```

**Key Performance Metrics:**
- **TensorNSGA3 Speedup:** 132x faster than CPU NSGA-III (0.041s vs 5.41s)
- **Previous GPU vs CPU:** 22.9x faster (0.235s vs 5.41s) 
- **TensorNSGA3 Improvement:** 5.7x faster than previous GPU implementation
- **Solution Quality:** 26 vs 12 Pareto-optimal solutions (GPU vs CPU)
- **Model Accuracy:** RMSE <0.05, R² >0.80 for all objectives
- **Economic Impact:** €15,000-50,000/year potential savings per greenhouse

## ⚙️ Configuration & Environment Variables

### Pipeline Configuration

```bash
# Experiment settings
START_DATE="2013-12-01"          # Dataset start date
END_DATE="2016-09-08"            # Dataset end date  
BATCH_SIZE="48"                  # Processing batch size
MIN_ERA_ROWS="200"               # Minimum era size

# Feature extraction
FEATURE_SET=minimal|efficient|comprehensive
N_JOBS=4                         # CPU parallelism
USE_SPARSE_FEATURES="true"       # Enable enhanced features

# GPU settings (for TensorNSGA3)
CUDA_VISIBLE_DEVICES=0           # GPU device ID
GPU_MEMORY_LIMIT=12GB            # VRAM limit
USE_GPU="true"                   # Enable GPU acceleration

# MOEA optimization
POPULATION_SIZE="100"            # NSGA-III population size
GENERATIONS="300"                # Evolution generations
N_RUNS="3"                       # Multiple runs for statistics

# Database
DATABASE_URL=postgresql://postgres:postgres@db:5432/postgres
POSTGRES_SHARED_BUFFERS=8GB      # Production only
```

### Algorithm Selection

```bash
# Choose optimization algorithm
ALGORITHM_TYPE="tensornsga3"     # Use real TensorNSGA3 (recommended)
ALGORITHM_TYPE="nsga3_gpu"       # Use custom GPU implementation  
ALGORITHM_TYPE="nsga3_cpu"       # Use CPU pymoo NSGA-III
```

## Deployment

### Local Machine Setup (Recommended)

**Minimum Requirements:**
- Docker & Docker Compose v2.0+
- 8GB+ RAM  
- 4+ CPU cores
- 20GB+ disk space

**For GPU Acceleration (TensorNSGA3):**
- NVIDIA GPU with 6GB+ VRAM
- NVIDIA Container Toolkit
- CUDA 11.8+ compatible drivers

**High-Performance Setup:**
- 32GB+ RAM
- 16+ CPU cores  
- RTX 4070+ or better GPU
- 100GB+ SSD storage

## Performance

| Environment | Configuration | Processing Time | Hardware Cost |
|-------------|--------------|-----------------|---------------|
| Local CPU | Minimal features | 8-12 hours | Standard PC |
| Local CPU | Efficient features | 15-20 hours | Standard PC |
| Local GPU+CPU | Comprehensive (TensorNSGA3) | 2-4 hours | RTX 4070+ GPU |
| High-Performance | Full pipeline + experiments | 1-2 hours | RTX 4080+ GPU |

## Troubleshooting

### Out of Memory
```bash
# Reduce batch sizes
BATCH_SIZE=500 docker compose up

# Or use fewer workers
GPU_WORKERS=2 CPU_WORKERS=4 docker compose up
```

### GPU Not Available
```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# Fall back to CPU
docker compose up  # Uses override.yml (CPU-only)
```

### Database Connection Issues
```bash
# Check database health
docker compose exec db pg_isready

# Reset database
docker compose down -v
docker compose up db
```

## Contributing

1. Use local development setup for testing
2. Ensure all tests pass
3. Document any new features
4. Follow existing code structure

## Related Documentation

- [Unified Pipeline Architecture](docs/UNIFIED_PIPELINE_ARCHITECTURE.md)
- [TensorNSGA3 Performance Gains](TENSORNSGA3_PERFORMANCE_GAINS_REPORT.md)
- [Experimental Findings Report](EXPERIMENTAL_FINDINGS_REPORT.md)
- [Feature Extraction Guide](docs/feature_extraction/README.md)
- [GPU Feature Extraction](docs/GPU_FEATURE_EXTRACTION_QUICK_START.md)
- [Era Detection Operations](docs/operations/ERA_DETECTION_OPERATIONS_GUIDE.md)
- [MOEA Integration Examples](docs/MOEA_INTEGRATION_EXAMPLE.md)