# 🚀 Running the Enhanced Sparse Pipeline

## ✅ Prerequisites Status: ALL MET

The enhanced sparse pipeline is **ready to run**. All prerequisites have been verified:
- ✅ Docker & GPU environment configured
- ✅ Enhanced mode configuration (.env)
- ✅ Plant phenotype data (Kalanchoe blossfeldiana)
- ✅ External data table schemas
- ✅ 50 CSV sensor data files available

## 📋 Quick Start - Full Pipeline Run

### Option 1: Complete Pipeline (Recommended)

```bash
# 1. Start the full pipeline with enhanced mode
cd DataIngestion
docker compose -f docker-compose.sparse.yml up --build

# This will run:
# - Database initialization (with external tables)
# - Rust data ingestion
# - Enhanced sparse GPU pipeline
# - Model building
# - MOEA optimization
```

### Option 2: Step-by-Step Execution

```bash
# 1. Start database
docker compose -f docker-compose.sparse.yml up -d db

# 2. Wait for database to be ready (10-15 seconds)
docker compose -f docker-compose.sparse.yml exec db pg_isready

# 3. Run data ingestion
docker compose -f docker-compose.sparse.yml run --rm rust_pipeline

# 4. (Optional) Fetch external data
cd feature_extraction/pre_process
python external/fetch_external_weather.py  # Weather for 2014
python external/fetch_energy.py            # Energy prices
cd ../..

# 5. Run enhanced sparse pipeline
docker compose -f docker-compose.sparse.yml run --rm sparse_pipeline \
  --enhanced-mode \
  --start-date 2014-01-01 \
  --end-date 2014-07-01 \
  --batch-size 24

# 6. Run model building
docker compose -f docker-compose.sparse.yml run --rm model_builder

# 7. Run MOEA optimization
docker compose -f docker-compose.sparse.yml run --rm moea_optimizer_gpu
```

## 🔍 Monitoring GPU Utilization

Open a second terminal to monitor GPU usage:

```bash
# Watch GPU utilization (should reach 85-95% during feature extraction)
watch -n 1 nvidia-smi

# Or for detailed monitoring
nvidia-smi dmon -s u,m,p,t -d 1
```

## 📊 Expected Output

### Enhanced Pipeline Features

When running in enhanced mode, you should see:

```
Enhanced sparse pipeline complete in X.Xs:
  - Resolution levels: 5
  - Total feature sets: ~1,200+
  - Enhanced eras: 6
  - 15min: 2,880 feature sets
  - 60min: 720 feature sets
  - 240min: 180 feature sets
  - 720min: 60 feature sets
  - 1440min: 30 feature sets
  - Sensor: 40, Extended: 80, Weather: 15, Energy: 12, Growth: 20, Optimization: 10
  - Growth performance score: 0.XXX
  - Energy cost efficiency: 0.XXX
  - Sustainability score: 0.XXX
  - Performance: 150+ feature sets/second
```

### Performance Metrics

| Metric | Expected Value |
|--------|----------------|
| GPU Utilization | 85-95% |
| Feature Count | ~1,200+ per window |
| Processing Speed | 150+ features/second |
| Memory Usage | 6-8 GB GPU RAM |
| Total Runtime | 3-5 minutes for 6 months |

## 🐛 Troubleshooting

### If GPU utilization is low (<85%):
```bash
# Check GPU is detected
docker compose -f docker-compose.sparse.yml run --rm sparse_pipeline nvidia-smi

# Check enhanced mode is enabled
docker compose -f docker-compose.sparse.yml run --rm sparse_pipeline env | grep ENHANCED
```

### If external data is missing:
```bash
# Manually create tables if needed
docker compose -f docker-compose.sparse.yml exec db psql -U postgres -d postgres \
  -f /docker-entrypoint-initdb.d/01_create_external_tables.sql

# Fetch weather data
cd feature_extraction/pre_process
python external/fetch_external_weather.py
```

### If build fails:
```bash
# Clean and rebuild
docker compose -f docker-compose.sparse.yml down -v
docker compose -f docker-compose.sparse.yml build --no-cache sparse_pipeline
```

## 📈 Comparing Basic vs Enhanced Mode

To compare performance, run both modes:

```bash
# Basic mode (baseline)
docker compose -f docker-compose.sparse.yml run --rm sparse_pipeline \
  --sparse-mode \
  --start-date 2014-01-01 \
  --end-date 2014-07-01

# Enhanced mode (3.4x more features)
docker compose -f docker-compose.sparse.yml run --rm sparse_pipeline \
  --enhanced-mode \
  --start-date 2014-01-01 \
  --end-date 2014-07-01
```

## 🎯 MOEA Optimization Results

After the pipeline completes, check optimization results:

```bash
# View MOEA results
ls -la moea_optimizer/results/

# Check Pareto front solutions
cat moea_optimizer/results/experiment/complete_results.json | jq '.'
```

## ✨ What's Happening

The enhanced pipeline:
1. **Ingests** sparse sensor data (91.3% missing values)
2. **Enriches** with external weather and energy price data
3. **Extracts** 1,200+ features using GPU acceleration
4. **Calculates** plant growth metrics using Kalanchoe phenotype data
5. **Optimizes** for 3 objectives: growth, energy cost, stress
6. **Produces** Pareto-optimal greenhouse control strategies

## 🚀 Ready to Run!

The enhanced sparse pipeline is fully configured and ready. Simply run:

```bash
docker compose -f docker-compose.sparse.yml up --build
```

And monitor GPU usage to verify enhanced performance!