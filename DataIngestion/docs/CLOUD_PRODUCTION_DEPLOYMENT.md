# Cloud Production Deployment Guide

## Overview
This guide covers deploying the complete greenhouse optimization pipeline to Google Cloud Platform with:
- **Real data only** (no synthetic targets)
- **End-to-end processing** from raw sensor data to optimization results
- **Cloud SQL with TimescaleDB** for scalable time-series storage
- **4× NVIDIA A100 GPUs** for parallel processing

## Architecture

```
Google Cloud Platform
├── Compute Engine (a2-highgpu-4g)
│   ├── 48 vCPUs
│   ├── 340GB RAM
│   ├── 4× NVIDIA A100 GPUs
│   └── 500GB SSD
├── Cloud SQL PostgreSQL 16
│   ├── TimescaleDB Extension
│   ├── 8 vCPUs, 52GB RAM
│   ├── 500GB SSD (auto-resize)
│   └── Private VPC Connection
└── Monitoring Stack
    ├── Prometheus
    ├── Grafana
    └── DCGM GPU Metrics
```

## Deployment Steps

### 1. Prerequisites
- Google Cloud account with billing enabled
- `gcloud` CLI installed and configured
- Terraform installed (v1.0+)
- GitHub repository access

### 2. Deploy Infrastructure

```bash
# Clone repository locally
git clone https://github.com/Fnux8890/Proactive-thesis.git
cd Proactive-thesis/DataIngestion/terraform/parallel-feature

# Configure Terraform variables
cat > terraform.tfvars <<EOF
project_id = "your-gcp-project-id"
region = "us-central1"
zone = "us-central1-a"
use_preemptible = false  # Use true for dev/test (80% cheaper)
EOF

# Deploy infrastructure
terraform init
terraform plan
terraform apply
```

### 3. What Happens Automatically

The Terraform deployment:
1. Creates GCP infrastructure (VM, Cloud SQL, networking)
2. Installs Docker, NVIDIA drivers, monitoring tools
3. Clones the repository
4. Configures TimescaleDB
5. Starts the production pipeline

### 4. Monitor Deployment

```bash
# Get instance IP
INSTANCE_IP=$(terraform output -raw instance_ip)

# SSH to instance
gcloud compute ssh feature-extraction-instance --zone=us-central1-a

# Check pipeline status
cd /opt/Proactive-thesis/DataIngestion
docker compose ps
tail -f /var/log/cloud-init-output.log
```

### 5. Access Services

- **Grafana Dashboard**: `http://<instance-ip>:3001` (admin/admin)
- **Prometheus**: `http://<instance-ip>:9090`
- **Pipeline Logs**: `docker compose logs -f <service-name>`

## Pipeline Stages & Timing

### Complete Pipeline (~4-6 hours for full dataset)

1. **Data Ingestion** (30-60 min)
   - Loads all CSV/JSON files from `/Data` directory
   - Parses and inserts into `sensor_data` table
   - Handles millions of sensor readings

2. **Preprocessing** (30-45 min)
   - Cleans and regularizes sensor data
   - Handles missing values and outliers
   - Creates `preprocessed_greenhouse_data`

3. **Era Detection** (15-30 min)
   - Detects operational periods using PELT/BOCPD/HMM
   - Creates era labels at multiple granularities
   - Filters eras with sufficient data (>200 rows)

4. **Feature Extraction** (60-120 min)
   - Extracts comprehensive tsfresh features
   - Uses GPU acceleration and parallel processing
   - Processes 500 eras per batch across all CPU cores

5. **Target Calculation** (15-30 min)
   - Calculates REAL metrics from sensor data:
     - Energy consumption from heating/lighting/ventilation
     - Plant growth from environmental conditions
     - Water usage from humidity and evapotranspiration
     - Crop quality from stability and optimal conditions

6. **Model Training** (30-60 min)
   - Trains LightGBM models for each objective
   - Uses GPU acceleration
   - Validates on real greenhouse data

7. **MOEA Optimization** (15-30 min)
   - Runs NSGA-III with GPU acceleration
   - Population: 200, Generations: 500
   - Finds Pareto-optimal control strategies

## Real Target Calculation

The production pipeline calculates actual targets from sensor data:

```python
# Energy Consumption (kWh)
- Heating energy from pipe temperatures and flow
- Lighting energy from lamp status × power rating × time
- Ventilation energy from vent positions × motor power

# Plant Growth (g/day)
- Temperature optimality (22°C optimal)
- Humidity optimality (70% optimal)
- CO2 concentration effect
- Daily light integral

# Water Usage (L)
- Humidity deficit integration
- Evapotranspiration estimates
- Irrigation system measurements

# Crop Quality (0-1 index)
- Environmental stability
- Optimal condition maintenance
- Stress factor minimization
```

## Monitoring & Validation

### Real-time Monitoring
```bash
# Watch pipeline progress
watch -n 10 'docker compose ps'

# GPU utilization
nvidia-smi -l 5

# Database growth
docker compose run --rm db psql -U postgres -d greenhouse -c "
SELECT 
    table_name,
    pg_size_pretty(pg_total_relation_size(table_name::regclass)) as size,
    (SELECT COUNT(*) FROM table_name) as row_count
FROM information_schema.tables
WHERE table_schema = 'public'
ORDER BY pg_total_relation_size(table_name::regclass) DESC;"
```

### Validation Checkpoints
```bash
# Run validation script
docker compose run --rm feature_extraction python scripts/validate_pipeline_data.py

# Check specific stage
docker compose logs --tail=100 <service-name>
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce BATCH_SIZE in docker-compose.prod.yml
   - Use preemptible instances with more RAM

2. **Slow Processing**
   - Check GPU utilization: `nvidia-smi`
   - Verify parallel settings: N_JOBS=-1
   - Monitor CPU usage: `htop`

3. **Database Connection Issues**
   - Check Cloud SQL is running
   - Verify private IP connectivity
   - Check firewall rules

### Debug Commands
```bash
# Check resource usage
docker stats

# View service logs
docker compose logs -f <service-name>

# Access database
docker compose run --rm db psql -U postgres -d greenhouse

# Test GPU
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

## Cost Optimization

### Development/Testing
- Use preemptible instances (80% cheaper)
- Stop instances when not in use
- Use smaller instance types for testing

### Production
- Schedule pipeline runs during off-peak hours
- Use committed use discounts
- Enable auto-shutdown after pipeline completion

### Estimated Costs
- **Full pipeline run**: ~$50-100
- **Preemptible run**: ~$10-20
- **Monthly storage**: ~$50-100

## Results & Outputs

After successful completion:

1. **Database Tables**
   - `sensor_data`: Raw sensor readings
   - `preprocessed_greenhouse_data`: Cleaned data
   - `era_labels_*`: Operational periods
   - `tsfresh_features`: Feature vectors with real targets
   - `optimization_results`: Pareto-optimal solutions

2. **Model Artifacts**
   - `/app/models/*/model.txt`: LightGBM models
   - `/app/models/*/scaler.joblib`: Feature scalers

3. **Optimization Results**
   - `/app/results/pareto_front.csv`: Optimal solutions
   - `/app/results/convergence_history.json`: Algorithm progress
   - Grafana dashboards with visualizations

## Next Steps

1. **Analyze Results**
   ```sql
   -- View Pareto-optimal solutions
   SELECT * FROM optimization_results
   WHERE is_pareto_optimal = true
   ORDER BY energy_consumption ASC;
   ```

2. **Deploy Control Strategies**
   - Select solutions from Pareto front
   - Implement in greenhouse control system
   - Monitor real-world performance

3. **Continuous Improvement**
   - Collect new data
   - Retrain models periodically
   - Refine optimization objectives

## Cleanup

```bash
# Stop pipeline
docker compose down

# Destroy infrastructure (saves costs)
cd terraform/parallel-feature
terraform destroy

# Or just stop the VM to preserve data
gcloud compute instances stop feature-extraction-instance --zone=us-central1-a
```