# Quick Cloud Deployment Guide

## Prerequisites
- Google Cloud account with billing enabled
- `gcloud` CLI installed and authenticated
- Terraform installed (v1.0+)

## Deploy in 3 Steps

### 1. Configure Variables
```bash
cd DataIngestion/terraform/parallel-feature

cat > terraform.tfvars <<EOF
project_id = "your-gcp-project-id"
region = "us-central1"
zone = "us-central1-a"
use_preemptible = false  # Set to true for 80% cost savings in dev/test
EOF
```

### 2. Deploy Infrastructure
```bash
terraform init
terraform plan
terraform apply -auto-approve
```

### 3. Monitor Progress
```bash
# Get instance IP
INSTANCE_IP=$(terraform output -raw instance_ip)

# SSH to instance
gcloud compute ssh greenhouse-production-a2 --zone=us-central1-a

# Monitor pipeline progress
tail -f /var/log/production-pipeline.log

# Check service status
cd /opt/Proactive-thesis/DataIngestion
docker compose ps
```

## What Gets Deployed

### Infrastructure
- **Compute**: a2-highgpu-4g instance (48 vCPUs, 340GB RAM, 4× A100 GPUs)
- **Database**: Cloud SQL PostgreSQL 16 with TimescaleDB (8 vCPUs, 52GB RAM)
- **Storage**: 500GB SSD for both compute and database
- **Network**: Private VPC with secure connectivity

### Pipeline Components
1. **Data Ingestion**: Loads all CSV/JSON sensor data
2. **Preprocessing**: Cleans and regularizes data
3. **Era Detection**: Identifies operational periods
4. **Feature Extraction**: GPU-accelerated feature computation
5. **Target Calculation**: Computes REAL metrics from sensor data
6. **Model Training**: Trains LightGBM surrogates
7. **MOEA Optimization**: Finds Pareto-optimal solutions

### Key Features
- **Real Data Only**: No synthetic targets - calculates actual energy, growth, water, quality
- **GPU Acceleration**: Uses all 4 A100s for parallel processing
- **TimescaleDB**: Scalable time-series storage with hypertables
- **Monitoring**: Grafana + Prometheus with GPU metrics

## Access Services

```bash
# Monitoring dashboards
echo "Grafana: http://$INSTANCE_IP:3001 (admin/admin)"
echo "Prometheus: http://$INSTANCE_IP:9090"

# Database connection
echo "Database: ${DB_HOST}:5432/greenhouse"
```

## Cost Estimates
- **Production run**: ~$50-100 for complete pipeline
- **Preemptible run**: ~$10-20 (dev/test)
- **Storage**: ~$50-100/month

## Cleanup
```bash
# Stop but preserve data
gcloud compute instances stop greenhouse-production-a2 --zone=us-central1-a

# Or destroy everything
terraform destroy -auto-approve
```

## Troubleshooting

### Check logs
```bash
# Pipeline log
tail -f /var/log/production-pipeline.log

# Service logs
docker compose logs -f <service-name>

# System log
tail -f /var/log/cloud-init-output.log
```

### Common issues
- **Out of memory**: Reduce BATCH_SIZE in docker-compose.prod.yml
- **GPU not found**: Check nvidia-smi output
- **Database connection**: Verify Cloud SQL is running and accessible

## Results Location
- Database tables: `sensor_data`, `preprocessed_greenhouse_data`, `tsfresh_features`
- Models: `/app/models/*/model.txt`
- MOEA results: `/app/results/pareto_front.csv`
- Logs: `/var/log/production-pipeline.log`