# Parallel Feature Extraction Deployment Guide

## Quick Start for Google Cloud Deployment

### Prerequisites
- Google Cloud account with billing enabled
- `gcloud` CLI installed and configured
- Terraform installed (optional, for automated deployment)

### Option 1: One-Command Terraform Deployment (Recommended)

```bash
# Clone the repository
git clone https://github.com/Fnux8890/Proactive-thesis.git
cd Proactive-thesis/DataIngestion/terraform/parallel-feature

# Set your project ID
export TF_VAR_project_id="your-gcp-project-id"

# Deploy everything
terraform init
terraform apply -auto-approve
```

This will:
1. Create an A2 high-GPU instance (4x A100, 48 vCPUs, 340GB RAM)
2. Set up Cloud SQL with TimescaleDB
3. Install all dependencies
4. Start the parallel feature extraction pipeline automatically

### Option 2: Manual Deployment

```bash
# Create the A2 instance
gcloud compute instances create feature-extraction-a2 \
  --machine-type=a2-highgpu-4g \
  --zone=us-central1-a \
  --boot-disk-size=500GB \
  --boot-disk-type=pd-ssd \
  --accelerator=type=nvidia-tesla-a100,count=4 \
  --maintenance-policy=TERMINATE \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud

# SSH into the instance
gcloud compute ssh feature-extraction-a2 --zone=us-central1-a

# On the instance, run:
curl -fsSL https://raw.githubusercontent.com/Fnux8890/Proactive-thesis/main/DataIngestion/terraform/parallel-feature/startup.sh | sudo bash
```

## What Happens Next

The system will automatically:

1. **Install Dependencies**: Docker, NVIDIA drivers, monitoring tools
2. **Clone Repository**: Gets latest code from GitHub
3. **Run Full Pipeline**:
   - Data ingestion (Rust)
   - Preprocessing (Python)
   - Era detection (Rust)
   - Parallel feature extraction (4 GPU + 8 CPU workers)

## Monitoring Progress

### Check Pipeline Status
```bash
# SSH to instance
gcloud compute ssh feature-extraction-a2 --zone=us-central1-a

# View logs
docker compose logs -f parallel-coordinator

# Check worker status
docker compose ps
```

### Access Monitoring Dashboards
Get the instance IP:
```bash
gcloud compute instances describe feature-extraction-a2 \
  --zone=us-central1-a \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
```

Then access:
- Grafana: `http://INSTANCE_IP:3001` (admin/admin)
- Prometheus: `http://INSTANCE_IP:9090`

## Expected Runtime

For the full dataset (2014-2015):
- Data ingestion: ~30 minutes
- Preprocessing: ~20 minutes
- Era detection: ~10 minutes
- Parallel feature extraction: ~2-4 hours (vs 20-40 hours single-threaded)

## Checking Results

```bash
# On the instance
docker compose exec db psql -U postgres -d postgres -c "
SELECT COUNT(*) as total_features FROM tsfresh_features;
"

# Download results
docker compose exec db pg_dump -U postgres -t tsfresh_features postgres > features.sql
gcloud compute scp feature-extraction-a2:~/features.sql . --zone=us-central1-a
```

## Costs

- A2 instance: $14.69/hour
- Cloud SQL: ~$0.50/hour
- **Total**: ~$15/hour
- **Estimated total cost**: $60-80 for complete run

## Cleanup

```bash
# If using Terraform
terraform destroy -auto-approve

# If manual
gcloud compute instances delete feature-extraction-a2 --zone=us-central1-a
```

## Troubleshooting

### GPU Not Detected
```bash
# Check NVIDIA drivers
nvidia-smi

# Restart Docker with GPU support
sudo systemctl restart docker
```

### Out of Memory
```bash
# Check memory usage
docker stats

# Reduce batch sizes
# Edit docker-compose.parallel-feature.yml
# Reduce GPU_THRESHOLD or memory limits
```

### Database Connection Issues
```bash
# Check database connectivity
docker compose exec parallel-coordinator ping db

# Check PgBouncer
docker compose logs pgbouncer
```