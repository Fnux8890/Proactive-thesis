# Pre-Deployment Checklist

## Before Running `terraform apply`

### 1. Prerequisites Setup
- [ ] Install Google Cloud SDK: `curl https://sdk.cloud.google.com | bash`
- [ ] Authenticate: `gcloud auth login`
- [ ] Set project: `gcloud config set project YOUR_PROJECT_ID`
- [ ] Enable required APIs:
```bash
gcloud services enable compute.googleapis.com
gcloud services enable sqladmin.googleapis.com
gcloud services enable servicenetworking.googleapis.com
gcloud services enable storage-api.googleapis.com
```

### 2. Choose Data Upload Method

#### Option A: If your repo is PUBLIC (Easiest)
- [ ] Ensure Data folder is committed to git
- [ ] No additional steps needed - startup script will clone repo

#### Option B: If your repo is PRIVATE
- [ ] Add GitHub personal access token to startup.sh:
```bash
# Replace line 54 in startup.sh:
git clone https://YOUR_TOKEN@github.com/Fnux8890/Proactive-thesis.git || true
```

#### Option C: Use Google Cloud Storage (Recommended for large data)
- [ ] Upload data to GCS first:
```bash
./upload_data_to_gcs.sh
```
- Benefits: Faster, more reliable, supports large files

#### Option D: Upload after deployment
- [ ] Deploy infrastructure first: `terraform apply`
- [ ] Then upload data: `./upload_data_direct.sh`

### 3. Configure Terraform
- [ ] Create terraform.tfvars:
```bash
cat > terraform.tfvars <<EOF
project_id = "your-gcp-project-id"
region = "us-central1"
zone = "us-central1-a"
use_preemptible = false  # true for dev/test (80% cheaper)
EOF
```

### 4. Verify Local Setup
- [ ] Check you're in the right directory:
```bash
pwd  # Should show: .../DataIngestion/terraform/parallel-feature
```
- [ ] Verify data exists:
```bash
ls -la ../../../Data/  # Should show CSV/JSON files
du -sh ../../../Data/  # Check total size
```

### 5. Cost Awareness
- [ ] Production instance: ~$2-4/hour when running
- [ ] Preemptible instance: ~$0.40-0.80/hour (can be interrupted)
- [ ] Cloud SQL: ~$0.50/hour
- [ ] Storage: ~$0.02/GB/month

## Deployment Commands

### Full Production Deployment
```bash
# 1. Upload data (choose one method)
./upload_data_to_gcs.sh  # Recommended

# 2. Deploy infrastructure
terraform init
terraform plan
terraform apply -auto-approve

# 3. Monitor deployment (in another terminal)
INSTANCE_IP=$(terraform output -raw instance_ip)
gcloud compute ssh greenhouse-production-a2 --zone=us-central1-a
tail -f /var/log/cloud-init-output.log
```

### Dev/Test Deployment (Cheaper)
```bash
# Use preemptible instances
sed -i 's/use_preemptible = false/use_preemptible = true/' terraform.tfvars
terraform apply -auto-approve
```

## Post-Deployment Verification

### 1. Check Infrastructure
```bash
# Get outputs
terraform output

# SSH to instance
gcloud compute ssh greenhouse-production-a2 --zone=us-central1-a
```

### 2. On the Instance
```bash
# Check data
ls -la /opt/Proactive-thesis/Data/

# Check pipeline status
cd /opt/Proactive-thesis/DataIngestion
docker compose ps

# Monitor pipeline
tail -f /var/log/production-pipeline.log

# Check GPU
nvidia-smi
```

### 3. Access Services
- Grafana: `http://<INSTANCE_IP>:3001` (admin/admin)
- Prometheus: `http://<INSTANCE_IP>:9090`

## Troubleshooting

### Data not found
```bash
# Check if data was uploaded
gsutil ls gs://YOUR_PROJECT_ID-feature-extraction-data/Data/

# Or check on instance
ls -la /opt/Proactive-thesis/Data/
```

### Pipeline not starting
```bash
# Check startup script log
sudo tail -f /var/log/cloud-init-output.log

# Check Docker
docker ps
docker compose logs
```

### Database connection issues
```bash
# Test connection
PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -U postgres -d greenhouse
```

## Clean Up After Testing
```bash
# Stop instance (preserves data)
gcloud compute instances stop greenhouse-production-a2 --zone=us-central1-a

# Or destroy everything
terraform destroy -auto-approve
```