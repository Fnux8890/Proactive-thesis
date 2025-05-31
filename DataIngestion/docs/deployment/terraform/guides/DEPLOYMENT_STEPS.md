# 🚀 Cloud Deployment Steps

Your data (448MB in `/Data` folder) needs to be available on the cloud instance. Here's exactly what to do:

## Step 1: Prepare Your Environment

```bash
# 1. Open terminal and navigate to terraform directory
cd DataIngestion/terraform/parallel-feature

# 2. Make sure you have gcloud CLI
gcloud --version  # If not installed: https://cloud.google.com/sdk/docs/install

# 3. Login and set project
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 4. Enable required APIs
gcloud services enable compute.googleapis.com sqladmin.googleapis.com servicenetworking.googleapis.com storage-api.googleapis.com
```

## Step 2: Configure Terraform

```bash
# Create your configuration
cat > terraform.tfvars <<EOF
project_id = "your-actual-project-id"  # Replace this!
region = "us-central1"
zone = "us-central1-a"
use_preemptible = false  # Set to true for 80% cost savings in dev
EOF
```

## Step 3: Handle Your Data (Choose ONE Option)

### Option A: If Your GitHub Repo is PUBLIC ✅
**Nothing to do!** The startup script will automatically clone your repo with all data.

### Option B: If Your GitHub Repo is PRIVATE 🔒
```bash
# Edit startup.sh line 54 to add your GitHub token:
# Get token from: https://github.com/settings/tokens
sed -i 's|git clone https://github.com/Fnux8890/Proactive-thesis.git|git clone https://YOUR_GITHUB_TOKEN@github.com/Fnux8890/Proactive-thesis.git|' startup.sh
```

### Option C: Upload to Google Cloud Storage First (RECOMMENDED) ☁️
```bash
# This is best for reliability and speed
./upload_data_to_gcs.sh

# This will:
# - Create a GCS bucket
# - Upload all 448MB of CSV/JSON files
# - The instance will download from GCS (faster than git)
```

### Option D: Upload After Deployment 📤
```bash
# Deploy first, upload later
# Use this if you want to test the infrastructure first
# Run ./upload_data_direct.sh after terraform apply
```

## Step 4: Deploy!

```bash
# Initialize Terraform
terraform init

# Check what will be created
terraform plan

# Deploy everything (takes ~10 minutes)
terraform apply -auto-approve
```

## Step 5: Monitor Deployment

```bash
# In a new terminal, watch the progress
INSTANCE_IP=$(terraform output -raw instance_ip)
echo "Instance IP: $INSTANCE_IP"

# SSH to the instance
gcloud compute ssh greenhouse-production-a2 --zone=us-central1-a

# Once connected, monitor the pipeline startup
tail -f /var/log/cloud-init-output.log  # Watch initial setup
tail -f /var/log/production-pipeline.log  # Watch pipeline progress
```

## Step 6: Verify Everything is Running

```bash
# On the instance:
cd /opt/Proactive-thesis/DataIngestion

# Check services
docker compose ps

# Verify data was downloaded
ls -la /opt/Proactive-thesis/Data/

# Check GPU status
nvidia-smi

# View pipeline progress
tail -f /var/log/production-pipeline.log
```

## Step 7: Access Monitoring

Open in your browser:
- Grafana: `http://<INSTANCE_IP>:3001` (username: admin, password: admin)
- Prometheus: `http://<INSTANCE_IP>:9090`

## 🎯 What Happens Automatically

1. **Infrastructure Creation** (~5 min)
   - A2 instance with 4× A100 GPUs
   - Cloud SQL with TimescaleDB
   - VPC networking

2. **Software Installation** (~5 min)
   - Docker, NVIDIA drivers
   - Repository clone
   - Data download (from git or GCS)

3. **Pipeline Execution** (~4-6 hours)
   - Data ingestion from CSV/JSON
   - Preprocessing & cleaning
   - Era detection
   - Feature extraction (GPU accelerated)
   - Real target calculation
   - Model training
   - MOEA optimization

## 💰 Cost Management

### During Development
```bash
# Stop instance when not using (saves ~$50/day)
gcloud compute instances stop greenhouse-production-a2 --zone=us-central1-a

# Restart when needed
gcloud compute instances start greenhouse-production-a2 --zone=us-central1-a
```

### After Testing
```bash
# Destroy everything to stop all charges
terraform destroy -auto-approve
```

## ⚠️ Common Issues

### "Permission denied" on APIs
```bash
# Enable missing APIs
gcloud services list --available | grep -E "(compute|sql|storage)"
gcloud services enable <service-name>
```

### Data not found
```bash
# Check if using correct method
# If using GCS:
gsutil ls gs://${PROJECT_ID}-feature-extraction-data/Data/

# If using git, check if repo cloned:
ls -la /opt/Proactive-thesis/
```

### Pipeline not starting
```bash
# Check logs
sudo journalctl -u cloud-final -f
docker compose logs
```

## 📞 Need Help?

1. Check pipeline logs: `/var/log/production-pipeline.log`
2. Check Docker logs: `docker compose logs <service-name>`
3. Verify environment: `./scripts/validate_cloud_env.sh`

---

**Ready to deploy? Start with Step 1!** 🚀