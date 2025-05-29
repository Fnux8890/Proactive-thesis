# Terraform Deployment for Production Pipeline

This Terraform configuration deploys the complete greenhouse optimization pipeline on Google Cloud using an A2 high-GPU instance.

## What This Deploys

### Compute Resources
- **A2 Instance**: `a2-highgpu-4g` with 4× NVIDIA A100 GPUs, 48 vCPUs, 340GB RAM
- **Cloud SQL**: PostgreSQL 16 with TimescaleDB extension (8 vCPUs, 52GB RAM)
- **Networking**: VPC with proper firewall rules
- **Storage**: SSD boot disk and persistent database storage

### Software Stack
- Docker and Docker Compose
- NVIDIA GPU drivers and container toolkit
- Full production pipeline with parallel processing
- Monitoring stack (Prometheus + Grafana)

## Prerequisites

1. **Google Cloud Account**
   - Project with billing enabled
   - Sufficient quota for A2 instances in your region
   - Required APIs enabled (Compute Engine, Cloud SQL)

2. **Local Tools**
   ```bash
   # Install Terraform
   brew install terraform  # macOS
   # or download from https://www.terraform.io/downloads

   # Install Google Cloud SDK
   brew install google-cloud-sdk  # macOS
   # or follow https://cloud.google.com/sdk/docs/install

   # Authenticate
   gcloud auth login
   gcloud auth application-default login
   ```

3. **IAM Permissions**
   - Compute Admin
   - Cloud SQL Admin
   - Service Account Admin
   - Storage Admin

## Quick Start

1. **Clone and Navigate**
   ```bash
   git clone https://github.com/Fnux8890/Proactive-thesis.git
   cd Proactive-thesis/DataIngestion/terraform/parallel-feature
   ```

2. **Configure Variables**
   ```bash
   cp terraform.tfvars.example terraform.tfvars
   # Edit terraform.tfvars with your project ID
   ```

3. **Validate Configuration**
   ```bash
   chmod +x validate.sh
   ./validate.sh
   ```

4. **Deploy**
   ```bash
   # Initialize Terraform
   terraform init

   # Review the plan
   terraform plan

   # Apply (this will take ~10 minutes)
   terraform apply
   ```

5. **Access Your Instance**
   ```bash
   # SSH into the instance
   gcloud compute ssh greenhouse-production-a2 --zone=us-central1-a

   # Or use the SSH command from Terraform output
   terraform output ssh_command
   ```

## What Happens on Startup

The instance automatically:
1. Installs Docker and NVIDIA drivers
2. Clones the repository
3. Configures environment variables
4. Starts the production pipeline with `docker-compose.production.yml`
5. Begins parallel processing across 4 GPUs and 48 vCPUs

## Monitoring

After deployment, access monitoring dashboards:
```bash
# Get the instance IP
INSTANCE_IP=$(terraform output -raw instance_ip)

# Access dashboards
echo "Grafana: http://$INSTANCE_IP:3000 (admin/admin)"
echo "Prometheus: http://$INSTANCE_IP:9090"
```

## Configuration Files

- `main.tf` - Main Terraform configuration
- `startup.sh` - Instance initialization script
- `terraform.tfvars` - Your project-specific variables
- `validate.sh` - Pre-deployment validation script

## Cost Optimization

### Preemptible Instances (60-80% savings)
```hcl
# In terraform.tfvars
use_preemptible = true
```
⚠️ Note: Preemptible instances can be terminated at any time. Only use for development or fault-tolerant workloads.

### Committed Use Discounts
For production workloads running 24/7, consider purchasing committed use contracts for additional savings.

## Validation

Run validation before deployment:
```bash
./validate.sh
```

This checks:
- Terraform syntax and formatting
- Required variables
- Google Cloud authentication
- Docker Compose file references
- Security best practices
- Resource specifications

## Outputs

After successful deployment:
```bash
# Show all outputs
terraform output

# Specific outputs
terraform output instance_ip
terraform output db_connection_name
terraform output ssh_command
terraform output monitoring_urls
```

## Updating the Deployment

To update the running pipeline:
```bash
# SSH into the instance
gcloud compute ssh greenhouse-production-a2 --zone=us-central1-a

# Pull latest changes
cd /opt/Proactive-thesis
git pull

# Restart services
cd DataIngestion
docker-compose -f docker-compose.yml -f docker-compose.production.yml down
docker-compose -f docker-compose.yml -f docker-compose.production.yml up -d
```

## Destroying Resources

To tear down all resources:
```bash
terraform destroy
```
⚠️ This will delete all data! Back up any important results first.

## Troubleshooting

### Instance Won't Start
- Check quota limits: `gcloud compute project-info describe --project=$PROJECT_ID`
- Verify A2 availability in your zone
- Check billing is enabled

### GPU Not Available
```bash
# On the instance
nvidia-smi  # Should show 4 GPUs
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### Database Connection Issues
- Ensure Cloud SQL Admin API is enabled
- Check VPC peering is established
- Verify firewall rules

### Monitoring Not Accessible
- Check firewall rules allow ports 3000, 9090
- Ensure services are running: `docker-compose ps`

## Architecture

The deployed system uses our simplified configuration approach:
- Base configuration + Production overrides
- Full parallel processing (4 GPU + 6 CPU workers)
- Comprehensive feature extraction
- Integrated monitoring
- Optimized for the A2 instance specifications

## Support

For issues:
1. Check the validation script output
2. Review Terraform logs
3. Check instance startup logs: `gcloud compute instances get-serial-port-output greenhouse-production-a2`
4. Open an issue on GitHub