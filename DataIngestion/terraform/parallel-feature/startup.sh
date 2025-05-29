#!/bin/bash
set -e

# Variables from Terraform
PROJECT_ID="${project_id}"
REGION="${region}"
DB_HOST="${db_host}"
DB_PASS="${db_pass}"

# Update system
apt-get update
apt-get upgrade -y

# Install dependencies
apt-get install -y \
    curl \
    git \
    build-essential \
    python3-pip \
    python3-dev \
    postgresql-client \
    htop \
    iotop \
    nvtop

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
usermod -aG docker $USER

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/download/v2.23.3/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install monitoring tools
pip3 install nvidia-ml-py3 gpustat

# Clone repository
cd /opt
git clone https://github.com/Fnux8890/Proactive-thesis.git || true
cd Proactive-thesis

# Download data from GCS if available
if gsutil ls "gs://${PROJECT_ID}-feature-extraction-data/Data/" >/dev/null 2>&1; then
    echo "Downloading data from Google Cloud Storage..."
    mkdir -p Data
    gsutil -m cp -r "gs://${PROJECT_ID}-feature-extraction-data/Data/*" Data/
    echo "Data download complete"
else
    echo "No data found in GCS, using data from git repository"
fi

cd DataIngestion

# Create .env file at DataIngestion level
cat > .env <<EOF
DB_HOST=${DB_HOST}
DB_PORT=5432
DB_NAME=greenhouse
DB_USER=postgres
DB_PASSWORD=${DB_PASS}
DATABASE_URL=postgresql://postgres:${DB_PASS}@${DB_HOST}:5432/greenhouse
REDIS_URL=redis://localhost:6379
GPU_WORKERS=4
CPU_WORKERS=8
EOF

# Create monitoring configuration
mkdir -p monitoring/dashboards monitoring/datasources

# Prometheus configuration
cat > monitoring/prometheus.yml <<EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9100']

  - job_name: 'dcgm'
    static_configs:
      - targets: ['localhost:9400']

  - job_name: 'redis'
    static_configs:
      - targets: ['localhost:9121']

  - job_name: 'postgres'
    static_configs:
      - targets: ['localhost:9187']
EOF

# Grafana datasource
cat > monitoring/datasources/prometheus.yml <<EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

# Create systemd service for auto-start
cat > /etc/systemd/system/feature-extraction.service <<EOF
[Unit]
Description=Feature Extraction Pipeline
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/Proactive-thesis/DataIngestion
ExecStart=/usr/local/bin/docker-compose -f docker-compose.yml -f docker-compose.production.yml up -d
ExecStop=/usr/local/bin/docker-compose -f docker-compose.yml -f docker-compose.production.yml down
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
systemctl daemon-reload
systemctl enable feature-extraction.service

# Setup GPU monitoring
nvidia-smi -pm 1
nvidia-smi -ac 1215,1410  # Set GPU clocks for A100

# Create startup check script
cat > /opt/startup-check.sh <<'EOF'
#!/bin/bash
echo "=== System Information ==="
echo "CPU: $(lscpu | grep 'Model name' | cut -d: -f2 | xargs)"
echo "RAM: $(free -h | grep Mem | awk '{print $2}')"
echo "GPUs: $(nvidia-smi -L | wc -l)x $(nvidia-smi -L | head -1 | cut -d: -f2 | cut -d'(' -f1)"
echo ""
echo "=== Docker Status ==="
docker --version
docker-compose --version
echo ""
echo "=== GPU Status ==="
nvidia-smi
echo ""
echo "=== Network Configuration ==="
ip addr show
echo ""
echo "=== Database Connectivity ==="
PGPASSWORD=${DB_PASS} psql -h ${DB_HOST} -U postgres -d greenhouse -c "SELECT version();" || echo "Database connection failed"
EOF

chmod +x /opt/startup-check.sh

# Run startup check
/opt/startup-check.sh > /var/log/startup-check.log 2>&1

# Start the production pipeline
cd /opt/Proactive-thesis/DataIngestion
# Mark as running on Google Cloud
export GOOGLE_CLOUD_PROJECT=${PROJECT_ID}

# Copy production overrides if not exists
if [ ! -f docker-compose.production.yml ]; then
    cp docker-compose.prod.yml docker-compose.production.yml 2>/dev/null || echo "Production override file will be created"
fi

# Make sure production pipeline script is executable
chmod +x run_production_pipeline.sh 2>/dev/null || true

# Start the production pipeline with real data processing
echo "Starting production pipeline with real data only..."
echo "Using TimescaleDB at ${DB_HOST}"
echo ""

# First, start the monitoring stack
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d prometheus grafana dcgm-exporter

# Wait for database to be ready
echo "Waiting for database to be ready..."
for i in {1..30}; do
    if PGPASSWORD=${DB_PASS} psql -h ${DB_HOST} -U postgres -d greenhouse -c "SELECT 1" >/dev/null 2>&1; then
        echo "Database is ready!"
        break
    fi
    echo "Waiting for database... ($i/30)"
    sleep 10
done

# Run the production pipeline script
echo "Starting full production pipeline..."
nohup ./run_production_pipeline.sh > /var/log/production-pipeline.log 2>&1 &

echo ""
echo "Production pipeline started!"
echo "Monitor progress with:"
echo "  tail -f /var/log/production-pipeline.log"
echo "  docker compose ps"
echo ""
echo "Access monitoring at:"
echo "  Grafana: http://$(curl -s ifconfig.me):3001 (admin/admin)"
echo "  Prometheus: http://$(curl -s ifconfig.me):9090"
echo ""
echo "Feature extraction setup completed!"