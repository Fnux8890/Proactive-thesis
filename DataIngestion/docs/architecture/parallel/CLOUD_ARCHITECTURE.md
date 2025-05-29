# Parallel Feature Extraction Architecture for Google Cloud A2

## Instance Specifications
- **Type**: a2-highgpu-4g
- **vCPUs**: 48
- **GPUs**: 4x NVIDIA A100 40GB
- **Memory**: 340 GiB
- **Cost**: $14.69/hour

## Architecture Design

### Container Distribution Strategy

```yaml
# Optimal resource allocation for parallel feature extraction
services:
  # 1. Coordinator/Orchestrator (1 container)
  coordinator:
    cpu: 2
    memory: 8GB
    tasks:
      - Era distribution
      - Progress monitoring
      - Result aggregation
      - Database connection pooling

  # 2. GPU Feature Workers (4 containers - 1 per GPU)
  gpu-worker-[0-3]:
    cpu: 4
    gpu: 1
    memory: 40GB
    tasks:
      - GPU-accelerated data preprocessing
      - Large era processing (>1M rows)
      - Feature selection (correlation matrices)
      - Bulk data transformations

  # 3. CPU Feature Workers (8 containers)
  cpu-worker-[0-7]:
    cpu: 4
    memory: 16GB
    tasks:
      - tsfresh feature extraction
      - Small/medium era processing
      - Feature aggregation
      - Database writes

  # 4. Database Proxy/Connection Pool
  pgbouncer:
    cpu: 2
    memory: 4GB
    max_connections: 200
    pool_mode: transaction
```

### Task Distribution Algorithm

```python
def distribute_eras(all_eras, gpu_workers=4, cpu_workers=8):
    """
    Smart era distribution based on size and complexity
    """
    # Sort eras by row count
    sorted_eras = sorted(all_eras, key=lambda x: x['row_count'], reverse=True)
    
    # Threshold for GPU processing (configurable)
    GPU_THRESHOLD = 1_000_000  # rows
    
    gpu_eras = [e for e in sorted_eras if e['row_count'] > GPU_THRESHOLD]
    cpu_eras = [e for e in sorted_eras if e['row_count'] <= GPU_THRESHOLD]
    
    # Balance GPU workload
    gpu_queues = [[] for _ in range(gpu_workers)]
    for i, era in enumerate(gpu_eras):
        gpu_queues[i % gpu_workers].append(era)
    
    # Balance CPU workload by total row count
    cpu_queues = balance_by_workload(cpu_eras, cpu_workers)
    
    return gpu_queues, cpu_queues
```

## Kubernetes Deployment (Recommended for Production)

### 1. GPU Worker Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: feature-gpu-workers
spec:
  replicas: 4
  selector:
    matchLabels:
      app: feature-gpu-worker
  template:
    metadata:
      labels:
        app: feature-gpu-worker
    spec:
      containers:
      - name: gpu-worker
        image: gcr.io/PROJECT_ID/feature-extraction-gpu:latest
        resources:
          requests:
            memory: "40Gi"
            cpu: "4"
            nvidia.com/gpu: 1
          limits:
            memory: "40Gi"
            cpu: "4"
            nvidia.com/gpu: 1
        env:
        - name: WORKER_TYPE
          value: "GPU"
        - name: WORKER_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: url
        - name: REDIS_URL
          value: "redis://redis-service:6379"
```

### 2. CPU Worker StatefulSet
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: feature-cpu-workers
spec:
  serviceName: cpu-workers
  replicas: 8
  selector:
    matchLabels:
      app: feature-cpu-worker
  template:
    metadata:
      labels:
        app: feature-cpu-worker
    spec:
      containers:
      - name: cpu-worker
        image: gcr.io/PROJECT_ID/feature-extraction-cpu:latest
        resources:
          requests:
            memory: "16Gi"
            cpu: "4"
          limits:
            memory: "16Gi"
            cpu: "4"
        env:
        - name: WORKER_TYPE
          value: "CPU"
        - name: TSFRESH_N_JOBS
          value: "4"
        - name: WORKER_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
```

### 3. Redis for Task Queue
```yaml
apiVersion: v1
kind: Service
metadata:
  name: redis-service
spec:
  selector:
    app: redis
  ports:
  - port: 6379
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
```

## Docker Compose Alternative (Development/Testing)

```yaml
version: '3.8'

services:
  coordinator:
    build:
      context: .
      dockerfile: coordinator.dockerfile
    environment:
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=${DATABASE_URL}
    depends_on:
      - redis
      - pgbouncer
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 8G

  gpu-worker-0:
    build:
      context: .
      dockerfile: feature_gpu.dockerfile
    environment:
      - WORKER_ID=gpu-0
      - WORKER_TYPE=GPU
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
        limits:
          cpus: '4'
          memory: 40G

  # Repeat for gpu-worker-1 through gpu-worker-3

  cpu-worker-0:
    build:
      context: .
      dockerfile: feature.dockerfile
    environment:
      - WORKER_ID=cpu-0
      - WORKER_TYPE=CPU
      - TSFRESH_N_JOBS=4
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 16G

  # Repeat for cpu-worker-1 through cpu-worker-7

  redis:
    image: redis:7-alpine
    command: redis-server --maxmemory 4gb --maxmemory-policy allkeys-lru
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 4G

  pgbouncer:
    image: pgbouncer/pgbouncer:latest
    environment:
      - DATABASES_HOST=${DB_HOST}
      - DATABASES_PORT=${DB_PORT}
      - DATABASES_DATABASE=${DB_NAME}
      - DATABASES_USER=${DB_USER}
      - DATABASES_PASSWORD=${DB_PASSWORD}
      - POOL_MODE=transaction
      - MAX_CLIENT_CONN=1000
      - DEFAULT_POOL_SIZE=25
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
```

## Deployment Strategy

### 1. Terraform Infrastructure (IaC)
```hcl
# main.tf
provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_compute_instance" "feature_extraction" {
  name         = "feature-extraction-a2"
  machine_type = "a2-highgpu-4g"
  zone         = var.zone

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 500
      type  = "pd-ssd"
    }
  }

  guest_accelerator {
    type  = "nvidia-tesla-a100"
    count = 4
  }

  scheduling {
    on_host_maintenance = "TERMINATE"
    automatic_restart   = false
    preemptible        = var.use_preemptible
  }

  network_interface {
    network = "default"
    access_config {}
  }

  metadata = {
    enable-oslogin = "TRUE"
  }

  metadata_startup_script = file("startup.sh")
}

# Cloud SQL for TimescaleDB
resource "google_sql_database_instance" "timescale" {
  name             = "timescale-instance"
  database_version = "POSTGRES_16"
  region           = var.region

  settings {
    tier = "db-highmem-8"
    
    database_flags {
      name  = "shared_preload_libraries"
      value = "timescaledb"
    }
    
    ip_configuration {
      ipv4_enabled    = true
      private_network = google_compute_network.vpc.id
    }
  }
}
```

### 2. Startup Script
```bash
#!/bin/bash
# startup.sh

# Install Docker and NVIDIA drivers
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Install kubectl and helm
sudo snap install kubectl --classic
sudo snap install helm --classic

# Clone repository and start services
git clone https://github.com/YOUR_REPO/Proactive-thesis.git
cd Proactive-thesis/DataIngestion/feature_extraction

# Start with Docker Compose or Kubernetes
docker compose -f docker-compose.parallel.yml up -d
# OR
# kubectl apply -f k8s/
```

## Monitoring and Optimization

### 1. Resource Monitoring
```yaml
# prometheus-values.yaml
prometheus:
  prometheusSpec:
    serviceMonitorSelectorNilUsesHelmValues: false
    additionalScrapeConfigs:
    - job_name: 'dcgm-exporter'
      static_configs:
      - targets: ['dcgm-exporter:9400']
```

### 2. Performance Metrics
- **GPU Utilization**: Target >80% for GPU workers
- **CPU Utilization**: Target 70-80% for CPU workers
- **Memory Usage**: Monitor for OOM kills
- **Queue Length**: Redis queue depth
- **Processing Time**: Era processing duration
- **Database Connections**: Active/idle connections

### 3. Auto-scaling Rules
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: cpu-worker-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: StatefulSet
    name: feature-cpu-workers
  minReplicas: 4
  maxReplicas: 16
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: External
    external:
      metric:
        name: redis_queue_length
        selector:
          matchLabels:
            queue: "cpu_tasks"
      target:
        type: AverageValue
        averageValue: "100"
```

## Cost Optimization

1. **Preemptible Instances**: Use for non-critical workloads (save 60-80%)
2. **Committed Use Discounts**: For sustained workloads
3. **Regional Distribution**: Distribute across regions for better pricing
4. **Workload Scheduling**: Process during off-peak hours
5. **Resource Right-sizing**: Monitor and adjust resource allocations

## Implementation Timeline

1. **Week 1**: Refactor code for parallel processing
2. **Week 2**: Create Docker images and test locally
3. **Week 3**: Deploy to GCP with Terraform
4. **Week 4**: Optimize and monitor performance