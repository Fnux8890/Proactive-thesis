# Enhanced Pipeline Integration Plan: Using Rust + Python GPU Code

## Objective
Integrate the existing Rust enhanced sparse pipeline with Python GPU acceleration code to create a proper hybrid architecture that leverages both CPU (Rust) and GPU (Python) capabilities.

## Current Situation
- We have ~42KB of Rust code for enhanced sparse pipeline that's not being used
- We have Python GPU acceleration scripts that aren't being called
- The Docker image has an old binary without the enhanced mode
- We used a temporary Python workaround instead of the real implementation

## Step-by-Step Integration Plan

### Phase 1: Clean Slate (30 minutes)

#### 1.1 Purge All Docker Images
```bash
# Stop all containers
docker compose -f docker-compose.enhanced.yml down
docker compose -f docker-compose.sparse.yml down
docker compose down

# Remove all project images
docker images | grep dataingestion | awk '{print $3}' | xargs docker rmi -f
docker images | grep model_builder | awk '{print $3}' | xargs docker rmi -f
docker images | grep gpu | awk '{print $3}' | xargs docker rmi -f

# Clean build cache
docker builder prune -af
```

#### 1.2 Verify Clean State
```bash
docker images | grep -E "(dataingestion|gpu|sparse|enhanced)"
# Should return nothing
```

### Phase 2: Verify Rust Code Has Enhanced Mode (15 minutes)

#### 2.1 Check main.rs
```bash
# Verify enhanced mode exists
grep -n "enhanced_mode" gpu_feature_extraction/src/main.rs
# Should show the flag and the condition check
```

#### 2.2 Verify Python Bridge
```bash
# Check python_bridge.rs exists and has subprocess calls
cat gpu_feature_extraction/src/python_bridge.rs | grep -A5 "Command::new"
```

#### 2.3 Verify GPU Python Scripts
```bash
# Check sparse_gpu_features.py has main entry point
grep -n "if __name__" gpu_feature_extraction/sparse_gpu_features.py
```

### Phase 3: Fix the Dockerfile (30 minutes)

#### 3.1 Create a Proper Enhanced Dockerfile
```dockerfile
# gpu_feature_extraction/Dockerfile.enhanced-fixed
# Stage 1: Rust builder
FROM rust:1.87-slim AS rust-builder

RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and build Rust
COPY Cargo.toml build.rs ./
COPY src ./src

# IMPORTANT: Ensure latest code is built
RUN touch src/main.rs && \
    RUST_LOG=debug cargo build --release --verbose

# Verify the binary has enhanced mode
RUN /app/target/release/gpu_feature_extraction --help | grep enhanced || \
    (echo "ERROR: Binary missing enhanced mode!" && exit 1)

# Stage 2: Runtime with Python GPU support
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    postgresql-client \
    python3.11 \
    python3-pip \
    python3.11-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python defaults
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

# Copy Rust binary
COPY --from=rust-builder /app/target/release/gpu_feature_extraction /app/

# Install Python dependencies PROPERLY
COPY requirements-gpu.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements-gpu.txt

# Verify Python packages
RUN python -c "import pandas, numpy, torch, cupy; print('All packages installed!')"

# Copy Python scripts
COPY *.py ./

# Create required directories
RUN mkdir -p /app/data /app/checkpoints /app/logs

ENV PYTHONUNBUFFERED=1
ENV RUST_LOG=info
ENV RUST_BACKTRACE=1

ENTRYPOINT ["/app/gpu_feature_extraction"]
```

#### 3.2 Create Proper requirements-gpu.txt
```txt
# gpu_feature_extraction/requirements-gpu.txt
pandas==2.2.0
numpy==1.24.3
scipy==1.11.4
scikit-learn==1.4.0
psycopg2-binary==2.9.9
sqlalchemy==2.0.25
torch==2.4.0
cupy-cuda12x==13.0.0
dask==2024.1.0
dask[dataframe]==2024.1.0
dask-cuda==24.04.0
```

### Phase 4: Build and Verify (45 minutes)

#### 4.1 Build Enhanced Image
```bash
cd DataIngestion
docker build -f gpu_feature_extraction/Dockerfile.enhanced-fixed \
    -t enhanced-sparse-pipeline:latest \
    --no-cache \
    --progress=plain \
    gpu_feature_extraction/
```

#### 4.2 Verify Binary Works
```bash
# Test enhanced mode exists
docker run --rm enhanced-sparse-pipeline:latest --help

# Should see:
# --enhanced-mode    Enable enhanced sparse pipeline with external data
# --sparse-mode      Enable sparse pipeline mode
```

#### 4.3 Verify Python Works
```bash
# Test Python dependencies
docker run --rm --entrypoint python enhanced-sparse-pipeline:latest \
    -c "import pandas, numpy, torch, cupy; print('Success!')"
```

### Phase 5: Update Docker Compose (15 minutes)

#### 5.1 Create New Enhanced Compose
```yaml
# docker-compose.enhanced-fixed.yml
services:
  db:
    # ... same as before ...

  rust_pipeline:
    # ... same as before ...

  enhanced_sparse_pipeline:
    image: enhanced-sparse-pipeline:latest
    container_name: enhanced_sparse_pipeline
    environment:
      DATABASE_URL: postgresql://postgres:postgres@db:5432/postgres
      RUST_LOG: gpu_feature_extraction=debug,info
      PYTHONUNBUFFERED: 1
      CUDA_VISIBLE_DEVICES: 0
    command: [
      "--database-url", "postgresql://postgres:postgres@db:5432/postgres",
      "--enhanced-mode",
      "--start-date", "2013-12-01",
      "--end-date", "2016-09-08",
      "--batch-size", "1000",
      "--features-table", "enhanced_sparse_features"
    ]
    volumes:
      - ./gpu_feature_extraction/checkpoints:/app/checkpoints:rw
      - ./gpu_feature_extraction/logs:/app/logs:rw
    depends_on:
      db:
        condition: service_healthy
      rust_pipeline:
        condition: service_completed_successfully
    networks:
      - pipeline-net
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Phase 6: Test Integration (30 minutes)

#### 6.1 Test Rust → Python Bridge
Create a test to verify the python bridge works:
```bash
# Create test script
cat > gpu_feature_extraction/test_integration.py << 'EOF'
#!/usr/bin/env python
import sys
import json

# Read from stdin (how Rust calls Python)
data = json.loads(sys.stdin.read())
print(f"Python received: {data['timestamps'][:5]}", file=sys.stderr)

# Return success
response = {
    "status": "success",
    "features": {"test_feature": 1.0},
    "metadata": {"num_samples": len(data['timestamps']), "gpu_used": False}
}
print(json.dumps(response))
EOF

chmod +x gpu_feature_extraction/test_integration.py
```

#### 6.2 Run Full Pipeline
```bash
# Clear old data
docker compose -f docker-compose.enhanced-fixed.yml exec db \
    psql -U postgres -c "TRUNCATE TABLE enhanced_sparse_features;"

# Run enhanced pipeline
docker compose -f docker-compose.enhanced-fixed.yml up enhanced_sparse_pipeline
```

### Phase 7: Monitor and Debug (30 minutes)

#### 7.1 Watch Logs
```bash
# In separate terminal
docker compose -f docker-compose.enhanced-fixed.yml logs -f enhanced_sparse_pipeline | grep -E "(Stage|GPU|Python)"
```

#### 7.2 Check Process Flow
The logs should show:
1. "Starting Enhanced sparse pipeline mode"
2. "Stage 1: Enhanced aggregation with external data"
3. "Stage 2: Conservative gap filling"
4. "Stage 3: Enhanced GPU feature extraction"
5. "Calling Python GPU service"
6. "Feature extraction successful: X features from Y samples (GPU: true)"

#### 7.3 Verify GPU Usage
```bash
# While pipeline is running
nvidia-smi
# Should show python process using GPU memory
```

### Phase 8: Troubleshooting Checklist

If things don't work:

1. **Binary doesn't have enhanced mode**
   - Check Cargo.toml has all required dependencies
   - Ensure src/main.rs is latest version
   - Force rebuild with `touch src/main.rs`

2. **Python subprocess fails**
   - Check Python scripts are executable
   - Verify JSON serialization/deserialization
   - Check stderr for Python errors

3. **GPU not detected**
   - Verify CUDA_VISIBLE_DEVICES is set
   - Check nvidia-docker runtime is configured
   - Test with simple cupy script first

4. **Database errors**
   - Ensure tables exist
   - Check connection string
   - Verify permissions

## Success Criteria

The pipeline is working correctly when:

1. ✅ Rust binary starts with "--enhanced-mode" flag
2. ✅ Logs show all 4 stages executing
3. ✅ Python subprocess is called (check logs)
4. ✅ GPU memory usage visible in nvidia-smi
5. ✅ Features are inserted into enhanced_sparse_features table
6. ✅ More than 2M features generated
7. ✅ Processing time is significantly faster than Python-only

## Estimated Timeline

- Phase 1-2: 45 minutes (cleanup and verification)
- Phase 3-4: 1 hour 15 minutes (Docker fixes and builds)
- Phase 5-6: 45 minutes (integration testing)
- Phase 7-8: 1 hour (monitoring and troubleshooting)

**Total: ~3.5 hours**

## Next Steps After Success

1. Benchmark Rust+GPU vs Python-only performance
2. Add external weather and energy data integration  
3. Implement proper changepoint detection
4. Create unit tests for python_bridge
5. Document the architecture with diagrams