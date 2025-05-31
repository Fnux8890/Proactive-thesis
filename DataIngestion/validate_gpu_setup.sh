#!/bin/bash
# Quick GPU validation script
# Checks if GPU is properly configured for the sparse pipeline

set -e

echo "============================================"
echo "GPU Setup Validation"
echo "Date: $(date)"
echo "============================================"

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $2"
    else
        echo -e "${RED}✗${NC} $2"
        return 1
    fi
}

# Check 1: nvidia-smi availability
echo ""
echo "1. Checking NVIDIA drivers..."
if command -v nvidia-smi &> /dev/null; then
    print_status 0 "nvidia-smi found"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
    print_status 1 "nvidia-smi not found - NVIDIA drivers may not be installed"
fi

# Check 2: Docker GPU runtime
echo ""
echo "2. Checking Docker GPU runtime..."
if docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    print_status 0 "Docker can access GPU"
else
    print_status 1 "Docker cannot access GPU - check nvidia-container-toolkit installation"
fi

# Check 3: Environment file
echo ""
echo "3. Checking environment configuration..."
if [ -f ".env.sparse" ]; then
    DISABLE_GPU=$(grep "^DISABLE_GPU=" .env.sparse | cut -d'=' -f2)
    if [ "$DISABLE_GPU" = "false" ]; then
        print_status 0 "DISABLE_GPU=false in .env.sparse"
    else
        print_status 1 "DISABLE_GPU is not set to false in .env.sparse (current: $DISABLE_GPU)"
    fi
else
    print_status 1 ".env.sparse file not found"
fi

# Check 4: Docker Compose GPU configuration
echo ""
echo "4. Checking Docker Compose configuration..."
if grep -q "capabilities: \[gpu\]" docker-compose.sparse.yml; then
    print_status 0 "GPU capabilities configured in docker-compose.sparse.yml"
else
    print_status 1 "GPU capabilities not found in docker-compose.sparse.yml"
fi

# Check 5: Build sparse pipeline container
echo ""
echo "5. Testing sparse pipeline build..."
if docker compose -f docker-compose.sparse.yml build sparse_pipeline > /tmp/build.log 2>&1; then
    print_status 0 "Sparse pipeline container built successfully"
else
    print_status 1 "Failed to build sparse pipeline container"
    echo "Check /tmp/build.log for details"
fi

# Check 6: Quick GPU test
echo ""
echo "6. Running quick GPU initialization test..."
cat > /tmp/test_gpu.sh << 'EOF'
#!/bin/bash
docker compose -f docker-compose.sparse.yml run --rm \
    -e SPARSE_MODE=true \
    -e SPARSE_START_DATE=2014-05-01 \
    -e SPARSE_END_DATE=2014-05-02 \
    -e DISABLE_GPU=false \
    sparse_pipeline \
    --sparse-mode \
    --start-date 2014-05-01 \
    --end-date 2014-05-02 \
    --batch-size 24 2>&1 | tee /tmp/gpu_test.log

# Check if GPU was initialized
if grep -q "CUDA context initialized" /tmp/gpu_test.log; then
    exit 0
elif grep -q "GPU disabled by environment variable" /tmp/gpu_test.log; then
    exit 2
else
    exit 1
fi
EOF

chmod +x /tmp/test_gpu.sh

# Start database for test
docker compose -f docker-compose.sparse.yml up -d db
sleep 10

# Run data ingestion for test date
echo "Preparing test data..."
docker compose -f docker-compose.sparse.yml run --rm \
    -e DATA_SOURCE_PATH=/app/data \
    rust_pipeline > /dev/null 2>&1

# Run GPU test
if bash /tmp/test_gpu.sh > /dev/null 2>&1; then
    print_status 0 "GPU successfully initialized in sparse pipeline"
    GPU_READY=true
elif [ $? -eq 2 ]; then
    print_status 1 "GPU disabled by environment variable"
    GPU_READY=false
else
    print_status 1 "GPU initialization failed"
    GPU_READY=false
fi

# Clean up
docker compose -f docker-compose.sparse.yml down > /dev/null 2>&1

# Summary
echo ""
echo "============================================"
echo "Validation Summary"
echo "============================================"

if [ "$GPU_READY" = true ]; then
    echo -e "${GREEN}✓ GPU is properly configured and ready for use${NC}"
    echo ""
    echo "You can now run the full performance test:"
    echo "  ./run_gpu_performance_test.sh"
else
    echo -e "${RED}✗ GPU configuration issues detected${NC}"
    echo ""
    echo "Please fix the issues above before running GPU tests."
    echo "Common fixes:"
    echo "1. Install NVIDIA drivers: https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/"
    echo "2. Install nvidia-container-toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    echo "3. Ensure DISABLE_GPU=false in .env.sparse"
    echo "4. Rebuild containers after fixes: docker compose -f docker-compose.sparse.yml build"
fi

echo ""
echo "Detailed logs available at:"
echo "  - Build log: /tmp/build.log"
echo "  - GPU test log: /tmp/gpu_test.log"