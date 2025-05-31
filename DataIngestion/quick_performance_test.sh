#!/bin/bash
# Quick performance test - single month for faster results

echo "===== QUICK PERFORMANCE TEST ====="
echo "Testing sparse pipeline for May 2014 only"
echo ""

# Ensure database is running
docker compose -f docker-compose.sparse.yml up -d db redis
sleep 5

# Test 1: CPU only
echo "Test 1: CPU-only processing..."
DISABLE_GPU=true docker run --rm \
    --network container:dataingestion-db-1 \
    -e DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres \
    -e RUST_LOG=gpu_feature_extraction=info \
    -e SPARSE_MODE=true \
    -e DISABLE_GPU=true \
    -v $(pwd)/gpu_feature_extraction/checkpoints:/tmp/gpu_sparse_pipeline:rw \
    dataingestion-gpu-sparse-pipeline:latest \
    --sparse-mode \
    --start-date 2014-05-01 \
    --end-date 2014-06-01 \
    --batch-size 24 2>&1 | tee /tmp/cpu_test.log

CPU_TIME=$(grep "Sparse pipeline complete in" /tmp/cpu_test.log | grep -oP '\d+\.\d+s' | sed 's/s//')
CPU_RATE=$(grep "Performance:" /tmp/cpu_test.log | grep -oP '\d+\.\d+ features/second' | grep -oP '\d+\.\d+')

echo -e "\nTest 2: GPU-enabled processing..."
rm -rf gpu_feature_extraction/checkpoints/*
DISABLE_GPU=false docker run --rm \
    --network container:dataingestion-db-1 \
    -e DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres \
    -e RUST_LOG=gpu_feature_extraction=info \
    -e SPARSE_MODE=true \
    -e DISABLE_GPU=false \
    -v $(pwd)/gpu_feature_extraction/checkpoints:/tmp/gpu_sparse_pipeline:rw \
    dataingestion-gpu-sparse-pipeline:latest \
    --sparse-mode \
    --start-date 2014-05-01 \
    --end-date 2014-06-01 \
    --batch-size 24 2>&1 | tee /tmp/gpu_test.log

GPU_TIME=$(grep "Sparse pipeline complete in" /tmp/gpu_test.log | grep -oP '\d+\.\d+s' | sed 's/s//')
GPU_RATE=$(grep "Performance:" /tmp/gpu_test.log | grep -oP '\d+\.\d+ features/second' | grep -oP '\d+\.\d+')

# Summary
echo -e "\n===== QUICK TEST RESULTS ====="
echo "Processing May 2014 data (1 month)"
echo ""
echo "CPU-only:"
echo "  Time: ${CPU_TIME}s"
echo "  Rate: ${CPU_RATE} features/second"
echo ""
echo "GPU-enabled:"
echo "  Time: ${GPU_TIME}s"
echo "  Rate: ${GPU_RATE} features/second"
echo ""

if [ -n "$CPU_TIME" ] && [ -n "$GPU_TIME" ]; then
    SPEEDUP=$(echo "scale=2; $CPU_TIME / $GPU_TIME" | bc)
    echo "Speedup: ${SPEEDUP}x"
fi

# Save results
mkdir -p docs/experiments/results
cat > docs/experiments/results/quick_test_$(date +%Y%m%d_%H%M%S).json << EOF
{
  "test": "quick_performance",
  "date": "$(date -Iseconds)",
  "month": "2014-05",
  "cpu": {
    "time_seconds": ${CPU_TIME:-0},
    "features_per_second": ${CPU_RATE:-0}
  },
  "gpu": {
    "time_seconds": ${GPU_TIME:-0},
    "features_per_second": ${GPU_RATE:-0}
  },
  "speedup": ${SPEEDUP:-0}
}
EOF

echo -e "\nResults saved to docs/experiments/results/"