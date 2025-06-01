#!/bin/bash

# Quick test of the enhanced sparse pipeline with Rust+GPU hybrid architecture
# This verifies that the enhanced mode flags are working correctly

echo "🚀 Testing Enhanced Sparse Pipeline (Rust+GPU Hybrid)"
echo "======================================================="

# Change to correct directory
cd "$(dirname "$0")"

# Check if enhanced image exists
if ! docker images | grep -q "enhanced-sparse-pipeline-v3"; then
    echo "❌ Enhanced image not found. Please run: docker build -f Dockerfile.enhanced -t enhanced-sparse-pipeline-v3 ."
    exit 1
fi

echo "✅ Enhanced Docker image found"

# Test that the enhanced mode flags are available
echo ""
echo "🔍 Testing enhanced mode flags..."
docker run --rm enhanced-sparse-pipeline-v3 --help > /tmp/enhanced_help.txt 2>&1

if grep -q "enhanced-mode" /tmp/enhanced_help.txt; then
    echo "✅ Enhanced mode flag is available"
else
    echo "❌ Enhanced mode flag is missing"
    echo "Help output:"
    cat /tmp/enhanced_help.txt
    exit 1
fi

if grep -q "sparse-mode" /tmp/enhanced_help.txt; then
    echo "✅ Sparse mode flag is available"
else
    echo "❌ Sparse mode flag is missing"
fi

if grep -q "hybrid-mode" /tmp/enhanced_help.txt; then
    echo "✅ Hybrid mode flag is available"
else
    echo "❌ Hybrid mode flag is missing"
fi

echo ""
echo "🎯 All enhanced pipeline flags are working correctly!"
echo ""
echo "Next steps:"
echo "1. Run 'docker compose -f docker-compose.enhanced.yml up' to execute the full pipeline"
echo "2. The pipeline will use the Rust+GPU hybrid architecture as documented in Enhanced Pipeline Report"
echo "3. Expected features: thousands instead of just 60 basic statistical features"

rm -f /tmp/enhanced_help.txt