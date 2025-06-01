#!/bin/bash

# Docker-based Benchmark Runner for Enhanced Sparse Pipeline
# This script runs benchmarks entirely within Docker containers

EXPERIMENT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$EXPERIMENT_DIR/../../../.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Enhanced Sparse Pipeline Docker Benchmark${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Timestamp: $TIMESTAMP"
echo "Project root: $PROJECT_ROOT"
echo ""

# Create results directory
RESULTS_DIR="$EXPERIMENT_DIR/results/${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

# Create benchmark compose file
cat > "$RESULTS_DIR/docker-compose.benchmark.yml" << 'EOF'
services:
  benchmark_runner:
    image: python:3.11-slim
    container_name: benchmark_runner
    working_dir: /workspace
    environment:
      DOCKER_HOST: unix:///var/run/docker.sock
      PYTHONUNBUFFERED: 1
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - ./:/workspace
      - ../../../:/project
    command: python run_benchmarks.py
    networks:
      - pipeline-net

networks:
  pipeline-net:
    external: true
    name: dataingestion_pipeline-net
EOF

# Create Python benchmark runner
cat > "$RESULTS_DIR/run_benchmarks.py" << 'EOF'
import subprocess
import json
import time
import os
from datetime import datetime
import csv

# Benchmark configurations
BENCHMARKS = [
    # Basic vs Enhanced - 1 month
    {"name": "basic_gpu_1mo", "mode": "basic", "gpu": True, "batch": 24, "start": "2014-01-01", "end": "2014-01-31"},
    {"name": "basic_cpu_1mo", "mode": "basic", "gpu": False, "batch": 24, "start": "2014-01-01", "end": "2014-01-31"},
    {"name": "enhanced_gpu_1mo", "mode": "enhanced", "gpu": True, "batch": 24, "start": "2014-01-01", "end": "2014-01-31"},
    {"name": "enhanced_cpu_1mo", "mode": "enhanced", "gpu": False, "batch": 24, "start": "2014-01-01", "end": "2014-01-31"},
    
    # Batch size comparison
    {"name": "enhanced_gpu_b12", "mode": "enhanced", "gpu": True, "batch": 12, "start": "2014-01-01", "end": "2014-01-31"},
    {"name": "enhanced_gpu_b48", "mode": "enhanced", "gpu": True, "batch": 48, "start": "2014-01-01", "end": "2014-01-31"},
    
    # Larger datasets
    {"name": "basic_gpu_3mo", "mode": "basic", "gpu": True, "batch": 24, "start": "2014-01-01", "end": "2014-03-31"},
    {"name": "enhanced_gpu_3mo", "mode": "enhanced", "gpu": True, "batch": 24, "start": "2014-01-01", "end": "2014-03-31"},
    
    # 6 month test
    {"name": "enhanced_gpu_6mo", "mode": "enhanced", "gpu": True, "batch": 24, "start": "2014-01-01", "end": "2014-06-30"},
]

def run_benchmark(config):
    """Run a single benchmark configuration."""
    print(f"\n{'='*60}")
    print(f"Running: {config['name']}")
    print(f"Mode: {config['mode']}, GPU: {config['gpu']}, Batch: {config['batch']}")
    print(f"Period: {config['start']} to {config['end']}")
    print(f"{'='*60}")
    
    # Set environment variables
    env = os.environ.copy()
    env['SPARSE_MODE'] = 'true'
    env['ENHANCED_MODE'] = 'true' if config['mode'] == 'enhanced' else 'false'
    env['DISABLE_GPU'] = 'false' if config['gpu'] else 'true'
    env['SPARSE_BATCH_SIZE'] = str(config['batch'])
    env['SPARSE_START_DATE'] = config['start']
    env['SPARSE_END_DATE'] = config['end']
    
    # Build command
    cmd = [
        'docker', 'compose', 
        '-f', '/project/DataIngestion/docker-compose.sparse.yml',
        'run', '--rm', 'sparse_pipeline'
    ]
    
    if config['mode'] == 'enhanced':
        cmd.extend(['--enhanced-mode'])
    else:
        cmd.extend(['--sparse-mode'])
    
    cmd.extend([
        '--start-date', config['start'],
        '--end-date', config['end'],
        '--batch-size', str(config['batch'])
    ])
    
    # Run benchmark
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        end_time = time.time()
        duration = end_time - start_time
        
        # Parse output
        output = result.stdout
        
        # Extract metrics
        metrics = {
            'name': config['name'],
            'mode': config['mode'],
            'device': 'gpu' if config['gpu'] else 'cpu',
            'batch_size': config['batch'],
            'start_date': config['start'],
            'end_date': config['end'],
            'duration': duration,
            'status': 'success' if result.returncode == 0 else 'failed'
        }
        
        # Parse feature count
        if 'Total feature sets:' in output:
            feature_count = int(output.split('Total feature sets:')[1].split()[0])
            metrics['feature_count'] = feature_count
        elif 'Window features:' in output:
            feature_count = int(output.split('Window features:')[1].split()[0])
            metrics['feature_count'] = feature_count
        else:
            metrics['feature_count'] = 0
        
        # Parse features per second
        if 'feature sets/second' in output:
            fps = float(output.split('feature sets/second')[0].split()[-1])
            metrics['features_per_second'] = fps
        elif 'features/second' in output:
            fps = float(output.split('features/second')[0].split()[-1])
            metrics['features_per_second'] = fps
        else:
            metrics['features_per_second'] = 0
        
        # Parse GPU utilization if available
        if config['gpu'] and 'GPU utilization:' in output:
            gpu_util = float(output.split('GPU utilization:')[1].split('%')[0])
            metrics['gpu_utilization'] = gpu_util
        else:
            metrics['gpu_utilization'] = 0
        
        print(f"\n✓ Completed in {duration:.2f}s")
        print(f"  Features: {metrics.get('feature_count', 0)}")
        print(f"  Speed: {metrics.get('features_per_second', 0):.1f} feat/s")
        if config['gpu']:
            print(f"  GPU Utilization: {metrics.get('gpu_utilization', 0):.1f}%")
        
        return metrics
        
    except Exception as e:
        print(f"\n✗ Failed: {str(e)}")
        return {
            'name': config['name'],
            'mode': config['mode'],
            'device': 'gpu' if config['gpu'] else 'cpu',
            'status': 'error',
            'error': str(e)
        }

def main():
    print("Starting Enhanced Sparse Pipeline Benchmarks")
    print(f"Running {len(BENCHMARKS)} benchmark configurations")
    
    # Results storage
    results = []
    
    # Ensure database is running
    print("\nEnsuring database is ready...")
    subprocess.run([
        'docker', 'compose', 
        '-f', '/project/DataIngestion/docker-compose.sparse.yml',
        'up', '-d', 'db'
    ])
    time.sleep(10)
    
    # Run benchmarks
    for config in BENCHMARKS:
        result = run_benchmark(config)
        results.append(result)
        
        # Cool down between tests
        time.sleep(5)
    
    # Save results
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    with open('benchmark_results.csv', 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    
    # Generate summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    # Calculate speedups
    basic_gpu = next((r for r in results if r['name'] == 'basic_gpu_1mo'), None)
    basic_cpu = next((r for r in results if r['name'] == 'basic_cpu_1mo'), None)
    enhanced_gpu = next((r for r in results if r['name'] == 'enhanced_gpu_1mo'), None)
    enhanced_cpu = next((r for r in results if r['name'] == 'enhanced_cpu_1mo'), None)
    
    if basic_gpu and basic_cpu and basic_gpu['status'] == 'success' and basic_cpu['status'] == 'success':
        speedup = basic_cpu['duration'] / basic_gpu['duration']
        print(f"Basic Mode GPU Speedup: {speedup:.1f}x")
    
    if enhanced_gpu and enhanced_cpu and enhanced_gpu['status'] == 'success' and enhanced_cpu['status'] == 'success':
        speedup = enhanced_cpu['duration'] / enhanced_gpu['duration']
        print(f"Enhanced Mode GPU Speedup: {speedup:.1f}x")
    
    if basic_gpu and enhanced_gpu and basic_gpu['status'] == 'success' and enhanced_gpu['status'] == 'success':
        feature_ratio = enhanced_gpu.get('feature_count', 0) / max(basic_gpu.get('feature_count', 1), 1)
        print(f"Feature Enhancement: {feature_ratio:.1f}x")
        print(f"Enhanced GPU Utilization: {enhanced_gpu.get('gpu_utilization', 0):.1f}%")
    
    print("\nBenchmark complete!")

if __name__ == "__main__":
    main()
EOF

# Change to results directory and run
cd "$RESULTS_DIR"

echo -e "${BLUE}Starting Docker-based benchmarks...${NC}"
echo "This will run multiple configurations and may take 10-20 minutes."
echo ""

# Run the benchmark container
docker compose -f docker-compose.benchmark.yml run --rm benchmark_runner

# Display results
echo -e "\n${GREEN}Benchmark Complete!${NC}"
echo "Results saved to: $RESULTS_DIR"

if [ -f "benchmark_results.csv" ]; then
    echo -e "\n${BLUE}Quick Results Summary:${NC}"
    column -t -s',' benchmark_results.csv | head -20
fi

if [ -f "benchmark_results.json" ]; then
    echo -e "\n${BLUE}Top Performers:${NC}"
    cat benchmark_results.json | \
        jq -r '.[] | select(.status=="success") | "\(.name): \(.features_per_second // 0) feat/s (\(.feature_count // 0) features in \(.duration | tonumber | round)s)"' | \
        sort -t':' -k2 -nr | head -5
fi