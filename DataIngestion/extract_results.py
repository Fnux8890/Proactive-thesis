#!/usr/bin/env python3
"""
Results Extraction Tool for Enhanced Sparse Pipeline
Extracts comprehensive features and MOEA results for analysis
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import psycopg2
from sqlalchemy import create_engine

def extract_enhanced_features(database_url, output_dir="./analysis_results"):
    """Extract enhanced features from database for analysis"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    engine = create_engine(database_url)
    
    print("🔄 Extracting enhanced features from database...")
    
    # Extract all enhanced features
    query = """
    SELECT 
        era_id,
        resolution,
        computed_at,
        sensor_features,
        extended_stats,
        weather_features,
        energy_features,
        growth_features,
        temporal_features,
        optimization_metrics
    FROM enhanced_sparse_features_full
    ORDER BY computed_at, era_id
    """
    
    df = pd.read_sql_query(query, engine)
    print(f"✅ Loaded {len(df)} enhanced feature records")
    
    # Flatten JSONB features for analysis
    flattened_features = []
    
    for idx, row in df.iterrows():
        feature_row = {
            'era_id': row['era_id'],
            'computed_at': row['computed_at'],
            'resolution': row['resolution']
        }
        
        # Extract features from JSONB columns
        jsonb_columns = ['sensor_features', 'extended_stats', 'weather_features', 
                        'energy_features', 'growth_features', 'temporal_features', 
                        'optimization_metrics']
        
        for col in jsonb_columns:
            if row[col] is not None:
                features = row[col] if isinstance(row[col], dict) else json.loads(row[col])
                for feature_name, feature_value in features.items():
                    feature_row[f"{col}_{feature_name}"] = feature_value
        
        flattened_features.append(feature_row)
    
    features_df = pd.DataFrame(flattened_features)
    
    # Save to multiple formats
    features_df.to_csv(output_path / "enhanced_features.csv", index=False)
    features_df.to_parquet(output_path / "enhanced_features.parquet", index=False)
    
    print(f"✅ Enhanced features saved to {output_path}")
    print(f"   - CSV: enhanced_features.csv ({len(features_df)} rows, {len(features_df.columns)} columns)")
    print(f"   - Parquet: enhanced_features.parquet")
    
    return features_df

def extract_moea_results(experiment_dir="./experiments/full_experiment", output_dir="./analysis_results"):
    """Extract MOEA optimization results"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    exp_path = Path(experiment_dir)
    results = {}
    
    print("🔄 Extracting MOEA results...")
    
    # Extract CPU results
    cpu_dir = exp_path / "moea_cpu" / "experiment"
    if cpu_dir.exists():
        cpu_results_file = cpu_dir / "complete_results.json"
        if cpu_results_file.exists():
            with open(cpu_results_file, 'r') as f:
                results['cpu'] = json.load(f)
            print("✅ CPU MOEA results loaded")
        
        # Load Pareto solutions
        pareto_files = list(cpu_dir.glob("*/pareto_*.npy"))
        if pareto_files:
            results['cpu_pareto'] = {}
            for file in pareto_files:
                results['cpu_pareto'][file.stem] = np.load(file)
            print(f"✅ CPU Pareto solutions loaded: {list(results['cpu_pareto'].keys())}")
    
    # Extract GPU results
    gpu_dir = exp_path / "moea_gpu" / "experiment"
    if gpu_dir.exists():
        gpu_results_file = gpu_dir / "complete_results.json"
        if gpu_results_file.exists():
            with open(gpu_results_file, 'r') as f:
                results['gpu'] = json.load(f)
            print("✅ GPU MOEA results loaded")
        
        # Load Pareto solutions
        pareto_files = list(gpu_dir.glob("*/pareto_*.npy"))
        if pareto_files:
            results['gpu_pareto'] = {}
            for file in pareto_files:
                results['gpu_pareto'][file.stem] = np.load(file)
            print(f"✅ GPU Pareto solutions loaded: {list(results['gpu_pareto'].keys())}")
    
    # Save consolidated results
    results_file = output_path / "moea_results.json"
    with open(results_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if 'pareto' in key:
                json_results[key] = {k: v.tolist() for k, v in value.items()}
            else:
                json_results[key] = value
        json.dump(json_results, f, indent=2)
    
    print(f"✅ MOEA results saved to {results_file}")
    
    return results

def generate_performance_summary(database_url, experiment_dir="./experiments/full_experiment", output_dir="./analysis_results"):
    """Generate comprehensive performance summary"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("📊 Generating performance summary...")
    
    # Extract data
    features_df = extract_enhanced_features(database_url, output_dir)
    moea_results = extract_moea_results(experiment_dir, output_dir)
    
    # Generate summary
    summary = {
        "experiment_timestamp": pd.Timestamp.now().isoformat(),
        "data_summary": {
            "total_features": len(features_df),
            "feature_columns": len(features_df.columns),
            "unique_eras": features_df['era_id'].nunique(),
            "resolutions": features_df['resolution'].unique().tolist(),
            "time_range": {
                "start": features_df['computed_at'].min().isoformat(),
                "end": features_df['computed_at'].max().isoformat()
            }
        },
        "performance_comparison": {}
    }
    
    # Add MOEA performance comparison
    if 'cpu' in moea_results and 'gpu' in moea_results:
        cpu_metrics = moea_results['cpu'][0] if moea_results['cpu'] else {}
        gpu_metrics = moea_results['gpu'][0] if moea_results['gpu'] else {}
        
        cpu_runtime = cpu_metrics.get('runtime', {}).get('total', 0)
        gpu_runtime = gpu_metrics.get('runtime', {}).get('total', 0)
        
        summary["performance_comparison"] = {
            "cpu_runtime_seconds": cpu_runtime,
            "gpu_runtime_seconds": gpu_runtime,
            "speedup_factor": cpu_runtime / gpu_runtime if gpu_runtime > 0 else 0,
            "cpu_solutions": cpu_metrics.get('metrics', {}).get('n_solutions', {}).get('mean', 0),
            "gpu_solutions": gpu_metrics.get('metrics', {}).get('n_solutions', {}).get('mean', 0),
            "cpu_hypervolume": cpu_metrics.get('metrics', {}).get('hypervolume', {}).get('mean', 0),
            "gpu_hypervolume": gpu_metrics.get('metrics', {}).get('hypervolume', {}).get('mean', 0)
        }
    
    # Save summary
    summary_file = output_path / "performance_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Performance summary saved to {summary_file}")
    
    # Display key metrics
    if summary["performance_comparison"]:
        perf = summary["performance_comparison"]
        print(f"\n📊 KEY PERFORMANCE METRICS:")
        print(f"   🚀 GPU Speedup: {perf['speedup_factor']:.1f}x faster")
        print(f"   ⏱️  CPU Runtime: {perf['cpu_runtime_seconds']:.2f}s")
        print(f"   ⚡ GPU Runtime: {perf['gpu_runtime_seconds']:.2f}s")
        print(f"   🎯 CPU Solutions: {perf['cpu_solutions']}")
        print(f"   🎯 GPU Solutions: {perf['gpu_solutions']}")
        print(f"   📈 CPU Hypervolume: {perf['cpu_hypervolume']:.3f}")
        print(f"   📈 GPU Hypervolume: {perf['gpu_hypervolume']:.3f}")
    
    return summary

def main():
    """Main extraction function"""
    
    # Configuration
    database_url = os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/postgres')
    experiment_dir = "./experiments/full_experiment"
    output_dir = "./analysis_results"
    
    print("🔬 Enhanced Sparse Pipeline Results Extraction")
    print("=" * 50)
    
    try:
        summary = generate_performance_summary(database_url, experiment_dir, output_dir)
        
        print(f"\n✅ EXTRACTION COMPLETE!")
        print(f"📁 Results saved to: {output_dir}")
        print(f"📄 Key files:")
        print(f"   - enhanced_features.csv (comprehensive feature data)")
        print(f"   - enhanced_features.parquet (compressed format)")
        print(f"   - moea_results.json (optimization results)")
        print(f"   - performance_summary.json (key metrics)")
        
        print(f"\n🚀 Ready for experimentation and analysis!")
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())