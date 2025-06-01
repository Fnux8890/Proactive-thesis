#!/usr/bin/env python3
"""
Full Experiment Evaluation Framework

Evaluates MOEA performance against real-world greenhouse data using LightGBM models.
Assesses energy efficiency, plant growth, and economic impact of optimization strategies.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import pickle

# Database and ML imports
import psycopg2
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FullExperimentEvaluator:
    """Comprehensive evaluation of MOEA optimization results"""
    
    def __init__(self, experiment_dir: str, database_url: str):
        self.experiment_dir = Path(experiment_dir)
        self.database_url = database_url
        self.results = {}
        
        # Create evaluation output directory
        self.output_dir = self.experiment_dir / "evaluation_results"
        self.output_dir.mkdir(exist_ok=True)
        
    def load_moea_results(self, device: str) -> Dict:
        """Load MOEA optimization results for CPU or GPU"""
        device_dir = self.experiment_dir / f"moea_{device}"
        results = {}
        
        if not device_dir.exists():
            logger.warning(f"MOEA results directory not found: {device_dir}")
            return results
            
        # Load Pareto front solutions
        pareto_files = list(device_dir.glob("*/pareto_*.npy"))
        if pareto_files:
            pareto_fronts = []
            for file in pareto_files:
                pareto_fronts.append(np.load(file))
            results['pareto_fronts'] = pareto_fronts
            
        # Load metrics
        metrics_files = list(device_dir.glob("*/metrics.json"))
        if metrics_files:
            metrics = []
            for file in metrics_files:
                with open(file, 'r') as f:
                    metrics.append(json.load(f))
            results['metrics'] = metrics
            
        # Load convergence history
        convergence_files = list(device_dir.glob("*/convergence.csv"))
        if convergence_files:
            convergence_data = []
            for file in convergence_files:
                convergence_data.append(pd.read_csv(file))
            results['convergence'] = convergence_data
            
        return results
    
    def load_lightgbm_models(self) -> Dict:
        """Load trained LightGBM models for validation"""
        models_dir = self.experiment_dir / "models"
        models = {}
        
        # Energy consumption model
        energy_model_path = models_dir / "energy_consumption_model.pt"
        if energy_model_path.exists():
            models['energy'] = joblib.load(energy_model_path)
            
        # Plant growth model
        growth_model_path = models_dir / "plant_growth_model.pt"
        if growth_model_path.exists():
            models['growth'] = joblib.load(growth_model_path)
            
        # Check for additional models
        for model_file in models_dir.glob("*.joblib"):
            model_name = model_file.stem
            models[model_name] = joblib.load(model_file)
            
        logger.info(f"Loaded {len(models)} models: {list(models.keys())}")
        return models
    
    def get_historical_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Retrieve historical greenhouse data for validation"""
        
        conn = psycopg2.connect(self.database_url)
        
        query = """
        SELECT 
            timestamp,
            temperature_greenhouse,
            humidity_relative,
            co2_concentration,
            light_par,
            energy_consumption,
            ventilation_rate,
            growth_rate,
            biomass_accumulation
        FROM enhanced_sparse_features_full 
        WHERE timestamp BETWEEN %s AND %s
        ORDER BY timestamp
        """
        
        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        conn.close()
        
        return df
    
    def validate_moea_solutions(self, models: Dict, historical_data: pd.DataFrame) -> Dict:
        """Validate MOEA solutions against real greenhouse performance"""
        
        validation_results = {}
        
        # CPU results validation
        cpu_results = self.load_moea_results('cpu')
        if 'pareto_fronts' in cpu_results:
            cpu_validation = self._validate_pareto_solutions(
                cpu_results['pareto_fronts'], models, historical_data, 'CPU'
            )
            validation_results['cpu'] = cpu_validation
            
        # GPU results validation
        gpu_results = self.load_moea_results('gpu')
        if 'pareto_fronts' in gpu_results:
            gpu_validation = self._validate_pareto_solutions(
                gpu_results['pareto_fronts'], models, historical_data, 'GPU'
            )
            validation_results['gpu'] = gpu_validation
            
        return validation_results
    
    def _validate_pareto_solutions(self, pareto_fronts: List[np.ndarray], 
                                 models: Dict, historical_data: pd.DataFrame,
                                 device: str) -> Dict:
        """Validate Pareto solutions using LightGBM models"""
        
        validation_results = {
            'device': device,
            'solution_quality': [],
            'energy_predictions': [],
            'growth_predictions': [],
            'real_world_performance': {},
            'economic_impact': {}
        }
        
        for i, pareto_front in enumerate(pareto_fronts):
            logger.info(f"Validating {device} Pareto front {i+1}/{len(pareto_fronts)}")
            
            # Extract decision variables (first 6 columns typically)
            n_vars = min(6, pareto_front.shape[1] // 2)  # Assume half are decisions, half objectives
            decisions = pareto_front[:, :n_vars]
            objectives = pareto_front[:, n_vars:]
            
            # Validate each solution
            for j, (decision_vars, objective_vals) in enumerate(zip(decisions, objectives)):
                
                # Create feature vector for LightGBM prediction
                features = self._create_feature_vector(decision_vars, historical_data)
                
                # Predict using LightGBM models
                predictions = {}
                if 'energy' in models:
                    predictions['energy'] = models['energy'].predict([features])[0]
                if 'growth' in models:
                    predictions['growth'] = models['growth'].predict([features])[0]
                
                # Calculate solution quality metrics
                quality_metrics = self._calculate_solution_quality(
                    decision_vars, objective_vals, predictions, historical_data
                )
                
                validation_results['solution_quality'].append(quality_metrics)
                validation_results['energy_predictions'].append(predictions.get('energy', 0))
                validation_results['growth_predictions'].append(predictions.get('growth', 0))
        
        # Calculate aggregate performance metrics
        validation_results['real_world_performance'] = self._calculate_real_world_performance(
            validation_results, historical_data
        )
        
        # Calculate economic impact
        validation_results['economic_impact'] = self._calculate_economic_impact(
            validation_results, historical_data
        )
        
        return validation_results
    
    def _create_feature_vector(self, decision_vars: np.ndarray, 
                             historical_data: pd.DataFrame) -> np.ndarray:
        """Create feature vector from decision variables for LightGBM prediction"""
        
        # Decision variables: [temp, humidity, co2, photoperiod, light_intensity, ventilation]
        temp_setpoint, humidity_target, co2_target = decision_vars[:3]
        photoperiod, light_intensity, ventilation_rate = decision_vars[3:6]
        
        # Create comprehensive feature vector based on historical patterns
        historical_means = historical_data.select_dtypes(include=[np.number]).mean()
        
        # Base features from decision variables
        features = [
            temp_setpoint,
            humidity_target, 
            co2_target,
            light_intensity,
            ventilation_rate,
            photoperiod
        ]
        
        # Add contextual features from historical data
        features.extend([
            historical_means.get('energy_consumption', 100),
            historical_means.get('growth_rate', 0.1),
            temp_setpoint - historical_means.get('temperature_greenhouse', 20),  # Temperature delta
            humidity_target - historical_means.get('humidity_relative', 60),    # Humidity delta
            co2_target - historical_means.get('co2_concentration', 400),        # CO2 delta
        ])
        
        return np.array(features)
    
    def _calculate_solution_quality(self, decision_vars: np.ndarray, 
                                  objective_vals: np.ndarray, predictions: Dict,
                                  historical_data: pd.DataFrame) -> Dict:
        """Calculate solution quality metrics"""
        
        # Energy efficiency (lower is better)
        energy_efficiency = 1.0 / (predictions.get('energy', 1) + 1e-6)
        
        # Growth effectiveness (higher is better)
        growth_effectiveness = predictions.get('growth', 0)
        
        # Operational feasibility (based on realistic ranges)
        temp_feasible = 15 <= decision_vars[0] <= 30
        humidity_feasible = 40 <= decision_vars[1] <= 90
        co2_feasible = 400 <= decision_vars[2] <= 1200
        
        feasibility_score = sum([temp_feasible, humidity_feasible, co2_feasible]) / 3.0
        
        # Stability metric (how close to historical operating points)
        historical_means = historical_data.select_dtypes(include=[np.number]).mean()
        temp_stability = 1.0 - abs(decision_vars[0] - historical_means.get('temperature_greenhouse', 20)) / 15
        humidity_stability = 1.0 - abs(decision_vars[1] - historical_means.get('humidity_relative', 60)) / 50
        
        stability_score = (temp_stability + humidity_stability) / 2.0
        
        return {
            'energy_efficiency': energy_efficiency,
            'growth_effectiveness': growth_effectiveness,
            'feasibility_score': max(0, feasibility_score),
            'stability_score': max(0, stability_score),
            'overall_quality': (energy_efficiency + growth_effectiveness + feasibility_score + stability_score) / 4.0
        }
    
    def _calculate_real_world_performance(self, validation_results: Dict, 
                                        historical_data: pd.DataFrame) -> Dict:
        """Calculate real-world performance metrics"""
        
        solution_qualities = validation_results['solution_quality']
        energy_predictions = validation_results['energy_predictions'] 
        growth_predictions = validation_results['growth_predictions']
        
        if not solution_qualities:
            return {}
        
        # Average performance metrics
        avg_energy_efficiency = np.mean([sq['energy_efficiency'] for sq in solution_qualities])
        avg_growth_effectiveness = np.mean([sq['growth_effectiveness'] for sq in solution_qualities])
        avg_feasibility = np.mean([sq['feasibility_score'] for sq in solution_qualities])
        avg_stability = np.mean([sq['stability_score'] for sq in solution_qualities])
        
        # Baseline performance from historical data
        historical_energy = historical_data['energy_consumption'].mean() if 'energy_consumption' in historical_data else 100
        historical_growth = historical_data['growth_rate'].mean() if 'growth_rate' in historical_data else 0.1
        
        # Performance improvements
        energy_improvement = (historical_energy - np.mean(energy_predictions)) / historical_energy * 100
        growth_improvement = (np.mean(growth_predictions) - historical_growth) / historical_growth * 100
        
        return {
            'avg_energy_efficiency': avg_energy_efficiency,
            'avg_growth_effectiveness': avg_growth_effectiveness,
            'avg_feasibility': avg_feasibility,
            'avg_stability': avg_stability,
            'energy_improvement_percent': energy_improvement,
            'growth_improvement_percent': growth_improvement,
            'overall_performance_score': (avg_energy_efficiency + avg_growth_effectiveness + avg_feasibility + avg_stability) / 4.0
        }
    
    def _calculate_economic_impact(self, validation_results: Dict, 
                                 historical_data: pd.DataFrame) -> Dict:
        """Calculate economic impact of optimization strategies"""
        
        energy_predictions = validation_results['energy_predictions']
        growth_predictions = validation_results['growth_predictions']
        
        if not energy_predictions or not growth_predictions:
            return {}
        
        # Economic parameters (Danish context)
        energy_price_kwh = 0.25  # EUR per kWh (approximate Danish spot price)
        plant_value_per_unit = 2.50  # EUR per plant unit
        operational_days_per_year = 365
        
        # Annual energy consumption and costs
        avg_daily_energy = np.mean(energy_predictions)
        annual_energy_consumption = avg_daily_energy * operational_days_per_year
        annual_energy_cost = annual_energy_consumption * energy_price_kwh
        
        # Annual plant production value
        avg_daily_growth = np.mean(growth_predictions)
        annual_plant_production = avg_daily_growth * operational_days_per_year
        annual_production_value = annual_plant_production * plant_value_per_unit
        
        # Historical baseline
        historical_energy = historical_data['energy_consumption'].mean() if 'energy_consumption' in historical_data else 100
        historical_growth = historical_data['growth_rate'].mean() if 'growth_rate' in historical_data else 0.1
        
        baseline_annual_energy_cost = historical_energy * operational_days_per_year * energy_price_kwh
        baseline_annual_production_value = historical_growth * operational_days_per_year * plant_value_per_unit
        
        # Economic improvements
        energy_cost_savings = baseline_annual_energy_cost - annual_energy_cost
        production_value_increase = annual_production_value - baseline_annual_production_value
        net_economic_benefit = energy_cost_savings + production_value_increase
        
        return {
            'annual_energy_cost_eur': annual_energy_cost,
            'annual_production_value_eur': annual_production_value,
            'energy_cost_savings_eur': energy_cost_savings,
            'production_value_increase_eur': production_value_increase,
            'net_economic_benefit_eur': net_economic_benefit,
            'roi_percent': (net_economic_benefit / baseline_annual_energy_cost) * 100 if baseline_annual_energy_cost > 0 else 0
        }
    
    def compare_cpu_gpu_performance(self) -> Dict:
        """Compare CPU vs GPU MOEA performance"""
        
        comparison = {
            'performance_metrics': {},
            'convergence_analysis': {},
            'solution_quality': {},
            'computational_efficiency': {}
        }
        
        # Load results for both devices
        cpu_results = self.load_moea_results('cpu')
        gpu_results = self.load_moea_results('gpu') 
        
        # Performance comparison
        if 'metrics' in cpu_results and 'metrics' in gpu_results:
            cpu_metrics = cpu_results['metrics'][0] if cpu_results['metrics'] else {}
            gpu_metrics = gpu_results['metrics'][0] if gpu_results['metrics'] else {}
            
            comparison['performance_metrics'] = {
                'cpu_hypervolume': cpu_metrics.get('hypervolume', 0),
                'gpu_hypervolume': gpu_metrics.get('hypervolume', 0),
                'cpu_igd': cpu_metrics.get('igd', float('inf')),
                'gpu_igd': gpu_metrics.get('igd', float('inf')),
                'cpu_runtime_seconds': cpu_metrics.get('runtime', 0),
                'gpu_runtime_seconds': gpu_metrics.get('runtime', 0)
            }
            
            # Calculate speedup
            if cpu_metrics.get('runtime', 0) > 0 and gpu_metrics.get('runtime', 0) > 0:
                speedup = cpu_metrics['runtime'] / gpu_metrics['runtime']
                comparison['computational_efficiency']['speedup'] = speedup
        
        return comparison
    
    def generate_comprehensive_report(self):
        """Generate comprehensive evaluation report"""
        
        logger.info("Starting comprehensive evaluation...")
        
        # Load models and data
        models = self.load_lightgbm_models()
        historical_data = self.get_historical_data("2013-12-01", "2016-09-08")
        
        # Validate MOEA solutions
        validation_results = self.validate_moea_solutions(models, historical_data)
        
        # Compare CPU vs GPU performance
        performance_comparison = self.compare_cpu_gpu_performance()
        
        # Generate evaluation report
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'experiment_directory': str(self.experiment_dir),
            'data_summary': {
                'historical_data_points': len(historical_data),
                'models_loaded': list(models.keys()),
                'evaluation_period': "2013-12-01 to 2016-09-08"
            },
            'validation_results': validation_results,
            'performance_comparison': performance_comparison,
            'key_findings': self._generate_key_findings(validation_results, performance_comparison),
            'recommendations': self._generate_recommendations(validation_results, performance_comparison)
        }
        
        # Save report
        report_path = self.output_dir / "comprehensive_evaluation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate visualizations
        self._generate_visualizations(validation_results, performance_comparison)
        
        # Generate markdown summary
        self._generate_markdown_summary(report)
        
        logger.info(f"Comprehensive evaluation completed. Results saved to {self.output_dir}")
        
        return report
    
    def _generate_key_findings(self, validation_results: Dict, comparison: Dict) -> List[str]:
        """Generate key findings from evaluation"""
        
        findings = []
        
        # CPU vs GPU findings
        if 'cpu' in validation_results and 'gpu' in validation_results:
            cpu_performance = validation_results['cpu'].get('real_world_performance', {})
            gpu_performance = validation_results['gpu'].get('real_world_performance', {})
            
            cpu_score = cpu_performance.get('overall_performance_score', 0)
            gpu_score = gpu_performance.get('overall_performance_score', 0)
            
            if abs(cpu_score - gpu_score) < 0.05:
                findings.append("CPU and GPU MOEA produce similar solution quality")
            elif gpu_score > cpu_score:
                findings.append(f"GPU MOEA produces superior solutions (score: {gpu_score:.3f} vs {cpu_score:.3f})")
            else:
                findings.append(f"CPU MOEA produces superior solutions (score: {cpu_score:.3f} vs {gpu_score:.3f})")
        
        # Performance improvement findings
        for device in ['cpu', 'gpu']:
            if device in validation_results:
                perf = validation_results[device].get('real_world_performance', {})
                energy_improvement = perf.get('energy_improvement_percent', 0)
                growth_improvement = perf.get('growth_improvement_percent', 0)
                
                if energy_improvement > 10:
                    findings.append(f"{device.upper()} optimization achieves {energy_improvement:.1f}% energy reduction")
                if growth_improvement > 10:
                    findings.append(f"{device.upper()} optimization achieves {growth_improvement:.1f}% growth improvement")
        
        # Economic impact findings
        for device in ['cpu', 'gpu']:
            if device in validation_results:
                economic = validation_results[device].get('economic_impact', {})
                net_benefit = economic.get('net_economic_benefit_eur', 0)
                roi = economic.get('roi_percent', 0)
                
                if net_benefit > 1000:
                    findings.append(f"{device.upper()} optimization provides €{net_benefit:.0f} annual benefit")
                if roi > 20:
                    findings.append(f"{device.upper()} optimization achieves {roi:.1f}% ROI")
        
        # Computational efficiency findings
        efficiency = comparison.get('computational_efficiency', {})
        speedup = efficiency.get('speedup', 1)
        if speedup > 2:
            findings.append(f"GPU acceleration provides {speedup:.1f}x speedup over CPU")
        
        return findings
    
    def _generate_recommendations(self, validation_results: Dict, comparison: Dict) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Performance-based recommendations
        if 'cpu' in validation_results and 'gpu' in validation_results:
            cpu_perf = validation_results['cpu'].get('real_world_performance', {})
            gpu_perf = validation_results['gpu'].get('real_world_performance', {})
            
            cpu_energy = cpu_perf.get('energy_improvement_percent', 0)
            gpu_energy = gpu_perf.get('energy_improvement_percent', 0)
            
            if max(cpu_energy, gpu_energy) > 15:
                recommendations.append("Implement MOEA-optimized control strategies for significant energy savings")
            
            if gpu_perf.get('avg_feasibility', 0) > cpu_perf.get('avg_feasibility', 0):
                recommendations.append("Use GPU-based optimization for more feasible control strategies")
        
        # Economic recommendations
        for device in ['cpu', 'gpu']:
            if device in validation_results:
                economic = validation_results[device].get('economic_impact', {})
                roi = economic.get('roi_percent', 0)
                
                if roi > 30:
                    recommendations.append(f"Deploy {device.upper()}-optimized control system for high ROI ({roi:.1f}%)")
        
        # Computational recommendations
        efficiency = comparison.get('computational_efficiency', {})
        speedup = efficiency.get('speedup', 1)
        
        if speedup > 4:
            recommendations.append("Use GPU acceleration for real-time optimization applications")
        elif speedup < 2:
            recommendations.append("Consider CPU implementation for cost-effective deployment")
        
        # Technical recommendations
        recommendations.extend([
            "Validate optimized strategies with additional real-world greenhouse trials",
            "Implement adaptive control system that adjusts to seasonal variations",
            "Monitor energy market prices for dynamic optimization objectives",
            "Consider multi-objective trade-offs based on business priorities"
        ])
        
        return recommendations
    
    def _generate_visualizations(self, validation_results: Dict, comparison: Dict):
        """Generate evaluation visualizations"""
        
        plt.style.use('seaborn-v0_8')
        
        # Performance comparison chart
        if 'cpu' in validation_results and 'gpu' in validation_results:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            devices = ['CPU', 'GPU']
            colors = ['#1f77b4', '#ff7f0e']
            
            # Energy improvement comparison
            energy_improvements = [
                validation_results['cpu'].get('real_world_performance', {}).get('energy_improvement_percent', 0),
                validation_results['gpu'].get('real_world_performance', {}).get('energy_improvement_percent', 0)
            ]
            
            axes[0, 0].bar(devices, energy_improvements, color=colors)
            axes[0, 0].set_title('Energy Improvement (%)')
            axes[0, 0].set_ylabel('Improvement %')
            
            # Growth improvement comparison
            growth_improvements = [
                validation_results['cpu'].get('real_world_performance', {}).get('growth_improvement_percent', 0),
                validation_results['gpu'].get('real_world_performance', {}).get('growth_improvement_percent', 0)
            ]
            
            axes[0, 1].bar(devices, growth_improvements, color=colors)
            axes[0, 1].set_title('Growth Improvement (%)')
            axes[0, 1].set_ylabel('Improvement %')
            
            # Economic benefit comparison
            economic_benefits = [
                validation_results['cpu'].get('economic_impact', {}).get('net_economic_benefit_eur', 0),
                validation_results['gpu'].get('economic_impact', {}).get('net_economic_benefit_eur', 0)
            ]
            
            axes[1, 0].bar(devices, economic_benefits, color=colors)
            axes[1, 0].set_title('Annual Economic Benefit (EUR)')
            axes[1, 0].set_ylabel('Benefit (EUR)')
            
            # Overall performance score
            performance_scores = [
                validation_results['cpu'].get('real_world_performance', {}).get('overall_performance_score', 0),
                validation_results['gpu'].get('real_world_performance', {}).get('overall_performance_score', 0)
            ]
            
            axes[1, 1].bar(devices, performance_scores, color=colors)
            axes[1, 1].set_title('Overall Performance Score')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_ylim(0, 1)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info("Visualizations generated successfully")
    
    def _generate_markdown_summary(self, report: Dict):
        """Generate markdown summary report"""
        
        markdown_content = f"""# Full Experiment Evaluation Report

## Executive Summary

This report evaluates the performance of MOEA-optimized greenhouse control strategies using the complete 2013-2016 dataset. The evaluation compares CPU and GPU implementations and assesses real-world performance using trained LightGBM models.

**Evaluation Date:** {report['evaluation_timestamp']}  
**Data Period:** {report['data_summary']['evaluation_period']}  
**Data Points:** {report['data_summary']['historical_data_points']:,}

## Key Findings

"""
        
        for finding in report['key_findings']:
            markdown_content += f"- {finding}\n"
        
        markdown_content += "\n## Performance Metrics\n\n"
        
        # Add performance tables
        if 'cpu' in report['validation_results'] and 'gpu' in report['validation_results']:
            markdown_content += """| Metric | CPU | GPU | Improvement |
|--------|-----|-----|-------------|
"""
            
            cpu_perf = report['validation_results']['cpu'].get('real_world_performance', {})
            gpu_perf = report['validation_results']['gpu'].get('real_world_performance', {})
            
            energy_cpu = cpu_perf.get('energy_improvement_percent', 0)
            energy_gpu = gpu_perf.get('energy_improvement_percent', 0)
            growth_cpu = cpu_perf.get('growth_improvement_percent', 0)
            growth_gpu = gpu_perf.get('growth_improvement_percent', 0)
            
            markdown_content += f"| Energy Improvement (%) | {energy_cpu:.1f} | {energy_gpu:.1f} | {gpu_energy - energy_cpu:.1f} |\n"
            markdown_content += f"| Growth Improvement (%) | {growth_cpu:.1f} | {growth_gpu:.1f} | {gpu_growth - growth_cpu:.1f} |\n"
        
        markdown_content += "\n## Economic Impact\n\n"
        
        for device in ['cpu', 'gpu']:
            if device in report['validation_results']:
                economic = report['validation_results'][device].get('economic_impact', {})
                benefit = economic.get('net_economic_benefit_eur', 0)
                roi = economic.get('roi_percent', 0)
                
                markdown_content += f"**{device.upper()} Optimization:**\n"
                markdown_content += f"- Annual Economic Benefit: €{benefit:,.0f}\n"
                markdown_content += f"- Return on Investment: {roi:.1f}%\n\n"
        
        markdown_content += "## Recommendations\n\n"
        
        for i, recommendation in enumerate(report['recommendations'], 1):
            markdown_content += f"{i}. {recommendation}\n"
        
        markdown_content += f"""
## Technical Details

- **Models Used:** {', '.join(report['data_summary']['models_loaded'])}
- **Validation Method:** LightGBM surrogate model validation
- **Evaluation Metrics:** Hypervolume, IGD, solution quality, economic impact
- **Results Directory:** `{report['experiment_directory']}`

## Files Generated

- `comprehensive_evaluation_report.json` - Complete evaluation data
- `performance_comparison.png` - Performance visualization
- `evaluation_summary.md` - This summary report

---

*Generated by Full Experiment Evaluation Framework*
"""
        
        # Save markdown report
        with open(self.output_dir / "evaluation_summary.md", 'w') as f:
            f.write(markdown_content)

def main():
    """Main evaluation function"""
    
    # Configuration
    experiment_dir = os.getenv('EXPERIMENT_DATA_DIR', '/app/experiment_data')
    database_url = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@db:5432/postgres')
    
    # Create evaluator
    evaluator = FullExperimentEvaluator(experiment_dir, database_url)
    
    # Run comprehensive evaluation
    try:
        report = evaluator.generate_comprehensive_report()
        
        print("=== FULL EXPERIMENT EVALUATION COMPLETE ===")
        print(f"Results saved to: {evaluator.output_dir}")
        print(f"Total data points analyzed: {report['data_summary']['historical_data_points']:,}")
        print(f"Models validated: {', '.join(report['data_summary']['models_loaded'])}")
        
        # Print key findings
        print("\nKEY FINDINGS:")
        for finding in report['key_findings']:
            print(f"  ✓ {finding}")
        
        # Print top recommendations
        print("\nTOP RECOMMENDATIONS:")
        for i, recommendation in enumerate(report['recommendations'][:3], 1):
            print(f"  {i}. {recommendation}")
        
        print(f"\nDetailed results: {evaluator.output_dir}/evaluation_summary.md")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()