# MOEA Integration with Enhanced Sparse Pipeline

## Overview

This document demonstrates how the enhanced sparse pipeline features are designed for multi-objective evolutionary algorithm (MOEA) optimization in greenhouse climate control systems. The pipeline now extracts features that directly support optimization of:

1. **Plant Growth Performance** (maximize)
2. **Energy Cost Efficiency** (minimize cost)
3. **Stress Minimization** (minimize plant stress)

## Enhanced Feature Categories for MOEA

### 1. Plant Growth Optimization Features

```rust
// Growth features from enhanced pipeline
pub struct GrowthObjective {
    pub growing_degree_days: f64,         // Cumulative heat units
    pub daily_light_integral: f64,        // Light quantity
    pub photoperiod_hours: f64,           // Day length
    pub temperature_optimality: f64,      // How close to optimal temp
    pub light_sufficiency: f64,           // Light adequacy ratio
    pub flowering_signal: f64,            // Flowering trigger (0/1)
    pub expected_growth_rate: f64,        // Combined growth score
}

// MOEA objective function
fn calculate_growth_score(features: &GrowthObjective) -> f64 {
    // Weighted combination of growth factors
    let base_score = features.temperature_optimality * 
                     features.light_sufficiency.min(1.0);
    
    // Bonus for optimal photoperiod (species-specific)
    let photoperiod_bonus = if features.flowering_signal > 0.5 { 1.1 } else { 1.0 };
    
    // Penalty for insufficient light
    let light_penalty = if features.light_sufficiency < 0.8 { 0.9 } else { 1.0 };
    
    base_score * photoperiod_bonus * light_penalty
}
```

### 2. Energy Cost Optimization Features

```rust
// Energy features for cost minimization
pub struct EnergyObjective {
    pub cost_weighted_consumption: f64,   // Total cost in DKK
    pub peak_offpeak_ratio: f64,         // Peak vs off-peak usage
    pub energy_efficiency_score: f64,    // kWh per unit output
    pub hours_until_cheap: f64,          // Predictive scheduling
    pub optimal_load_shift_hours: f64,   // Load shifting potential
}

// MOEA objective function
fn calculate_cost_efficiency(features: &EnergyObjective) -> f64 {
    // Minimize cost while maintaining efficiency
    let base_efficiency = features.energy_efficiency_score;
    
    // Penalty for peak hour usage
    let peak_penalty = 1.0 / (1.0 + features.peak_offpeak_ratio * 0.1);
    
    // Reward for load shifting capability
    let shift_bonus = 1.0 + (features.optimal_load_shift_hours / 24.0) * 0.2;
    
    base_efficiency * peak_penalty * shift_bonus
}
```

### 3. Environmental Stress Minimization

```rust
// Stress factors for plant health
pub struct StressObjective {
    pub stress_degree_days: f64,         // Cumulative stress
    pub temp_differential_std: f64,      // Temperature volatility
    pub humidity_stress_hours: f64,      // Hours outside optimal range
    pub co2_depletion_events: f64,       // CO2 shortage incidents
    pub ventilation_effectiveness: f64,   // Air circulation quality
}

// MOEA objective function
fn calculate_stress_minimization(features: &StressObjective) -> f64 {
    // Lower values are better (minimize stress)
    let temp_stress = 1.0 / (1.0 + features.stress_degree_days);
    let volatility_penalty = 1.0 / (1.0 + features.temp_differential_std);
    let environmental_health = features.ventilation_effectiveness;
    
    temp_stress * volatility_penalty * environmental_health
}
```

## MOEA Configuration Example

### Objective Function Implementation

```python
# Python example for MOEA optimizer integration
import numpy as np
from pymoo.core.problem import Problem

class GreenhouseOptimizationProblem(Problem):
    """
    Multi-objective greenhouse climate control optimization
    
    Variables:
    - Temperature setpoint (18-25°C)
    - Lamp intensity (0-100%)
    - CO2 setpoint (400-1200 ppm)
    - Ventilation rate (0-100%)
    
    Objectives:
    1. Maximize plant growth performance
    2. Minimize energy costs
    3. Minimize plant stress
    """
    
    def __init__(self):
        super().__init__(
            n_var=4,  # Control variables
            n_obj=3,  # Objectives
            n_constr=2,  # Constraints
            xl=np.array([18.0, 0.0, 400.0, 0.0]),  # Lower bounds
            xu=np.array([25.0, 100.0, 1200.0, 100.0])  # Upper bounds
        )
        
        # Load enhanced pipeline features for current conditions
        self.current_features = self.load_enhanced_features()
        
    def load_enhanced_features(self):
        """Load features from enhanced sparse pipeline"""
        # Connect to feature database or call pipeline directly
        return {
            'weather_temp_external': 15.0,
            'weather_solar_radiation': 300.0,
            'energy_price_current': 0.25,  # DKK/kWh
            'phenotype_base_temp': 10.0,
            'phenotype_optimal_day': 22.0,
            'phenotype_optimal_night': 18.0,
            'phenotype_light_requirement': 12.0,
        }
    
    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate control strategies using enhanced features
        
        X: Control variables [temp_setpoint, lamp_intensity, co2_setpoint, vent_rate]
        """
        
        # Extract control variables
        temp_setpoint = X[:, 0]
        lamp_intensity = X[:, 1]
        co2_setpoint = X[:, 2]
        vent_rate = X[:, 3]
        
        # Calculate objectives using enhanced features
        f1 = self.calculate_growth_objective(temp_setpoint, lamp_intensity, co2_setpoint)
        f2 = self.calculate_cost_objective(temp_setpoint, lamp_intensity, vent_rate)
        f3 = self.calculate_stress_objective(temp_setpoint, vent_rate)
        
        # MOEA minimizes, so negate growth (which we want to maximize)
        out["F"] = np.column_stack([-f1, f2, f3])
        
        # Constraints
        g1 = self.temperature_stability_constraint(temp_setpoint)
        g2 = self.energy_budget_constraint(lamp_intensity)
        out["G"] = np.column_stack([g1, g2])
    
    def calculate_growth_objective(self, temp_setpoint, lamp_intensity, co2_setpoint):
        """Calculate plant growth performance using enhanced features"""
        
        # Temperature optimality (using phenotype data)
        optimal_temp = self.current_features['phenotype_optimal_day']
        temp_optimality = 1.0 - np.abs(temp_setpoint - optimal_temp) / 10.0
        temp_optimality = np.clip(temp_optimality, 0.0, 1.0)
        
        # Light sufficiency (DLI calculation)
        daily_light_integral = (lamp_intensity / 100.0) * 50.0 * 12.0 / 1000000.0  # mol/m²/day
        light_requirement = self.current_features['phenotype_light_requirement']
        light_sufficiency = daily_light_integral / light_requirement
        
        # CO2 enhancement factor
        co2_enhancement = np.minimum(1.0 + (co2_setpoint - 400.0) / 800.0 * 0.3, 1.3)
        
        # Growing degree days accumulation
        base_temp = self.current_features['phenotype_base_temp']
        gdd_rate = np.maximum(0.0, temp_setpoint - base_temp) / 24.0
        
        # Combined growth score
        growth_score = temp_optimality * \
                      np.minimum(light_sufficiency, 1.0) * \
                      co2_enhancement * \
                      (1.0 + gdd_rate / 20.0)
        
        return growth_score
    
    def calculate_cost_objective(self, temp_setpoint, lamp_intensity, vent_rate):
        """Calculate energy cost using enhanced energy features"""
        
        # Current energy price from enhanced features
        energy_price = self.current_features['energy_price_current']
        external_temp = self.current_features['weather_temp_external']
        
        # Heating cost (temperature differential)
        temp_diff = np.maximum(0.0, temp_setpoint - external_temp)
        heating_cost = temp_diff * 0.5 * energy_price  # kW/°C * price
        
        # Lighting cost
        lighting_cost = (lamp_intensity / 100.0) * 2.0 * energy_price  # 2kW max * price
        
        # Ventilation cost
        ventilation_cost = (vent_rate / 100.0) * 0.5 * energy_price  # 0.5kW max * price
        
        # Peak hour penalty (from enhanced energy features)
        peak_penalty = 1.2 if energy_price > 0.3 else 1.0
        
        total_cost = (heating_cost + lighting_cost + ventilation_cost) * peak_penalty
        
        return total_cost
    
    def calculate_stress_objective(self, temp_setpoint, vent_rate):
        """Calculate plant stress using enhanced stress features"""
        
        optimal_temp = self.current_features['phenotype_optimal_day']
        
        # Temperature stress (distance from optimal)
        temp_stress = np.abs(temp_setpoint - optimal_temp) / 5.0
        
        # Ventilation stress (too little or too much)
        optimal_vent = 30.0  # Optimal ventilation rate
        vent_stress = np.abs(vent_rate - optimal_vent) / 50.0
        
        # Environmental volatility (from weather coupling features)
        weather_volatility = 0.1  # From enhanced features
        
        total_stress = temp_stress + vent_stress + weather_volatility
        
        return total_stress
    
    def temperature_stability_constraint(self, temp_setpoint):
        """Constraint: Temperature changes should be gradual"""
        # Assuming previous setpoint is available
        previous_setpoint = 21.0  # From state
        max_change = 2.0  # Maximum °C change per hour
        
        change = np.abs(temp_setpoint - previous_setpoint)
        return change - max_change  # <= 0
    
    def energy_budget_constraint(self, lamp_intensity):
        """Constraint: Total energy use within budget"""
        max_energy_budget = 5.0  # kW max
        estimated_usage = (lamp_intensity / 100.0) * 2.0 + 2.0  # Base load
        
        return estimated_usage - max_energy_budget  # <= 0

# MOEA Algorithm Configuration
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions

# Reference directions for 3 objectives
ref_dirs = get_reference_directions("uniform", 3, n_partitions=12)

# Configure NSGA-III algorithm
algorithm = NSGA3(
    pop_size=92,  # Population size
    ref_dirs=ref_dirs,
    eliminate_duplicates=True
)

# Run optimization
from pymoo.optimize import minimize

problem = GreenhouseOptimizationProblem()
result = minimize(
    problem,
    algorithm,
    termination=('n_gen', 100),  # 100 generations
    seed=1,
    verbose=True
)

# Extract Pareto optimal solutions
pareto_front = result.F
pareto_solutions = result.X

print(f"Found {len(pareto_solutions)} Pareto optimal solutions")
print(f"Best growth score: {-np.min(pareto_front[:, 0]):.3f}")
print(f"Lowest cost: {np.min(pareto_front[:, 1]):.3f}")
print(f"Lowest stress: {np.min(pareto_front[:, 2]):.3f}")
```

## Integration with Enhanced Pipeline

### Real-time Feature Updates

```python
class GreenhouseController:
    """Real-time greenhouse controller using MOEA optimization"""
    
    def __init__(self):
        self.sparse_pipeline = EnhancedSparsePipeline()
        self.moea_problem = GreenhouseOptimizationProblem()
        self.control_history = []
        
    async def update_control_strategy(self):
        """Update control strategy based on current conditions"""
        
        # Get latest enhanced features from pipeline
        current_time = datetime.utcnow()
        window_start = current_time - timedelta(hours=24)
        
        # Run enhanced pipeline for recent data
        pipeline_results = await self.sparse_pipeline.run_enhanced_pipeline(
            window_start, 
            current_time
        )
        
        # Extract MOEA-relevant features
        latest_features = self.extract_moea_features(pipeline_results)
        
        # Update MOEA problem with current features
        self.moea_problem.current_features = latest_features
        
        # Run quick MOEA optimization (limited generations for real-time)
        optimal_strategy = self.run_moea_optimization(max_generations=20)
        
        # Apply control strategy
        await self.apply_control_strategy(optimal_strategy)
        
        return optimal_strategy
    
    def extract_moea_features(self, pipeline_results):
        """Extract MOEA-relevant features from pipeline results"""
        
        # Get latest feature set (highest resolution)
        latest_features = {}
        
        if "15min" in pipeline_results.multiresolution_features:
            feature_sets = pipeline_results.multiresolution_features["15min"]
            if feature_sets:
                latest_set = feature_sets[-1]  # Most recent
                
                # Extract growth features
                latest_features.update({
                    'growth_gdd': latest_set.growth_features.get('growing_degree_days', 0.0),
                    'growth_dli': latest_set.growth_features.get('daily_light_integral', 0.0),
                    'growth_temp_optimality': latest_set.growth_features.get('temperature_optimality', 0.0),
                    'growth_stress_dd': latest_set.growth_features.get('stress_degree_days', 0.0),
                })
                
                # Extract energy features
                latest_features.update({
                    'energy_cost_weighted': latest_set.energy_features.get('cost_weighted_consumption', 0.0),
                    'energy_efficiency': latest_set.energy_features.get('energy_efficiency_score', 0.0),
                    'energy_peak_ratio': latest_set.energy_features.get('peak_offpeak_ratio', 1.0),
                })
                
                # Extract weather coupling
                latest_features.update({
                    'weather_temp_diff': latest_set.weather_features.get('temp_differential_mean', 0.0),
                    'weather_solar_eff': latest_set.weather_features.get('solar_efficiency', 0.0),
                })
                
                # Extract optimization metrics
                latest_features.update(latest_set.optimization_metrics)
        
        return latest_features
    
    def run_moea_optimization(self, max_generations=50):
        """Run MOEA optimization with current features"""
        
        # Quick optimization for real-time control
        from pymoo.optimize import minimize
        
        result = minimize(
            self.moea_problem,
            algorithm,
            termination=('n_gen', max_generations),
            verbose=False
        )
        
        # Select strategy from Pareto front based on current priorities
        strategy = self.select_strategy_from_pareto(result.F, result.X)
        
        return strategy
    
    def select_strategy_from_pareto(self, pareto_front, pareto_solutions):
        """Select best strategy from Pareto front based on current conditions"""
        
        # Example: Balance all objectives equally
        normalized_front = pareto_front / np.max(pareto_front, axis=0)
        combined_scores = np.sum(normalized_front, axis=1)
        best_idx = np.argmin(combined_scores)
        
        return pareto_solutions[best_idx]
    
    async def apply_control_strategy(self, strategy):
        """Apply optimized control strategy to greenhouse systems"""
        
        temp_setpoint, lamp_intensity, co2_setpoint, vent_rate = strategy
        
        # Send control signals to greenhouse systems
        control_commands = {
            'temperature_setpoint': float(temp_setpoint),
            'lamp_intensity_percent': float(lamp_intensity),
            'co2_setpoint_ppm': float(co2_setpoint),
            'ventilation_rate_percent': float(vent_rate),
            'timestamp': datetime.utcnow(),
        }
        
        # Log for analysis
        self.control_history.append(control_commands)
        
        print(f"Applied control strategy: {control_commands}")
        
        return control_commands

# Usage example
async def main():
    controller = GreenhouseController()
    
    # Run control loop
    while True:
        try:
            strategy = await controller.update_control_strategy()
            print(f"Optimal strategy: Temp={strategy[0]:.1f}°C, "
                  f"Light={strategy[1]:.0f}%, CO2={strategy[2]:.0f}ppm, "
                  f"Vent={strategy[3]:.0f}%")
            
            # Wait for next control cycle (e.g., every 15 minutes)
            await asyncio.sleep(15 * 60)
            
        except Exception as e:
            print(f"Control error: {e}")
            await asyncio.sleep(60)  # Shorter retry interval

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## Benefits of Enhanced Integration

### 1. Data-Driven Optimization
- **Real-time features**: Current sensor, weather, and energy conditions
- **Historical patterns**: Learning from past performance
- **Predictive capabilities**: Weather and energy price forecasting

### 2. Multi-Domain Objectives
- **Plant biology**: Species-specific growth requirements
- **Economics**: Energy cost optimization with market awareness
- **Sustainability**: Environmental impact minimization

### 3. Adaptive Control
- **Dynamic objectives**: Weights change based on growth stage
- **Seasonal adaptation**: Different strategies for different times of year
- **Market responsiveness**: Energy usage optimization based on prices

### 4. Validation Metrics
- **Growth performance**: Actual vs predicted growth rates
- **Cost effectiveness**: Energy cost per unit growth
- **Stress indicators**: Plant health monitoring

This integration transforms the greenhouse control system from reactive to predictive, leveraging the rich feature set from the enhanced sparse pipeline to make optimal decisions across multiple objectives simultaneously.