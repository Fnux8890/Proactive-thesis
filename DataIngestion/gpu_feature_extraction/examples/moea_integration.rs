use anyhow::Result;
use std::collections::HashMap;

/// Example showing how MOEA would use the enhanced features
/// for multi-objective optimization of greenhouse control

#[derive(Debug, Clone)]
struct ControlParameters {
    temperature_setpoint: f32,
    co2_setpoint: f32,
    light_hours: f32,
    ventilation_rate: f32,
}

#[derive(Debug)]
struct OptimizationObjectives {
    plant_growth: f64,      // Maximize
    energy_cost: f64,       // Minimize
    resource_efficiency: f64, // Maximize
}

/// Simulated MOEA fitness evaluation using extracted features
fn evaluate_control_strategy(
    params: &ControlParameters,
    features: &HashMap<String, f64>,
) -> OptimizationObjectives {
    // Plant growth objective based on environmental conditions
    let gdd = features.get("gdd_phenotype_specific").unwrap_or(&0.0);
    let dli = features.get("dli_mol_m2_d").unwrap_or(&0.0);
    let photoperiod = features.get("photoperiod_hours").unwrap_or(&0.0);
    
    // Growth model (simplified)
    let optimal_temp_deviation = (params.temperature_setpoint - 22.0).abs();
    let temp_factor = 1.0 - (optimal_temp_deviation / 10.0).min(1.0);
    
    let optimal_dli = 12.0; // mol/m²/day for Kalanchoe
    let light_factor = (dli / optimal_dli).min(1.5); // Saturating response
    
    let photoperiod_factor = if *photoperiod < 8.0 {
        photoperiod / 8.0 // Short day plant
    } else {
        1.0 - ((photoperiod - 8.0) / 16.0).min(0.5)
    };
    
    let plant_growth = gdd * temp_factor * light_factor * photoperiod_factor;
    
    // Energy cost objective
    let base_energy_cost = features.get("total_energy_cost").unwrap_or(&0.0);
    let peak_ratio = features.get("peak_hours_ratio").unwrap_or(&0.5);
    
    // Additional cost from control parameters
    let heating_cooling_cost = (params.temperature_setpoint - 20.0).abs() * 0.1;
    let lighting_cost = params.light_hours * 0.05;
    let ventilation_cost = params.ventilation_rate * 0.02;
    
    let energy_cost = base_energy_cost + heating_cooling_cost + lighting_cost + ventilation_cost
        + peak_ratio * 0.2; // Peak hour penalty
    
    // Resource efficiency objective
    let temp_efficiency = features.get("thermal_coupling_slope").unwrap_or(&0.0).abs();
    let solar_efficiency = features.get("solar_efficiency_ratio").unwrap_or(&0.0);
    let co2_efficiency = 1.0 / (1.0 + (params.co2_setpoint - 400.0) / 1000.0);
    
    let resource_efficiency = (temp_efficiency + solar_efficiency + co2_efficiency) / 3.0;
    
    OptimizationObjectives {
        plant_growth,
        energy_cost,
        resource_efficiency,
    }
}

/// Example Pareto front evaluation
fn find_pareto_optimal_solutions(
    feature_sets: Vec<HashMap<String, f64>>,
) -> Vec<(ControlParameters, OptimizationObjectives)> {
    let mut solutions = Vec::new();
    
    // Generate candidate solutions
    let temp_range = vec![18.0, 20.0, 22.0, 24.0, 26.0];
    let co2_range = vec![400.0, 600.0, 800.0, 1000.0];
    let light_range = vec![0.0, 8.0, 12.0, 16.0];
    let vent_range = vec![0.0, 0.25, 0.5, 0.75, 1.0];
    
    for features in &feature_sets {
        for &temp in &temp_range {
            for &co2 in &co2_range {
                for &light in &light_range {
                    for &vent in &vent_range {
                        let params = ControlParameters {
                            temperature_setpoint: temp,
                            co2_setpoint: co2,
                            light_hours: light,
                            ventilation_rate: vent,
                        };
                        
                        let objectives = evaluate_control_strategy(&params, features);
                        solutions.push((params, objectives));
                    }
                }
            }
        }
    }
    
    // Simple Pareto dominance check
    let mut pareto_front = Vec::new();
    
    for (i, (params_i, obj_i)) in solutions.iter().enumerate() {
        let mut is_dominated = false;
        
        for (j, (_, obj_j)) in solutions.iter().enumerate() {
            if i != j {
                // Check if solution j dominates solution i
                let dominates = obj_j.plant_growth >= obj_i.plant_growth &&
                               obj_j.energy_cost <= obj_i.energy_cost &&
                               obj_j.resource_efficiency >= obj_i.resource_efficiency &&
                               (obj_j.plant_growth > obj_i.plant_growth ||
                                obj_j.energy_cost < obj_i.energy_cost ||
                                obj_j.resource_efficiency > obj_i.resource_efficiency);
                
                if dominates {
                    is_dominated = true;
                    break;
                }
            }
        }
        
        if !is_dominated {
            pareto_front.push((params_i.clone(), obj_i.clone()));
        }
    }
    
    pareto_front
}

/// Demonstrate decision making based on features
fn select_control_strategy(
    features: &HashMap<String, f64>,
    preference: &str,
) -> ControlParameters {
    // Extract key features for decision making
    let current_temp = features.get("air_temp_c_mean").unwrap_or(&20.0);
    let outside_temp = features.get("outside_temp_c_mean").unwrap_or(&15.0);
    let energy_price = features.get("energy_price_mean").unwrap_or(&0.15);
    let solar_radiation = features.get("solar_radiation_mean").unwrap_or(&200.0);
    
    match preference {
        "growth" => {
            // Maximize growth regardless of cost
            ControlParameters {
                temperature_setpoint: 22.0,
                co2_setpoint: 1000.0,
                light_hours: 16.0,
                ventilation_rate: 0.5,
            }
        }
        "cost" => {
            // Minimize cost while maintaining minimum growth
            ControlParameters {
                temperature_setpoint: if outside_temp > 18.0 { outside_temp + 2.0 } else { 18.0 },
                co2_setpoint: 600.0,
                light_hours: if solar_radiation > 300.0 { 0.0 } else { 8.0 },
                ventilation_rate: if current_temp > 25.0 { 0.75 } else { 0.25 },
            }
        }
        "balanced" => {
            // Balance between growth and cost
            let temp_target = if energy_price > 0.20 { 20.0 } else { 22.0 };
            let light_supplement = if solar_radiation < 200.0 { 12.0 } else { 8.0 };
            
            ControlParameters {
                temperature_setpoint: temp_target,
                co2_setpoint: 800.0,
                light_hours: light_supplement,
                ventilation_rate: 0.5,
            }
        }
        _ => {
            // Default conservative strategy
            ControlParameters {
                temperature_setpoint: 20.0,
                co2_setpoint: 600.0,
                light_hours: 8.0,
                ventilation_rate: 0.5,
            }
        }
    }
}

fn main() -> Result<()> {
    println!("MOEA Integration Example");
    println!("========================\n");
    
    // Simulate feature sets from different time windows
    let mut feature_sets = Vec::new();
    
    // Winter morning scenario
    let mut winter_morning = HashMap::new();
    winter_morning.insert("gdd_phenotype_specific".to_string(), 8.5);
    winter_morning.insert("dli_mol_m2_d".to_string(), 6.0);
    winter_morning.insert("photoperiod_hours".to_string(), 10.0);
    winter_morning.insert("air_temp_c_mean".to_string(), 18.0);
    winter_morning.insert("outside_temp_c_mean".to_string(), 5.0);
    winter_morning.insert("solar_radiation_mean".to_string(), 50.0);
    winter_morning.insert("energy_price_mean".to_string(), 0.25);
    winter_morning.insert("peak_hours_ratio".to_string(), 0.8);
    winter_morning.insert("total_energy_cost".to_string(), 45.0);
    winter_morning.insert("thermal_coupling_slope".to_string(), 0.7);
    winter_morning.insert("solar_efficiency_ratio".to_string(), 0.3);
    feature_sets.push(winter_morning);
    
    // Summer afternoon scenario
    let mut summer_afternoon = HashMap::new();
    summer_afternoon.insert("gdd_phenotype_specific".to_string(), 15.0);
    summer_afternoon.insert("dli_mol_m2_d".to_string(), 18.0);
    summer_afternoon.insert("photoperiod_hours".to_string(), 0.0);
    summer_afternoon.insert("air_temp_c_mean".to_string(), 26.0);
    summer_afternoon.insert("outside_temp_c_mean".to_string(), 22.0);
    summer_afternoon.insert("solar_radiation_mean".to_string(), 600.0);
    summer_afternoon.insert("energy_price_mean".to_string(), 0.10);
    summer_afternoon.insert("peak_hours_ratio".to_string(), 0.2);
    summer_afternoon.insert("total_energy_cost".to_string(), 15.0);
    summer_afternoon.insert("thermal_coupling_slope".to_string(), 0.9);
    summer_afternoon.insert("solar_efficiency_ratio".to_string(), 0.8);
    feature_sets.push(summer_afternoon);
    
    // Test different control strategies
    println!("Testing control strategies:\n");
    
    for (i, features) in feature_sets.iter().enumerate() {
        let scenario = if i == 0 { "Winter Morning" } else { "Summer Afternoon" };
        println!("Scenario: {}", scenario);
        println!("-" * 50);
        
        // Test different preferences
        for preference in &["growth", "cost", "balanced"] {
            let strategy = select_control_strategy(features, preference);
            let objectives = evaluate_control_strategy(&strategy, features);
            
            println!("\n{} preference:", preference);
            println!("  Control: T={:.1}°C, CO2={:.0}ppm, Light={:.1}h, Vent={:.2}",
                strategy.temperature_setpoint,
                strategy.co2_setpoint,
                strategy.light_hours,
                strategy.ventilation_rate
            );
            println!("  Results: Growth={:.2}, Cost={:.2}, Efficiency={:.2}",
                objectives.plant_growth,
                objectives.energy_cost,
                objectives.resource_efficiency
            );
        }
        println!("\n");
    }
    
    // Find Pareto optimal solutions
    println!("Finding Pareto optimal solutions...");
    let pareto_front = find_pareto_optimal_solutions(feature_sets);
    
    println!("\nPareto front ({} solutions):", pareto_front.len());
    println!("{:<20} {:<20} {:<20} {:<20}", "Temperature", "Growth", "Cost", "Efficiency");
    println!("{}", "-".repeat(80));
    
    // Show top 10 diverse solutions
    for (params, objectives) in pareto_front.iter().take(10) {
        println!("{:<20.1} {:<20.2} {:<20.2} {:<20.2}",
            params.temperature_setpoint,
            objectives.plant_growth,
            objectives.energy_cost,
            objectives.resource_efficiency
        );
    }
    
    println!("\nMOEA integration example completed!");
    Ok(())
}