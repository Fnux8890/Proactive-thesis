// Kernel modules - GPU functionality moved to Python
// These modules now contain CPU-only implementations or stubs

pub mod actuator_dynamics;
pub mod economic_features;
pub mod enhanced_features;
pub mod entropy_complexity;
pub mod environment_coupling;
pub mod frequency_domain;
pub mod psychrometric;
pub mod rolling_statistics_extended;
pub mod stress_counters;
pub mod temporal_dependencies;
pub mod thermal_time;
pub mod wavelet_features;

// Note: GPU kernel functions have been removed
// Use the enhanced_features module for CPU implementations