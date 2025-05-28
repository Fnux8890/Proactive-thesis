use polars::prelude::*;
use std::collections::HashSet;
use crate::io::coverage;
use crate::optimal_signals::OptimalSignals;

/// Validates that user-specified columns exist and are numeric
pub fn validate_user_columns(columns: &[String], df: &DataFrame) -> Vec<String> {
    let all_df_cols: HashSet<String> = df.get_column_names().into_iter().map(|s| s.to_string()).collect();
    
    columns.iter()
        .filter(|col_name| {
            if !all_df_cols.contains(*col_name) {
                log::warn!("User-specified column '{}' not found in DataFrame. It will be ignored.", col_name);
                false
            } else {
                match df.column(col_name).expect("Column checked for existence").dtype() {
                    DataType::Float32 | DataType::Float64 |
                    DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 |
                    DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => true,
                    _ => {
                        log::warn!("User-specified column '{}' is not a numeric type (found {:?}). It will be ignored for signal processing.", 
                            col_name, df.column(col_name).unwrap().dtype());
                        false
                    }
                }
            }
        })
        .cloned()
        .collect()
}

/// Selects optimal signals that meet the coverage threshold
pub fn select_optimal_signals(df: &DataFrame, min_coverage: f64) -> Vec<String> {
    let optimal_signals = OptimalSignals::new();
    let preferred_signals = optimal_signals.get_all();
    
    let mut optimal_selected: Vec<String> = Vec::new();
    for &signal in &preferred_signals {
        if let Ok(column) = df.column(signal) {
            if let Some(series) = column.as_series() {
                let cov = coverage(series);
                if cov >= min_coverage as f32 {
                    optimal_selected.push(signal.to_string());
                    log::info!("Selected optimal signal '{}' with coverage: {:.2}%", signal, cov * 100.0);
                } else {
                    log::debug!("Optimal signal '{}' has insufficient coverage: {:.2}%", signal, cov * 100.0);
                }
            }
        }
    }
    
    optimal_selected
}

/// Selects all numeric columns that meet the coverage threshold
pub fn select_numeric_columns_by_coverage(df: &DataFrame, min_coverage: f64, exclude_cols: &[&str]) -> Vec<String> {
    let mut selected = Vec::new();
    
    for col_name in df.get_column_names() {
        if exclude_cols.contains(&col_name.as_str()) {
            continue;
        }
        
        if let Ok(column) = df.column(col_name) {
            match column.dtype() {
                DataType::Float32 | DataType::Float64 |
                DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 |
                DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
                    if let Some(series) = column.as_series() {
                        let cov = coverage(series);
                        log::debug!("Column '{}' coverage: {:.2}%", col_name, cov * 100.0);
                        if cov >= min_coverage as f32 {
                            selected.push(col_name.to_string());
                        }
                    }
                }
                _ => log::debug!("Column '{}' is not numeric (type: {:?}), skipping", col_name, column.dtype()),
            }
        }
    }
    
    log::info!("Auto-selected {} numeric columns based on coverage threshold {}", selected.len(), min_coverage);
    selected
}

/// Performs the complete column selection logic based on CLI arguments
pub fn select_columns(df: &DataFrame, signal_cols: &[String], min_coverage: f64, time_col: &str) -> Vec<String> {
    if !signal_cols.is_empty() {
        log::info!("Using user-provided signal_cols: {:?}", signal_cols);
        validate_user_columns(signal_cols, df)
    } else {
        log::info!("Auto-selecting signal columns based on min_coverage: {} and optimal signals", min_coverage);
        
        // First, try to use optimal signals that meet coverage threshold
        let optimal_selected = select_optimal_signals(df, min_coverage);
        
        if !optimal_selected.is_empty() {
            log::info!("Using {} optimal signals that meet coverage threshold", optimal_selected.len());
            optimal_selected
        } else {
            // Otherwise, fall back to selecting all numeric columns with sufficient coverage
            log::info!("No optimal signals meet coverage threshold. Falling back to all numeric columns.");
            let exclude_cols = vec!["time", time_col];
            select_numeric_columns_by_coverage(df, min_coverage, &exclude_cols)
        }
    }
}