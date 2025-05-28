use clap::Parser;
use std::path::PathBuf;
use anyhow::{Result, Context};
use rayon::prelude::*;
use serde_json;

use std::collections::HashSet;
use std::sync::Arc;
use polars::datatypes::TimeUnit;
use polars::prelude::DynamicGroupOptions;
use polars::lazy::frame::IntoLazy;
use polars::prelude::{
    lit, // Added to fix E0425
    Column,
    ChunkCompareIneq, // For series.gt()
    NamedFrom, SortMultipleOptions,
    DataFrame, Series, DataType, Expr, Duration, ClosedWindow, FillNullStrategy, Label, StartBy,
    col,
};

mod io;
mod level_a;
mod level_b;
mod level_c;
mod db_hybrid;
mod column_selection;
mod optimal_signals;

// Use hybrid DB module that auto-detects table structure
use crate::db_hybrid::EraDb;
use crate::column_selection::select_columns;
use crate::optimal_signals::OptimalSignals;
use crate::io::coverage;

/// Where the program should read the input data from.
/// * `--input-parquet  some_file.parquet`  (old behaviour)
/// * `--db-dsn  postgresql://…`            (new behaviour)
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Cli {
    /// Optional parquet-file path (legacy mode)
    #[clap(long, value_parser)]
    input_parquet: Option<PathBuf>,

    /// Directory to save the output Parquet files with era labels.
    #[clap(long, value_parser)]
    output_dir: Option<PathBuf>, // Making it optional for now, to align with input_parquet being optional. We'll need logic to ensure one mode (file or DB) is chosen.

    /// Suffix to append to the output filenames.
    #[clap(long)]
    output_suffix: Option<String>,

    /// Postgres connection string.
    /// Falls back to $DB_DSN when not supplied explicitly.
    #[clap(long, env = "DB_DSN")]
    db_dsn: Option<String>,

    /// Name of the table / materialised view that holds the
    /// densified, resampled features you want to detect eras on.
    #[clap(long, default_value = "preprocessed_features")]
    db_table: String,


    #[clap(long, default_value_t = 48)] // Default min_size for PELT (e.g. 1 day for 5T data if 12*24=288, 48 ~ 4h)
    pelt_min_size: usize,
    #[clap(long, default_value_t = 200.0)] // Expected run length for BOCPD
    bocpd_lambda: f64,
    #[clap(long, default_value_t = 5)] // Reduced default HMM states
    hmm_states: usize,
    #[clap(long, default_value_t = 20)] // Reduced default HMM iterations
    hmm_iterations: usize,

    #[clap(long, default_value_t = 250)] // Max value for u8 quantization (e.g., 250 means values 0-249)
    quant_max_val: u8,

    // New arguments from the plan
    #[clap(long, default_value = "5m")]
    resample_every: String,

    #[clap(long, default_value_t = 0.9)]
    min_coverage: f32,

    #[clap(long, value_delimiter = ',', value_parser)]
    signal_cols: Vec<String>,

    #[clap(long, default_value_t = true, action = clap::ArgAction::Set)] // Use Set to allow --include-time false
    include_time: bool,

    /// Enable parallel processing for multiple signals
    #[clap(long, default_value_t = true)]
    parallel_processing: bool,

    /// Maximum number of signals to process concurrently
    #[clap(long, default_value_t = 12)]
    max_concurrent_signals: usize,
}


/// Parses a JSON string and extracts numeric values for specified keys
fn parse_json_features(json_str: &str, all_keys: &std::collections::HashSet<String>) -> Result<std::collections::HashMap<String, Option<f64>>> {
    let mut result = std::collections::HashMap::new();
    
    match serde_json::from_str::<serde_json::Value>(json_str) {
        Ok(serde_json::Value::Object(map)) => {
            for key in all_keys {
                let value = map.get(key).and_then(|val| {
                    match val {
                        serde_json::Value::Number(n) => n.as_f64(),
                        _ => {
                            log::warn!("Non-numeric value found for key '{}': {:?}", key, val);
                            None
                        }
                    }
                });
                result.insert(key.clone(), value);
            }
        }
        Ok(other) => {
            log::warn!("Expected JSON object but got: {:?}", other);
            for key in all_keys {
                result.insert(key.clone(), None);
            }
        }
        Err(e) => {
            log::error!("Failed to parse JSON: {}. Content: {}", e, json_str);
            for key in all_keys {
                result.insert(key.clone(), None);
            }
        }
    }
    
    Ok(result)
}

/// Helper function to persist a given era level's DataFrame to the database.
/// Connects to DB, builds CSV, and copies data for a specific signal, level, and stage.



/// Quantizes a signal from f64 to u8 values
fn quantize_signal_f64(signal: &[f64], max_val: u8) -> (Vec<u8>, f64, f64) {
    if signal.is_empty() {
        return (Vec::new(), 0.0, 0.0);
    }
    
    let min_val = signal.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val_signal = signal.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    
    if (max_val_signal - min_val).abs() < 1e-9 {
        // All values are the same
        return (vec![0; signal.len()], min_val, max_val_signal);
    }
    
    let range = max_val_signal - min_val;
    
    let quantized: Vec<u8> = signal
        .iter()
        .map(|&v| {
            let normalized = (v - min_val) / range;
            let scaled = normalized * max_val as f64;
            scaled.round() as u8
        })
        .collect();
    
    (quantized, min_val, max_val_signal)
}

fn main() -> Result<()> {
    eprintln!("XXXXX MAIN FUNCTION ENTERED XXXXX"); // Test: Print to stderr immediately
    println!("RUST_PROGRAM_STARTED_DIRECT_PRINTLN_HOOK");
    // Initialize logger FIRST - before any other operations
    println!("RUST_PROGRAM_ENV_LOGGER_INIT_ABOUT_TO_START");
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    println!("RUST_PROGRAM_ENV_LOGGER_INIT_COMPLETED");
    
    log::info!("Era Detector starting up...");
    log::info!("RUST_LOG level: {}", std::env::var("RUST_LOG").unwrap_or_else(|_| "info (default)".to_string()));
    
    // Wrap the main logic to ensure errors are logged
    if let Err(e) = run_main() {
        log::error!("Era Detector failed with error: {}", e);
        log::error!("Error chain:");
        let mut source = e.source();
        let mut depth = 0;
        while let Some(err) = source {
            depth += 1;
            log::error!("  {}: {}", depth, err);
            source = err.source();
        }
        return Err(e);
    }
    log::info!("Era Detector completed successfully");
    Ok(())
}

fn run_main() -> Result<()> {
    // Ultra-early log to stderr, bypasses 'log' crate.
    eprintln!("--- era_detector run_main CALLED (raw stderr check) ---");

    // Early log using the 'log' crate. Confirms logger is initialized.
    log::info!("--- era_detector run_main started (logger check) ---");

    // let cli = Cli::parse(); // Original line
    let cli = match Cli::try_parse() {
        Ok(parsed_cli) => {
            // Using log::info! here assumes logger is initialized.
            // For an even earlier diagnostic, one could use eprintln! here too.
            log::info!("CLI arguments parsed successfully via try_parse().");
            parsed_cli
        }
        Err(e) => {
            // Print directly to stderr, which Docker should capture.
            eprintln!("CRITICAL_ERROR_CLAP_PARSING_FAILED: {}", e);
            // Optionally, to ensure the full help text is also printed on error:
            // e.print().expect("Failed to print clap error details to stderr");
            return Err(anyhow::anyhow!("CLI argument parsing failed").context(e));
        }
    };
    
    log::info!("Parsed CLI arguments: {:?}", cli);
    log::info!("DB_DSN from env/args: {:?}", cli.db_dsn);

    if cli.input_parquet.is_none() && cli.db_dsn.is_none() {
        log::error!("Neither --input-parquet nor --db-dsn provided (DB_DSN env var also not set)");
        anyhow::bail!("❌ Supply either --input-parquet or --db-dsn (or set DB_DSN env var)");
    }

    // Moved EraDb initialization earlier with explicit logging
    log::info!("Attempting to initialize EraDb. Provided DSN: {:?}", cli.db_dsn);
    let era_db = match EraDb::new(cli.db_dsn.as_deref()) {
        Ok(instance) => {
            log::info!("SUCCESS: EraDb initialized successfully.");
            Arc::new(instance)
        }
        Err(e) => {
            // Convert 'e' to an anyhow::Error if it isn't already, then add context.
            // If 'e' comes from a function returning Result<_, anyhow::Error>,
            // it's already an anyhow::Error. If it's Result<_, OtherErrorType>,
            // anyhow!(e) will wrap it.
            let init_error = anyhow::anyhow!(e).context(format!(
                "EraDb initialization failed. DSN used: {:?}",
                cli.db_dsn
            ));
            log::error!("CRITICAL_ERROR_DB_INITIALIZATION_FAILED: {:?}", init_error);
            
            // Log the full error chain
            let mut source = init_error.source();
            let mut depth = 0;
            while let Some(err_source) = source {
                depth += 1;
                log::error!("  Cause [{}]: {}", depth, err_source);
                source = err_source.source();
            }
            return Err(init_error);
        }
    };

    log::info!("Era Detector application core logic started (after DB init attempt).");
    log::info!("Version: {}", env!("CARGO_PKG_VERSION"));


    let start_time_main = std::time::Instant::now();

    // The CLI arguments are already logged via log::debug!("CLI arguments parsed: {:?}", cli);
    // If more detailed startup info is needed, it can be added here.

    let mut df_main = if let Some(ref parquet_path) = cli.input_parquet {
        log::info!("Loading data from Parquet file: {:?}", parquet_path);
        io::read_parquet_to_polars_df(parquet_path)
            .with_context(|| format!("Failed to read input parquet: {:?}", parquet_path))?
    } else {
        // input_parquet is None, so db_dsn must be Some (guaranteed by earlier check)
        log::info!("Loading data from database table: {}", cli.db_table);
        // EraDb instance `era_db` is already initialized above and is an Arc.
        // We can use it directly if we don't need a mutable borrow or ownership transfer.
        // For load_feature_df, we just need a reference.
        // Note: The original plan was to initialize EraDb here, but it's already initialized above.
        // We'll use the existing `era_db` (Arc-wrapped) directly. 
        // This requires `EraDb::new` to have been called with `cli.db_dsn.as_deref()` already, which is now the case.
        era_db.load_feature_df(&cli.db_table, None)
            .with_context(|| format!("Failed to load feature DataFrame from DB table: {}", cli.db_table))?
    };
    log::info!("Loaded DataFrame. Shape: {:?}, Load time: {:.2?}", df_main.shape(), start_time_main.elapsed());

    // Ensure 'time' column is Datetime
    if df_main.column("time")?.dtype() != &DataType::Datetime(TimeUnit::Microseconds, None) &&
       df_main.column("time")?.dtype() != &DataType::Datetime(TimeUnit::Nanoseconds, None) &&
       df_main.column("time")?.dtype() != &DataType::Datetime(TimeUnit::Milliseconds, None) {
        df_main = df_main.lazy().with_column(
            col("time").cast(DataType::Datetime(TimeUnit::Microseconds, None))
        ).collect()?;
        log::info!("Casted 'time' column to Datetime(tu=us).");
    }
    
    let sort_options = SortMultipleOptions::new().with_order_descending(false);
    df_main = df_main.sort(["time"], sort_options)?;

    // Check if 'features' column exists (indicating JSONB data that needs parsing)
    if df_main.column("features").is_ok() && df_main.column("features")?.dtype() == &DataType::String {
        log::info!("'features' column found (JSONB as string). Parsing and unnesting...");

        // Parse JSON manually using serde_json with improved error handling
        let features_series = df_main.column("features")?;
        let features_ca = features_series.str()?;
        
        // Collect all unique keys from the JSON objects
        let mut all_keys = std::collections::HashSet::new();
        let mut parse_errors = 0;
        for (idx, opt_str) in features_ca.into_iter().enumerate() {
            if let Some(json_str) = opt_str {
                match serde_json::from_str::<serde_json::Value>(json_str) {
                    Ok(serde_json::Value::Object(map)) => {
                        all_keys.extend(map.keys().cloned());
                    }
                    Ok(_other) => {
                        log::warn!("Row {}: Expected JSON object but got non-object value", idx);
                        parse_errors += 1;
                    }
                    Err(e) => {
                        log::error!("Row {}: JSON parse error: {}. Content preview: {}...", 
                            idx, e, &json_str[..json_str.len().min(100)]);
                        parse_errors += 1;
                    }
                }
            }
        }
        
        if parse_errors > 0 {
            log::warn!("Encountered {} JSON parsing errors out of {} rows", parse_errors, df_main.height());
        }
        
        log::info!("Found {} unique keys in JSON features", all_keys.len());
        
        // Create new columns for each key found
        let mut new_columns = vec![];
        for key in &all_keys {
            let mut values = Vec::with_capacity(df_main.height());
            for opt_str in features_ca.into_iter() {
                if let Some(json_str) = opt_str {
                    let parsed_values = parse_json_features(json_str, &all_keys)?;
                    values.push(parsed_values.get(key).copied().flatten());
                } else {
                    values.push(None);
                }
            }
            let new_series = Series::new(key.clone().into(), values);
            new_columns.push(new_series);
        }
        
        // Add all new columns to the dataframe
        for series in new_columns {
            df_main = df_main.lazy()
                .with_column(lit(series))
                .collect()?;
        }


        // Drop the original 'features' string column as it's now redundant
        df_main = df_main.drop("features").context("Failed to drop original 'features' column")?;

        log::info!("'features' column parsed and unnested. DataFrame shape: {:?}, columns: {:?}", 
                   df_main.shape(), df_main.get_column_names());
    } else {
        log::debug!("'features' column not found or not String. Assuming direct columnar structure from DB.");
    }

    // --- Start: New Column Selection Logic ---
    log::info!("Starting column selection based on coverage and CLI arguments...");
    let mut selected_columns_intermediate: Vec<String>;

    if !cli.signal_cols.is_empty() {
        log::info!("Using user-provided signal_cols: {:?}", cli.signal_cols);
        selected_columns_intermediate = cli.signal_cols.clone();
        let all_df_cols: HashSet<String> = df_main.get_column_names().into_iter().map(|s| s.to_string()).collect();
        
        selected_columns_intermediate.retain(|col_name| {
            if !all_df_cols.contains(col_name) {
                log::warn!("User-specified column '{}' not found in DataFrame. It will be ignored.", col_name);
                false
            } else {
                match df_main.column(col_name).expect("Column checked for existence").dtype() {
                    DataType::Float32 | DataType::Float64 |
                    DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 |
                    DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => true,
                    _ => {
                        log::warn!("User-specified column '{}' is not a numeric type (found {:?}). It will be ignored for signal processing.", col_name, df_main.column(col_name).unwrap().dtype());
                        false
                    }
                }
            }
        });
    } else {
        log::info!("Auto-selecting signal columns based on min_coverage: {} and optimal signals", cli.min_coverage);
        
        // Initialize optimal signals
        let optimal_signals = OptimalSignals::new();
        let preferred_signals = optimal_signals.get_all();
        
        // First, try to use optimal signals that meet coverage threshold
        let mut optimal_selected: Vec<String> = Vec::new();
        for &signal in &preferred_signals {
            if let Ok(column) = df_main.column(signal) {
                if let Some(series) = column.as_series() {
                    let cov = coverage(series);
                    if cov >= cli.min_coverage {
                        optimal_selected.push(signal.to_string());
                        log::info!("Selected optimal signal '{}' with coverage: {:.2}%", signal, cov * 100.0);
                    } else {
                        log::debug!("Optimal signal '{}' has insufficient coverage: {:.2}%", signal, cov * 100.0);
                    }
                }
            }
        }
        
        // If we have optimal signals, use them
        if !optimal_selected.is_empty() {
            selected_columns_intermediate = optimal_selected;
        } else {
            // Fall back to original logic if no optimal signals meet threshold
            log::warn!("No optimal signals meet coverage threshold, falling back to general selection");
            selected_columns_intermediate = df_main
                .get_columns()
                .par_iter()
                .filter_map(|s: &Column| {
                    // Match on the result of s.as_series()
                    if let Some(actual_series_ref) = s.as_series() {
                        // Now 'actual_series_ref' is &Series
                        match s.dtype() {
                            DataType::Float32 | DataType::Float64 |
                            DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 |
                            DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
                                if coverage(actual_series_ref) >= cli.min_coverage {
                                    Some(s.name().to_string())
                                } else {
                                    log::debug!(
                                        "Column '{}' (dtype: {:?}) excluded due to coverage: {:.2} < {}",
                                        s.name(),
                                        s.dtype(),
                                        coverage(actual_series_ref),
                                        cli.min_coverage
                                    );
                                    None // This None is for filter_map, if coverage is too low
                                }
                            }   
                            _ => {
                                log::trace!(
                                    "Column '{}' (dtype: {:?}) ignored (not a numeric signal type for auto-selection).",
                                    s.name(),
                                    s.dtype()
                                );
                                None // This None is for filter_map, if not numeric
                            }
                        }
                    } else {
                        log::warn!(
                            "Column '{}' could not be represented as a Series for coverage check during auto-selection.",
                            s.name()
                        );
                        None
                    }
                })
                .collect();
        }
    }

    let mut final_selected_cols: Vec<String> = Vec::new();
    let mut seen_cols: HashSet<String> = HashSet::new();

    // 1. Ensure 'time' column is included and first, if it exists and was implicitly/explicitly selected or needed
    let time_col_name = "time";
    if df_main.column(time_col_name).is_ok() { // Check if 'time' column actually exists
        // Conditionally include the "time" column if not already present and flag is true, or if no specific signals were given.
        let time_col_present = selected_columns_intermediate.iter().any(|name| name == "time");
        if !time_col_present && (cli.include_time || cli.signal_cols.is_empty()) {
            log::info!("Adding 'time' column to selected columns as it's essential and --include-time is true (or no specific signals provided).");
            selected_columns_intermediate.insert(0, "time".to_string()); // Add to the beginning for convention
        } else if time_col_present && !cli.include_time && !cli.signal_cols.is_empty() {
            log::info!("User provided 'time' in signal_cols, but --include-time is false. 'time' will be kept as per user's explicit list.");
        } else if !time_col_present && !cli.include_time && !cli.signal_cols.is_empty() {
            log::warn!("'time' column is not selected (due to --include-time=false and not in user's list) but is crucial for time series processing. This might lead to errors downstream if 'time' is implicitly required by operations like resampling.");
        } 
    }

    // 2. Handle 'dli_sum' based on whether user specified --signal-cols
    let dli_sum_col_name = "dli_sum";
    if cli.signal_cols.is_empty() && df_main.column(dli_sum_col_name).is_ok() {
        if selected_columns_intermediate.contains(&dli_sum_col_name.to_string()) {
            log::debug!("'{}' is present in auto-selected intermediate columns and will be processed if it remains in final_selected_cols.", dli_sum_col_name);
        } else {
            log::debug!("'{}' was not auto-selected (e.g., due to coverage or type). It will not be processed unless explicitly in --signal-cols.", dli_sum_col_name);
        }
    } else if !cli.signal_cols.is_empty() {
        if selected_columns_intermediate.contains(&dli_sum_col_name.to_string()) {
            log::debug!("'{}' was included in user's --signal-cols and will be processed.", dli_sum_col_name);
        } else {
            log::debug!("'{}' was not in user's --signal-cols. It will not be processed.", dli_sum_col_name);
        }
    }

    // 3. Add other columns from the intermediate selection, ensuring uniqueness and existence
    for col_name in &selected_columns_intermediate {
        if df_main.column(col_name).is_ok() { 
            if seen_cols.insert(col_name.clone()) { 
                final_selected_cols.push(col_name.clone());
            }
        }
    }
    
    if !seen_cols.contains(time_col_name) && df_main.column(time_col_name).is_ok() {
        log::warn!("'time' column was not in selected signals but is required for resampling. Adding it.");
        final_selected_cols.insert(0, time_col_name.to_string()); 
        seen_cols.insert(time_col_name.to_string());
    }

    if final_selected_cols.is_empty() {
        log::error!("No columns selected after filtering. Please check input data, min_coverage, or signal_cols argument.");
        return Err(anyhow::anyhow!("No columns selected for processing. Check input data, min_coverage, or signal_cols."));
    }

    // Apply concurrent signals limit if needed
    let signal_count = final_selected_cols.iter().filter(|&name| name != "time").count();
    if signal_count > cli.max_concurrent_signals {
        log::warn!("Selected {} signals exceeds max_concurrent_signals limit of {}. Truncating to limit.", 
                  signal_count, cli.max_concurrent_signals);
        // Keep time column and limit other signals
        let mut limited_cols = vec![];
        let mut non_time_count = 0;
        for col in final_selected_cols {
            if col == "time" || non_time_count < cli.max_concurrent_signals {
                limited_cols.push(col.clone()); // Clone to avoid move error
                if col != "time" {
                    non_time_count += 1;
                }
            }
        }
        final_selected_cols = limited_cols;
    }

    log::info!("Final selected columns for processing: {:?}", final_selected_cols);

    let selected_columns_str: Vec<&str> = final_selected_cols.iter().map(|s| s.as_str()).collect();
    let mut feature_df = df_main.select(selected_columns_str) 
        .with_context(|| format!("Failed to select final columns: {:?}", final_selected_cols))?;
    log::info!("Created feature_df with selected columns. Shape: {:?}", feature_df.shape());

    // Check for NULL time values before resampling
    let time_nulls = feature_df.column("time")?.null_count();
    if time_nulls > 0 {
        log::warn!("Found {} NULL time values, filtering them out", time_nulls);
        feature_df = feature_df.lazy()
            .filter(col("time").is_not_null())
            .collect()
            .with_context(|| "Failed to filter out NULL time values")?;
    }
    
    // Ensure we have data after filtering
    if feature_df.height() == 0 {
        log::error!("No data remaining after filtering NULL time values");
        return Err(anyhow::anyhow!("No valid data with non-NULL timestamps"));
    }

    log::info!("Starting resampling with interval: {}", cli.resample_every);
    let resampling_start_time = std::time::Instant::now();

    if final_selected_cols.iter().filter(|&name| name != "time").count() == 0 {
        log::error!("No signal columns (other than 'time') were selected or survived validation. Cannot proceed with resampling aggregation.");
        return Err(anyhow::anyhow!("No signal columns selected for resampling aggregation."));
    }

    let mut agg_exprs: Vec<Expr> = Vec::new();
    for col_name_str in &final_selected_cols {
        if col_name_str != "time" { 
            agg_exprs.push(col(col_name_str).mean().alias(col_name_str));
        }
    }

    feature_df = feature_df.lazy()
        .group_by_dynamic(
            col("time"),
            [], 
            DynamicGroupOptions {
                every: Duration::parse(&cli.resample_every),
                period: Duration::parse(&cli.resample_every),
                offset: Duration::parse("0s"),
                include_boundaries: true,
                closed_window: ClosedWindow::Left, 
                label: Label::Left, 
                start_by: StartBy::WindowBound, 
                index_column: "time".into() 
            }
        )
        .agg(agg_exprs)
        .collect()
        .with_context(|| format!("Failed to resample DataFrame with interval '{}'", cli.resample_every))?;

    log::info!("Resampled feature_df. Shape: {:?}, Resampling time: {:.2?}", feature_df.shape(), resampling_start_time.elapsed());

    let fill_start_time = std::time::Instant::now();
    feature_df = feature_df
        .fill_null(FillNullStrategy::Forward(None))? 
        .fill_null(FillNullStrategy::Backward(None))?; 
    log::info!("Filled nulls in resampled feature_df. Shape: {:?}, Fill time: {:.2?}", feature_df.shape(), fill_start_time.elapsed());

    let feature_df_arc = Arc::new(feature_df);
    let time_col_series_arc = Arc::new(feature_df_arc.column("time")?.clone());


    // Process signals either in parallel or sequentially based on CLI flag
    let signal_columns: Vec<_> = final_selected_cols.iter()
        .filter(|&name| name.as_str() != "time")
        .collect();
    
    // Adaptive parallelism based on available resources
    let num_cpus = num_cpus::get();
    let optimal_parallelism = std::cmp::min(
        signal_columns.len(),
        std::cmp::min(cli.max_concurrent_signals, num_cpus)
    );
    
    let processing_results: Vec<Result<()>> = if cli.parallel_processing && signal_columns.len() > 1 {
        log::info!("Processing {} signals in parallel (optimal parallelism: {}, CPUs: {})", 
            signal_columns.len(), optimal_parallelism, num_cpus);
        
        // Configure rayon thread pool
        rayon::ThreadPoolBuilder::new()
            .num_threads(optimal_parallelism)
            .build_global()
            .unwrap_or_else(|e| log::warn!("Failed to set thread pool size: {}", e));
        
        signal_columns.par_iter().map(|&signal_name_to_process| {
            let era_db_clone = Arc::clone(&era_db);
            let feature_df_task_local = Arc::clone(&feature_df_arc);
            let time_col_series_task_local = Arc::clone(&time_col_series_arc);
            
            let process_signal_task = || -> Result<()> {
            log::info!("\n\n=== Processing signal: '{}' ===", signal_name_to_process);

            let pelt_start_time = std::time::Instant::now();
            log::info!("\n--- Level A: PELT Segmentation on '{}' ---", signal_name_to_process);
            let pelt_signal_column = feature_df_task_local
                .column(signal_name_to_process)
                .with_context(|| format!("PELT signal column '{}' not found in feature_df", signal_name_to_process))?;
            let pelt_signal_vec = io::series_to_vec_f64(pelt_signal_column.as_series().expect("Series conversion failed for PELT"))?;
            let level_a_bocpd_threshold = 0.5;
            let pelt_bkps_indices = level_a::detect_changepoints_level_a(
                &pelt_signal_vec, 
                cli.bocpd_lambda, 
                cli.pelt_min_size, 
                level_a_bocpd_threshold
            )?;
            log::info!("Level A for '{}': PELT-like (BOCPD based) detected {} breakpoints. Indices: {:?}. Time: {:.2?}", 
                signal_name_to_process, pelt_bkps_indices.len(), pelt_bkps_indices, pelt_start_time.elapsed());

            let mut era_a_values = vec![0i32; feature_df_task_local.height()];
            let mut current_era_a = 0;
            let mut last_idx_a = 0;
            for &bkp_idx in &pelt_bkps_indices {
                if bkp_idx > last_idx_a && bkp_idx <= feature_df_task_local.height() {
                    for i in last_idx_a..bkp_idx { era_a_values[i] = current_era_a; }
                }
                last_idx_a = bkp_idx;
                current_era_a += 1;
            }
            if last_idx_a < feature_df_task_local.height() { 
                for i in last_idx_a..feature_df_task_local.height() { era_a_values[i] = current_era_a; }
            }
            let era_a_series = Series::new("era_level_A".into(), era_a_values);
            let df_out_a = DataFrame::new(vec![time_col_series_task_local.as_ref().clone(), era_a_series.into()])?;
            
            // Note: Deletion is now handled inside save_era_labels as part of transaction
            era_db_clone.as_ref().save_era_labels(
                &df_out_a, 
                "era_level_A", 
                signal_name_to_process, 
                'A', 
                "PELT", 
                "era_labels_level_a"
            ).with_context(|| format!("Failed to save Level A labels to DB for signal '{}'", signal_name_to_process))?;
            log::info!("Saved Level A labels for '{}' to database table 'era_labels_level_a'", signal_name_to_process);

            let bocpd_start_time = std::time::Instant::now();
            log::info!("\n--- Level B: BOCPD on '{}' ---", signal_name_to_process);
            let bocpd_signal_column = feature_df_task_local
                .column(signal_name_to_process)
                .with_context(|| format!("BOCPD signal column '{}' not found in feature_df", signal_name_to_process))?;
            let bocpd_signal_vec = io::series_to_vec_f64(bocpd_signal_column.as_series().expect("Series conversion failed for BOCPD"))?;
            let cp_probs_b = level_b::bocpd_probabilities(&bocpd_signal_vec, cli.bocpd_lambda)?;
            log::info!("Level B for '{}': BOCPD probabilities calculated. Time: {:.2?}", signal_name_to_process, bocpd_start_time.elapsed());
            
            let cp_probs_b_series = Series::new("cp_prob_level_B".into(), cp_probs_b);
            let bool_chunked = cp_probs_b_series.gt(0.5)?;
            let mut cum_label: i32 = 0;
            let era_labels: Vec<i32> = bool_chunked
                .into_iter()
                .map(|opt_b| {
                    if let Some(b) = opt_b {
                        if b { cum_label += 1; }
                    }
                    cum_label
                })
                .collect();
            let era_b_series = Series::new("era_level_B".into(), era_labels);
            let df_out_b = DataFrame::new(vec![time_col_series_task_local.as_ref().clone(), cp_probs_b_series.into(), era_b_series.into()])?;
            era_db_clone.as_ref().save_era_labels(
                &df_out_b, 
                "era_level_B", 
                signal_name_to_process, 
                'B', 
                "BOCPD", 
                "era_labels_level_b"
            ).with_context(|| format!("Failed to save Level B labels to DB for signal '{}'", signal_name_to_process))?;
            log::info!("Saved Level B labels for '{}' to database table 'era_labels_level_b'", signal_name_to_process);

            let hmm_start_time = std::time::Instant::now();
            log::info!("\n--- Level C: HMM Viterbi on '{}' (Quantized up to max value: {}) ---", signal_name_to_process, cli.quant_max_val);
            let hmm_signal_series_cont = feature_df_task_local.column(signal_name_to_process)
                .with_context(|| format!("HMM signal column '{}' not found in feature_df", signal_name_to_process))?;
            let hmm_signal_series_materialized = hmm_signal_series_cont.as_materialized_series();
            let hmm_signal_series_discrete = io::quantize_series_to_u8(&hmm_signal_series_materialized, cli.quant_max_val)?;
            let min_discrete: Option<u8> = hmm_signal_series_discrete.min()?;
            let max_discrete: Option<u8> = hmm_signal_series_discrete.max()?;
            log::info!("Level C for '{}': Signal quantized for HMM. Min: {:?}, Max: {:?}", signal_name_to_process, min_discrete, max_discrete);
            let hmm_signal_vec_u8 = io::series_to_vec_u8(&hmm_signal_series_discrete)?;
            
            let viterbi_states_c = level_c::viterbi_path_from_observations(
                &hmm_signal_vec_u8, 
                cli.hmm_states, 
                cli.quant_max_val
            )?;
            log::info!("Level C for '{}': HMM Viterbi path calculated. Num states in path: {}, Time: {:.2?}", 
                signal_name_to_process,
                viterbi_states_c.iter().max().map_or(0, |m| m + 1), 
                hmm_start_time.elapsed()
            );
            let era_c_series = Series::new("era_level_C".into(), viterbi_states_c.iter().map(|&x| x as i32).collect::<Vec<i32>>());
            let df_out_c = DataFrame::new(vec![time_col_series_task_local.as_ref().clone(), era_c_series.into()])?;
            era_db_clone.as_ref().save_era_labels(
                &df_out_c, 
                "era_level_C", 
                signal_name_to_process, 
                'C', 
                "HMM", 
                "era_labels_level_c"
            ).with_context(|| format!("Failed to save Level C labels to DB for signal '{}'", signal_name_to_process))?;
            log::info!("Saved Level C labels for '{}' to database table 'era_labels_level_c'", signal_name_to_process);

            Ok(())
        };
        // Execute the processing for the current signal and log any errors.
            process_signal_task()
        }).collect()
    } else {
        log::info!("Processing {} signals sequentially", signal_columns.len());
        signal_columns.iter().map(|&signal_name_to_process| {
            let era_db_clone = Arc::clone(&era_db);
            let feature_df_task_local = Arc::clone(&feature_df_arc);
            let time_col_series_task_local = Arc::clone(&time_col_series_arc);
            
            let process_signal_task = || -> Result<()> {
                log::info!("\n\n=== Processing signal: '{}' ===", signal_name_to_process);

                let pelt_start_time = std::time::Instant::now();
                log::info!("\n--- Level A: PELT Segmentation on '{}' ---", signal_name_to_process);
                let pelt_signal_column = feature_df_task_local
                    .column(signal_name_to_process)
                    .with_context(|| format!("PELT signal column '{}' not found in feature_df", signal_name_to_process))?;
                let pelt_signal_vec = io::series_to_vec_f64(pelt_signal_column.as_series().expect("Series conversion failed for PELT"))?;
                let level_a_bocpd_threshold = 0.5;
                let pelt_bkps_indices = level_a::detect_changepoints_level_a(
                    &pelt_signal_vec, 
                    cli.bocpd_lambda, 
                    cli.pelt_min_size, 
                    level_a_bocpd_threshold
                )?;
                log::info!("Level A for '{}': PELT-like (BOCPD based) detected {} breakpoints. Indices: {:?}. Time: {:.2?}", 
                    signal_name_to_process, pelt_bkps_indices.len(), pelt_bkps_indices, pelt_start_time.elapsed());

                let mut era_a_values = vec![0i32; feature_df_task_local.height()];
                let mut current_era_a = 0;
                let mut last_idx_a = 0;
                for &bkp_idx in &pelt_bkps_indices {
                    if bkp_idx > last_idx_a && bkp_idx <= feature_df_task_local.height() {
                        for i in last_idx_a..bkp_idx { era_a_values[i] = current_era_a; }
                    }
                    last_idx_a = bkp_idx;
                    current_era_a += 1;
                }
                if last_idx_a < feature_df_task_local.height() { 
                    for i in last_idx_a..feature_df_task_local.height() { era_a_values[i] = current_era_a; }
                }
                let era_a_series = Series::new("era_level_A".into(), era_a_values);
                let df_out_a = DataFrame::new(vec![time_col_series_task_local.as_ref().clone(), era_a_series.into()])?;

                // Note: Deletion is now handled inside save_era_labels as part of transaction
                era_db_clone.as_ref().save_era_labels(
                    &df_out_a, 
                    "era_level_A", 
                    signal_name_to_process, 
                    'A', 
                    "PELT", 
                    "era_labels_level_a"
                ).with_context(|| format!("Failed to save Level A labels to DB for signal '{}'", signal_name_to_process))?;
                log::info!("Saved Level A labels for '{}' to database table 'era_labels_level_a'", signal_name_to_process);

                let bocpd_start_time = std::time::Instant::now();
                log::info!("\n--- Level B: BOCPD on '{}' ---", signal_name_to_process);
                let bocpd_signal_column = feature_df_task_local
                    .column(signal_name_to_process)
                    .with_context(|| format!("BOCPD signal column '{}' not found in feature_df", signal_name_to_process))?;
                let bocpd_signal_vec = io::series_to_vec_f64(bocpd_signal_column.as_series().expect("Series conversion failed for BOCPD"))?;
                let cp_probs_b = level_b::bocpd_probabilities(&bocpd_signal_vec, cli.bocpd_lambda)?;
                log::info!("Level B for '{}': BOCPD probabilities calculated. Time: {:.2?}", signal_name_to_process, bocpd_start_time.elapsed());
                
                let cp_probs_b_series = Series::new("cp_prob_level_B".into(), cp_probs_b);
                let bool_chunked = cp_probs_b_series.gt(0.5)?;
                let mut cum_label: i32 = 0;
                let era_labels: Vec<i32> = bool_chunked
                    .into_iter()
                    .map(|opt_b| {
                        match opt_b {
                            Some(true) => { cum_label += 1; cum_label }
                            _ => cum_label,
                        }
                    })
                    .collect();
                let era_b_series = Series::new("era_level_B".into(), era_labels);
                let df_out_b = DataFrame::new(vec![time_col_series_task_local.as_ref().clone(), era_b_series.into()])?;

                // Note: Deletion is now handled inside save_era_labels as part of transaction
                era_db_clone.as_ref().save_era_labels(
                    &df_out_b, 
                    "era_level_B", 
                    signal_name_to_process, 
                    'B', 
                    "BOCPD", 
                    "era_labels_level_b"
                ).with_context(|| format!("Failed to save Level B labels to DB for signal '{}'", signal_name_to_process))?;
                log::info!("Saved Level B labels for '{}' to database table 'era_labels_level_b'", signal_name_to_process);

                let hmm_start_time = std::time::Instant::now();
                log::info!("\n--- Level C: HMM on '{}' ---", signal_name_to_process);
                let hmm_signal_column = feature_df_task_local
                    .column(signal_name_to_process)
                    .with_context(|| format!("HMM signal column '{}' not found in feature_df", signal_name_to_process))?;
                let hmm_signal_vec = io::series_to_vec_f64(hmm_signal_column.as_series().expect("Series conversion failed for HMM"))?;
                let (quantized_data, _min_val, _max_val) = quantize_signal_f64(&hmm_signal_vec, cli.quant_max_val); // Prefixed unused vars
                let predicted_states = level_c::viterbi_path_from_observations(&quantized_data, cli.hmm_states, cli.quant_max_val)?;
                let final_states_as_i32: Vec<i32> = predicted_states.into_iter().map(|s| s as i32).collect();
                log::info!("Level C for '{}': HMM predicted {} states. Time: {:.2?}", signal_name_to_process, cli.hmm_states, hmm_start_time.elapsed());
                
                let era_c_series = Series::new("era_level_C".into(), final_states_as_i32);
                let df_out_c = DataFrame::new(vec![time_col_series_task_local.as_ref().clone(), era_c_series.into()])?;

                // Note: Deletion is now handled inside save_era_labels as part of transaction
                era_db_clone.as_ref().save_era_labels(
                    &df_out_c, 
                    "era_level_C", 
                    signal_name_to_process, 
                    'C', 
                    "HMM", 
                    "era_labels_level_c"
                ).with_context(|| format!("Failed to save Level C labels to DB for signal '{}'", signal_name_to_process))?;
                log::info!("Saved Level C labels for '{}' to database table 'era_labels_level_c'", signal_name_to_process);

                log::info!("\n=== Completed processing signal: '{}' ===", signal_name_to_process);
                Ok(())
            };
            
            process_signal_task()
        }).collect()
    };

    // Check for any errors in processing
    let mut had_errors = false;
    for (i, result) in processing_results.into_iter().enumerate() {
        if let Err(e) = result {
            had_errors = true;
            let signal_name = signal_columns.get(i).map(|s| s.as_str()).unwrap_or("unknown");
            log::error!("Error processing signal '{}': {:?}", signal_name, e);
        }
    }

    if had_errors {
        return Err(anyhow::anyhow!("One or more signals failed to process. Check logs for details."));
    }

    log::info!("\n--- Rust Era Detection Script Finished Successfully. Total Time: {:.2?} ---", start_time_main.elapsed());
    Ok(())
}
