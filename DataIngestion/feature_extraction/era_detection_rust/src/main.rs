use clap::Parser;
use polars::prelude::*;
use polars::prelude::FillNullStrategy;
use std::path::PathBuf;
use anyhow::{Result, Context};

mod io;
mod level_a;
mod level_b;
mod level_c;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Cli {
    #[clap(long, value_parser)]
    input_parquet: PathBuf,
    #[clap(long)]
    output_suffix: String,
    #[clap(long, default_value = "/app/data/processed/")]
    output_dir: PathBuf,
    #[clap(long, default_value_t = 48)] // Default min_size for PELT (e.g. 1 day for 5T data if 12*24=288, 48 ~ 4h)
    pelt_min_size: usize,
    #[clap(long, default_value_t = 200.0)] // Expected run length for BOCPD
    bocpd_lambda: f64,
    #[clap(long, default_value_t = 5)] // Reduced default HMM states
    hmm_states: usize,
    #[clap(long, default_value_t = 20)] // Reduced default HMM iterations
    hmm_iterations: usize,
    #[clap(long, default_value = "dli_sum")]
    pelt_signal_col: String, // Column for PELT
    #[clap(long, default_value = "dli_sum")]
    b_level_signal_col: String,
    #[clap(long, default_value = "dli_sum")]
    c_level_signal_col: String,
    #[clap(long, default_value_t = 250)] // Max value for u8 quantization (e.g., 250 means values 0-249)
    quant_max_val: u8,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let start_time_main = std::time::Instant::now();

    println!("--- Rust Era Detection Script ---");
    println!("Input Parquet: {:?}", cli.input_parquet);
    println!("Output Suffix: {}", cli.output_suffix);
    println!("Output Dir: {:?}", cli.output_dir);
    println!("Pelt Min Size: {}", cli.pelt_min_size);
    println!("BOCPD Lambda: {}", cli.bocpd_lambda);
    println!("HMM States: {}", cli.hmm_states);
    println!("HMM Iterations: {}", cli.hmm_iterations);
    println!("Quantization Max Value (0-{}): {}", cli.quant_max_val.saturating_sub(1), cli.quant_max_val);


    let mut df_main = io::read_parquet_to_polars_df(&cli.input_parquet)
        .with_context(|| format!("Failed to read input parquet: {:?}", cli.input_parquet))?;
    println!("Loaded DataFrame. Shape: {:?}, Time: {:.2?}", df_main.shape(), start_time_main.elapsed());

    // Ensure 'time' column is Datetime
    if df_main.column("time")?.dtype() != &DataType::Datetime(TimeUnit::Microseconds, None) &&
       df_main.column("time")?.dtype() != &DataType::Datetime(TimeUnit::Nanoseconds, None) &&
       df_main.column("time")?.dtype() != &DataType::Datetime(TimeUnit::Milliseconds, None) {
        df_main = df_main.lazy().with_column(
            col("time").cast(DataType::Datetime(TimeUnit::Microseconds, None))
        ).collect()?;
        println!("Casted 'time' column to Datetime(tu=us).");
    }
    
    let sort_options = SortMultipleOptions::new().with_order_descending(false);
    df_main = df_main.sort(["time"], sort_options)?;
    let time_col_series = df_main.column("time")?.clone();


    let segmentation_cols = [
        "dli_sum", "pipe_temp_1_c", "pipe_temp_2_c",
        "outside_temp_c", "radiation_w_m2", "co2_measured_ppm"
    ];
    let mut feature_df = df_main.select(segmentation_cols)?;
    
    let fill_start_time = std::time::Instant::now();
    feature_df = feature_df
        .fill_null(FillNullStrategy::Forward(None))?
        .fill_null(FillNullStrategy::Backward(None))?;
    println!("Filled nulls in feature_df. Shape: {:?}, Time: {:.2?}", feature_df.shape(), fill_start_time.elapsed());


    // --- Level A: PELT ---
    let pelt_start_time = std::time::Instant::now();
    println!("\n--- Level A: PELT Segmentation on '{}' ---", cli.pelt_signal_col);
    let pelt_signal_column = feature_df
        .column(&cli.pelt_signal_col)
        .with_context(|| format!("PELT signal column '{}' not found", cli.pelt_signal_col))?;
    let pelt_signal_vec = io::series_to_vec_f64(pelt_signal_column.as_series())?;
    let level_a_bocpd_threshold = 0.5;
    let pelt_bkps_indices = level_a::detect_changepoints_level_a(
        &pelt_signal_vec, 
        cli.bocpd_lambda, // Using general BOCPD lambda for consistency, or define a specific one for Level A
        cli.pelt_min_size, 
        level_a_bocpd_threshold
    )?;
    println!("PELT-like (BOCPD based) detected {} breakpoints. Indices: {:?}. Time: {:.2?}", pelt_bkps_indices.len(), pelt_bkps_indices, pelt_start_time.elapsed());

    let mut era_a_values = vec![0i32; df_main.height()];
    let mut current_era_a = 0;
    let mut last_idx_a = 0;
    for &bkp_idx in &pelt_bkps_indices {
        if bkp_idx > last_idx_a && bkp_idx <= df_main.height() {
            for i in last_idx_a..bkp_idx { era_a_values[i] = current_era_a; }
        }
        last_idx_a = bkp_idx;
        current_era_a += 1;
    }
    if last_idx_a < df_main.height() { // Label the last segment
        for i in last_idx_a..df_main.height() { era_a_values[i] = current_era_a; }
    }
    let era_a_series = Series::new("era_level_A".into(), era_a_values);
    let mut df_out_a = DataFrame::new(vec![time_col_series.clone(), era_a_series.into()])?;
    let path_a = cli.output_dir.join(format!("{}_era_labels_levelA.parquet", cli.output_suffix));
    io::write_polars_df_to_parquet(&mut df_out_a, &path_a)?;
    println!("Saved Level A labels to {:?}", path_a);

    // --- Level B: BOCPD ---
    let bocpd_start_time = std::time::Instant::now();
    println!("\n--- Level B: BOCPD on '{}' ---", cli.b_level_signal_col);
    let bocpd_signal_column = feature_df
        .column(&cli.b_level_signal_col)
        .with_context(|| format!("BOCPD signal column '{}' not found", cli.b_level_signal_col))?;
    let bocpd_signal_vec = io::series_to_vec_f64(bocpd_signal_column.as_series())?;
    let cp_probs_b = level_b::bocpd_probabilities(&bocpd_signal_vec, cli.bocpd_lambda)?;
    println!("BOCPD probabilities calculated. Time: {:.2?}", bocpd_start_time.elapsed());
    
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
    let mut df_out_b = DataFrame::new(vec![time_col_series.clone(), cp_probs_b_series.into(), era_b_series.into()])?;
    let path_b = cli.output_dir.join(format!("{}_era_labels_levelB.parquet", cli.output_suffix));
    io::write_polars_df_to_parquet(&mut df_out_b, &path_b)?;
    println!("Saved Level B labels to {:?}", path_b);

    // --- Level C: HMM Viterbi ---
    /*
    let hmm_start_time = std::time::Instant::now();
    println!("\n--- Level C: HMM Viterbi on '{}' (Quantized up to max value: {}) ---", cli.c_level_signal_col, cli.quant_max_val);
    let hmm_signal_series_cont = feature_df.column(&cli.c_level_signal_col)
        .with_context(|| format!("HMM signal column '{}' not found", cli.c_level_signal_col))?;
    let hmm_signal_series_discrete = io::quantize_series_to_u8(&hmm_signal_series_cont, cli.quant_max_val)?;
    let min_discrete: Option<u8> = hmm_signal_series_discrete.min()?;
    let max_discrete: Option<u8> = hmm_signal_series_discrete.max()?;
    println!("Signal quantized for HMM. Min: {:?}, Max: {:?}", min_discrete, max_discrete);
    let hmm_signal_vec_u8 = io::series_to_vec_u8(&hmm_signal_series_discrete)?;
    
    let viterbi_states_c = level_c::viterbi_path_from_observations(&hmm_signal_vec_u8, cli.hmm_states, cli.hmm_iterations)?;
    println!("HMM Viterbi path calculated. Num states: {}, Time: {:.2?}", viterbi_states_c.iter().max().map_or(0, |m| m + 1), hmm_start_time.elapsed());
    let era_c_series = Series::new("era_level_C".into(), viterbi_states_c.iter().map(|&x| x as i32).collect::<Vec<i32>>());
    let mut df_out_c = DataFrame::new(vec![time_col_series.clone(), era_c_series.into()])?;
    let path_c = cli.output_dir.join(format!("{}_era_labels_levelC.parquet", cli.output_suffix));
    io::write_polars_df_to_parquet(&mut df_out_c, &path_c)?;
    println!("Saved Level C labels to {:?}", path_c);
    */

    println!("\n--- Rust Era Detection Script Finished Successfully. Total Time: {:.2?} ---", start_time_main.elapsed());
    Ok(())
}
