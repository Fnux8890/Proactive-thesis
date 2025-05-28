use polars::prelude::*;
use polars::prelude::{RoundSeries, RoundMode};
use std::path::Path;
use anyhow::{Result, Context};
use std::fs::File;
// just for rode rabbit
pub fn read_parquet_to_polars_df(file_path: &Path) -> Result<DataFrame> {
    let file = File::open(file_path).with_context(|| format!("Failed to open parquet file: {:?}", file_path))?;
    ParquetReader::new(file)
        .finish()
        .with_context(|| format!("Failed to read parquet schema/metadata from file: {:?}", file_path))
}

#[allow(dead_code)]
pub fn write_polars_df_to_parquet(df: &mut DataFrame, output_path: &Path) -> Result<()> {
    let file = File::create(output_path)
        .with_context(|| format!("Failed to create output file: {:?}", output_path))?;
    ParquetWriter::new(file)
        .finish(df)
        .with_context(|| format!("Failed to write DataFrame to parquet: {:?}", output_path))?;
    Ok(())
}

pub fn series_to_vec_f64(series: &Series) -> Result<Vec<f64>> {
    let s_f64 = series.cast(&DataType::Float64)
        .with_context(|| format!("Failed to cast series '{}' to Float64", series.name()))?;
    let ca: &Float64Chunked = s_f64.f64()?;
    Ok(ca.into_iter().filter_map(|opt_val| opt_val).collect())
}

/// Calculate the coverage (non-null ratio) of a series
pub fn coverage(series: &Series) -> f32 {
    (series.len() - series.null_count()) as f32 / series.len() as f32
}

#[allow(dead_code)]
pub fn series_to_vec_u8(series: &Series) -> Result<Vec<u8>> {
    let s_u8 = series.cast(&DataType::UInt8)
        .with_context(|| format!("Failed to cast series '{}' to UInt8. Ensure values are in 0-255 range.", series.name()))?;
    let ca: &UInt8Chunked = s_u8.u8()?;
    Ok(ca.into_iter().filter_map(|opt_val| opt_val).collect())
}

#[allow(dead_code)]
pub fn quantize_series_to_u8(series: &Series, max_val_u8: u8) -> Result<Series> {
    let series_name = series.name().clone();
    if series.null_count() == series.len() || series.len() == 0 {
        let ca = UInt8Chunked::full_null(series_name.clone(), series.len());
        return Ok(ca.into_series().with_name(series_name));
    }

    let s_f64 = series.cast(&DataType::Float64)
        .with_context(|| format!("Failed to cast series '{}' to Float64 for quantization", series.name()))?;
    
    let min_val_opt: Option<f64> = s_f64.min()?;
    let max_val_opt: Option<f64> = s_f64.max()?;

    match (min_val_opt, max_val_opt) {
        (Some(min_val), Some(max_val)) => {
            if (max_val - min_val).abs() < 1e-9 {
                let ca = UInt8Chunked::full(series_name.clone(), 0u8, series.len());
                return Ok(ca.into_series().with_name(series_name));
            }

            let max_u8_f64 = max_val_u8 as f64;
            
            let scaled_series_f64 = (s_f64.clone() - min_val) / (max_val - min_val) * max_u8_f64;
            
            let quantized_series = scaled_series_f64
                .round(0, RoundMode::HalfAwayFromZero)?
                .cast(&DataType::UInt8)
                .with_context(|| "Failed to cast scaled series to UInt8 during quantization")?;

            Ok(quantized_series.with_name(series_name.clone()).clone())
        },
        _ => {
            let ca = UInt8Chunked::full_null(series_name.clone(), series.len());
            Ok(ca.into_series().with_name(series_name))
        }
    }
}
