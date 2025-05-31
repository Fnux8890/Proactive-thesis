use anyhow::Result;
use polars::prelude::*;

pub fn handle_empty_hours(df: DataFrame) -> Result<DataFrame> {
    if df.height() == 0 {
        warn!("Empty DataFrame encountered, returning empty result");
        return Ok(df);
    }
    
    // Filter out hours with no data at all
    let mask = df.column("total_records")?
        .gt(0)?;
    
    Ok(df.filter(&mask)?)
}

pub fn validate_physics_constraints(
    value: f32,
    prev_value: Option<f32>,
    min_val: f32,
    max_val: f32,
    max_change: f32,
) -> Option<f32> {
    // Check bounds
    if value < min_val || value > max_val {
        return None;
    }
    
    // Check rate of change
    if let Some(prev) = prev_value {
        if (value - prev).abs() > max_change {
            return None;
        }
    }
    
    Some(value)
}
