#[cfg(test)]
pub mod test_helpers {
    use polars::prelude::*;
    use anyhow::Result;
    
    /// Test the coverage calculation logic independently
    pub fn calculate_coverage_metrics(df: DataFrame) -> Result<DataFrame> {
        let temp_counts = df.column("temp_count")?
            .cast(&DataType::Float32)?
            .f32()?
            .clone();
        let temp_coverage = temp_counts / 10.0;
        
        let co2_counts = df.column("co2_count")?
            .cast(&DataType::Float32)?
            .f32()?
            .clone();
        let co2_coverage = co2_counts / 10.0;
        
        let humidity_counts = df.column("humidity_count")?
            .cast(&DataType::Float32)?
            .f32()?
            .clone();
        let humidity_coverage = humidity_counts / 10.0;
        
        // Convert to series for arithmetic operations
        let temp_series = temp_coverage.into_series();
        let co2_series = co2_coverage.into_series();
        let humidity_series = humidity_coverage.into_series();
        
        // Calculate overall coverage as average of three sensors
        let sum_coverage = (&temp_series + &co2_series)?;
        let sum_coverage_final = (&sum_coverage + &humidity_series)?;
        let overall_coverage = &sum_coverage_final / 3.0;
        
        let mut df = df;
        df.with_column(temp_series.with_name("temp_coverage"))?;
        df.with_column(co2_series.with_name("co2_coverage"))?;
        df.with_column(humidity_series.with_name("humidity_coverage"))?;
        df.with_column(overall_coverage.with_name("overall_coverage"))?;
        
        Ok(df)
    }
    
    /// Test viable hours filtering logic
    pub fn filter_viable_hours(df: DataFrame, min_coverage: f32) -> Result<DataFrame> {
        let mask = df.column("overall_coverage")?
            .gt(min_coverage)?;
        
        Ok(df.filter(&mask)?)
    }
}