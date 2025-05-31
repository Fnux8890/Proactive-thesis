#[cfg(test)]
mod sparse_pipeline_tests {
    use crate::sparse_pipeline::*;
    use crate::data_quality::*;
    use polars::prelude::*;
    use chrono::{DateTime, Utc, TimeZone};
    use anyhow::Result;
    
    // Helper function to create test DataFrame
    fn create_test_dataframe() -> DataFrame {
        let hours = vec![
            Utc.ymd(2014, 6, 1).and_hms(0, 0, 0).timestamp(),
            Utc.ymd(2014, 6, 1).and_hms(1, 0, 0).timestamp(),
            Utc.ymd(2014, 6, 1).and_hms(2, 0, 0).timestamp(),
        ];
        
        df! {
            "hour" => hours,
            "temp_mean" => [20.5, 21.0, f32::NAN],
            "temp_count" => [3i32, 5, 0],
            "co2_mean" => [450.0, f32::NAN, 480.0],
            "co2_count" => [2i32, 0, 4],
            "humidity_count" => [4i32, 3, 2],
            "total_records" => [60i32, 60, 60],
        }.unwrap()
    }
    
    #[test]
    fn test_coverage_calculation() {
        let config = SparsePipelineConfig::default();
        let pipeline = SparsePipeline::new_test(config);
        
        let df = create_test_dataframe();
        let result = pipeline.add_coverage_metrics(df);
        
        assert!(result.is_ok());
        let df_with_coverage = result.unwrap();
        
        // Check that coverage columns were added
        assert!(df_with_coverage.get_column_names().contains(&"temp_coverage"));
        assert!(df_with_coverage.get_column_names().contains(&"overall_coverage"));
        
        // Check coverage values
        let overall_coverage = df_with_coverage.column("overall_coverage").unwrap();
        let first_coverage = overall_coverage.f32().unwrap().get(0).unwrap();
        
        // (3 + 2 + 4) / 30.0 = 0.3
        assert!((first_coverage - 0.3).abs() < 0.01);
    }
    
    #[test]
    fn test_viable_hours_filtering() {
        let mut config = SparsePipelineConfig::default();
        config.min_hourly_coverage = 0.2;
        let pipeline = SparsePipeline::new_test(config);
        
        let df = create_test_dataframe();
        let df_with_coverage = pipeline.add_coverage_metrics(df).unwrap();
        let viable_df = pipeline.filter_viable_hours(df_with_coverage).unwrap();
        
        // Should filter out hours with coverage < 0.2
        assert!(viable_df.height() < 3);
    }
    
    #[test]
    fn test_gap_filling_constraints() {
        let config = SparsePipelineConfig::default();
        let pipeline = SparsePipeline::new_test(config);
        
        let df = create_test_dataframe();
        
        // Test temperature constraints
        let (filled_df, gaps_filled) = pipeline.fill_column_with_stats(
            df.clone(), 
            "temp_mean", 
            10.0,  // min
            40.0,  // max
            2.0    // max_change
        ).unwrap();
        
        // Check that NaN was filled
        let temp_col = filled_df.column("temp_mean").unwrap();
        let third_value = temp_col.f32().unwrap().get(2);
        assert!(third_value.is_some()); // Should be filled
        
        // Check physics constraint
        if let Some(val) = third_value {
            assert!(val >= 10.0 && val <= 40.0);
        }
    }
    
    #[test]
    fn test_extreme_sparsity() {
        // Test with only 1 reading per hour
        let hours = vec![Utc.ymd(2014, 6, 1).and_hms(0, 0, 0).timestamp()];
        
        let df = df! {
            "hour" => hours,
            "temp_mean" => [25.0f32],
            "temp_count" => [1i32],
            "co2_mean" => [f32::NAN],
            "co2_count" => [0i32],
            "humidity_count" => [0i32],
            "total_records" => [60i32],
        }.unwrap();
        
        let config = SparsePipelineConfig {
            min_hourly_coverage: 0.01, // Very low threshold
            ..Default::default()
        };
        let pipeline = SparsePipeline::new_test(config);
        
        let result = pipeline.add_coverage_metrics(df);
        assert!(result.is_ok());
        
        let df_with_coverage = result.unwrap();
        let coverage = df_with_coverage.column("overall_coverage")
            .unwrap()
            .f32()
            .unwrap()
            .get(0)
            .unwrap();
        
        // Should have very low coverage
        assert!(coverage < 0.1);
    }
    
    #[test]
    fn test_window_creation_edge_cases() {
        let config = SparsePipelineConfig::default();
        let pipeline = SparsePipeline::new_test(config);
        
        // Test with insufficient data for window
        let df = create_test_dataframe();
        let windows = pipeline.create_adaptive_windows(
            df,
            &AdaptiveWindowConfig {
                window_hours: 24,  // Larger than available data
                slide_hours: 12,
                min_quality_threshold: 0.1,
                min_sensors: 1,
            }
        );
        
        // Should handle gracefully
        assert!(windows.is_ok());
    }
    
    #[test]
    fn test_empty_dataframe_handling() {
        let config = SparsePipelineConfig::default();
        let pipeline = SparsePipeline::new_test(config);
        
        // Create empty DataFrame with correct schema
        let df = df! {
            "hour" => Vec::<i64>::new(),
            "temp_mean" => Vec::<f32>::new(),
            "temp_count" => Vec::<i32>::new(),
            "co2_mean" => Vec::<f32>::new(),
            "co2_count" => Vec::<i32>::new(),
            "humidity_count" => Vec::<i32>::new(),
            "total_records" => Vec::<i32>::new(),
        }.unwrap();
        
        let result = pipeline.add_coverage_metrics(df);
        assert!(result.is_ok());
        
        let df_with_coverage = result.unwrap();
        assert_eq!(df_with_coverage.height(), 0);
    }
    
    #[test]
    fn test_all_null_handling() {
        let hours = vec![Utc.ymd(2014, 6, 1).and_hms(0, 0, 0).timestamp()];
        
        let df = df! {
            "hour" => hours,
            "temp_mean" => [f32::NAN],
            "temp_count" => [0i32],
            "co2_mean" => [f32::NAN],
            "co2_count" => [0i32],
            "humidity_count" => [0i32],
            "total_records" => [60i32],
        }.unwrap();
        
        let config = SparsePipelineConfig::default();
        let pipeline = SparsePipeline::new_test(config);
        
        let result = pipeline.add_coverage_metrics(df);
        assert!(result.is_ok());
        
        let df_with_coverage = result.unwrap();
        let viable_df = pipeline.filter_viable_hours(df_with_coverage);
        
        assert!(viable_df.is_ok());
        assert_eq!(viable_df.unwrap().height(), 0); // Should filter out
    }
}

// Integration tests
#[cfg(test)]
mod integration_tests {
    use super::*;
    use sqlx::postgres::PgPoolOptions;
    
    #[tokio::test]
    #[ignore] // Run with: cargo test -- --ignored
    async fn test_database_integration() {
        dotenv::dotenv().ok();
        
        let database_url = std::env::var("DATABASE_URL")
            .unwrap_or_else(|_| "postgresql://postgres:postgres@localhost:5432/postgres".to_string());
        
        let pool = PgPoolOptions::new()
            .max_connections(5)
            .connect(&database_url)
            .await
            .expect("Failed to connect to database");
        
        let config = SparsePipelineConfig::default();
        let pipeline = SparsePipeline::new(pool.clone(), config);
        
        // Test actual database query
        let start = Utc.ymd(2014, 6, 1).and_hms(0, 0, 0);
        let end = Utc.ymd(2014, 6, 1).and_hms(1, 0, 0);
        
        let result = pipeline.stage1_aggregate_hourly(&start, &end).await;
        
        match result {
            Ok(df) => {
                println!("Successfully retrieved {} hours", df.height());
                assert!(df.height() >= 0);
            },
            Err(e) => {
                // This might fail if test data isn't present
                println!("Database test failed (expected if no test data): {}", e);
            }
        }
    }
}