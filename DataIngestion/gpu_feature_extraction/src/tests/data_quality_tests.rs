#[cfg(test)]
mod data_quality_tests {
    use crate::data_quality::*;
    use polars::prelude::*;
    
    fn create_test_df_good_quality() -> DataFrame {
        df! {
            "air_temp_c" => [20.0f32, 20.5, 21.0, 21.2, 20.8],
            "co2_measured_ppm" => [400.0f32, 420.0, 430.0, 425.0, 410.0],
            "relative_humidity_percent" => [60.0f32, 62.0, 61.0, 63.0, 60.5],
        }.unwrap()
    }
    
    fn create_test_df_poor_quality() -> DataFrame {
        df! {
            "air_temp_c" => [20.0f32, f32::NAN, f32::NAN, 30.0, f32::NAN],
            "co2_measured_ppm" => [f32::NAN, f32::NAN, 430.0, f32::NAN, f32::NAN],
            "relative_humidity_percent" => [60.0f32, f32::NAN, f32::NAN, f32::NAN, f32::NAN],
        }.unwrap()
    }
    
    #[test]
    fn test_quality_metrics_good_data() {
        let analyzer = DataQualityAnalyzer::new();
        let df = create_test_df_good_quality();
        
        let metrics = analyzer.analyze_window(&df).unwrap();
        
        // Good data should have high scores
        assert!(metrics.coverage > 0.9);
        assert!(metrics.continuity > 0.8);
        assert!(metrics.consistency > 0.8);
        assert!(metrics.overall_score > 0.8);
    }
    
    #[test]
    fn test_quality_metrics_poor_data() {
        let analyzer = DataQualityAnalyzer::new();
        let df = create_test_df_poor_quality();
        
        let metrics = analyzer.analyze_window(&df).unwrap();
        
        // Poor data should have low scores
        assert!(metrics.coverage < 0.5);
        assert!(metrics.continuity < 0.5);
        assert!(metrics.overall_score < 0.5);
    }
    
    #[test]
    fn test_adaptive_window_config() {
        // Test high quality metrics
        let high_quality = DataQualityMetrics {
            overall_score: 0.9,
            coverage: 0.95,
            continuity: 0.9,
            consistency: 0.85,
            sensor_availability: 0.9,
        };
        
        let config = AdaptiveWindowConfig::from_quality_metrics(&high_quality);
        assert_eq!(config.window_hours, 12); // Should use small window
        assert_eq!(config.slide_hours, 3);
        
        // Test low quality metrics
        let low_quality = DataQualityMetrics {
            overall_score: 0.3,
            coverage: 0.2,
            continuity: 0.3,
            consistency: 0.4,
            sensor_availability: 0.3,
        };
        
        let config = AdaptiveWindowConfig::from_quality_metrics(&low_quality);
        assert_eq!(config.window_hours, 48); // Should use large window
        assert_eq!(config.slide_hours, 12);
    }
    
    #[test]
    fn test_empty_dataframe_quality() {
        let analyzer = DataQualityAnalyzer::new();
        let df = DataFrame::empty();
        
        let result = analyzer.analyze_window(&df);
        
        // Should handle empty DataFrame gracefully
        assert!(result.is_err() || result.unwrap().overall_score == 0.0);
    }
    
    #[test]
    fn test_single_row_quality() {
        let analyzer = DataQualityAnalyzer::new();
        let df = df! {
            "air_temp_c" => [20.0f32],
            "co2_measured_ppm" => [400.0f32],
            "relative_humidity_percent" => [60.0f32],
        }.unwrap();
        
        let metrics = analyzer.analyze_window(&df).unwrap();
        
        // Single row should have perfect coverage but zero continuity
        assert_eq!(metrics.coverage, 1.0);
        assert_eq!(metrics.continuity, 0.0);
    }
    
    #[test]
    fn test_outlier_detection() {
        let analyzer = DataQualityAnalyzer::new();
        let df = df! {
            "air_temp_c" => [20.0f32, 21.0, 22.0, 100.0, 20.5], // 100.0 is outlier
            "co2_measured_ppm" => [400.0f32, 420.0, 430.0, 425.0, 410.0],
            "relative_humidity_percent" => [60.0f32, 62.0, 61.0, 63.0, 60.5],
        }.unwrap();
        
        let metrics = analyzer.analyze_window(&df).unwrap();
        
        // Consistency should be lower due to outlier
        assert!(metrics.consistency < 0.8);
    }
}