#[cfg(test)]
mod simple_tests {
    use polars::prelude::*;
    
    #[test]
    fn test_dataframe_operations() {
        // Test basic DataFrame operations work
        let df = df! {
            "temp_count" => [3i32, 5, 0],
            "co2_count" => [2i32, 0, 4],
        }.unwrap();
        
        assert_eq!(df.height(), 3);
        assert_eq!(df.width(), 2);
    }
    
    #[test]
    fn test_series_arithmetic() {
        // Test series arithmetic operations
        let s1 = Series::new("a", &[1.0f32, 2.0, 3.0]);
        let s2 = Series::new("b", &[4.0f32, 5.0, 6.0]);
        
        let sum = (&s1 + &s2).unwrap();
        let expected = Series::new("", &[5.0f32, 7.0, 9.0]);
        
        assert_eq!(sum, expected);
    }
    
    #[test]
    fn test_coverage_calculation() {
        // Test coverage calculation logic
        let counts = Series::new("counts", &[3.0f32, 5.0, 8.0]);
        let coverage = &counts / 10.0;
        
        match coverage.get(0).unwrap() {
            AnyValue::Float32(v) => assert!((v - 0.3f32).abs() < 0.001),
            _ => panic!("Expected Float32"),
        }
        match coverage.get(1).unwrap() {
            AnyValue::Float32(v) => assert!((v - 0.5f32).abs() < 0.001),
            _ => panic!("Expected Float32"),
        }
        match coverage.get(2).unwrap() {
            AnyValue::Float32(v) => assert!((v - 0.8f32).abs() < 0.001),
            _ => panic!("Expected Float32"),
        }
    }
}