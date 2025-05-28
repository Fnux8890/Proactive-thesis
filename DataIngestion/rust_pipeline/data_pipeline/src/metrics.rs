use once_cell::sync::Lazy;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Global metrics instance
pub static METRICS: Lazy<Mutex<Metrics>> = Lazy::new(|| Mutex::new(Metrics::new()));

/// Pipeline metrics tracker
#[derive(Debug, Default)]
pub struct Metrics {
    pub total_files_attempted: u64,
    pub total_files_successful: u64,
    pub total_files_failed: u64,
    pub total_records_parsed: u64,
    pub total_records_inserted: u64,
    pub total_bytes_processed: u64,
    pub processing_times: HashMap<String, Duration>,
    pub start_time: Option<Instant>,
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            start_time: Some(Instant::now()),
            ..Default::default()
        }
    }

    pub fn record_file_attempt(&mut self) {
        self.total_files_attempted += 1;
    }

    pub fn record_file_success(&mut self, records: u64) {
        self.total_files_successful += 1;
        self.total_records_parsed += records;
    }

    pub fn record_file_failure(&mut self) {
        self.total_files_failed += 1;
    }

    pub fn record_insertion(&mut self, count: u64) {
        self.total_records_inserted += count;
    }

    pub fn record_processing_time(&mut self, operation: String, duration: Duration) {
        self.processing_times.insert(operation, duration);
    }

    #[allow(dead_code)]
    pub fn record_bytes_processed(&mut self, bytes: u64) {
        self.total_bytes_processed += bytes;
    }

    pub fn get_total_duration(&self) -> Duration {
        self.start_time
            .map(|start| start.elapsed())
            .unwrap_or_default()
    }

    pub fn get_throughput(&self) -> f64 {
        let duration_secs = self.get_total_duration().as_secs_f64();
        if duration_secs > 0.0 {
            self.total_records_parsed as f64 / duration_secs
        } else {
            0.0
        }
    }

    pub fn print_summary(&self) {
        let duration = self.get_total_duration();
        println!("\n========== Pipeline Metrics Summary ==========");
        println!("Total Duration: {:.2?}", duration);
        println!("Files Attempted: {}", self.total_files_attempted);
        println!("Files Successful: {}", self.total_files_successful);
        println!("Files Failed: {}", self.total_files_failed);
        println!("Records Parsed: {}", self.total_records_parsed);
        println!("Records Inserted: {}", self.total_records_inserted);
        println!("Bytes Processed: {:.2} MB", self.total_bytes_processed as f64 / 1_048_576.0);
        println!("Throughput: {:.2} records/sec", self.get_throughput());
        
        if !self.processing_times.is_empty() {
            println!("\nProcessing Times:");
            for (op, duration) in &self.processing_times {
                println!("  {}: {:.2?}", op, duration);
            }
        }
        println!("=============================================\n");
    }
}

/// Helper macro to time an operation
#[macro_export]
macro_rules! time_operation {
    ($name:expr, $op:expr) => {{
        let start = std::time::Instant::now();
        let result = $op;
        let duration = start.elapsed();
        crate::metrics::METRICS.lock().record_processing_time($name.to_string(), duration);
        result
    }};
}