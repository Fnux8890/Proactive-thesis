use crate::config::FileConfig;
use crate::data_models::ParsedRecord;
use crate::errors::PipelineError;
use crate::file_processor;
use crossbeam_channel::{bounded, Receiver, Sender};
use indicatif::{ProgressBar, ProgressStyle};
use log::{error, info};
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

/// Result of processing a single file
#[derive(Debug)]
pub struct FileProcessResult {
    #[allow(dead_code)]
    pub config_index: usize,
    pub file_path: String,
    pub records: Vec<ParsedRecord>,
    pub error: Option<String>,
    #[allow(dead_code)]
    pub processing_time_ms: u128,
}

/// Parallel file processor using Rayon
pub struct ParallelProcessor {
    #[allow(dead_code)]
    num_workers: usize,
}

impl ParallelProcessor {
    pub fn new() -> Self {
        let num_workers = num_cpus::get();
        info!("Initializing ParallelProcessor with {} workers", num_workers);
        Self { num_workers }
    }

    #[allow(dead_code)]
    pub fn with_workers(num_workers: usize) -> Self {
        info!("Initializing ParallelProcessor with {} custom workers", num_workers);
        Self { num_workers }
    }

    /// Process multiple file configs in parallel
    pub fn process_files(&self, configs: Vec<FileConfig>) -> Vec<FileProcessResult> {
        let total_files = configs.len();
        info!("Starting parallel processing of {} files", total_files);

        // Create progress bar
        let progress = Arc::new(ProgressBar::new(total_files as u64));
        progress.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        // Process files in parallel using Rayon
        let results: Vec<FileProcessResult> = configs
            .into_par_iter()
            .enumerate()
            .map(|(index, config)| {
                let start = Instant::now();
                let file_path = config.container_path.to_string_lossy().to_string();
                let progress_clone = Arc::clone(&progress);

                let result = match file_processor::process_file(&config) {
                    Ok(records) => {
                        let processing_time = start.elapsed().as_millis();
                        info!(
                            "Successfully processed {} records from {} in {}ms",
                            records.len(),
                            file_path,
                            processing_time
                        );
                        FileProcessResult {
                            config_index: index,
                            file_path,
                            records,
                            error: None,
                            processing_time_ms: processing_time,
                        }
                    }
                    Err(e) => {
                        let processing_time = start.elapsed().as_millis();
                        error!("Failed to process {}: {}", file_path, e);
                        FileProcessResult {
                            config_index: index,
                            file_path,
                            records: Vec::new(),
                            error: Some(e.to_string()),
                            processing_time_ms: processing_time,
                        }
                    }
                };

                progress_clone.inc(1);
                result
            })
            .collect();

        progress.finish_with_message("File processing completed");
        results
    }
}

/// Producer-Consumer pattern for batch processing
#[allow(dead_code)]
pub struct BatchProcessor<T: Send + 'static> {
    sender: Sender<Option<T>>,
    receiver: Receiver<Option<T>>,
    buffer_size: usize,
}

impl<T: Send + 'static> BatchProcessor<T> {
    #[allow(dead_code)]
    pub fn new(buffer_size: usize) -> Self {
        let (sender, receiver) = bounded(buffer_size);
        Self {
            sender,
            receiver,
            buffer_size,
        }
    }

    #[allow(dead_code)]
    pub fn sender(&self) -> Sender<Option<T>> {
        self.sender.clone()
    }

    #[allow(dead_code)]
    pub fn receiver(&self) -> Receiver<Option<T>> {
        self.receiver.clone()
    }

    /// Process items in batches
    #[allow(dead_code)]
    pub async fn process_batches<F, Fut>(
        &self,
        batch_size: usize,
        mut processor: F,
    ) -> Result<usize, PipelineError>
    where
        F: FnMut(Vec<T>) -> Fut,
        Fut: std::future::Future<Output = Result<usize, PipelineError>>,
    {
        let mut batch = Vec::with_capacity(batch_size);
        let mut total_processed = 0;

        loop {
            match self.receiver.recv() {
                Ok(Some(item)) => {
                    batch.push(item);
                    if batch.len() >= batch_size {
                        let current_batch = std::mem::replace(&mut batch, Vec::with_capacity(batch_size));
                        total_processed += processor(current_batch).await?;
                    }
                }
                Ok(None) => {
                    // Process remaining items
                    if !batch.is_empty() {
                        total_processed += processor(batch).await?;
                    }
                    break;
                }
                Err(e) => {
                    error!("Channel receive error: {}", e);
                    return Err(PipelineError::ChannelError(e.to_string()));
                }
            }
        }

        Ok(total_processed)
    }
}

/// Glob file expander using parallel processing
pub fn expand_globs_parallel(configs: &[FileConfig]) -> Vec<FileConfig> {
    configs
        .par_iter()
        .flat_map(|config| {
            let path_str = config.container_path.to_string_lossy();
            if path_str.contains('*') || path_str.contains('?') {
                match glob::glob(&path_str) {
                    Ok(paths) => {
                        let expanded: Vec<FileConfig> = paths
                            .filter_map(|entry| entry.ok())
                            .map(|path| {
                                let mut specific_config = config.clone();
                                specific_config.container_path = path;
                                specific_config
                            })
                            .collect();
                        info!("Expanded glob {} to {} files", path_str, expanded.len());
                        expanded
                    }
                    Err(e) => {
                        error!("Invalid glob pattern {}: {}", path_str, e);
                        vec![config.clone()]
                    }
                }
            } else {
                vec![config.clone()]
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_parallel_processor_creation() {
        let processor = ParallelProcessor::new();
        assert!(processor.num_workers > 0);
    }

    #[test]
    fn test_batch_processor_creation() {
        let processor: BatchProcessor<String> = BatchProcessor::new(100);
        assert_eq!(processor.buffer_size, 100);
    }
}