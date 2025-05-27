use log::{error, warn};
use std::future::Future;
use std::time::Duration;
use tokio::time::sleep;

/// Configuration for retry logic
#[derive(Debug, Clone)]
pub struct RetryConfig {
    pub max_attempts: u32,
    pub initial_delay: Duration,
    pub max_delay: Duration,
    pub exponential_base: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            exponential_base: 2.0,
        }
    }
}

/// Execute an async operation with exponential backoff retry
pub async fn retry_with_backoff<F, Fut, T, E>(
    config: &RetryConfig,
    operation_name: &str,
    mut operation: F,
) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
    E: std::fmt::Display,
{
    let mut attempt = 0;
    let mut delay = config.initial_delay;

    loop {
        attempt += 1;
        
        match operation().await {
            Ok(result) => {
                if attempt > 1 {
                    warn!(
                        "Operation '{}' succeeded after {} attempts",
                        operation_name, attempt
                    );
                }
                return Ok(result);
            }
            Err(error) => {
                if attempt >= config.max_attempts {
                    error!(
                        "Operation '{}' failed after {} attempts. Final error: {}",
                        operation_name, attempt, error
                    );
                    return Err(error);
                }

                warn!(
                    "Operation '{}' failed (attempt {}/{}): {}. Retrying in {:?}...",
                    operation_name, attempt, config.max_attempts, error, delay
                );

                sleep(delay).await;

                // Calculate next delay with exponential backoff
                delay = Duration::from_millis(
                    (delay.as_millis() as f64 * config.exponential_base) as u64
                );
                if delay > config.max_delay {
                    delay = config.max_delay;
                }
            }
        }
    }
}

/// Retry configuration for database operations
pub fn db_retry_config() -> RetryConfig {
    RetryConfig {
        max_attempts: 5,
        initial_delay: Duration::from_millis(200),
        max_delay: Duration::from_secs(30),
        exponential_base: 2.0,
    }
}

/// Retry configuration for file operations
pub fn file_retry_config() -> RetryConfig {
    RetryConfig {
        max_attempts: 3,
        initial_delay: Duration::from_millis(50),
        max_delay: Duration::from_secs(5),
        exponential_base: 1.5,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[tokio::test]
    async fn test_retry_success_on_third_attempt() {
        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_clone = attempts.clone();

        let config = RetryConfig {
            max_attempts: 3,
            initial_delay: Duration::from_millis(10),
            max_delay: Duration::from_millis(100),
            exponential_base: 2.0,
        };

        let result = retry_with_backoff(&config, "test_operation", || {
            let attempts = attempts_clone.clone();
            async move {
                let current = attempts.fetch_add(1, Ordering::SeqCst) + 1;
                if current < 3 {
                    Err("simulated failure")
                } else {
                    Ok("success")
                }
            }
        })
        .await;

        assert_eq!(result, Ok("success"));
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_retry_exhausted() {
        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_clone = attempts.clone();

        let config = RetryConfig {
            max_attempts: 2,
            initial_delay: Duration::from_millis(10),
            max_delay: Duration::from_millis(100),
            exponential_base: 2.0,
        };

        let result = retry_with_backoff(&config, "test_operation", || {
            let attempts = attempts_clone.clone();
            async move {
                attempts.fetch_add(1, Ordering::SeqCst);
                Err::<(), &str>("persistent failure")
            }
        })
        .await;

        assert_eq!(result, Err("persistent failure"));
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
    }
}