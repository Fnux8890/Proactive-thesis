use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub database_url: String,
    pub batch_size: usize,
    pub gpu_device_id: u32,
    pub features: FeatureConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureConfig {
    pub rolling_window_sizes: Vec<usize>,
    pub enable_spectral: bool,
    pub enable_entropy: bool,
    pub enable_wavelet: bool,
    pub enable_cross_features: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            database_url: String::from("postgresql://postgres:postgres@localhost:5432/greenhouse"),
            batch_size: 1000,
            gpu_device_id: 0,
            features: FeatureConfig::default(),
        }
    }
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            rolling_window_sizes: vec![60, 300, 900, 3600], // 1min, 5min, 15min, 1h
            enable_spectral: true,
            enable_entropy: true,
            enable_wavelet: true,
            enable_cross_features: true,
        }
    }
}

impl Config {
    #[allow(dead_code)]
    pub fn from_env() -> Result<Self> {
        let mut config = Self::default();
        
        if let Ok(url) = std::env::var("DATABASE_URL") {
            config.database_url = url;
        }
        
        if let Ok(batch_size) = std::env::var("GPU_BATCH_SIZE") {
            config.batch_size = batch_size.parse()?;
        }
        
        if let Ok(device_id) = std::env::var("CUDA_VISIBLE_DEVICES") {
            config.gpu_device_id = device_id.parse()?;
        }
        
        Ok(config)
    }
}