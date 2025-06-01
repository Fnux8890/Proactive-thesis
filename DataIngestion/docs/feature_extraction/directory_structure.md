# Feature Extraction Directory Structure

```
feature_extraction/
├── config/                     # Configuration files
│   └── data_processing_config.json
├── utils/                      # Utility scripts
│   ├── pipeline_status.py
│   ├── validate_pipeline.py
│   └── run_with_validation.sh
├── parallel/                   # Parallel processing implementation
│   ├── coordinator.py
│   ├── worker_base.py
│   ├── gpu_worker.py
│   └── cpu_worker.py
├── feature/                    # CPU feature extraction
│   ├── extract_features*.py
│   └── feature_utils.py
├── feature-gpu/               # GPU feature extraction
│   └── extract_features_gpu.py
├── pre_process/               # Preprocessing pipeline
│   ├── preprocess.py
│   └── core/
├── era_detection_rust/        # Era detection (Rust)
│   └── src/
├── benchmarks/                # Performance benchmarks
├── tests/                     # Test files
│   └── docker/
├── db_utils_optimized.py      # Core database utilities
├── docker-compose.feature.yml # Feature extraction compose
├── pyproject.toml            # Python project config
└── README.md                 # Main documentation
```

## Key Directories

- **config/**: All configuration files
- **utils/**: Utility and validation scripts
- **parallel/**: Parallel processing workers
- **feature/**: Traditional CPU-based extraction
- **feature-gpu/**: GPU-accelerated extraction
- **pre_process/**: Data preprocessing pipeline
