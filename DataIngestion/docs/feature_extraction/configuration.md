# Feature Extraction Configuration

This directory contains configuration files for the feature extraction pipeline.

## Files

### `data_processing_config.json`
Main configuration file containing:
- Feature definitions and parameters
- Sensor column mappings
- Processing settings
- Feature extraction profiles (minimal, efficient, comprehensive)

## Usage

The configuration is loaded by various components:
- Preprocessing pipeline
- Feature extraction workers
- Validation scripts

## Configuration Structure

```json
{
  "features": {
    "sensor_name": {
      "type": "numeric|categorical|boolean",
      "unit": "unit_of_measurement",
      "profile": "minimal|efficient|comprehensive"
    }
  },
  "processing": {
    "batch_size": 1000,
    "n_jobs": 4
  }
}
```

## Environment Overrides

Configuration values can be overridden with environment variables:
- `FEATURE_SET`: minimal|efficient|comprehensive
- `BATCH_SIZE`: Processing batch size
- `N_JOBS`: Parallel jobs for CPU processing
