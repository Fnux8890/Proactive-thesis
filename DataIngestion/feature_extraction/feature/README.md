# Feature Extraction Module

This module implements time-series feature extraction using tsfresh for the greenhouse climate control optimization system.

## Quick Start

### Running Feature Extraction Independently

To run feature extraction on existing era labels without running the full pipeline:

```bash
# From the feature_extraction directory
./run_feature_extraction.sh --skip-era-detection true --era-level B
```

### Building the Docker Container

```bash
# Build the feature extraction container
docker build -f feature/feature.dockerfile -t feature-extraction .

# Or using docker-compose
docker compose -f docker-compose.feature.yml build
```

### Running with Docker Compose

```bash
# Run feature extraction with existing era labels
SKIP_ERA_DETECTION=true docker compose -f docker-compose.feature.yml up

# Run with GPU support
USE_GPU=true docker compose -f docker-compose.feature.yml up
```

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and adjust:

- `SKIP_ERA_DETECTION`: Set to `true` to use existing era labels (speeds up testing)
- `ERA_LEVEL`: Which era detection level to use (A, B, or C)
- `FEATURE_SET`: Feature extraction complexity (minimal, efficient, comprehensive)
- `USE_GPU`: Enable GPU acceleration for feature extraction
- `BATCH_SIZE`: Number of eras to process in parallel

### Era Detection Levels

- **Level A (PELT)**: Major operational changes, fewer but longer eras
- **Level B (BOCPD)**: Medium-term patterns, balanced segmentation
- **Level C (HMM)**: Short-term state transitions, many short eras

## Architecture

### Components

1. **extract_features_enhanced.py**: Main feature extraction pipeline
2. **extract_features_direct.py**: Wrapper that can skip era detection
3. **feature_utils.py**: Utility functions for feature processing
4. **db_utils.py**: Database connection and operations

### Design Patterns

- **Repository Pattern**: Data access through `TimescaleDBRepository`
- **Factory Pattern**: Feature extractor creation
- **Pipeline Pattern**: Data transformation stages
- **Strategy Pattern**: Different feature selection methods

### Optimal Signal Selection

The module automatically selects optimal signals based on coverage analysis:

**Primary Signals** (highest coverage):
- `dli_sum`: Daily Light Integral
- `radiation_w_m2`: Solar radiation
- `outside_temp_c`: External temperature
- `co2_measured_ppm`: CO2 concentration
- `air_temp_c`: Internal temperature

**Secondary Signals**:
- `pipe_temp_1_c`: Heating system
- `curtain_1_percent`: Light control
- `humidity_deficit_g_m3`: Humidity control
- `heating_setpoint_c`: Temperature setpoint
- `vpd_hpa`: Vapor pressure deficit

## Testing

### Quick Test with Existing Era Labels

```bash
# Ensure database has era labels
# Set skip flag and run
SKIP_ERA_DETECTION=true ./run_feature_extraction.sh --era-level B --feature-set minimal
```

### Full Pipeline Test

```bash
# Run from parent DataIngestion directory
docker compose up feature_extraction
```

## Troubleshooting

### "Era label table does not exist"

This means era detection hasn't been run yet. Either:
1. Run era detection first: `docker compose up era_detector`
2. Or check if you're using the correct `ERA_LEVEL` (A, B, or C)

### GPU Memory Issues

Reduce `BATCH_SIZE` or set `USE_GPU=false` for CPU processing.

### Connection Refused

Ensure the database is running:
```bash
docker compose -f docker-compose.feature.yml up -d db
```

## Performance Tips

1. **Use GPU acceleration** when available (`USE_GPU=true`)
2. **Adjust batch size** based on available memory
3. **Use efficient feature set** for faster processing
4. **Process larger eras** by increasing `MIN_ERA_ROWS`

## Output

Features are saved to the `tsfresh_features` table with:
- Era metadata (id, level, stage, time range)
- Extracted time-series features
- Signal information
- Processing timestamps