# Feature Extraction Utilities

This directory contains utility scripts for pipeline validation and monitoring.

## Scripts

### `pipeline_status.py`
Checks the status of the GPU-enhanced feature extraction pipeline.

```bash
python utils/pipeline_status.py
```

### `validate_pipeline.py`
Pre-execution validation script that checks:
- Python file formatting (Black, isort)
- Configuration files (YAML, JSON)
- GPU availability
- Database connectivity

```bash
python utils/validate_pipeline.py [--fix]
```

### `run_with_validation.sh`
Shell script that runs validation before executing the pipeline.

```bash
./utils/run_with_validation.sh
```

## Note

One-time fix scripts have been removed. If you need to perform code formatting:
- Use `black .` for Python formatting
- Use `isort .` for import sorting
- Use `ruff check .` for linting
