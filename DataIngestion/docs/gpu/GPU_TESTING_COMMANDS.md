# GPU Feature Extraction Testing Commands

## Era Detection Parameter Changes Applied

The era detection parameters have been updated to create more reasonable era sizes:

| Parameter | Old Value | New Value | Effect |
|-----------|-----------|-----------|---------|
| `pelt-min-size` | 48 (4 hours) | 288 (24 hours) | Prevents tiny eras |
| `bocpd-lambda` | 200.0 | 50.0 | More sensitive to operational changes |
| `hmm-states` | 5 | 10 | More granular state detection |

## Testing Sequence

### 1. Clean Previous Era Detection Results
```bash
# Remove old era labels
docker compose run --rm db psql -U postgres -c "DROP TABLE IF EXISTS era_labels_level_a, era_labels_level_b, era_labels_level_c CASCADE;"

# Clean up feature tables
docker compose run --rm db psql -U postgres -c "DROP TABLE IF EXISTS gpu_features_level_a, gpu_features_level_b, gpu_features_level_c CASCADE;"
```

### 2. Re-run Era Detection with New Parameters
```bash
# Build and run era detection with new parameters
docker compose build era_detector
docker compose up era_detector
```

### 3. Verify New Era Sizes
```bash
# Check Level A eras
docker compose run --rm db psql -U postgres -c "
SELECT COUNT(*) as total_eras, 
       AVG(EXTRACT(EPOCH FROM (end_time - start_time))/3600) as avg_hours,
       MIN(EXTRACT(EPOCH FROM (end_time - start_time))/3600) as min_hours,
       MAX(EXTRACT(EPOCH FROM (end_time - start_time))/3600) as max_hours
FROM era_labels_level_a;"

# Check Level B eras
docker compose run --rm db psql -U postgres -c "
SELECT COUNT(*) as total_eras,
       AVG(EXTRACT(EPOCH FROM (end_time - start_time))/3600) as avg_hours
FROM era_labels_level_b;"

# Check Level C eras
docker compose run --rm db psql -U postgres -c "
SELECT COUNT(*) as total_eras,
       AVG(EXTRACT(EPOCH FROM (end_time - start_time))/3600) as avg_hours
FROM era_labels_level_c;"
```

### 4. Test GPU Feature Extraction

#### Build GPU Feature Extraction Container
```bash
docker compose build gpu_feature_extraction
```

#### Test Level A (should now work with fixed shared memory and reasonable era sizes)
```bash
# Test with limited eras first
docker compose run --rm -e CUDA_LAUNCH_BLOCKING=1 gpu_feature_extraction_level_a \
  --era-level A \
  --max-eras 1 \
  --batch-size 100 \
  --min-era-rows 10
```

#### If successful, run full Level A
```bash
docker compose up gpu_feature_extraction_level_a
```

#### Test Sequential Execution (A → B → C)
```bash
# This runs all three levels sequentially as configured
docker compose up gpu_feature_extraction_level_a gpu_feature_extraction_level_b gpu_feature_extraction_level_c
```

## Monitoring Commands

### Check GPU Memory Usage
```bash
# In another terminal while GPU extraction is running
nvidia-smi -l 1
```

### Check Container Logs
```bash
# Follow logs for Level A
docker compose logs -f gpu_feature_extraction_level_a

# Check for CUDA errors
docker compose logs gpu_feature_extraction_level_a | grep -i "cuda\|error"
```

### Verify Features Were Computed
```bash
# Check feature counts
docker compose run --rm db psql -U postgres -c "
SELECT 'Level A' as level, COUNT(*) as features FROM gpu_features_level_a
UNION ALL
SELECT 'Level B', COUNT(*) FROM gpu_features_level_b  
UNION ALL
SELECT 'Level C', COUNT(*) FROM gpu_features_level_c;"
```

## Expected Results

With the updated parameters:
- **Level A**: Should have 10-50 eras spanning days to weeks (not years)
- **Level B**: Should have 50-200 eras spanning hours to days
- **Level C**: Should have 200-1000 eras spanning hours

The GPU feature extraction should:
- No longer encounter `CUDA_ERROR_INVALID_VALUE` (shared memory fixed)
- Process reasonable data sizes per era (not millions of rows)
- Complete successfully for all three levels

## Troubleshooting

If issues persist:
1. Check era sizes match expectations (step 3)
2. Enable debug mode: `RUST_LOG=debug docker compose up gpu_feature_extraction_level_a`
3. Run with CUDA debugging: `CUDA_LAUNCH_BLOCKING=1` to get better error messages
4. Check database logs: `docker compose logs db`

## Production Deployment

Once testing is successful:
```bash
# Use production compose file
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```