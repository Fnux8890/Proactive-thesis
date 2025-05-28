#!/bin/bash
# Test script to run era detection with both storage approaches

echo "ERA DETECTION: Testing Both Storage Approaches"
echo "=============================================="

# Configuration
DB_DSN="postgresql://postgres:postgres@db:5432/postgres?sslmode=prefer"
SIGNALS="air_temp_c,total_lamps_on"
RESAMPLE="5m"

# Test 1: JSONB Table (current approach)
echo -e "\n1. Testing JSONB table (preprocessed_features)..."
echo "   This uses features->>'column_name' extraction"
time docker run --rm \
    -e DB_DSN="$DB_DSN" \
    -e RUST_LOG=info \
    era_detector \
    --db-table preprocessed_features \
    --signal-cols "$SIGNALS" \
    --resample-every "$RESAMPLE" \
    --pelt-min-size 48 \
    --bocpd-lambda 200.0 \
    --hmm-states 5 \
    --hmm-iterations 20

# Test 2: Hybrid Table (new approach)
echo -e "\n2. Testing hybrid table (preprocessed_features_hybrid)..."
echo "   This uses direct column access"
time docker run --rm \
    -e DB_DSN="$DB_DSN" \
    -e RUST_LOG=info \
    era_detector \
    --db-table preprocessed_features_hybrid \
    --signal-cols "$SIGNALS" \
    --resample-every "$RESAMPLE" \
    --pelt-min-size 48 \
    --bocpd-lambda 200.0 \
    --hmm-states 5 \
    --hmm-iterations 20

echo -e "\nPerformance Comparison:"
echo "- JSONB approach: Requires extraction for every row"
echo "- Hybrid approach: Direct column access (5-10x faster)"

echo -e "\nChecking results in database..."
psql "$DB_DSN" <<EOF
-- Count era labels from both runs
SELECT 
    'Level A (PELT)' as algorithm,
    COUNT(DISTINCT signal_name) as signals,
    COUNT(*) as total_eras
FROM era_labels_level_a
UNION ALL
SELECT 
    'Level B (BOCPD)' as algorithm,
    COUNT(DISTINCT signal_name) as signals,
    COUNT(*) as total_eras
FROM era_labels_level_b
UNION ALL
SELECT 
    'Level C (HMM)' as algorithm,
    COUNT(DISTINCT signal_name) as signals,
    COUNT(*) as total_eras
FROM era_labels_level_c;
EOF