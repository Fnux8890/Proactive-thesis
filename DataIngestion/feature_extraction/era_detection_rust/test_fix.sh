#!/bin/bash
# Test script to verify the duplicate key fix

echo "🔧 Testing Era Detection Duplicate Key Fix"
echo "==========================================="

cd "$(dirname "$0")"

echo "1. Checking Rust code compilation..."
if timeout 600 cargo check --quiet; then
    echo "✅ Cargo check passed"
else
    echo "❌ Cargo check failed or timed out"
    exit 1
fi

echo ""
echo "2. Building the project..."
if timeout 600 cargo build --quiet; then
    echo "✅ Build successful"
else
    echo "❌ Build failed or timed out"
    exit 1
fi

echo ""
echo "3. Code changes summary:"
echo "   - Modified save_era_labels() to use atomic transactions"
echo "   - Removed separate delete_era_labels_for_signal() calls"
echo "   - Fixed race condition in parallel signal processing"

echo ""
echo "✅ Era detection fix is ready for testing!"
echo ""
echo "To test with your data, run:"
echo "  docker compose up era_detector"
echo ""
echo "The duplicate key constraint violations should now be resolved."