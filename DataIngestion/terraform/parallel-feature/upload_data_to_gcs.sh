#!/bin/bash
# Script to upload local data files to Google Cloud Storage before deployment

# Check if gcloud is installed
if ! command -v gsutil &> /dev/null; then
    echo "❌ gsutil not found. Please install Google Cloud SDK first."
    exit 1
fi

# Get project ID from terraform.tfvars or prompt
if [ -f "terraform.tfvars" ]; then
    PROJECT_ID=$(grep project_id terraform.tfvars | cut -d'"' -f2)
else
    read -p "Enter your GCP project ID: " PROJECT_ID
fi

BUCKET_NAME="${PROJECT_ID}-feature-extraction-data"
DATA_DIR="../../../Data"

echo "==================================="
echo "Uploading Data to Google Cloud Storage"
echo "==================================="
echo "Project: $PROJECT_ID"
echo "Bucket: gs://$BUCKET_NAME"
echo "Data directory: $DATA_DIR"
echo ""

# Create bucket if it doesn't exist
echo "Creating bucket (if not exists)..."
gsutil mb -p "$PROJECT_ID" "gs://$BUCKET_NAME" 2>/dev/null || echo "Bucket already exists"

# Count files to upload
FILE_COUNT=$(find "$DATA_DIR" -type f \( -name "*.csv" -o -name "*.json" \) | wc -l)
echo "Found $FILE_COUNT data files to upload"
echo ""

# Upload all CSV and JSON files
echo "Uploading data files..."
gsutil -m cp -r "$DATA_DIR"/*.csv "$DATA_DIR"/*.json "gs://$BUCKET_NAME/Data/" 2>/dev/null || true

# Upload nested directories
for dir in "$DATA_DIR"/*/ ; do
    if [ -d "$dir" ]; then
        dirname=$(basename "$dir")
        echo "Uploading $dirname/..."
        gsutil -m cp -r "$dir" "gs://$BUCKET_NAME/Data/" 2>/dev/null || true
    fi
done

echo ""
echo "✅ Data upload complete!"
echo ""

# Show bucket contents
echo "Bucket contents:"
gsutil ls -l "gs://$BUCKET_NAME/Data/" | head -20
echo "..."
echo ""

# Show total size
TOTAL_SIZE=$(gsutil du -s "gs://$BUCKET_NAME" | awk '{print $1}')
echo "Total data size: $(numfmt --to=iec-i --suffix=B $TOTAL_SIZE)"
echo ""
echo "Next steps:"
echo "1. Run: terraform apply"
echo "2. The startup script will download this data to the instance"