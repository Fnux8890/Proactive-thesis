#!/bin/bash
# Script to upload data directly to the instance after deployment

# Get instance details
INSTANCE_NAME="greenhouse-production-a2"
ZONE="us-central1-a"
DATA_DIR="../../../Data"

echo "==================================="
echo "Direct Data Upload to Instance"
echo "==================================="

# Check if instance exists
if ! gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" >/dev/null 2>&1; then
    echo "❌ Instance $INSTANCE_NAME not found. Deploy infrastructure first:"
    echo "   terraform apply"
    exit 1
fi

# Get instance IP
INSTANCE_IP=$(gcloud compute instances describe "$INSTANCE_NAME" \
    --zone="$ZONE" \
    --format="get(networkInterfaces[0].accessConfigs[0].natIP)")

echo "Instance: $INSTANCE_NAME"
echo "IP: $INSTANCE_IP"
echo "Data directory: $DATA_DIR"
echo ""

# Create remote data directory
echo "Creating remote directory..."
gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command="sudo mkdir -p /opt/Proactive-thesis/Data && sudo chown -R \$USER:\$USER /opt/Proactive-thesis"

# Count files
FILE_COUNT=$(find "$DATA_DIR" -type f \( -name "*.csv" -o -name "*.json" \) | wc -l)
TOTAL_SIZE=$(du -sh "$DATA_DIR" | cut -f1)

echo "Uploading $FILE_COUNT files ($TOTAL_SIZE)..."
echo "This may take a while depending on your connection speed..."

# Upload using gcloud (more reliable than scp for large transfers)
gcloud compute scp --recurse "$DATA_DIR" "$INSTANCE_NAME":/opt/Proactive-thesis/ --zone="$ZONE"

echo ""
echo "✅ Upload complete!"
echo ""

# Verify upload
echo "Verifying upload..."
gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command="ls -la /opt/Proactive-thesis/Data/ | head -10"

echo ""
echo "Next steps:"
echo "1. SSH to instance: gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo "2. Check pipeline status: docker compose ps"
echo "3. Monitor logs: tail -f /var/log/production-pipeline.log"