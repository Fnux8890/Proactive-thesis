# (Optional: Remove old binary if it exists)
Remove-Item -Path ./data_pipeline_binary -ErrorAction SilentlyContinue

# Build the builder stage with the updated main.rs
docker build --target builder -t rust_pipeline_builder --no-cache .

# Create container and capture ID
$containerId = docker create --name temp_builder rust_pipeline_builder

# Copy the new binary out
docker cp "$($containerId):/usr/src/app/target/release/data_pipeline" ./data_pipeline_binary

# Remove the temp container
docker rm temp_builder