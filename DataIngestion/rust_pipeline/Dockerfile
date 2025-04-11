# --- Build Stage ---
# Use the official Rust image as the builder
FROM rust:1.78 AS builder

# Set the working directory
WORKDIR /usr/src/app

# Copy Cargo.toml and Cargo.lock from the correct subfolder
COPY data_pipeline/Cargo.toml data_pipeline/Cargo.lock ./

# Build dependencies only (another cache layer)
# Create a dummy main.rs to allow this step
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release
RUN rm -f target/release/deps/data_pipeline* # Adjust dummy artifact name if needed

# Copy the actual source code from the correct subfolder
COPY data_pipeline/src ./src

# Build the application
RUN cargo build --release

# --- Run Stage ---
# Use a minimal image for the final stage
FROM debian:bookworm-slim

# Set the working directory
WORKDIR /usr/local/bin

# Create a directory for configuration
RUN mkdir -p /app/config

# Copy the compiled binary from the builder stage
COPY --from=builder /usr/src/app/target/release/data_pipeline .

# Copy the configuration file from the host (build context) into the image
# Assumes data_files.json is at data_pipeline/data_files.json relative to build context
COPY ./data_pipeline/data_files.json /app/config/data_files.json

# === DEBUG STEP: Verify the copied file ===
RUN ls -l /app/config/
# === END DEBUG STEP ===

# (Optional) Set permissions if needed, though default should be fine
# RUN chown nonroot:nonroot /app/config/data_files.json

# Command to run the application (adjust if needed)
CMD ["./data_pipeline"] 