# --- Build Stage ---
# We will skip the separate builder stage for now
# FROM rust:1.78 AS builder
# ... (rest of builder stage commented out or removed)

# --- Run Stage ---
# Use a full Rust image temporarily to include cargo
# FROM debian:bookworm-slim 
FROM rust:1.86

# Set the working directory
WORKDIR /usr/src/app

# Create directories needed
RUN mkdir -p /app/config /app/sql_scripts

# Copy configuration first (less likely to change)
COPY ./data_pipeline/data_files.json /app/config/data_files.json
COPY ./data_pipeline/sql_scripts/ /app/sql_scripts

# Install dependencies for Postgres client and SSL
RUN apt-get update -qq && apt-get install -y --no-install-recommends ca-certificates libpq5 && rm -rf /var/lib/apt/lists/*

# Copy Cargo files 
COPY data_pipeline/Cargo.toml data_pipeline/Cargo.lock ./

# Build dependencies (cache layer)
# Create dummy src to allow this
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release
RUN rm -rf src target/release/deps/data_pipeline* # Clean up dummy files

# Copy the actual source code
COPY data_pipeline/src ./src

# Build the application IN the final stage
RUN cargo build --release

# Verify binary exists
RUN ls -l target/release/data_pipeline

# Add a volume mount point for logs
VOLUME /app/logs

# Set the final command to run the newly built binary
CMD echo "Attempting to execute data_pipeline from final stage build..." && ./target/release/data_pipeline 