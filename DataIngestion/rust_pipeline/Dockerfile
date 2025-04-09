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

# Copy the built binary from the builder stage (adjust binary name if needed based on Cargo.toml)
COPY --from=builder /usr/src/app/target/release/data_pipeline .

# Set the entrypoint (adjust binary name if needed)
CMD ["./data_pipeline"] 