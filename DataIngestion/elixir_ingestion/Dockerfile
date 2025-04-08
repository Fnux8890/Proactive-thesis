# Build stage
FROM elixir:1.18.2-slim AS builder

WORKDIR /app

# Install dependencies needed for building using apt-get
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git ca-certificates

# Copy the data processing script
COPY process_data_file.exs .

# Final stage - runtime image
FROM elixir:1.18.2-slim

WORKDIR /app

# Create results directory
RUN mkdir -p results

# Copy required files from the builder stage
COPY --from=builder /app/process_data_file.exs .

# Set environment variables
ENV MIX_ENV=prod

# No need for specific health check packages, Elixir can handle this
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD elixir -e "System.stop(0)"

# Use ENTRYPOINT as per user requirements
ENTRYPOINT ["elixir", "process_data_file.exs"]
