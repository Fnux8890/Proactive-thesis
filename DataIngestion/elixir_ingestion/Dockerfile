# Build stage
FROM elixir:1.18.2-alpine AS builder

WORKDIR /app

# Install dependencies needed for building
RUN apk add --no-cache build-base git

# Install hex and rebar
RUN mix local.hex --force && \
    mix local.rebar --force

# Copy the data processing script
COPY process_data_file.exs .

# Final stage - runtime image
FROM elixir:1.18.2-alpine

WORKDIR /app

# Create results directory
RUN mkdir -p results

# Copy required files from the builder stage
COPY --from=builder /app/process_data_file.exs .

# Set environment variables
ENV MIX_ENV=prod

# Add a health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD elixir -e "System.stop(0)"

# Use ENTRYPOINT as per user requirements
ENTRYPOINT ["elixir", "process_data_file.exs"]
