FROM elixir:1.16-alpine as builder

# Install build dependencies
RUN apk add --no-cache build-base git

# Set environment variables
ENV MIX_ENV=prod

# Create app directory and copy the Elixir projects into it
WORKDIR /app

# Install hex package manager and rebar
RUN mix local.hex --force && \
    mix local.rebar --force

# Copy configuration files
COPY config ./config
COPY mix.exs .
COPY mix.lock .

# Fetch dependencies
RUN mix deps.get --only prod
RUN mix deps.compile

# Copy lib files
COPY lib ./lib
COPY priv ./priv

# Compile the project
RUN mix compile

# Build release
RUN mix release

# Start a new build stage for the final image
FROM alpine:3.19

# Install runtime dependencies
RUN apk add --no-cache bash openssl libstdc++ ncurses-libs

# Set environment variables
ENV MIX_ENV=prod

WORKDIR /app

# Copy the release from the builder stage
COPY --from=builder /app/_build/prod/rel/ingestion_service ./

# Expose port for the Phoenix API
EXPOSE 4000

# Start the Phoenix server
CMD ["/app/bin/ingestion_service", "start"] 