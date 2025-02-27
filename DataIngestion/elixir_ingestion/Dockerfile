FROM elixir:1.16-alpine AS build

# Install build dependencies using apk (Alpine-based)
RUN apk update && apk add --no-cache \
    git \
    build-base \
    nodejs \
    npm \
    bash \
    curl

# Set production environment
ENV MIX_ENV=prod

# Set working directory and cache dependencies by copying mix files first
WORKDIR /app
COPY mix.exs mix.lock ./

# Install Hex, Rebar and production dependencies
RUN mix local.hex --force && mix local.rebar --force
RUN mix deps.get --only prod

# Copy the rest of the project and compile assets
COPY . .
RUN mix deps.compile && mix compile && mix phx.digest

# Build the release
RUN mix release

# -----------------------------------------------------------------------------
FROM alpine:3.17 AS app

# Install runtime dependencies including C++ standard libraries and inotify-tools
RUN apk add --no-cache openssl ncurses-libs libstdc++ libgcc inotify-tools

# Set environment variables
ENV MIX_ENV=prod \
    APP_NAME=ingestion_service \
    PORT=4000

WORKDIR /app

# Copy the release from the build stage
COPY --from=build /app/_build/prod/rel/ingestion_service ./

# Create data directory if it doesn't exist
RUN mkdir -p /app/data

# Create verification directory
RUN mkdir -p /app/verification

# Expose the port used by the Phoenix app
EXPOSE 4000

# Create a simple health-check script and add a HEALTHCHECK instruction
RUN echo '#!/bin/sh\necho "OK"' > health.sh && chmod +x health.sh
HEALTHCHECK CMD ["/app/health.sh"]

# Define default command to start the release
ENTRYPOINT ["/app/bin/ingestion_service", "start"]