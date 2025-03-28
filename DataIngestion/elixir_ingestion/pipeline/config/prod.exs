import Config

# Production environment configuration
# Note: Most settings come from environment variables

# Configure the logger for production - less verbose
config :logger,
  level: :info,
  backends: [:console, {LoggerFileBackend, :error_log}, {LoggerFileBackend, :info_log}]
