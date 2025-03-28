import Config

# Helper function to parse boolean strings
defmodule StringHelper do
  def to_boolean("true"), do: true
  def to_boolean("false"), do: false
  def to_boolean(1), do: true
  def to_boolean(0), do: false
  def to_boolean(nil), do: false
end

# Configure application settings
config :pipeline,
  # Default watch directory for file watcher
  watch_dir: System.get_env("WATCH_DIR", "/app/data"),

  # Default connection pool settings
  redis_host: System.get_env("REDIS_HOST", "localhost"),
  redis_port: String.to_integer(System.get_env("REDIS_PORT", "6379")),
  redis_password: System.get_env("REDIS_PASSWORD", ""),
  redis_database: String.to_integer(System.get_env("REDIS_DATABASE", "0")),
  pool_size: String.to_integer(System.get_env("REDIS_POOL_SIZE", "10")),

  # File processing settings
  producer_timeout_ms: String.to_integer(System.get_env("PRODUCER_TIMEOUT_MS", "3600000")),

  # Log directory settings
  log_dir: System.get_env("LOG_DIR", "/app/results"),
  debug_logs_enabled: StringHelper.to_boolean(System.get_env("DEBUG_LOGS_ENABLED", "true"))

# Configure the Elixir Logger
config :logger,
  level: :info,
  backends: [:console, {LoggerFileBackend, :error_log}, {LoggerFileBackend, :info_log}]

# Configure file logging backends
config :logger, :error_log,
  path: Path.join(Application.get_env(:pipeline, :log_dir, "/app/results"), "error.log"),
  level: :error

config :logger, :info_log,
  path: Path.join(Application.get_env(:pipeline, :log_dir, "/app/results"), "info.log"),
  level: :info

# Import environment specific config
import_config "#{config_env()}.exs"
