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
  # Default watch directory, can be overridden by env var
  watch_dir: System.get_env("PIPELINE_WATCH_DIR", "/app/data"),

  # Default connection pool settings
  redis_host: System.get_env("REDIS_HOST", "redis"),
  redis_port: String.to_integer(System.get_env("REDIS_PORT", "6379")),
  redis_password: System.get_env("REDIS_PASSWORD", ""),
  redis_pool_size: String.to_integer(System.get_env("REDIS_POOL_SIZE", "5")),
  redis_pool_timeout: String.to_integer(System.get_env("REDIS_POOL_TIMEOUT", "5000")),

  # Default concurrency settings for dispatcher
  dispatcher_concurrency: String.to_integer(System.get_env("DISPATCHER_CONCURRENCY", "4")),

  # Default file processing settings
  file_encoding: System.get_env("FILE_ENCODING", "utf-8"),

  # Application name (used for logging context)
  app_name: :pipeline,

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
