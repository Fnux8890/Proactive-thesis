import Config

# Development environment configuration
config :pipeline,
  # Use a development directory for file watching
  watch_dir: System.get_env("WATCH_DIR", "./test/fixtures"),

  # Development Redis settings
  redis_host: System.get_env("REDIS_HOST", "localhost"),
  redis_port: String.to_integer(System.get_env("REDIS_PORT", "6379")),
  redis_password: System.get_env("REDIS_PASSWORD", ""),
  redis_database: String.to_integer(System.get_env("REDIS_DATABASE", "0")),
  pool_size: String.to_integer(System.get_env("REDIS_POOL_SIZE", "10")),

  # Development settings
  producer_timeout_ms: String.to_integer(System.get_env("PRODUCER_TIMEOUT_MS", "3600000")),

  # Log directory settings
  log_dir: System.get_env("LOG_DIR", "./logs"),
  debug_logs_enabled: true

# More verbose logging for development
config :logger,
  level: :debug,
  backends: [
    :console,
    {LoggerFileBackend, :error_log},
    {LoggerFileBackend, :info_log},
    {LoggerFileBackend, :debug_log}
  ]

# Add debug log file
config :logger, :debug_log,
  path: Path.join(Application.get_env(:pipeline, :log_dir, "./logs"), "debug.log"),
  level: :debug
