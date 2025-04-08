import Config

# Runtime configuration file.
# Loaded only in releases, allows overriding config with environment variables.

config :pipeline, Pipeline.Supervisor,
  # Example: Override watch_dir using an environment variable
  watch_dir: System.get_env("PIPELINE_WATCH_DIR", "/app/data"),
  redis_host: System.get_env("REDIS_HOST", "redis"), # Use REDIS_HOST if set, else default to 'redis'
  redis_port: String.to_integer(System.get_env("REDIS_PORT", "6379")) # Removed trailing comma
  # Add other runtime configurations here as needed
  # For example, database connection details if they differ at runtime
  # db_hostname: System.get_env("DB_HOST"),
  # db_password: System.get_env("DB_PASSWORD")

# Configure Logger level based on runtime environment if desired
# config :logger, level: String.to_existing_atom(System.get_env("LOG_LEVEL", "info"))
