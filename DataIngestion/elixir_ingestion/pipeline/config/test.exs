import Config

# Test environment configuration
config :pipeline,
  # Use a test directory for file watching
  watch_dir: System.tmp_dir(),

  # Test Redis settings (use a separate database for tests)
  redis_host: "localhost",
  redis_port: 6379,
  redis_password: "",
  # Use database 15 for tests
  redis_database: 15,
  # Smaller pool for tests
  pool_size: 5,

  # Shorter timeouts for tests
  producer_timeout_ms: 1000,

  # Test log directory
  log_dir: System.tmp_dir(),
  debug_logs_enabled: true

# Configure logger for tests - less verbose
config :logger,
  level: :warning,
  backends: [:console]

# Don't start the application supervision tree automatically
config :pipeline, start_supervision: false

# Inject mocks via configuration for dependency injection
config :pipeline, :redis_client, MockRedisClient
config :pipeline, :state_store, MockStateStore
config :pipeline, :producer, MockProducer
config :pipeline, :file_system, MockFileSystem
