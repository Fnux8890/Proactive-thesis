import Config

# Production environment configuration
# Note: Most settings come from environment variables

# Configure the logger for production - less verbose
config :logger,
  level: :info,
  backends: [:console, {LoggerFileBackend, :error_log}, {LoggerFileBackend, :info_log}]

# Configure your database
#
# The MIX_TEST_PARTITION environment variable can be used
# to provide built-in isolation in CI/ELIXIR_ASSERT_TIMEOUT tests.
# See `mix help test` for more information.
config :logger, :console,
  level: :debug,
  format: "$time $metadata[$level] $message\n",
  metadata: [:request_id]

# ## Using releases (Elixir v1.9+)
#
# If you are doing OTP releases, you need to instruct Phoenix
# to start each relevant endpoint:
#
#     config :pipeline, PipelineWeb.Endpoint,
#       server: true
#
# Then you can assemble a release by calling `mix release`.
# See `mix help release` for more information.
