import Config

# Configure the endpoint for test mode
config :web_service, WebServiceWeb.Endpoint,
  http: [ip: {127, 0, 0, 1}, port: 4002],
  secret_key_base: "3BYKBdPJHw7fVDiQyvSAuE91yNdV5W/lH2WqXBxjyFc5/0uuqkGRATlfp4qbTKWe",
  server: false

# Print only warnings and errors during test
config :logger, level: :warning

# Initialize plugs at runtime for faster test compilation
config :phoenix, :plug_init_mode, :runtime
