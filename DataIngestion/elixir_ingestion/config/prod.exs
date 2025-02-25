import Config

# For production, don't forget to configure the url host to something meaningful,
# Phoenix uses this for generating URLs and maintaining session state
config :ingestion_service, IngestionService.Endpoint,
  url: [host: "localhost", port: 4000],
  cache_static_manifest: "priv/static/cache_manifest.json"

# Configure your database
config :ingestion_service, IngestionService.Repo,
  pool_size: 10

# Do not print debug messages in production
config :logger, level: :info

# Runtime production configuration, including config for releases
config :ingestion_service, IngestionService.Endpoint, server: true 