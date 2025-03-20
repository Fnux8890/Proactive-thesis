import Config

config :pipeline,
  watch_dir: "data",
  redis_host: "localhost",
  redis_port: 6379

  if Mix.env() == :prod do
    config :pipeline,
      watch_dir: System.get_env("DATA_SOURCE_PATH", "/app/data"),
      redis_url: System.get_env("REDIS_URL", "redis://redis:6379")
  end
