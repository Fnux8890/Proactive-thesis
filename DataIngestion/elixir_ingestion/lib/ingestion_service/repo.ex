defmodule IngestionService.Repo do
  use Ecto.Repo,
    otp_app: :ingestion_service,
    adapter: Ecto.Adapters.Postgres
    
  # Add hooks for telemetry
  def init(_type, config) do
    # Dynamically configure the repository from environment variables
    database_url = System.get_env("DATABASE_URL")
    config = if database_url do
      Keyword.put(config, :url, database_url)
    else
      config
    end
    
    {:ok, config}
  end
end 