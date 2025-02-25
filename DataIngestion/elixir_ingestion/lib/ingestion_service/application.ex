defmodule IngestionService.Application do
  # See https://hexdocs.pm/elixir/Application.html
  # for more information on OTP Applications
  @moduledoc false

  use Application

  @impl true
  def start(_type, _args) do
    # Store application start time for monitoring
    Application.put_env(:ingestion_service, :started_at, DateTime.utc_now())
    
    children = [
      # Start the Ecto repository
      IngestionService.Repo,
      
      # Start the Redix connection
      {Redix, name: :redix, host: System.get_env("REDIS_HOST", "redis"), port: String.to_integer(System.get_env("REDIS_PORT", "6379"))},
      
      # Start Finch for HTTP requests
      {Finch, name: IngestionService.Finch},
      
      # Start the PubSub system
      {Phoenix.PubSub, name: IngestionService.PubSub},
      
      # Start the Telemetry supervisor
      IngestionService.Telemetry,
      
      # Start the ingestion pipeline supervisor
      IngestionService.Pipeline.Supervisor,
      
      # Start the endpoint when the application starts
      IngestionService.Endpoint
    ]

    # See https://hexdocs.pm/elixir/Supervisor.html
    # for other strategies and supported options
    opts = [strategy: :one_for_one, name: IngestionService.Supervisor]
    
    # Start metrics collection
    :telemetry.attach(
      "ingestion-metrics",
      [:ingestion_service, :pipeline, :process],
      &IngestionService.Telemetry.handle_event/4,
      nil
    )

    Supervisor.start_link(children, opts)
  end

  # Tell Phoenix to update the endpoint configuration
  # whenever the application is updated.
  @impl true
  def config_change(changed, _new, removed) do
    IngestionService.Endpoint.config_change(changed, removed)
    :ok
  end
end 