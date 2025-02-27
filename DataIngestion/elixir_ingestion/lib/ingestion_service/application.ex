defmodule IngestionService.Application do
  # See https://hexdocs.pm/elixir/Application.html
  # for more information on OTP Applications
  @moduledoc false

  use Application

  @impl true
  def start(_type, _args) do
    # Store application start time for uptime metrics
    Application.put_env(:ingestion_service, :start_time, System.monotonic_time())

    children = [
      # Start the Ecto repository
      IngestionService.Repo,

      # Start Redis connection
      {Redix, {System.get_env("REDIS_URL", "redis://localhost:6379"), [name: :redix]}},

      # Start Finch HTTP client
      {Finch, name: IngestionService.Finch},

      # Start the PubSub system
      {Phoenix.PubSub, name: IngestionService.PubSub},

      # Start Telemetry supervisor
      IngestionService.Telemetry,

      # Start Circuit Breaker
      {IngestionService.Resilience.CircuitBreaker,
       name: IngestionService.Resilience.CircuitBreaker},

      # Start the Dynamic Supervisor for dynamic pipelines
      {DynamicSupervisor, strategy: :one_for_one, name: IngestionService.DynamicSupervisor},

      # Start the Dynamic Pipeline manager
      IngestionService.Pipeline.DynamicPipeline,

      # Start the Metadata Catalog Service
      IngestionService.Metadata.CatalogService,

      # Start the default ingestion pipeline
      IngestionService.Pipeline.Supervisor,

      # Start the endpoint when the application starts
      IngestionServiceWeb.Endpoint
    ]

    # Attach telemetry handlers
    :ok = IngestionService.Telemetry.attach_handlers()

    # See https://hexdocs.pm/elixir/Supervisor.html
    # for other strategies and supported options
    opts = [strategy: :one_for_one, name: IngestionService.Supervisor]
    Supervisor.start_link(children, opts)
  end

  # Tell Phoenix to update the endpoint configuration
  # whenever the application is updated.
  @impl true
  def config_change(changed, _new, removed) do
    IngestionServiceWeb.Endpoint.config_change(changed, removed)
    :ok
  end
end
