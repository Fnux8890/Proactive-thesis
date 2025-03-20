defmodule WebService.Application do
  # See https://hexdocs.pm/elixir/Application.html
  # for more information on OTP Applications
  @moduledoc false

  use Application

  @impl true
  def start(_type, _args) do
    # Get port from environment
    _port = String.to_integer(System.get_env("PORT") || "4000")

    # Define children for all environments
    base_children = [
      # Start the Telemetry supervisor
      WebServiceWeb.Telemetry,
      
      # Start the Phoenix PubSub system
      {Phoenix.PubSub, name: WebService.PubSub},
      
      # Start a Registry for file processing statuses
      {Registry, keys: :unique, name: WebService.ProcessingRegistry},
      
      # Start the results monitor
      WebService.ResultsMonitor,
      
      # Start the Phoenix endpoint (this will use the Phoenix router and serve assets)
      {WebServiceWeb.Endpoint, []}
    ]

    # Add optional services based on environment
    children = if System.get_env("DISABLE_REDIS") do
      base_children
    else
      # Include Redis cache if not disabled
      base_children ++ [{WebService.Cache.RedisCache, []}]
    end

    # Define supervision options
    opts = [strategy: :one_for_one, name: WebService.Supervisor]

    # Start the supervision tree
    Supervisor.start_link(children, opts)
  end
end
