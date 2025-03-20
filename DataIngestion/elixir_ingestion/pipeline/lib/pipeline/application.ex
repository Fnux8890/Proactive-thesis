defmodule Pipeline.Application do
  # See https://hexdocs.pm/elixir/Application.html
  # for more information on OTP Applications
  @moduledoc """
  The pipeline application service.

  This is the enry point for the pipeline application which sets up
  the supervision tree and starts all necessary processes.
  """

  use Application

  @impl true
  def start(_type, _args) do
    children = [
      # Just use ConnectionHandler for Redis
      {ConnectionHandler.Supervisor, []},

      Pipeline.Supervisor
    ]

    opts = [strategy: :one_for_one, name: Pipeline.Application.Supervisor]
    Supervisor.start_link(children, opts)
  end
end
