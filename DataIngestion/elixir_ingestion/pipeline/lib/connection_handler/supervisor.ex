defmodule ConnectionHandler.Supervisor do
  use Supervisor
  require Logger

  def start_link(opts) do
    Supervisor.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @impl true
  def init(_opts) do
    Logger.info("Starting Connection Handler Supervisor")

    children = [
      {Registry, keys: :unique, name: ConnectionHandler.Registry},
      {ConnectionHandler.PoolSupervisor, []}
    ]

    Supervisor.init(children, strategy: :one_for_one)
  end
end
