defmodule ConnectionHandler.PoolSupervisor do
  use Supervisor
  require Logger

  @pool_size 5  # Number of connections in the pool

  def start_link(opts) do
    Supervisor.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @impl true
  def init(_opts) do
    Logger.info("Starting Redis Connection Pool with #{@pool_size} connections")

    children =
      for id <- 1..@pool_size do
        Supervisor.child_spec(
          {ConnectionHandler.Connection, [id: id]},
          id: {ConnectionHandler.Connection, id}
        )
      end

    Supervisor.init(children, strategy: :one_for_one)
  end

  def get_connection do
    # Pick a random worker from the pool
    id = :rand.uniform(@pool_size)
    ConnectionHandler.Connection.get_connection(id)
  end
end
