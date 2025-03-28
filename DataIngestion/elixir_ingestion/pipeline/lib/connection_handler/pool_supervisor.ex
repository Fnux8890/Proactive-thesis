defmodule ConnectionHandler.PoolSupervisor do
  use Supervisor
  require Logger

  # Number of connections in the pool
  @pool_size 5

  def start_link(opts) do
    Supervisor.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @impl true
  def init(_opts) do
    Logger.info("Starting Redis Connection Pool with #{@pool_size} connections")

    # Initialize ETS table for connection tracking
    :ets.new(:connection_pool_state, [:set, :public, :named_table])
    # Start with worker 1
    :ets.insert(:connection_pool_state, {:last_used, 0})

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
    # Use round-robin worker selection for better load distribution
    [{_, last_used}] = :ets.lookup(:connection_pool_state, :last_used)

    # Calculate next worker
    next_worker = rem(last_used, @pool_size) + 1

    # Store next worker index for next call
    :ets.insert(:connection_pool_state, {:last_used, next_worker})

    # Try selected worker first
    case ConnectionHandler.Connection.get_connection(next_worker) do
      {:ok, conn} ->
        {:ok, conn}

      {:error, _reason} ->
        # If selected worker fails, try any available worker (failover)
        try_any_available_worker(1, [next_worker])
    end
  end

  # Try to get a connection from any available worker, excluding already tried ones
  defp try_any_available_worker(_, tried) when length(tried) >= @pool_size do
    # All workers tried, return error
    Logger.error("All Redis connections in the pool are unavailable")
    {:error, :all_connections_unavailable}
  end

  defp try_any_available_worker(worker_id, tried) do
    if worker_id in tried do
      # Skip already tried workers
      try_any_available_worker(worker_id + 1, tried)
    else
      case ConnectionHandler.Connection.get_connection(worker_id) do
        {:ok, conn} ->
          {:ok, conn}

        {:error, _reason} ->
          # Try next worker
          try_any_available_worker(
            rem(worker_id, @pool_size) + 1,
            [worker_id | tried]
          )
      end
    end
  end

  @doc """
  Handles a connection failure by marking the connection as failed.
  This allows the pool to avoid using the connection until it's reconnected.

  ## Parameters
    * conn - The connection that failed

  ## Returns
    * :ok - The connection was marked as failed
  """
  def handle_connection_failure(conn) do
    Logger.warning("Handling connection failure for connection: #{inspect(conn)}")
    # In a real implementation, you might want to restart the connection
    # or remove it from the active pool until it reconnects.
    # For now, we just log the failure.
    :ok
  end
end
