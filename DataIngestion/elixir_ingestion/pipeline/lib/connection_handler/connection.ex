defmodule ConnectionHandler.Connection do
  use GenServer
  require Logger

  @reconnect_interval 5000
  @default_redis_url "redis://localhost:6379"

  # Client API
  def start_link(opts) do
    id = Keyword.fetch!(opts, :id)
    name = via_tuple(id)
    GenServer.start_link(__MODULE__, [id: id], name: name)
  end

  def get_connection(id) do
    case Registry.lookup(ConnectionHandler.Registry, id) do
      [{pid, _}] -> GenServer.call(pid, :get_connection)
      [] -> {:error, :not_found}
    end
  end

  defp via_tuple(id) do
    {:via, Registry, {ConnectionHandler.Registry, id}}
  end

  # Server callbacks
  @impl true
  def init(opts) do
    id = Keyword.fetch!(opts, :id)
    Logger.info("Starting Redis connection ##{id}")

    # Connect immediately and schedule reconnection if needed
    {:ok, %{id: id, conn: nil}, {:continue, :connect}}
  end

  @impl true
  def handle_continue(:connect, state) do
    case connect() do
      {:ok, conn} ->
        Logger.info("Redis connection ##{state.id} established")
        {:noreply, %{state | conn: conn}}

      {:error, reason} ->
        Logger.error("Failed to connect to Redis: #{inspect(reason)}")
        # Schedule reconnection
        Process.send_after(self(), :reconnect, @reconnect_interval)
        {:noreply, %{state | conn: nil}}
    end
  end

  @impl true
  def handle_call(:get_connection, _from, %{conn: nil} = state) do
    # Try to connect if no connection
    case connect() do
      {:ok, conn} ->
        {:reply, {:ok, conn}, %{state | conn: conn}}

      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call(:get_connection, _from, %{conn: conn} = state) do
    {:reply, {:ok, conn}, state}
  end

  @impl true
  def handle_info(:reconnect, state) do
    case connect() do
      {:ok, conn} ->
        Logger.info("Redis connection ##{state.id} re-established")
        {:noreply, %{state | conn: conn}}

      {:error, reason} ->
        Logger.error("Failed to reconnect to Redis: #{inspect(reason)}")
        # Schedule another reconnection
        Process.send_after(self(), :reconnect, @reconnect_interval)
        {:noreply, state}
    end
  end

  @impl true
  def handle_info({:DOWN, _ref, :process, conn, reason}, %{conn: conn} = state) do
    Logger.error("Redis connection ##{state.id} crashed: #{inspect(reason)}")
    Process.send_after(self(), :reconnect, @reconnect_interval)
    {:noreply, %{state | conn: nil}}
  end

  defp connect do
    redis_url = System.get_env("REDIS_URL") || @default_redis_url
    uri = URI.parse(redis_url)
    host = uri.host
    port = uri.port

    password =
      case uri.userinfo do
        nil -> nil
        userinfo ->
          case String.split(userinfo, ":", parts: 2) do
            [_user, pass] -> to_charlist(pass)
            _ -> nil # No password part found
          end
      end

    Logger.info("Attempting to connect to Redis at #{uri.host}:#{uri.port}")

    # Build connection options
    opts = [
      host: host,
      port: port,
      socket_opts: [keepalive: true],
      backoff_max: 5000,
      timeout: 5000,
      sync_connect: true
    ]

    # Add password if present
    opts = if password, do: Keyword.put(opts, :password, password), else: opts

    case Redix.start_link(opts) do
      {:ok, conn} ->
        Process.monitor(conn)
        # Test connection
        case Redix.command(conn, ["PING"]) do
          {:ok, "PONG"} ->
            Logger.info("Redis connection verified with PING")
            {:ok, conn}
          error ->
            Logger.error("Redis connection test failed: #{inspect(error)}")
            error
        end
      error ->
        Logger.error("Redis connection failed: #{inspect(error)}")
        error
    end
  end
end
