defmodule WebService.Cache.RedisCache do
  @moduledoc """
  Redis cache service for file operations and metadata.
  Provides caching capabilities for file content and directory listings
  to improve performance of the File Watcher component.
  """
  use GenServer
  require Logger
  
  @default_ttl 3600 # 1 hour default cache TTL
  @reconnect_interval 5000 # 5 seconds between reconnection attempts
  
  # Client API

  @doc """
  Start the Redis cache server
  """
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Get an item from the cache.
  Returns {:ok, value} if found, {:error, :not_found} if not found.
  """
  def get(key) do
    GenServer.call(__MODULE__, {:get, key})
  end

  @doc """
  Store an item in the cache with optional TTL (in seconds).
  Returns :ok on success, {:error, reason} on failure.
  """
  def put(key, value, ttl \\ @default_ttl) do
    GenServer.call(__MODULE__, {:put, key, value, ttl})
  end

  @doc """
  Delete an item from the cache.
  Returns :ok on success, {:error, reason} on failure.
  """
  def delete(key) do
    GenServer.call(__MODULE__, {:delete, key})
  end

  @doc """
  Clear all cached data related to a specific folder.
  This is useful when files are added or removed from a folder.
  """
  def clear_folder_cache(folder) do
    GenServer.call(__MODULE__, {:clear_folder, folder})
  end

  @doc """
  Invalidate the entire cache.
  Use with caution.
  """
  def flush_all do
    GenServer.call(__MODULE__, :flush_all)
  end

  # Server Callbacks

  @impl true
  def init(_opts) do
    # Schedule connection attempt
    schedule_connect()
    {:ok, %{redix_conn: nil, connected: false}}
  end

  @impl true
  def handle_info(:connect, state) do
    # Get Redis URL from environment or use default
    redis_url = System.get_env("REDIS_URL") || "redis://localhost:6379"
    Logger.info("Attempting to connect to Redis at #{redis_url}")
    
    case Redix.start_link(redis_url) do
      {:ok, conn} -> 
        Logger.info("Redis cache connected successfully")
        {:noreply, %{state | redix_conn: conn, connected: true}}
      {:error, reason} ->
        Logger.error("Failed to connect to Redis: #{inspect(reason)}")
        # Schedule reconnection attempt
        schedule_connect()
        {:noreply, %{state | redix_conn: nil, connected: false}}
    end
  end

  defp schedule_connect() do
    Process.send_after(self(), :connect, @reconnect_interval)
  end

  @impl true
  def handle_call(_, _from, %{connected: false} = state) do
    {:reply, {:error, :redis_not_connected}, state}
  end

  @impl true
  def handle_call({:get, key}, _from, state) do
    result = 
      case Redix.command(state.redix_conn, ["GET", cache_key(key)]) do
        {:ok, nil} -> 
          {:error, :not_found}
        {:ok, value} -> 
          {:ok, :erlang.binary_to_term(value)}
        {:error, reason} ->
          Logger.error("Redis cache get error: #{inspect(reason)}")
          handle_connection_error(state)
          {:error, reason}
      end
    {:reply, result, state}
  rescue
    e ->
      Logger.error("Redis cache get exception: #{inspect(e)}")
      {:reply, {:error, :cache_error}, state}
  end

  @impl true
  def handle_call({:put, key, value, ttl}, _from, state) do
    result = 
      try do
        binary_value = :erlang.term_to_binary(value)
        
        case Redix.pipeline(state.redix_conn, [
          ["SET", cache_key(key), binary_value],
          ["EXPIRE", cache_key(key), ttl]
        ]) do
          {:ok, ["OK", 1]} -> 
            :ok
          {:ok, _} -> 
            {:error, :partial_success}
          {:error, reason} -> 
            Logger.error("Redis cache put error: #{inspect(reason)}")
            handle_connection_error(state)
            {:error, reason}
        end
      rescue
        e ->
          Logger.error("Redis cache put exception: #{inspect(e)}")
          {:error, :cache_error}
      end
    
    {:reply, result, state}
  end

  @impl true
  def handle_call({:delete, key}, _from, state) do
    result = 
      try do
        case Redix.command(state.redix_conn, ["DEL", cache_key(key)]) do
          {:ok, 1} -> :ok
          {:ok, 0} -> {:error, :not_found}
          {:error, reason} -> 
            Logger.error("Redis cache delete error: #{inspect(reason)}")
            handle_connection_error(state)
            {:error, reason}
        end
      rescue
        e ->
          Logger.error("Redis cache delete exception: #{inspect(e)}")
          {:error, :cache_error}
      end
    
    {:reply, result, state}
  end

  @impl true
  def handle_call({:clear_folder, folder}, _from, state) do
    result = 
      try do
        pattern = "file_watcher:#{folder}:*"
        
        with {:ok, keys} <- Redix.command(state.redix_conn, ["KEYS", pattern]),
             keys when keys != [] <- keys,
             {:ok, _} <- Redix.command(state.redix_conn, ["DEL" | keys]) do
          :ok
        else
          {:ok, []} -> {:error, :not_found}
          {:error, reason} -> 
            Logger.error("Redis cache clear folder error: #{inspect(reason)}")
            handle_connection_error(state)
            {:error, reason}
        end
      rescue
        e ->
          Logger.error("Redis cache clear folder exception: #{inspect(e)}")
          {:error, :cache_error}
      end
    
    {:reply, result, state}
  end

  @impl true
  def handle_call(:flush_all, _from, state) do
    result = 
      try do
        case Redix.command(state.redix_conn, ["FLUSHDB"]) do
          {:ok, "OK"} -> :ok
          {:error, reason} -> 
            Logger.error("Redis cache flush error: #{inspect(reason)}")
            handle_connection_error(state)
            {:error, reason}
        end
      rescue
        e ->
          Logger.error("Redis cache flush exception: #{inspect(e)}")
          {:error, :cache_error}
      end
    
    {:reply, result, state}
  end

  # Handle connection errors by scheduling a reconnection
  defp handle_connection_error(state) do
    if state.connected do
      # Mark connection as closed and schedule reconnect
      Process.send(self(), :connect, [])
    end
  end

  # Private functions

  # Generate a consistent cache key for the file watcher data
  defp cache_key(key) do
    "file_watcher:#{key}"
  end
end
