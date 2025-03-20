defmodule FileWatcher.StateStore do
  use GenServer
  require Logger

  @redis_key_prefix "pipeline:file_watcher:file:"
  @redis_files_set "pipeline:file_watcher:files"
  @redis_metadata_key "pipeline:file_watcher:metadata"
  @max_retries 5
  @retry_delay_ms 1000

  # Client API
  def start_link(_opts) do
    GenServer.start_link(__MODULE__, [], name: __MODULE__)
  end

  def save_state(state) do
    GenServer.cast(__MODULE__, {:save_state, state})
  end

  def load_state do
    GenServer.call(__MODULE__, :load_state)
  end

  # Server callbacks
  @impl true
  def init(_) do
    Logger.info("Starting FileWatcher StateStore")
    {:ok, %{}}
  end

  @impl true
  def handle_cast({:save_state, state}, server_state) do
    # Extract files from state
    files = Map.get(state, :files, %{})
    file_count = map_size(files)
    Logger.debug("Saving #{file_count} files to Redis")

    # Save metadata separately
    metadata = Map.drop(state, [:files])
    save_metadata(metadata)

    # Save each file individually with delay
    files
    |> Enum.each(fn {path, file_info} ->
      save_file(path, file_info)
      # Add larger delay between files
      Process.sleep(100)
    end)

    Logger.debug("Completed saving all files")
    {:noreply, server_state}
  end

  defp save_metadata(metadata) do
    if map_size(metadata) > 0 do
      case ConnectionHandler.Client.set(@redis_metadata_key, Jason.encode!(metadata)) do
        {:ok, _} ->
          Logger.debug("Saved metadata successfully")

        {:error, reason} ->
          Logger.error("Failed to save metadata: #{inspect(reason)}")
      end
    end
  end

  # Instead of direct Redix calls
  defp save_file(path, file_info) do
    file_key = @redis_key_prefix <> path
    encoded_info = Jason.encode!(file_info)

    case ConnectionHandler.Client.set(file_key, encoded_info) do
      {:ok, _} ->
        case ConnectionHandler.Client.sadd(@redis_files_set, path) do
          {:ok, _} ->
            Logger.debug("Saved file: #{path}")
            :ok

          error ->
            error
        end

      error ->
        error
    end
  end

  @impl true
  def handle_call(:load_state, _from, server_state) do
    # First, get all file paths from the set
    case ConnectionHandler.Client.smembers(@redis_files_set) do
      {:ok, paths} ->
        if Enum.empty?(paths) do
          Logger.debug("No existing files found in Redis")

          # Check if we have metadata
          case ConnectionHandler.Client.get(@redis_metadata_key) do
            {:ok, nil} -> {:reply, %{}, server_state}
            {:ok, metadata_json} -> {:reply, Jason.decode!(metadata_json), server_state}
            {:error, _} -> {:reply, %{}, server_state}
          end
        else
          # Load each file individually (could be optimized with pipeline)
          files =
            Enum.reduce(paths, %{}, fn path, acc ->
              case ConnectionHandler.Client.get(@redis_key_prefix <> path) do
                {:ok, nil} -> acc
                {:ok, json_data} -> Map.put(acc, path, Jason.decode!(json_data))
                {:error, _} -> acc
              end
            end)

          # Get metadata
          metadata =
            case ConnectionHandler.Client.get(@redis_metadata_key) do
              {:ok, nil} -> %{}
              {:ok, metadata_json} -> Jason.decode!(metadata_json)
              {:error, _} -> %{}
            end

          # Combine into state
          state = Map.put(metadata, :files, files)
          Logger.debug("Loaded #{map_size(files)} files from Redis")
          {:reply, state, server_state}
        end

      {:error, reason} ->
        Logger.error("Failed to load file paths from Redis: #{inspect(reason)}")
        {:reply, %{}, server_state}
    end
  end
end
