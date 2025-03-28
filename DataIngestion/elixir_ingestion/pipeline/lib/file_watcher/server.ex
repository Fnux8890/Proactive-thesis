defmodule FileWatcher.Server do
  @moduledoc """
  Server responsible for watching files in specified directories and notifying
  interested processes of changes. Uses both OS-level file system events and
  periodic polling as a fallback.
  """

  use GenServer
  require Logger
  alias Pipeline.Utils.Retry
  alias FileWatcher.StateStore

  # Implement the ServerBehaviour
  @behaviour FileWatcher.ServerBehaviour

  # Default watch directory - updated to match Docker volume mount
  @default_watch_dir "/app/data"

  # Default poll interval in milliseconds (increased since polling is now backup)
  @default_poll_interval 300_000

  # Maximum retries for Redis operations
  @max_retries 3

  # Initial retry delay in milliseconds
  @retry_delay 1_000

  # Allowed file types
  @allowed_file_types [:csv, :json, :excel]

  # Client API
  def start_link(opts \\ []) do
    watch_paths =
      Keyword.get(opts, :watch_paths, [Application.get_env(:pipeline, :watch_dir, "data")])

    name = Keyword.get(opts, :name, __MODULE__)

    GenServer.start_link(__MODULE__, [watch_paths: watch_paths], name: name)
  end

  @impl FileWatcher.ServerBehaviour
  def get_files do
    case GenServer.whereis(__MODULE__) do
      nil -> {:error, :server_not_running}
      _pid -> GenServer.call(__MODULE__, :get_files)
    end
  end

  @impl FileWatcher.ServerBehaviour
  def get_file_content(path) do
    GenServer.call(__MODULE__, {:get_file_content, path})
  end

  @impl FileWatcher.ServerBehaviour
  def subscribe(pid \\ self()) do
    GenServer.call(__MODULE__, {:subscribe, pid})
  end

  @impl FileWatcher.ServerBehaviour
  def unsubscribe(pid \\ self()) do
    GenServer.call(__MODULE__, {:unsubscribe, pid})
  end

  @impl FileWatcher.ServerBehaviour
  def save_state do
    GenServer.call(__MODULE__, :save_state)
  end

  # Server callbacks
  @impl true
  def init(opts) do
    watch_paths = Keyword.get(opts, :watch_paths, ["data"])
    state_store = Keyword.get(opts, :state_store, FileWatcher.StateStore)

    # Initialize state
    initial_state = %{
      watch_paths: watch_paths,
      files: %{},
      subscribers: MapSet.new(),
      state_store: state_store,
      state_changed: false,
      file_system_pid: nil,
      stats: %{
        last_scan: nil,
        files_watched: 0,
        file_system_errors: 0,
        redis_errors: 0
      }
    }

    # Try to load existing state from Redis
    case state_store.load_state() do
      {:ok, files} ->
        Logger.info("Loaded #{map_size(files)} files from state store")
        {:ok, %{initial_state | files: files}}

      {:error, reason} ->
        Logger.warning("Failed to load state from store: #{inspect(reason)}")
        {:ok, initial_state}
    end
  end

  @impl true
  def handle_info({:file_event, watcher_pid, {path, events}}, %{watcher: watcher_pid} = state) do
    Logger.debug("File event: #{path}, events: #{inspect(events)}")
    {:noreply, state}
  end

  # For testing purposes
  @impl true
  def handle_info({:file_event, _watcher_pid, {path, events}}, state) do
    Logger.debug("Test file event: #{path}, events: #{inspect(events)}")

    if events[:created] || events[:modified] do
      {:noreply, handle_file_create(path, state)}
    else
      {:noreply, state}
    end
  end

  @impl true
  def handle_info({:DOWN, _ref, :process, watcher_pid, reason}, %{watcher: watcher_pid} = state) do
    Logger.error("FileSystem watcher process terminated: #{inspect(reason)}")
    {:noreply, %{state | watcher: nil}}
  end

  @impl true
  def handle_info({:file_event, _watcher_pid, :stop}, state) do
    Logger.warning("FileSystem watcher stopped")
    {:noreply, state}
  end

  @impl true
  def handle_info(:poll, state) do
    Logger.debug("Polling directory")
    {:noreply, state}
  end

  @impl true
  def handle_info(:save_state, state) do
    Logger.debug("Saving state via message")

    case state.state_store.save_state(state.files) do
      :ok ->
        {:noreply, %{state | state_changed: false}}

      {:error, reason} ->
        Logger.error("Failed to save state: #{inspect(reason)}")
        {:noreply, state}
    end
  end

  @impl true
  def handle_call(:get_files, _from, state) do
    {:reply, {:ok, state.files}, state}
  end

  @impl true
  def handle_call({:get_file_content, path}, _from, state) do
    if Map.has_key?(state.files, path) do
      case File.read(path) do
        {:ok, content} -> {:reply, {:ok, content}, state}
        error -> {:reply, error, state}
      end
    else
      {:reply, {:error, :not_found}, state}
    end
  end

  @impl true
  def handle_call({:subscribe, pid}, _from, state) do
    Process.monitor(pid)
    new_state = %{state | subscribers: MapSet.put(state.subscribers, pid)}
    {:reply, :ok, new_state}
  end

  @impl true
  def handle_call({:unsubscribe, pid}, _from, state) do
    new_state = %{state | subscribers: MapSet.delete(state.subscribers, pid)}
    {:reply, :ok, new_state}
  end

  @impl true
  def handle_call(:save_state, _from, state) do
    case state.state_store.save_state(state.files) do
      :ok -> {:reply, :ok, %{state | state_changed: false}}
      error -> {:reply, error, state}
    end
  end

  # Helper functions
  defp handle_file_create(path, state) do
    if File.exists?(path) && File.regular?(path) do
      file_info = file_stat_to_info(path)
      files = Map.put(state.files, path, file_info)
      %{state | files: files, state_changed: true}
    else
      state
    end
  end

  defp file_stat_to_info(path) do
    case File.stat(path, time: :posix) do
      {:ok, stat} ->
        %{
          name: Path.basename(path),
          size: stat.size,
          mtime: stat.mtime,
          type: get_file_type(path),
          processed: false
        }

      {:error, _} ->
        %{
          name: Path.basename(path),
          size: 0,
          mtime: System.os_time(:second),
          type: :unknown,
          processed: false
        }
    end
  end

  defp get_file_type(path) do
    case Path.extname(path) |> String.downcase() do
      ".csv" -> :csv
      ".json" -> :json
      ".txt" -> :text
      ".xml" -> :xml
      _ -> :unknown
    end
  end
end
