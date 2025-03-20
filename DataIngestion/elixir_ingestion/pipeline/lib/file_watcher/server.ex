defmodule FileWatcher.Server do
  use GenServer
  require Logger

  # Client API
  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def get_files do
    GenServer.call(__MODULE__, :get_files)
  end

  def get_file_content(file_path) do
    GenServer.call(__MODULE__, {:get_file_content, file_path})
  end

  # Server callbacks
  @impl true
  def init(opts) do
    Logger.info("Starting FileWatcher Server, watching directory: #{opts[:watch_dir]}")

    # Create watch directory if it doesn't exist
    watch_dir = opts[:watch_dir]
    File.mkdir_p!(watch_dir)

    # Initial state (without watcher_pid)
    initial_state = %{
      watch_dir: watch_dir,
      watcher_pid: nil,
      files: %{},
      last_scan: nil
    }

    # Start file system watcher
    case FileSystem.start_link(dirs: [watch_dir]) do
      {:ok, watcher_pid} ->
        # Subscribe to file system events
        FileSystem.subscribe(watcher_pid)

        # Update initial state with watcher_pid
        initial_state = %{initial_state | watcher_pid: watcher_pid}

        # Try to restore state from Redis
        state =
          case FileWatcher.StateStore.load_state() do
            %{} = saved_state when map_size(saved_state) > 0 ->
              Map.merge(initial_state, saved_state)
            _ ->
              initial_state
          end

        # Do initial scan
        state = scan_directory(state)

        # Save initial state
        FileWatcher.StateStore.save_state(%{files: state.files, last_scan: state.last_scan})

        {:ok, state}

      :ignore ->
        # Can't use file system watcher, continue with fallback mode
        Logger.warning("FileSystem watcher not available, using polling mode")

        # Try to restore state without watcher
        state =
          case FileWatcher.StateStore.load_state() do
            %{} = saved_state when map_size(saved_state) > 0 ->
              Map.merge(initial_state, saved_state)
            _ ->
              initial_state
          end

        # Do initial scan anyway
        state = scan_directory(state)

        # Schedule periodic polling instead of events
        Process.send_after(self(), :poll_directory, 30_000)

        {:ok, state}

      {:error, reason} ->
        Logger.error("Failed to start FileSystem watcher: #{inspect(reason)}")
        {:stop, reason}
    end
  end

  @impl true
  def handle_info({:file_event, _pid, {path, events}}, state) do
    Logger.debug("File event: #{path}, events: #{inspect(events)}")

    # Process the file event
    state = process_file_event(state, path, events)

    # Save updated state to Redis
    FileWatcher.StateStore.save_state(%{files: state.files, last_scan: state.last_scan})

    {:noreply, state}
  end

  @impl true
  def handle_info(:poll_directory, state) do
    # Scan directory for changes
    state = scan_directory(state)

    # Schedule next poll
    Process.send_after(self(), :poll_directory, 30_000)

    # Save state to Redis
    FileWatcher.StateStore.save_state(%{files: state.files, last_scan: state.last_scan})

    {:noreply, state}
  end

  @impl true
  def handle_call(:get_files, _from, state) do
    {:reply, state.files, state}
  end

  @impl true
  def handle_call({:get_file_content, file_path}, _from, state) do
    response = case File.read(file_path) do
      {:ok, content} -> {:ok, content}
      {:error, reason} -> {:error, reason}
    end

    {:reply, response, state}
  end

  # Private functions
  defp scan_directory(state) do
    Logger.debug("Scanning directory: #{state.watch_dir}")

    try do
      files = scan_recursively(state.watch_dir, %{})
      %{state | files: files, last_scan: DateTime.utc_now()}
    rescue
      e ->
        Logger.error("Error scanning directory #{state.watch_dir}: #{inspect(e)}")
        state
    end
  end

  defp scan_recursively(dir, acc) do
    File.ls!(dir)
    |> Enum.reduce(acc, fn entry, acc ->
      path = Path.join(dir, entry)

      cond do
        File.regular?(path) ->
          # Process regular file
          stats = File.stat!(path)

          # Convert mtime tuple to ISO string format
          mtime_string = format_datetime(stats.mtime)

          Map.put(acc, path, %{
            name: entry,
            size: stats.size,
            mtime: mtime_string,
            type: detect_file_type(entry)
          })

        File.dir?(path) ->
          # Recursively scan subdirectory
          Logger.debug("Scanning subdirectory: #{path}")
          scan_recursively(path, acc)

        true ->
          # Skip other types (symlinks, etc.)
          acc
      end
    end)
  end

  # Convert Erlang datetime tuple to string
  defp format_datetime({{year, month, day}, {hour, min, sec}}) do
    "#{year}-#{pad(month)}-#{pad(day)}T#{pad(hour)}:#{pad(min)}:#{pad(sec)}Z"
  end

  defp pad(num) when num < 10, do: "0#{num}"
  defp pad(num), do: "#{num}"

  defp process_file_event(state, path, events) do
    # Skip events for non-regular files and files outside watch directory
    if not String.starts_with?(path, state.watch_dir) or not File.regular?(path) do
      state
    else
      cond do
        # File was created or modified
        Enum.any?(events, &(&1 in [:created, :modified, :renamed])) ->
          case File.stat(path) do
            {:ok, stats} ->
              file_name = Path.basename(path)
              # Convert mtime tuple to ISO string here too
              mtime_string = format_datetime(stats.mtime)

              updated_files = Map.put(state.files, path, %{
                name: file_name,
                size: stats.size,
                mtime: mtime_string,
                type: detect_file_type(file_name)
              })
              %{state | files: updated_files, last_scan: DateTime.utc_now()}

            {:error, reason} ->
              Logger.debug("Could not stat file #{path}: #{inspect(reason)}")
              state
          end

        # File was deleted
        :deleted in events ->
          updated_files = Map.delete(state.files, path)
          %{state | files: updated_files, last_scan: DateTime.utc_now()}

        # Other events - no change to state
        true ->
          state
      end
    end
  end

  defp detect_file_type(file_name) do
    extension = Path.extname(file_name) |> String.downcase()

    case extension do
      ".csv" -> :csv
      ".json" -> :json
      ".xml" -> :xml
      ".txt" -> :text
      ".xlsx" -> :excel
      ".xls" -> :excel
      _ -> :unknown
    end
  end
end
