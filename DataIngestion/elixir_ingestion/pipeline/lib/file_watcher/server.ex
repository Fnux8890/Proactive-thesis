defmodule FileWatcher.Server do
  @moduledoc """
  Scans watch directories, identifies new files based on Redis state,
  and queues them for processing via Redis.
  """
  use GenServer
  require Logger

  # --- Configuration ---
  # Check every 5 seconds
  @default_poll_interval 5_000
  # HASH: path -> "discovered" | "processing" | "processed" | "failed"
  @redis_state_hash "file_processing_state"
  # LIST: [path1, path2, ...]
  @redis_queue_list "files_to_process"
  @allowed_extensions [".csv", ".json"]
  # --- Dependencies ---
  # Make these configurable via opts or Application config
  @redis_client ConnectionHandler.Client
  # Name of the producer to notify
  @producer_dispatcher Producer.Dispatcher

  # Client API
  def start_link(opts \\ []) do
    # Get config directly from Application env, ignore opts for these
    # FORCE the watch path for debugging
    watch_paths = ["/app/data"]
    Logger.warning("[FileWatcher.Server] Forcing watch_paths to: #{inspect(watch_paths)}")

    poll_interval = Application.get_env(:pipeline, :poll_interval, @default_poll_interval)

    name = Keyword.get(opts, :name, __MODULE__)
    # Ensure both watch_paths and poll_interval are passed to init/1
    init_args = [watch_paths: watch_paths, poll_interval: poll_interval]
    GenServer.start_link(__MODULE__, init_args, name: name)
  end

  # Server Callbacks
  @impl true
  def init(opts) do
    # ---> Fetch args but don't use them yet <---
    watch_paths = Keyword.fetch!(opts, :watch_paths)
    poll_interval = Keyword.fetch!(opts, :poll_interval)

    Logger.info("Starting FileWatcher.Server (fetch args, return empty state test)... watching: #{inspect(watch_paths)}")

    # Schedule the first scan/poll with a small delay
    # to allow other processes (like Dispatcher) to initialize.
    Logger.info("Scheduling initial scan/poll in 100ms...")
    Process.send_after(self(), :poll, 100)

    # Restore original state map return
    state = %{
      watch_paths: watch_paths,
      poll_interval: poll_interval,
      poll_timer: nil # Keep track of timer if needed for cancellation
    }

    Logger.debug("FileWatcher.Server init: State map created, about to return {:ok, state}")
    {:ok, state}
  end

  @impl true
  def handle_info(:poll, state) do
    # Restore original logic
    Logger.debug("Polling for new files...")
    scan_and_queue_files(state.watch_paths)
    # Schedule the *next* poll using the configured interval
    new_timer_ref = schedule_poll(state.poll_interval)
    {:noreply, %{state | poll_timer: new_timer_ref}}
  end

  # Catch-all for unexpected messages
  @impl true
  def handle_info(msg, state) do
    Logger.warning("Received unexpected message: #{inspect(msg)}")
    {:noreply, state}
  end

  # --- Private Helpers ---

  defp schedule_poll(interval) do
    Logger.debug("Scheduling next poll in #{interval}ms")
    Process.send_after(self(), :poll, interval)
  end

  defp scan_and_queue_files(watch_paths) do
    Logger.debug("Starting scan for watch paths: #{inspect(watch_paths)}")
    new_files_count =
      Enum.reduce(watch_paths, 0, fn path, acc ->
        acc + find_and_queue_files_recursive(path)
      end)

    Logger.debug("[FileWatcher] Finished scan. new_files_count = #{new_files_count}")

    if new_files_count > 0 do
      Logger.info("Scan complete. Found and queued #{new_files_count} new files.")
      Logger.debug("[FileWatcher] Attempting to cast :check_work to #{@producer_dispatcher}")
      GenServer.cast(@producer_dispatcher, :check_work)
    else
      Logger.debug("Scan complete. No new files found to queue.")
    end
  end

  # Recursive helper to find and queue files
  defp find_and_queue_files_recursive(path) do
    case File.ls(path) do
      {:ok, entries} ->
        Enum.reduce(entries, 0, fn entry, acc ->
          full_path = Path.join(path, entry)
          Logger.debug("[FileWatcher] Processing entry: #{full_path}")
          cond do
            File.regular?(full_path) and allowed_extension?(full_path) ->
              Logger.debug("[FileWatcher] Attempting maybe_queue_file for: #{full_path}")
              queued_count = maybe_queue_file(full_path)
              Logger.debug("[FileWatcher] maybe_queue_file result for #{full_path}: #{queued_count}")
              acc + queued_count
            File.dir?(full_path) ->
              Logger.debug("[FileWatcher] Recursing into directory: #{full_path}")
              acc + find_and_queue_files_recursive(full_path)
            true ->
              # Skip other types (symlinks, etc.) or log if needed
              Logger.debug("Skipping non-regular/non-directory entry: #{full_path}")
              acc
          end
        end)

      {:error, :enoent} ->
        Logger.error("Watch directory does not exist: #{path}")
        0

      {:error, reason} ->
        Logger.error("Failed to list directory #{path}: #{inspect(reason)}")
        0
    end
  end

  defp allowed_extension?(path) do
    Path.extname(path) in @allowed_extensions
  end

  # Checks Redis and queues the file if it's new or failed previously
  defp maybe_queue_file(file_path) do
    case @redis_client.hget(@redis_state_hash, file_path) do
      # File is completely new
      {:ok, nil} ->
        queue_file(file_path)

      # File failed before, retry
      {:ok, "failed"} ->
        Logger.info("Re-queueing previously failed file: #{file_path}")
        queue_file(file_path)

      # File failed permanently, do not retry
      {:ok, "permanently_failed"} ->
        Logger.warning("Skipping re-queue for permanently failed file: #{file_path}")
        0

      # File exists with other state (discovered, processing, processed)
      {:ok, _state} ->
        # Not a new file to queue
        0

      {:error, reason} ->
        Logger.error("Redis HGET error for #{file_path}: #{inspect(reason)}")
        # Avoid queueing on error
        0
    end
  end

  # Adds file to Redis queue and updates its state to "discovered"
  defp queue_file(file_path) do
    # Pipeline: 1. Set state to discovered, 2. Push to queue
    commands = [
      ["HSET", @redis_state_hash, file_path, "discovered"],
      ["LPUSH", @redis_queue_list, file_path]
    ]

    case @redis_client.pipeline(commands) do
      {:ok, [hset_result, _lpush_result]} ->
        # We check HSET result: 1 means new field, 0 means updated field
        # Both are acceptable here.
        if hset_result == 0 || hset_result == 1 do
          # Indicate one file was queued
          1
        else
          Logger.error(
            "Redis pipeline failed to HSET/LPUSH for #{file_path}. HSET result: #{inspect(hset_result)}"
          )

          # Consider attempting to revert or cleanup state if possible
          0
        end

      {:error, reason} ->
        Logger.error("Redis pipeline error queueing #{file_path}: #{inspect(reason)}")
        # Failed to queue
        0
    end
  end
end
