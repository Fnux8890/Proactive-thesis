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
    # Ensure watch_paths is a list
    watch_paths =
      Keyword.get(opts, :watch_paths, Application.get_env(:pipeline, :watch_dir, ["data"]))

    watch_paths = if is_list(watch_paths), do: watch_paths, else: [watch_paths]

    poll_interval = Keyword.get(opts, :poll_interval, @default_poll_interval)
    name = Keyword.get(opts, :name, __MODULE__)

    GenServer.start_link(__MODULE__, [watch_paths: watch_paths, poll_interval: poll_interval],
      name: name
    )
  end

  # Server Callbacks
  @impl true
  def init(opts) do
    watch_paths = Keyword.fetch!(opts, :watch_paths)
    poll_interval = Keyword.fetch!(opts, :poll_interval)

    Logger.info("Starting FileWatcher.Server, watching: #{inspect(watch_paths)}")

    # Schedule the first poll
    schedule_poll(poll_interval)

    state = %{
      watch_paths: watch_paths,
      poll_interval: poll_interval,
      # Will be set by schedule_poll
      poll_timer: nil
    }

    {:ok, state}
  end

  @impl true
  def handle_info(:poll, state) do
    Logger.debug("Polling for new files...")
    scan_and_queue_files(state.watch_paths)
    new_timer_ref = schedule_poll(state.poll_interval)
    {:noreply, %{state | poll_timer: new_timer_ref}}
  end

  # Catch-all for unexpected messages
  @impl true
  def handle_info(msg, state) do
    Logger.warn("Received unexpected message: #{inspect(msg)}")
    {:noreply, state}
  end

  # --- Private Helpers ---

  defp schedule_poll(interval) do
    Process.send_after(self(), :poll, interval)
  end

  defp scan_and_queue_files(watch_paths) do
    new_files_count =
      Enum.reduce(watch_paths, 0, fn path, acc ->
        acc + scan_directory(path)
      end)

    if new_files_count > 0 do
      Logger.info("Found and queued #{new_files_count} new files.")
      # Notify the producer that new work might be available
      # Use cast for fire-and-forget notification
      GenServer.cast(@producer_dispatcher, :check_work)
    else
      Logger.debug("No new files found in this poll.")
    end
  end

  defp scan_directory(dir_path) do
    case File.ls(dir_path) do
      {:ok, entries} ->
        Enum.reduce(entries, 0, fn entry, acc ->
          full_path = Path.join(dir_path, entry)

          if File.regular?(full_path) && allowed_extension?(full_path) do
            acc + maybe_queue_file(full_path)
          else
            # Optionally log skipping directory or non-allowed file type
            acc
          end
        end)

      {:error, reason} ->
        Logger.error("Failed to list directory #{dir_path}: #{inspect(reason)}")
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

      # File exists with state (discovered, processing, processed)
      {:ok, state} ->
        Logger.debug("Skipping known file: #{file_path} with state: #{state}")
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
          Logger.debug("Queued file: #{file_path}")
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
