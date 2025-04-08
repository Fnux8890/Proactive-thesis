defmodule Producer.FileWatcherConnector do
  @moduledoc """
  Connects the FileWatcher to the FileQueueProducer.

  This connector:
  - Monitors the FileWatcher for new files
  - Sends discovered files to the FileQueueProducer
  - Tracks which files have been enqueued to avoid duplicates
  - Logs its activities to a text file
  """
  use GenServer
  require Logger

  # Default poll interval in milliseconds
  # @default_poll_interval 3_000

  # Log file path for connector activities
  @connector_log_file "connector_activity.log"

  # Results directory path
  @results_dir "/app/results"

  # Summary file in results directory
  @summary_file_path Path.join(@results_dir, "file_processing_summary.txt")

  # Debug log file in results directory
  @debug_log_file Path.join(@results_dir, "connector_debug.log")

  @processed_files_key "file_watcher:processed_files"

  # Use Dispatcher's keys
  @dispatcher_state_hash "file_processing_state"
  @dispatcher_queue_list "files_to_process"
  @dispatcher_pid Producer.Dispatcher # Assuming registered name

  # Client API

  @doc """
  Starts the file watcher connector.

  ## Parameters
    * opts - Options for the connector
      * :name - Optional name for the connector
      * :poll_interval - Interval to poll for files in ms (default: 3000)
      * :producer - The FileQueueProducer to send files to (required)

  ## Returns
    * {:ok, pid} - PID of the started connector
    * {:error, reason} - If starting the connector failed
  """
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Triggers an immediate check for new files.

  ## Parameters
    * connector - The connector to trigger (PID or name)

  ## Returns
    * :ok - Check was triggered
  """
  def check_for_files(connector \\ __MODULE__) do
    GenServer.cast(connector, :check_files)
  end

  @doc """
  Gets statistics about the connector's operations.

  ## Parameters
    * connector - The connector to query (PID or name)

  ## Returns
    * Map with connector statistics
  """
  def get_stats(connector \\ __MODULE__) do
    GenServer.call(connector, :get_stats)
  end

  @doc """
  Checks if a file has already been enqueued for processing.

  ## Parameters
    * path - The file path to check
    * redis_client - (Optional) The Redis client module to use

  ## Returns
    * true - If the file has been enqueued
    * false - If the file has not been enqueued or an error occurred
  """
  def is_file_enqueued?(path, redis_client \\ nil) do
    client =
      redis_client || Application.get_env(:pipeline, :redis_client, ConnectionHandler.Client)

    case client.sismember(@processed_files_key, path) do
      {:ok, 1} ->
        true

      {:ok, 0} ->
        false

      {:error, reason} ->
        Logger.error("Failed to check if file is enqueued: #{inspect(reason)}")
        false
    end
  end

  # GenServer callbacks

  @impl true
  def init(opts) do
    Logger.info("[Producer.FileWatcherConnector] Initializing with opts: #{inspect(opts)}")
    producer =
      Keyword.get(
        opts,
        :producer,
        Application.get_env(:pipeline, :producer, Producer.FileQueueProducer)
      )

    redis_client =
      Keyword.get(
        opts,
        :redis_client,
        Application.get_env(:pipeline, :redis_client, ConnectionHandler.Client)
      )

    # Log PID for debugging restart issues
    current_pid = inspect(self())

    # Initialize state
    state = %{
      producer: producer,
      redis_client: redis_client,
      enqueued_files: MapSet.new(),
      stats: %{
        files_found: 0,
        files_enqueued: 0,
        errors: 0,
        last_check: nil
      },
      pid: current_pid,
      start_time: System.system_time(:millisecond)
    }

    # Create log file directory if it doesn't exist
    log_dir = Path.dirname(@connector_log_file)
    File.mkdir_p!(log_dir)

    # Create results directory if it doesn't exist
    File.mkdir_p!(@results_dir)

    # Initialize debug log file with header if it doesn't exist
    unless File.exists?(@debug_log_file) do
      debug_header =
        "# FileWatcherConnector Debug Log\n" <>
          "# Created: #{NaiveDateTime.utc_now() |> NaiveDateTime.to_string()}\n" <>
          "# Format: [TIMESTAMP] [PID] [OPERATION] [DETAILS]\n\n"

      File.write!(@debug_log_file, debug_header)
    end

    # Log startup with PID to debug log
    log_debug(
      "STARTUP",
      "FileWatcherConnector started with PID #{current_pid}, opts: #{inspect(opts)}"
    )

    # Initialize summary file with header if it doesn't exist
    unless File.exists?(@summary_file_path) do
      summary_header =
        "# File Processing Summary\n" <>
          "# Created: #{NaiveDateTime.utc_now() |> NaiveDateTime.to_string()}\n" <>
          "# Format: [TIMESTAMP] [FILE_ID] [STATUS] [FILE_PATH] [DETAILS]\n\n"

      File.write!(@summary_file_path, summary_header)
    end

    # Write startup entry to summary file
    append_to_summary(
      "CONNECTOR",
      "STARTUP",
      "FileWatcherConnector started with PID #{current_pid}"
    )

    # Log startup
    log_activity("FileWatcherConnector started with PID #{current_pid}")

    {:ok, state}
  end

  @impl true
  def handle_cast(:check_files, state) do
    Logger.debug("[Producer.FileWatcherConnector] handle_cast(:check_files) called.")
    # Perform file check immediately
    log_debug("CHECK_REQUEST", "Immediate file check requested")
    new_state = check_files(state)
    {:noreply, new_state}
  end

  @impl true
  def handle_call(:get_stats, _from, state) do
    # Return current statistics
    {:reply, state.stats, state}
  end

  @impl true
  def handle_info({:file_update, file_info}, state) do
    Logger.debug("[Producer.FileWatcherConnector] handle_info(:file_update, #{inspect(file_info.path)}) called.")
    file_path = file_info.path
    redis_client = state.redis_client

    # 1. Quick check: Has this connector instance *just* processed this path?
    #    (Avoids rapid duplicate events before Redis state updates)
    case redis_client.sismember(@processed_files_key, file_path) do
      {:ok, 1} ->
        log_debug("FILE_SKIP_LOCAL", "Path recently processed by this connector: #{file_path}")
        {:noreply, state}

      {:ok, 0} ->
        # 2. Check Dispatcher's state: Is it already known (processing, processed, failed)?
        case redis_client.hget(@dispatcher_state_hash, file_path) do
          {:ok, nil} ->
            # File is not known to the dispatcher, proceed to enqueue
            log_debug("FILE_NEW", "Attempting to enqueue new file: #{file_path}")

            # 3. Add to local processed set *before* pushing to prevent race conditions
            case redis_client.sadd(@processed_files_key, file_path) do
              {:ok, 1} ->
                 # 4. Push to Dispatcher's queue list (LPUSH for FIFO)
                case redis_client.lpush(@dispatcher_queue_list, file_path) do
                  {:ok, _length} ->
                    log_debug("FILE_ENQUEUED", "Successfully pushed to Redis list '#{@dispatcher_queue_list}': #{file_path}")
                    # 5. Notify Dispatcher to check for work
                    Producer.Dispatcher.check_work(@dispatcher_pid)
                    # Update stats if needed (omitted for brevity, add back if required)
                    {:noreply, state}

                  {:error, reason} ->
                    log_debug("REDIS_LPUSH_ERROR", "Failed to LPUSH #{file_path}: #{inspect(reason)}")
                    # Remove from local set since enqueue failed
                    redis_client.srem(@processed_files_key, file_path)
                    # Update stats if needed
                    {:noreply, state}
                end # case lpush

              {:ok, 0} ->
                # Should not happen if sismember check worked, but handle defensively
                log_debug("FILE_SKIP_SADD", "File already in local set (race?): #{file_path}")
                {:noreply, state}

              {:error, reason} ->
                 log_debug("REDIS_SADD_ERROR", "Failed to SADD #{file_path}: #{inspect(reason)}")
                 # Update stats if needed
                 {:noreply, state}

            end # case sadd

          {:ok, _status} ->
            # File is known to dispatcher (e.g., "processing", "processed", "failed")
            log_debug("FILE_SKIP_DISPATCHER", "File already known by dispatcher: #{file_path}")
            # Ensure it's in the local set too, in case this connector restarted
            redis_client.sadd(@processed_files_key, file_path)
            {:noreply, state}

          {:error, reason} ->
            log_debug("REDIS_HGET_ERROR", "Failed check dispatcher state for #{file_path}: #{inspect(reason)}")
            {:noreply, state}

        end # case hget

      {:error, reason} ->
        log_debug("REDIS_SISMEMBER_ERROR", "Failed check local set for #{file_path}: #{inspect(reason)}")
        {:noreply, state}
    end # case sismember
  end

  @impl true
  def handle_info({:file_removed, file_path}, state) do
    log_debug("FILE_REMOVED", "File removed: #{file_path}")
    log_activity("File removed: #{file_path}")
    {:noreply, state}
  end

  @impl true
  def handle_info(:check_files, state) do
    Logger.debug("[Producer.FileWatcherConnector] handle_info(:check_files) called.")
    # Calculate uptime in seconds
    uptime_ms = System.system_time(:millisecond) - state.start_time
    uptime_sec = div(uptime_ms, 1000)

    log_debug("SCHEDULED_CHECK", "Periodic file check triggered (uptime: #{uptime_sec}s)")

    # Check for new files
    new_state = check_files(state)

    # Schedule next check
    schedule_check(state.reconcile_timer)

    {:noreply, new_state}
  end

  # Private helper functions

  # Check for new files from FileWatcher
  defp check_files(state) do
    try do
      # Update last check timestamp
      now = System.system_time(:millisecond)

      log_debug(
        "CHECK_START",
        "Beginning file check, enqueued files count: #{MapSet.size(state.enqueued_files)}"
      )

      # Get files from FileWatcher.Server using its get_files function with retries - Removed Block
      # files_map =
      #   retry_with_backoff(
      #     fn ->
      #       try do
      #         case FileWatcher.Server.get_files() do
      #           %{} = files when is_map(files) ->
      #             log_debug(
      #               "FILES_RECEIVED",
      #               "Received #{map_size(files)} files from FileWatcher.Server"
      #             )
      #
      #             files
      #
      #           other ->
      #             log_debug(
      #               "UNEXPECTED_RESPONSE",
      #               "Unexpected response from FileWatcher.Server: #{inspect(other)}"
      #             )
      #
      #             %{}
      #         end
      #       rescue
      #         e ->
      #           log_debug(
      #             "WATCHER_ERROR",
      #             "Error calling FileWatcher.Server.get_files(): #{inspect(e)}"
      #           )
      #
      #           %{}
      #       end
      #     end,
      #     3,
      #     500 # initial backoff 500ms
      #   )
      files_map = %{} # Placeholder - this logic should be driven by Producer.Dispatcher

      # Process the files
      processed_count = process_files(files_map, state)

      # Update statistics
      new_stats = %{
        state.stats
        | files_found: state.stats.files_found + map_size(files_map),
          files_enqueued: state.stats.files_enqueued + processed_count,
          last_check: now
      }

      log_debug(
        "CHECK_END",
        "File check completed. Files found: #{map_size(files_map)}, Enqueued: #{processed_count}"
      )

      %{state | stats: new_stats}
    rescue
      e ->
        Logger.error("Error during file check: #{inspect(e)}")

        log_debug(
          "CHECK_ERROR",
          "Error during file check: #{inspect(e)} Backtrace: #{inspect(__STACKTRACE__)}"
        )

        %{state | stats: %{state.stats | errors: state.stats.errors + 1}}
    end
  end

  # Process files and enqueue them if not already enqueued
  defp process_files(files_map, state) do
    try do
      # Filter out files already enqueued
      new_files =
        Enum.filter(files_map, fn {path, _} ->
          not MapSet.member?(state.enqueued_files, path)
        end)

      log_debug(
        "NEW_FILES",
        "Found #{length(new_files)} new files out of #{length(files_map)} total files"
      )

      # Log how many new files found
      log_activity("Found #{length(new_files)} new files")

      # Enqueue each new file
      log_debug("ENQUEUE_START", "Starting to enqueue #{length(new_files)} files")

      result =
        Enum.reduce(new_files, {0, 0, state.enqueued_files}, fn {path, info}, # Changed _ to info
                                                                {enqueued, errors, enqueued_set} ->
          log_debug("ENQUEUE_FILE", "Attempting to enqueue file: #{path} with info: #{inspect(info)}")
          # Construct the map expected by enqueue_file
          file = Map.put(info, :path, path)

          case enqueue_file(file, state.producer) do # Pass the constructed file map
            {:ok, file_id} ->
              # File successfully enqueued
              log_debug(
                "ENQUEUE_SUCCESS",
                "Successfully enqueued file: #{path} with ID: #{file_id}"
              )

              {enqueued + 1, errors, MapSet.put(enqueued_set, path)}

            {:error, reason} ->
              # Error enqueuing file
              log_debug(
                "ENQUEUE_ERROR",
                "Failed to enqueue file: #{path}, reason: #{inspect(reason)}"
              )

              {enqueued, errors + 1, enqueued_set}

            _ ->
              # Unexpected response
              log_debug(
                "ENQUEUE_UNEXPECTED",
                "Unexpected response from producer for file: #{path}"
              )

              {enqueued, errors + 1, enqueued_set}
          end
        end)
        |> then(fn {enqueued, errors, new_enqueued_set} ->
          # Update state with new enqueued files set
          log_debug(
            "ENQUEUE_COMPLETE",
            "Finished enqueuing files, success: #{enqueued}, errors: #{errors}"
          )

          {enqueued, errors, %{state | enqueued_files: new_enqueued_set}}
        end)

      log_debug("PROCESS_RESULT", "Process files result: #{inspect(result)}")
      result
    rescue
      e ->
        # Log unexpected error
        error_message = "Unexpected error processing files: #{inspect(e)}"
        log_activity(error_message)

        log_debug(
          "PROCESS_ERROR",
          "#{error_message}\nStacktrace: #{inspect(__STACKTRACE__)}"
        )

        # Write error to summary file
        append_to_summary(
          "CONNECTOR",
          "ERROR",
          "Unexpected error processing files",
          "#{inspect(e)}"
        )

        # Return state with error count incremented
        {0, 1, state}
    end
  end

  # Enqueue a file to the producer
  defp enqueue_file(file, producer) do
    try do
      # Create metadata from file info
      metadata = %{
        discovered_at: file.discovered_at,
        size: file.size,
        type: file.type
      }

      log_debug(
        "PRODUCER_CALL",
        "Calling Producer.FileQueueProducer.enqueue_file for: #{file.path}"
      )

      # Enqueue the file
      case Producer.FileQueueProducer.enqueue_file(producer, file.path, metadata) do
        {:ok, file_id} ->
          # Log successful enqueue
          log_activity("Enqueued file to producer: #{file.path} (ID: #{file_id})")

          log_debug(
            "PRODUCER_SUCCESS",
            "Successfully enqueued file to producer: #{file.path} (ID: #{file_id})"
          )

          # Write to summary file
          append_to_summary(
            "CONNECTOR",
            "FORWARDED",
            file.path,
            "Sent to producer with ID: #{file_id}"
          )

          {:ok, file_id}

        {:error, reason} = error ->
          # Log error
          error_message = "Failed to enqueue file #{file.path}: #{inspect(reason)}"
          log_activity(error_message)
          log_debug("PRODUCER_ERROR", error_message)

          # Write error to summary file
          append_to_summary(
            "CONNECTOR",
            "ERROR",
            "Failed to enqueue file: #{file.path}",
            "#{inspect(reason)}"
          )

          error
      end
    rescue
      e ->
        # Log error
        error_message = "Exception enqueuing file #{file.path}: #{inspect(e)}"
        log_activity(error_message)

        log_debug(
          "PRODUCER_EXCEPTION",
          "#{error_message}\nStacktrace: #{inspect(__STACKTRACE__)}"
        )

        # Write error to summary file
        append_to_summary(
          "CONNECTOR",
          "ERROR",
          "Exception enqueuing file: #{file.path}",
          "#{inspect(e)}"
        )

        {:error, e}
    end
  end

  # Schedule next file check
  defp schedule_check(interval) do
    log_debug("SCHEDULE", "Scheduling next check in #{interval}ms")
    Process.send_after(self(), :check_files, interval)
  end

  # Log connector activity to text file
  defp log_activity(message) do
    timestamp = NaiveDateTime.utc_now() |> NaiveDateTime.to_string()
    log_entry = "[#{timestamp}] #{message}\n"

    # Append to log file
    File.write!(@connector_log_file, log_entry, [:append])

    # Also log to console
    Logger.info(message)
  end

  # Log detailed debug information to results directory
  defp log_debug(operation, details) do
    timestamp = NaiveDateTime.utc_now() |> NaiveDateTime.to_string()
    pid = inspect(self())
    log_entry = "[#{timestamp}] [#{pid}] [#{operation}] #{details}\n"

    # Ensure results directory exists
    File.mkdir_p!(@results_dir)

    # Append to debug log file
    File.write!(@debug_log_file, log_entry, [:append])
  end

  # Append entry to summary file in results directory
  defp append_to_summary(component, status, message, details \\ "") do
    timestamp = NaiveDateTime.utc_now() |> NaiveDateTime.to_string()
    details_text = if details == "", do: "", else: " - #{details}"
    summary_entry = "[#{timestamp}] [#{component}] [#{status}] [#{message}]#{details_text}\n"

    # Ensure results directory exists
    File.mkdir_p!(@results_dir)

    # Append to summary file
    File.write!(@summary_file_path, summary_entry, [:append])
  end
end
