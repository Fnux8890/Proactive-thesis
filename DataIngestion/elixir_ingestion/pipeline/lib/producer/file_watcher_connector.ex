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
  @default_poll_interval 3_000

  # Log file path for connector activities
  @connector_log_file "connector_activity.log"

  # Results directory path
  @results_dir "/app/results"

  # Summary file in results directory
  @summary_file_path Path.join(@results_dir, "file_processing_summary.txt")

  # Debug log file in results directory
  @debug_log_file Path.join(@results_dir, "connector_debug.log")

  @processed_files_key "file_watcher:processed_files"
  @reconcile_interval 60_000

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

    # Subscribe to file events
    :ok = FileWatcher.Server.subscribe()

    # Schedule periodic reconciliation
    timer_ref = Process.send_after(self(), :reconcile, @reconcile_interval)

    # Log PID for debugging restart issues
    current_pid = inspect(self())

    # Initialize state
    state = %{
      producer: producer,
      redis_client: redis_client,
      reconcile_timer: timer_ref,
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
  def handle_info(:reconcile, state) do
    # Reconcile with FileWatcher.Server
    case FileWatcher.Server.get_files() do
      {:ok, files} ->
        log_debug("RECONCILE", "Received #{map_size(files)} files from FileWatcher")
        process_reconciliation(files, state)

      {:error, reason} ->
        log_debug("RECONCILE_ERROR", "Failed to get files: #{inspect(reason)}")
        Logger.error("Reconciliation failed to get files from FileWatcher: #{inspect(reason)}")
    end

    # Schedule next reconciliation
    schedule_reconciliation(state.reconcile_timer)
    {:noreply, state}
  end

  @impl true
  def handle_info({:file_update, file_info}, state) do
    # Check if file is already processed using Redis
    file_path = file_info.path

    case is_file_enqueued?(file_path, state.redis_client) do
      true ->
        # File already processed, skip
        log_debug("FILE_SKIP", "File already processed: #{file_path}")
        {:noreply, state}

      false ->
        # Process new file
        log_debug("FILE_NEW", "Processing new file: #{file_path}")

        # Create metadata from file info
        metadata = %{
          size: Map.get(file_info, :size),
          type: Map.get(file_info, :type),
          discovered_at: Map.get(file_info, :mtime)
        }

        # Send to producer
        send(state.producer, {:enqueue_file, file_path, metadata})

        # Mark file as processed in Redis
        case state.redis_client.sadd(@processed_files_key, file_path) do
          {:ok, _} ->
            log_debug("FILE_MARKED", "Marked file as processed in Redis: #{file_path}")

          {:error, reason} ->
            log_debug("REDIS_ERROR", "Failed to mark file in Redis: #{inspect(reason)}")
        end

        {:noreply, state}
    end
  end

  @impl true
  def handle_info({:file_removed, file_path}, state) do
    log_debug("FILE_REMOVED", "File removed: #{file_path}")
    log_activity("File removed: #{file_path}")
    {:noreply, state}
  end

  @impl true
  def handle_info(:check_files, state) do
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

  # Process reconciliation with FileWatcher
  defp process_reconciliation(files, state) do
    Enum.each(files, fn {file_path, file_info} ->
      case is_file_enqueued?(file_path, state.redis_client) do
        false ->
          # File not processed yet, send it to producer
          metadata = %{
            size: Map.get(file_info, "size"),
            type: Map.get(file_info, "type"),
            discovered_at: Map.get(file_info, "mtime")
          }

          send(state.producer, {:enqueue_file, file_path, metadata})

          # Mark as processed
          state.redis_client.sadd(@processed_files_key, file_path)

        true ->
          # Already processed, skip
          :ok
      end
    end)
  end

  # Schedule reconciliation
  defp schedule_reconciliation(interval) do
    Process.send_after(self(), :reconcile, interval)
  end

  # Check for new files from FileWatcher
  defp check_files(state) do
    try do
      # Update last check timestamp
      now = System.system_time(:millisecond)

      log_debug(
        "CHECK_START",
        "Beginning file check, enqueued files count: #{MapSet.size(state.enqueued_files)}"
      )

      # Get files from FileWatcher.Server using its get_files function with retries
      files_map =
        retry_with_backoff(
          fn ->
            try do
              case FileWatcher.Server.get_files() do
                %{} = files when is_map(files) ->
                  log_debug(
                    "FILES_RECEIVED",
                    "Received #{map_size(files)} files from FileWatcher.Server"
                  )

                  files

                other ->
                  log_debug(
                    "UNEXPECTED_RESPONSE",
                    "Unexpected response from FileWatcher.Server: #{inspect(other)}"
                  )

                  %{}
              end
            rescue
              e ->
                log_debug(
                  "WATCHER_ERROR",
                  "Error calling FileWatcher.Server.get_files(): #{inspect(e)}"
                )

                %{}
            end
          end,
          3,
          1000
        )

      log_debug(
        "WATCHER_RESPONSE",
        "Received response from FileWatcher.Server, file count: #{map_size(files_map)}"
      )

      # Log some file paths for debugging
      if map_size(files_map) > 0 do
        sample_paths =
          files_map
          |> Enum.take(5)
          |> Enum.map(fn {path, _} -> path end)
          |> Enum.join(", ")

        log_debug("SAMPLE_FILES", "Sample file paths: #{sample_paths}")
      end

      # Convert files map to list format
      files =
        files_map
        |> Enum.map(fn {path, info} ->
          %{
            path: path,
            discovered_at: Map.get(info, :discovered_at, info.mtime),
            size: info.size,
            type: info.type
          }
        end)

      log_debug("FILES_PARSED", "Converted #{length(files)} files to internal format")

      # Only process CSV and JSON files
      filtered_files = Enum.filter(files, fn file -> file.type in [:csv, :json] end)

      if length(filtered_files) < length(files) do
        log_debug(
          "FILES_FILTERED",
          "Filtered out #{length(files) - length(filtered_files)} non-CSV/JSON files"
        )
      end

      # Process files
      log_debug("PROCESS_START", "Starting to process #{length(filtered_files)} files")
      {enqueued, errors, new_state} = process_files(filtered_files, state)

      log_debug(
        "PROCESS_COMPLETE",
        "Finished processing files, enqueued: #{enqueued}, errors: #{errors}"
      )

      # Update statistics
      new_stats = %{
        files_found: state.stats.files_found + length(filtered_files),
        files_enqueued: state.stats.files_enqueued + enqueued,
        errors: state.stats.errors + errors,
        last_check: now
      }

      # Write status to summary file
      append_to_summary(
        "CONNECTOR",
        "CHECK",
        "Found #{length(filtered_files)} files, enqueued #{enqueued}, errors: #{errors}"
      )

      # Return updated state
      log_debug("CHECK_COMPLETE", "File check completed successfully")
      %{new_state | stats: new_stats}
    rescue
      e ->
        # Log unexpected error
        error_message = "Unexpected error in file check: #{inspect(e)}"
        log_activity(error_message)
        log_debug("CHECK_ERROR", "#{error_message}\nStacktrace: #{inspect(__STACKTRACE__)}")

        # Write error to summary file
        append_to_summary(
          "CONNECTOR",
          "ERROR",
          "Unexpected error during file check",
          "#{inspect(e)}"
        )

        # Update error count
        now = System.system_time(:millisecond)
        new_stats = %{state.stats | errors: state.stats.errors + 1, last_check: now}
        %{state | stats: new_stats}
    end
  end

  # Retry a function with exponential backoff
  defp retry_with_backoff(fun, max_retries, initial_delay) do
    retry_with_backoff(fun, max_retries, initial_delay, 0)
  end

  defp retry_with_backoff(fun, max_retries, initial_delay, attempt) do
    if attempt >= max_retries do
      try do
        fun.()
      rescue
        e ->
          Logger.error("All retry attempts failed: #{inspect(e)}")
          reraise e, __STACKTRACE__
      end
    else
      try do
        fun.()
      rescue
        e ->
          delay = (initial_delay * :math.pow(2, attempt)) |> round()
          Logger.debug("Retry attempt #{attempt + 1} failed, retrying in #{delay}ms")
          Process.sleep(delay)
          retry_with_backoff(fun, max_retries, initial_delay, attempt + 1)
      end
    end
  end

  # Process files and enqueue them if not already enqueued
  defp process_files(files, state) do
    try do
      # Filter out files already enqueued
      new_files =
        Enum.filter(files, fn file ->
          not MapSet.member?(state.enqueued_files, file.path)
        end)

      log_debug(
        "NEW_FILES",
        "Found #{length(new_files)} new files out of #{length(files)} total files"
      )

      # Log how many new files found
      log_activity("Found #{length(new_files)} new files")

      # Enqueue each new file
      log_debug("ENQUEUE_START", "Starting to enqueue #{length(new_files)} files")

      result =
        Enum.reduce(new_files, {0, 0, state.enqueued_files}, fn file,
                                                                {enqueued, errors, enqueued_set} ->
          log_debug("ENQUEUE_FILE", "Attempting to enqueue file: #{file.path}")

          case enqueue_file(file, state.producer) do
            {:ok, file_id} ->
              # File successfully enqueued
              log_debug(
                "ENQUEUE_SUCCESS",
                "Successfully enqueued file: #{file.path} with ID: #{file_id}"
              )

              {enqueued + 1, errors, MapSet.put(enqueued_set, file.path)}

            {:error, reason} ->
              # Error enqueuing file
              log_debug(
                "ENQUEUE_ERROR",
                "Failed to enqueue file: #{file.path}, reason: #{inspect(reason)}"
              )

              {enqueued, errors + 1, enqueued_set}

            _ ->
              # Unexpected response
              log_debug(
                "ENQUEUE_UNEXPECTED",
                "Unexpected response from producer for file: #{file.path}"
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
