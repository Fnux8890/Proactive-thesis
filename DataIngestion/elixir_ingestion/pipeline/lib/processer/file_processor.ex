defmodule Processor.FileProcessor do
  @moduledoc """
  Dynamically started worker that processes a single file (CSV or JSON).
  Parses the file content and reports completion/failure to the Producer.Dispatcher.
  """
  use GenServer
  require Logger
  require NimbleCSV # Ensure NimbleCSV is available

  # --- Configuration ---
  # Path inside Docker usually
  @results_log_path "/app/results/processor_log.txt"
  @redis_parsed_data_queue "parsed_data_queue" # Target queue for parsed data
  @redis_client ConnectionHandler.Client # Assuming Redis client
  @batch_size 100 # Default batch size

  # Client API (usually only called by Supervisor)
  def start_link({file_path, slot_id, producer_pid}) do
    # Use GenServer.start_link directly as it's managed by DynamicSupervisor
    GenServer.start_link(__MODULE__, {file_path, slot_id, producer_pid})
  end

  # Server Callbacks
  @impl true
  def init({file_path, slot_id, producer_pid}) do
    Logger.info("[Processor.FileProcessor] Initializing for #{file_path} in slot #{slot_id}.")
    Logger.debug("Processor #{inspect(self())} starting for file #{file_path} in slot #{slot_id}")

    state = %{
      file_path: file_path,
      slot_id: slot_id,
      producer_pid: producer_pid
    }

    # Start processing immediately after init
    {:ok, state, {:continue, :process_file}}
  end

  @impl true
  def handle_continue(:process_file, state) do
    Logger.debug("[Processor.FileProcessor] handle_continue(:process_file) for #{state.file_path}")
    log_processor_activity(state.file_path, state.slot_id)

    try do
      case process_and_parse_file(state.file_path) do
        {:ok, list_of_maps} ->
          parsed_count = length(list_of_maps)
          Logger.info("Successfully parsed #{parsed_count} records from: #{state.file_path}")

          # Attempt to send parsed data downstream via Redis
          case send_data_to_queue(list_of_maps) do
            :ok ->
              Logger.debug("Successfully sent parsed data to queue '#{@redis_parsed_data_queue}' for #{state.file_path}")
              # Notify producer of success AFTER sending data
              send(state.producer_pid, {:processing_complete, self(), state.file_path, state.slot_id})
              # Stop normally
              {:stop, :normal, state}

            {:error, reason} ->
              Logger.error("Failed to send parsed data to Redis for #{state.file_path}. Reason: #{inspect(reason)}")
              # Notify producer of failure (as data couldn't be passed on)
              send(state.producer_pid, {:processing_failed, self(), state.file_path, state.slot_id, {:redis_send_failed, reason}})
              # Stop with shutdown reason
              {:stop, {:shutdown, {:processing_failed, :redis_send_failed}}, state}
          end

        {:error, reason} ->
          Logger.error("Processor failed parsing #{state.file_path}. Reason: #{inspect(reason)}")
          # Notify producer of failure
          send(state.producer_pid, {:processing_failed, self(), state.file_path, state.slot_id, reason})
          # Stop with shutdown reason
          {:stop, {:shutdown, {:processing_failed, reason}}, state}
      end
    rescue
      e ->
        # Catch unexpected errors during the process/parse phase
        stacktrace = __STACKTRACE__
        reason = {:exception, e}
        Logger.error("Processor encountered exception for #{state.file_path}. Error: #{inspect(e)}")
        Logger.debug("Stacktrace: #{inspect(stacktrace)}")
        # Notify producer of failure
        send(state.producer_pid, {:processing_failed, self(), state.file_path, state.slot_id, reason})
        # Stop with shutdown reason
        {:stop, {:shutdown, {:processing_failed, reason}}, state}
    end
  end

  # Catch-all for unexpected messages during processing
  @impl true
  def handle_info(msg, state) do
    Logger.warning("Processor for #{state.file_path} received unexpected message: #{inspect(msg)}")
    {:noreply, state}
  end

  # --- Private Helpers ---

  defp process_and_parse_file(file_path) do
    case Path.extname(file_path) do
      ".csv" ->
        Logger.debug("Detected CSV file: #{file_path}")
        with {:ok, content} <- File.read(file_path) do
          parse_csv_content(content)
        else
          {:error, reason} -> {:error, {:file_read_error, reason}}
        end

      ".json" ->
        Logger.debug("Detected JSON file: #{file_path}")
        with {:ok, content} <- File.read(file_path) do
          parse_json_content(content)
        else
          {:error, reason} -> {:error, {:file_read_error, reason}}
        end

      ext ->
        Logger.warning("Unsupported file type '#{ext}' for file: #{file_path}")
        {:error, :unsupported_type}
    end
  end

  defp parse_csv_content(content) do
    try do
      lines = String.split(content, "\n", trim: true)

      if Enum.empty?(lines) do
        {:ok, []} # Empty file
      else
        # Get header line and remaining data lines
        [_header_line | data_lines] = lines

        # For now, just parse the data lines, skipping the header
        data_content = Enum.join(data_lines, "\n")

        data_content
        |> NimbleCSV.parse_string(separator: ";", headers: false)
        # Stream mapping/chunking might need adjustment if headers were relied upon
        # |> Stream.map(&convert_row_to_map(&1)) # Still commented out
        |> Stream.chunk_every(@batch_size)
        |> Enum.reduce_while({:ok, []}, fn
             {:error, _reason} = err, _acc -> {:halt, err}
             # Since headers: false, each row is a list of strings
             row_list, {:ok, acc_list} when is_list(row_list) ->
               # We need a way to convert this list to a map later
               # For now, just collect the lists
               {:cont, {:ok, [row_list | acc_list]}}
             _other, acc -> {:cont, acc} # Ignore if not a list
           end)
        |> case do
             {:ok, list_of_lists_reversed} ->
               {:ok, Enum.reverse(list_of_lists_reversed)} # Now it's a list of lists
             {:error, reason} ->
               Logger.error("NimbleCSV parsing failed: #{inspect(reason)}")
               {:error, :csv_parse_error, reason}
           err = {:error, _type, _reason} ->
              # Handle specific NimbleCSV error tuple if needed
              Logger.error("NimbleCSV parsing failed directly: #{inspect(err)}")
              {:error, :csv_parse_error, err}
        end
      end
    rescue
      e in NimbleCSV.ParseError ->
        Logger.error("Caught NimbleCSV.ParseError: #{inspect(e)}")
        {:error, :csv_parse_error, e}
      e -> # Catch other potential errors during processing
        Logger.error("Exception during CSV parsing/processing: #{inspect(e)} \\nStacktrace: #{inspect(__STACKTRACE__)}")
        {:error, :csv_processing_exception, e}
    end
  end

  defp parse_json_content(content) do
    try do
      case Jason.decode(content) do
        {:ok, data} ->
          # Check the type *after* successful decoding
          cond do
            is_list(data) ->
              {:ok, data}
            is_map(data) ->
              # Handle case where JSON is a single object, wrap it in a list
              {:ok, [data]}
            true ->
              Logger.error("Parsed JSON but got unexpected type: #{inspect(data)}")
              {:error, :json_unexpected_type}
          end
        {:error, reason} ->
          {:error, :json_decode_error, reason}
      end
    rescue
      e -> # Catch other potential errors
        {:error, :json_processing_exception, e}
    end
  end

  defp send_data_to_queue(list_of_maps) do
    # Only proceed if there's data to send
    if list_of_maps == [] do
      :ok # Nothing to send for empty files
    else
      try do
        # Serialize the data to JSON
        json_payload = Jason.encode!(list_of_maps)
        # Push to the head of the Redis list (LPUSH) using generic command
        @redis_client.command(["LPUSH", @redis_parsed_data_queue, json_payload])
      rescue
        e in Jason.EncodeError ->
          {:error, {:json_encode_failed, e}}
        e -> # Catch potential Redis communication errors (depends on client library)
          {:error, {:redis_push_failed, e}}
      end
      |> case do
           {:ok, _} -> :ok # Check for {:ok, _} assuming Redis client returns this on success
           # Handle potential client-specific error formats here if needed
           {:error, reason} -> {:error, reason} # Propagate Redis errors
           error = {:error, _} -> error # Propagate Jason errors
           other ->
             # Log unexpected successful return value from Redis client if it's not {:ok, _}
             Logger.warning("Unexpected success value from Redis LPUSH: #{inspect(other)}")
             :ok # Assume success if not an explicit error tuple
         end
    end
  end

  defp log_processor_activity(file_path, slot_id) do
    timestamp = DateTime.utc_now() |> DateTime.to_iso8601()
    log_entry = "[#{timestamp}] Slot #{slot_id}: Processing file #{file_path}\\n"

    # Ensure results directory exists
    File.mkdir_p!(Path.dirname(@results_log_path))

    # Append to log file - Use File.write! for simplicity here,
    # consider a dedicated logging process or Logger backend for high throughput.
    case File.write(@results_log_path, log_entry, [:append, :utf8]) do
      :ok -> :ok
      {:error, reason} -> Logger.error("Failed to write to processor log: #{inspect(reason)}")
    end
  end
end
