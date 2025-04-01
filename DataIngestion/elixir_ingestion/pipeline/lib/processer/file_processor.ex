defmodule Processor.FileProcessor do
  @moduledoc """
  Dynamically started worker that simulates processing a single file.
  Logs activity and reports completion/failure to the Producer.Dispatcher.
  """
  use GenServer
  require Logger

  # --- Configuration ---
  # Path inside Docker usually
  @results_log_path "/app/results/processor_log.txt"

  # Client API (usually only called by Supervisor)
  def start_link({file_path, slot_id, producer_pid}) do
    # Use GenServer.start_link directly as it's managed by DynamicSupervisor
    GenServer.start_link(__MODULE__, {file_path, slot_id, producer_pid})
  end

  # Server Callbacks
  @impl true
  def init({file_path, slot_id, producer_pid}) do
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
    log_processor_activity(state.file_path, state.slot_id)

    try do
      # Simulate work - replace with actual file reading/processing
      simulate_work(state.file_path)

      Logger.debug("Processor finished work for: #{state.file_path}")
      # Notify producer of success
      send(state.producer_pid, {:processing_complete, self(), state.file_path, state.slot_id})
      # Stop normally
      {:stop, :normal, state}
    rescue
      e ->
        Logger.error("Processor failed for #{state.file_path}. Error: #{inspect(e)}")
        stacktrace = System.stacktrace()
        Logger.debug("Stacktrace: #{inspect(stacktrace)}")
        # Notify producer of failure
        send(state.producer_pid, {:processing_failed, self(), state.file_path, state.slot_id, e})
        # Stop with shutdown reason
        {:stop, {:shutdown, :processing_failed}, state}
    end
  end

  # Catch-all for unexpected messages during processing
  @impl true
  def handle_info(msg, state) do
    Logger.warn("Processor for #{state.file_path} received unexpected message: #{inspect(msg)}")
    {:noreply, state}
  end

  # --- Private Helpers ---

  defp log_processor_activity(file_path, slot_id) do
    timestamp = DateTime.utc_now() |> DateTime.to_iso8601()
    log_entry = "[#{timestamp}] Slot #{slot_id}: Received file #{file_path}\n"

    # Ensure results directory exists
    File.mkdir_p!(Path.dirname(@results_log_path))

    # Append to log file - Use File.write! for simplicity here,
    # consider a dedicated logging process or Logger backend for high throughput.
    case File.write(@results_log_path, log_entry, [:append, :utf8]) do
      :ok -> :ok
      {:error, reason} -> Logger.error("Failed to write to processor log: #{inspect(reason)}")
    end
  end

  defp simulate_work(file_path) do
    # Simulate varying work time based on file path hash
    # Simulate 100-600ms work
    :timer.sleep(:rand.uniform(500) + 100)

    # Simulate occasional failures
    # if :rand.uniform(10) == 1 do
    #  raise "Simulated processing error for #{file_path}"
    # end

    Logger.debug("Simulated work done for #{file_path}")
    :ok
  end
end
