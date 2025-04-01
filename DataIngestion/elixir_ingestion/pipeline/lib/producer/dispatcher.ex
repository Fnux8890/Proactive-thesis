defmodule Producer.Dispatcher do
  @moduledoc """
  Manages a fixed number of concurrent file processors.
  Fetches work from a Redis queue and dispatches it to dynamically started processors.
  Updates file state in Redis.
  """
  use GenServer
  require Logger

  # --- Configuration ---
  @max_concurrent 4
  # HASH: path -> state
  @redis_state_hash "file_processing_state"
  # LIST: [path1, path2, ...]
  @redis_queue_list "files_to_process"
  # --- Dependencies ---
  @redis_client ConnectionHandler.Client
  # Name of the DynamicSupervisor for processors
  @processor_supervisor Processor.Supervisor

  # Client API
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, [], name: name)
  end

  # Ask the dispatcher to check for work (e.g., after a processor finishes)
  def check_work(pid \\ __MODULE__) do
    GenServer.cast(pid, :check_work)
  end

  # Server Callbacks
  @impl true
  def init(_opts) do
    Logger.info("Starting Producer.Dispatcher with #{@max_concurrent} concurrent slots.")

    state = %{
      # %{file_path => %{processor_pid: pid, slot_id: integer, monitor_ref: reference}}
      in_progress: %{},
      # Slots identified by integers 1 to N
      available_slots: MapSet.new(1..@max_concurrent)
    }

    # Immediately check for work on startup
    GenServer.cast(self(), :check_work)
    {:ok, state}
  end

  # Triggered by FileWatcher or self when a slot frees up
  @impl true
  def handle_cast(:check_work, state) do
    new_state = dispatch_pending_work(state)
    {:noreply, new_state}
  end

  # Called by Processor when done
  @impl true
  def handle_info({:processing_complete, processor_pid, file_path, slot_id}, state) do
    case state.in_progress[file_path] do
      %{processor_pid: ^processor_pid} ->
        Logger.info("File processing completed: #{file_path} by slot #{slot_id}")
        # Stop monitoring the processor BEFORE updating state
        ref = state.in_progress[file_path].monitor_ref
        Process.demonitor(ref, [:flush])

        # Update Redis state
        case @redis_client.hset(@redis_state_hash, file_path, "processed") do
          {:ok, _} ->
            :ok

          {:error, reason} ->
            Logger.error(
              "Redis HSET error updating #{file_path} to processed: #{inspect(reason)}"
            )
        end

        # Free up slot and remove from in_progress
        new_in_progress = Map.delete(state.in_progress, file_path)
        new_available_slots = MapSet.put(state.available_slots, slot_id)
        new_state = %{state | in_progress: new_in_progress, available_slots: new_available_slots}

        # Check for more work immediately
        new_state = dispatch_pending_work(new_state)
        {:noreply, new_state}

      _ ->
        # Message from an unexpected processor or for a file not tracked?
        Logger.warn(
          "Received :processing_complete from unexpected processor #{inspect(processor_pid)} for file #{file_path}"
        )

        {:noreply, state}
    end
  end

  # Called by Processor on error
  @impl true
  def handle_info({:processing_failed, processor_pid, file_path, slot_id, reason}, state) do
    case state.in_progress[file_path] do
      %{processor_pid: ^processor_pid} ->
        Logger.error(
          "File processing failed: #{file_path} by slot #{slot_id}. Reason: #{inspect(reason)}"
        )

        ref = state.in_progress[file_path].monitor_ref
        Process.demonitor(ref, [:flush])

        # Update Redis state
        case @redis_client.hset(@redis_state_hash, file_path, "failed") do
          {:ok, _} ->
            :ok

          {:error, redis_reason} ->
            Logger.error(
              "Redis HSET error updating #{file_path} to failed: #{inspect(redis_reason)}"
            )
        end

        # Free up slot and remove from in_progress
        new_in_progress = Map.delete(state.in_progress, file_path)
        new_available_slots = MapSet.put(state.available_slots, slot_id)
        new_state = %{state | in_progress: new_in_progress, available_slots: new_available_slots}

        # Check for more work immediately
        new_state = dispatch_pending_work(new_state)
        {:noreply, new_state}

      _ ->
        Logger.warn(
          "Received :processing_failed from unexpected processor #{inspect(processor_pid)} for file #{file_path}"
        )

        {:noreply, state}
    end
  end

  # Handle processor crashing
  @impl true
  def handle_info({:DOWN, ref, :process, _pid, reason}, state) do
    # Find which file/slot this monitor reference belongs to
    case Enum.find(state.in_progress, fn {_, v} -> v.monitor_ref == ref end) do
      {file_path, %{slot_id: slot_id}} ->
        Logger.error(
          "Processor for file #{file_path} (slot #{slot_id}) terminated unexpectedly. Reason: #{inspect(reason)}"
        )

        # Mark as failed in Redis
        case @redis_client.hset(@redis_state_hash, file_path, "failed") do
          {:ok, _} ->
            :ok

          {:error, redis_reason} ->
            Logger.error(
              "Redis HSET error updating crashed #{file_path} to failed: #{inspect(redis_reason)}"
            )
        end

        # Free up slot and remove from in_progress
        new_in_progress = Map.delete(state.in_progress, file_path)
        new_available_slots = MapSet.put(state.available_slots, slot_id)
        new_state = %{state | in_progress: new_in_progress, available_slots: new_available_slots}

        # Check for more work immediately
        new_state = dispatch_pending_work(new_state)
        {:noreply, new_state}

      nil ->
        # Monitor ref not found, maybe already handled?
        Logger.warn("Received :DOWN signal with unknown monitor ref: #{inspect(ref)}")
        {:noreply, state}
    end
  end

  @impl true
  def handle_info(msg, state) do
    Logger.warn("Received unexpected message: #{inspect(msg)}")
    {:noreply, state}
  end

  # --- Private Helpers ---

  # Tries to fill available slots with work from Redis queue
  defp dispatch_pending_work(state) do
    # How many can we potentially start?
    num_to_try = MapSet.size(state.available_slots)

    if num_to_try > 0 do
      Logger.debug("Checking for work to fill #{num_to_try} available slots...")
      # Recursively try to fetch and dispatch work
      do_dispatch_work(state, num_to_try)
    else
      Logger.debug("No available slots, not checking for work.")
      state
    end
  end

  # Recursive helper to attempt dispatching
  # No more attempts needed
  defp do_dispatch_work(state, 0), do: state

  defp do_dispatch_work(state, attempts_left) do
    # Try to get one file path from the right end of the list (FIFO if LPUSHed)
    case @redis_client.rpop(@redis_queue_list) do
      # Queue is empty
      {:ok, nil} ->
        Logger.debug("Redis queue '#{@redis_queue_list}' is empty.")
        # Stop trying
        state

      # Got a file path
      {:ok, file_path} ->
        Logger.debug("Dequeued file path: #{file_path}")
        # Mark as processing *before* starting the processor
        case @redis_client.hset(@redis_state_hash, file_path, "processing") do
          {:ok, _} ->
            # Get an available slot ID
            {slot_id, remaining_slots} = MapSet.pop(state.available_slots)

            # Start the processor
            child_spec = {Processor.FileProcessor, {file_path, slot_id, self()}}

            case DynamicSupervisor.start_child(@processor_supervisor, child_spec) do
              {:ok, processor_pid} ->
                Logger.info(
                  "Dispatched file #{file_path} to processor #{inspect(processor_pid)} in slot #{slot_id}"
                )

                # Monitor the processor
                monitor_ref = Process.monitor(processor_pid)
                # Update state
                new_in_progress =
                  Map.put(state.in_progress, file_path, %{
                    processor_pid: processor_pid,
                    slot_id: slot_id,
                    monitor_ref: monitor_ref
                  })

                new_state = %{
                  state
                  | in_progress: new_in_progress,
                    available_slots: remaining_slots
                }

                # Recurse to try filling more slots
                do_dispatch_work(new_state, attempts_left - 1)

              {:error, reason} ->
                Logger.error("Failed to start processor for #{file_path}: #{inspect(reason)}")
                # Failed to start processor, put the file back? Or mark failed?
                # For simplicity, mark failed. Could implement retry or requeue.
                @redis_client.hset(@redis_state_hash, file_path, "failed")
                # Don't change available_slots, try next attempt
                do_dispatch_work(state, attempts_left - 1)
            end

          {:error, reason} ->
            Logger.error(
              "Redis HSET error marking #{file_path} as processing: #{inspect(reason)}"
            )

            # Don't proceed with this file, try the next attempt
            do_dispatch_work(state, attempts_left - 1)
        end

      # Error popping from Redis queue
      {:error, reason} ->
        Logger.error("Redis RPOP error on queue '#{@redis_queue_list}': #{inspect(reason)}")
        # Stop trying on Redis error
        state
    end
  end
end
