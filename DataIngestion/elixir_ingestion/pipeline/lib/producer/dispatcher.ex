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
    Logger.debug("[Dispatcher] handle_cast(:check_work) received.")
    new_state = dispatch_pending_work(state)
    {:noreply, new_state}
  end

  # Called by Processor when done
  @impl true
  def handle_info({:processing_complete, pid, file_path, slot_id}, state) do
    Logger.info("File processing completed: #{file_path} by slot #{slot_id} (PID: #{inspect(pid)})")

    case find_by_pid(state.in_progress, pid) do
      {file_path_from_state, %{monitor_ref: ref}} ->
        # Ensure the message matches the tracked processor
        if file_path_from_state == file_path do
          Process.demonitor(ref, [:flush])

          # Record completion metric using record_stage_end
          Pipeline.Tracking.Metrics.record_stage_end(file_path, :dispatcher_completion, :completed)

          # Update Redis state to 'processed'
          case @redis_client.command(["HSET", "file_processing_state", file_path, "processed"]) do
            {:ok, _} ->
              Logger.debug("Successfully updated Redis state to 'processed' for #{file_path}")
            {:error, reason} ->
              Logger.error("Failed to update Redis state to 'processed' for #{file_path}: #{inspect(reason)}")
          end

          # Remove from in_progress and add slot back
          new_state = %{
            state
            | in_progress: Map.delete(state.in_progress, file_path),
              available_slots: MapSet.put(state.available_slots, slot_id)
          }
          # Call dispatch_pending_work directly, as it handles the logic and returns state
          new_state = dispatch_pending_work(new_state)
          {:noreply, new_state}
        else
          Logger.warning("Received :processing_complete for PID #{inspect(pid)} with mismatched file paths: Expected '#{file_path_from_state}', got '#{file_path}'. Ignoring.")
          {:noreply, state}
        end
      nil ->
        Logger.warning("Received :processing_complete from unexpected/untracked processor #{inspect(pid)} for file #{file_path}")
        {:noreply, state}
    end
  end

  # Called by Processor on error
  @impl true
  def handle_info({:processing_failed, processor_pid, file_path, slot_id, reason}, state) do
    Logger.debug("[Dispatcher] Received :processing_failed for PID #{inspect(processor_pid)}, File: #{file_path}, Slot: #{slot_id}")
    Logger.debug("[Dispatcher] State BEFORE failed: #{inspect(state)}")
    # Look up by PID
    case find_by_pid(state.in_progress, processor_pid) do
      {^file_path, %{slot_id: ^slot_id, monitor_ref: ref}} ->
        Logger.error(
          "File processing failed: #{file_path} by slot #{slot_id}. Reason: #{inspect(reason)}"
        )

        Process.demonitor(ref, [:flush])

        # Update Redis state to permanently_failed
        case @redis_client.hset(@redis_state_hash, file_path, "permanently_failed") do
          {:ok, _} ->
            Logger.debug("Successfully updated Redis state to 'permanently_failed' for #{file_path}")
            :ok
          {:error, redis_reason} ->
            Logger.error(
              "Redis HSET error updating #{file_path} to permanently_failed: #{inspect(redis_reason)}"
            )
        end

        # Free up slot and remove from in_progress
        new_in_progress = Map.delete(state.in_progress, file_path)
        new_available_slots = MapSet.put(state.available_slots, slot_id)
        new_state = %{state | in_progress: new_in_progress, available_slots: new_available_slots}

        # Check for more work immediately
        Logger.debug("[Dispatcher] State AFTER failed: #{inspect(new_state)}")
        new_state = dispatch_pending_work(new_state)
        {:noreply, new_state}

      _ -> # PID or slot doesn't match
        Logger.warning(
          "Received :processing_failed from unexpected/untracked processor #{inspect(processor_pid)} for file #{file_path}"
        )

        {:noreply, state}
    end
  end

  # Handle processor crashing
  @impl true
  def handle_info({:DOWN, ref, :process, pid, reason}, state) do
    Logger.debug("[Dispatcher] Received :DOWN for Ref: #{inspect(ref)}, PID: #{inspect(pid)}, Reason: #{inspect(reason)}")
    Logger.debug("[Dispatcher] State BEFORE DOWN: #{inspect(state)}")
    # Find file/slot using the monitor reference
    case find_by_ref(state.in_progress, ref) do
      {file_path, %{slot_id: slot_id}} -> # Found the entry
        # Only mark as failed if the reason is NOT normal shutdown
        if reason not in [:normal, :shutdown] do
          Logger.error(
            "Processor for file #{file_path} (slot #{slot_id}) terminated unexpectedly. Reason: #{inspect(reason)}"
          )
          # Mark as failed in Redis
          case @redis_client.hset(@redis_state_hash, file_path, "failed") do
            {:ok, _} -> :ok
            {:error, redis_reason} ->
              Logger.error(
                "Redis HSET error updating crashed #{file_path} to failed: #{inspect(redis_reason)}"
              )
          end
        else
          # Normal termination, assume handled by :processing_complete or :processing_failed
          Logger.info("Processor for file #{file_path} (slot #{slot_id}) terminated normally. Reason: #{inspect(reason)}")
          # DO NOT mark as failed here
        end

        # Free up slot and remove from in_progress REGARDLESS of reason
        new_in_progress = Map.delete(state.in_progress, file_path)
        new_available_slots = MapSet.put(state.available_slots, slot_id)
        new_state = %{state | in_progress: new_in_progress, available_slots: new_available_slots}

        # Check for more work immediately
        Logger.debug("[Dispatcher] State AFTER DOWN: #{inspect(new_state)}")
        new_state = dispatch_pending_work(new_state)
        {:noreply, new_state}

      nil -> # Monitor ref not found in state
        # Monitor ref not found, maybe already handled?
        Logger.warning("Received :DOWN signal with unknown monitor ref: #{inspect(ref)}")
        {:noreply, state}
    end
  end

  @impl true
  def handle_info(msg, state) do
    Logger.warning("Received unexpected message: #{inspect(msg)}")
    {:noreply, state}
  end

  # --- Private Helpers ---

  # Helper to find file info by PID
  defp find_by_pid(in_progress_map, pid) do
    Enum.find(in_progress_map, fn {_, info} -> info.processor_pid == pid end)
  end

  # Helper to find file info by monitor reference
  defp find_by_ref(in_progress_map, ref) do
    Enum.find(in_progress_map, fn {_, info} -> info.monitor_ref == ref end)
  end

  # Tries to fill available slots with work from Redis queue
  defp dispatch_pending_work(state) do
    # Check if slots are available
    Logger.debug("[Dispatcher] Checking for work. Available slots: #{MapSet.size(state.available_slots)}")
    if MapSet.size(state.available_slots) > 0 do
      # Take one available slot ID (using Enum.random for simplicity)
      slot_id = Enum.random(state.available_slots)
      remaining_slots = MapSet.delete(state.available_slots, slot_id) # Tentatively remove slot

      Logger.debug("[Dispatcher] Attempting RPOP from #{@redis_queue_list} for slot #{slot_id}")
      # Try to get the next item from Redis
      case @redis_client.rpop(@redis_queue_list) do
        # Combined clause for successful RPOP
        {:ok, file_path} when is_binary(file_path) ->
          Logger.debug("[Dispatcher] RPOP result: #{inspect(file_path)}")

          # *** CHECK IF ALREADY IN PROGRESS ***
          if Map.has_key?(state.in_progress, file_path) do
            Logger.warning("[Dispatcher] Attempted to dispatch file '#{file_path}' which is already tracked in 'in_progress'. Skipping dispatch, releasing slot #{slot_id}, and checking for more work.")
            # Put slot back and recursively check work *without* modifying state further for this file
            dispatch_pending_work(%{state | available_slots: MapSet.put(state.available_slots, slot_id)})
          else
            # *** FILE IS NOT IN PROGRESS, PROCEED WITH DISPATCH ***
            Logger.debug("[Dispatcher] Found file in queue: #{file_path}, attempting to dispatch to slot #{slot_id}")
            # Use CORRECT metrics function
            Pipeline.Tracking.Metrics.record_item_received(file_path)
            Logger.debug("[Dispatcher] Incremented :received metric for #{file_path}")

            # Update Redis state to "processing" IMMEDIATELY
            # to prevent race conditions if this process crashes before starting worker
            Logger.debug("[Dispatcher] Attempting HSET state to 'processing' for #{file_path}")
            case @redis_client.hset(@redis_state_hash, file_path, "processing") do
              {:ok, _} ->
                Logger.debug("[Dispatcher] HSET successful. Attempting to start worker for #{file_path}")
                # Start the processor task
                processor_args = {file_path, slot_id, self()}
                # Define the child spec explicitly with restart: :temporary
                child_spec = %{
                  id: Processor.FileProcessor,
                  start: {Processor.FileProcessor, :start_link, [processor_args]},
                  restart: :temporary,
                  type: :worker
                }
                # Use DynamicSupervisor to start the worker with the custom spec
                case DynamicSupervisor.start_child(@processor_supervisor, child_spec) do
                    {:ok, processor_pid} ->
                      Logger.info("[Dispatcher] Dispatched #{file_path} to processor #{inspect processor_pid} in slot #{slot_id}")
                      # Monitor the processor
                      monitor_ref = Process.monitor(processor_pid)
                      # Add to in_progress map
                      new_in_progress = Map.put(state.in_progress, file_path, %{
                        processor_pid: processor_pid,
                        slot_id: slot_id,
                        monitor_ref: monitor_ref
                      })
                      # Update state: slot is now officially occupied, track in_progress
                      new_state = %{state | available_slots: remaining_slots, in_progress: new_in_progress}
                      # Recursively call to try and fill more slots if available
                      dispatch_pending_work(new_state)

                    {:error, reason} ->
                      Logger.error("Failed to start processor for #{file_path}: #{inspect(reason)}")
                      # Failed to start worker. Mark as failed in Redis to avoid potential loops.
                      Logger.error("Marking file '#{file_path}' as failed in Redis due to worker start failure.")
                      @redis_client.hset(@redis_state_hash, file_path, "failed")
                      # Put slot back and try to dispatch other work
                      dispatch_pending_work(%{state | available_slots: MapSet.put(state.available_slots, slot_id)})
                 end
              {:error, redis_reason} ->
                 # Failed to set state to 'processing'
                 Logger.error("Failed HSET 'processing' for #{file_path}: #{inspect(redis_reason)}. Releasing slot #{slot_id}.")
                 # Put slot back and try to dispatch other work
                 dispatch_pending_work(%{state | available_slots: MapSet.put(state.available_slots, slot_id)})
            end
          end # End of if Map.has_key? check

        {:ok, nil} -> # Queue empty
          Logger.debug("[Dispatcher] Work queue '#{@redis_queue_list}' is empty.")
          %{state | available_slots: MapSet.put(state.available_slots, slot_id)}


        {:error, reason} -> # Redis error
          Logger.error("[Dispatcher] Redis RPOP error from '#{@redis_queue_list}': #{inspect(reason)}")
          # Put slot back, return current state, hope it resolves later
          %{state | available_slots: MapSet.put(state.available_slots, slot_id)}
        other -> # Unexpected result from RPOP
          Logger.error("[Dispatcher] Unexpected RPOP result from '#{@redis_queue_list}': #{inspect(other)}")
          %{state | available_slots: MapSet.put(state.available_slots, slot_id)}
      end
    else
      Logger.debug("[Dispatcher] No available processing slots.")
      state # No slots available, return current state
    end
  end
end
