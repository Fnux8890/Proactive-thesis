defmodule Producer.FileQueueProducer do
  @moduledoc """
  Producer for files that manages a queue to control processing rate.

  This producer:
  - Maintains a queue of files to be processed
  - Uses GenStage backpressure to control processing rate
  - Uses UUIDs for file identification
  - Implements timeout handling for stuck files
  """
  use GenStage
  require Logger

  # 1 hour default timeout
  @producer_timeout_ms 3_600_000

  @doc """
  Starts a new FileQueueProducer process.

  ## Options
    * `:name` - The name to register the producer process (default: __MODULE__)
    * `:queue_check_interval` - Milliseconds between queue checks (default: 5000)
  """
  def start_link(opts) do
    name = Keyword.get(opts, :name, __MODULE__)
    queue_check_interval = Keyword.get(opts, :queue_check_interval, 5_000)
    GenStage.start_link(__MODULE__, [queue_check_interval: queue_check_interval], name: name)
  end

  @doc """
  Enqueues a file for processing.

  ## Parameters
    * `producer` - The producer to enqueue to (PID or name)
    * `file_path` - Path to the file to be processed
    * `metadata` - Optional metadata for the file

  ## Returns
    * `{:ok, file_id}` - ID assigned to the file
    * `{:error, reason}` - If enqueuing failed
  """
  def enqueue_file(producer \\ __MODULE__, file_path, metadata \\ %{}) do
    GenStage.call(producer, {:enqueue_file, file_path, metadata})
  end

  @doc """
  Gets the current queue status.

  ## Parameters
    * `producer` - The producer to query (PID or name)

  ## Returns
    * Map with queue statistics
  """
  def queue_status(producer \\ __MODULE__) do
    GenStage.call(producer, :queue_status)
  end

  @impl true
  def init(opts) do
    # Get config from options or map
    # Support both keyword list and map for options
    opts = if is_list(opts), do: opts, else: Map.to_list(opts)

    queue_check_interval = Keyword.get(opts, :queue_check_interval, 5_000)
    timeout_ms = Application.get_env(:pipeline, :producer_timeout_ms, @producer_timeout_ms)
    name = Keyword.get(opts, :name, __MODULE__)
    buffer_size = Keyword.get(opts, :buffer_size, 20)
    file_timeout = Keyword.get(opts, :file_timeout, timeout_ms)
    dispatch_method = Keyword.get(opts, :dispatch_method, :push)

    # Create initial state
    state = %{
      events: [],
      buffer_size: buffer_size,
      dispatch_method: dispatch_method,
      file_timeout: file_timeout,
      name: name,
      demand: 0,
      in_progress_files: %{}
    }

    Logger.info("FileQueueProducer initialized with options: #{inspect(opts)}")

    # Schedule periodic checks for queue status and timeouts
    if queue_check_interval > 0 do
      schedule_queue_check(queue_check_interval)
      schedule_timeout_check(file_timeout)
    end

    {:producer, state}
  end

  @impl true
  def handle_demand(demand, state) do
    # Add the demand to existing demand
    total_demand = state.demand + demand
    Logger.debug("Received demand for #{demand} event(s), total demand: #{total_demand}")

    # Dispatch events to meet demand if possible
    {events_to_dispatch, remaining_events} =
      if length(state.events) > 0 do
        events_to_send = Enum.take(state.events, total_demand)
        remaining = Enum.drop(state.events, total_demand)
        {events_to_send, remaining}
      else
        {[], state.events}
      end

    dispatched_count = length(events_to_dispatch)
    remaining_demand = total_demand - dispatched_count

    # Update state
    new_state = %{state | events: remaining_events, demand: remaining_demand}

    if dispatched_count > 0 do
      Logger.debug("Dispatched #{dispatched_count} event(s)")
    end

    {:noreply, events_to_dispatch, new_state}
  end

  @impl true
  def handle_call({:enqueue_file, file_path, metadata}, _from, state) do
    # Ensure the file exists
    if File.exists?(file_path) do
      # Generate a UUID for the file
      file_id = UUID.uuid4()

      # Create the file item with the UUID
      file_item = %{
        id: file_id,
        path: file_path,
        metadata: metadata,
        enqueued_at: DateTime.utc_now()
      }

      # Check if we've reached the maximum queue size
      queue_size = length(state.events)

      if queue_size >= state.buffer_size do
        Logger.warning(
          "Queue limit (#{state.buffer_size}) reached. File #{file_path} not enqueued."
        )

        {:reply, {:error, :queue_full}, state}
      else
        # Add to the queue
        new_events = [file_item | state.events]

        Logger.info("Enqueued file #{file_id}: #{file_path}")
        Logger.debug("File metadata: #{inspect(metadata)}")

        # Dispatch if there's pending demand
        if state.demand > 0 do
          {:reply, {:ok, file_id}, elem(dispatch_files(%{state | events: new_events}), 1)}
        else
          {:reply, {:ok, file_id}, %{state | events: new_events}}
        end
      end
    else
      Logger.warning("File not found: #{file_path}")
      {:reply, {:error, :file_not_found}, state}
    end
  end

  @impl true
  def handle_call(:queue_status, _from, state) do
    status = %{
      queue_size: length(state.events),
      pending_demand: state.demand,
      in_progress: map_size(state.in_progress_files),
      max_concurrent: 5
    }

    {:reply, status, state}
  end

  @impl true
  def handle_call({:file_complete, file_id}, _from, state) do
    # Check if file is in progress
    if Map.has_key?(state.in_progress_files, file_id) do
      # Remove from in-progress tracking
      new_in_progress = Map.delete(state.in_progress_files, file_id)
      new_state = %{state | in_progress_files: new_in_progress}

      Logger.info("File #{file_id} processing completed")
      {:reply, :ok, state}
    else
      # File wasn't found in our tracking
      Logger.warning("Cannot complete unknown file ID: #{file_id}")
      {:reply, {:error, :unknown_file_id}, state}
    end
  end

  @impl true
  def handle_call(:queue_state, _from, state) do
    # Build queue state information
    queue_state = %{
      queue_size: length(state.events),
      in_progress: map_size(state.in_progress_files),
      pending_demand: state.demand,
      events: state.events,
      in_progress_files: state.in_progress_files
    }

    {:reply, queue_state, state}
  end

  @impl true
  def handle_info({:file_processed, file_id}, state) do
    if Map.has_key?(state.in_progress_files, file_id) do
      # Remove from in-progress set
      new_in_progress = Map.delete(state.in_progress_files, file_id)
      new_state = %{state | in_progress_files: new_in_progress}

      Logger.info("File #{file_id} processing completed")

      # Try to dispatch more files if there's demand
      dispatch_files(new_state)
    else
      # File wasn't in our in-progress map, ignore
      {:noreply, [], state}
    end
  end

  @impl true
  def handle_info(:check_queue, state) do
    # Check if there's demand and dispatch files if needed
    Logger.debug("Checking queue: demand=#{state.demand}, queue size=#{length(state.events)}")

    result =
      if state.demand > 0 do
        dispatch_files(state)
      else
        {:noreply, [], state}
      end

    # Re-schedule the check
    schedule_queue_check(5_000)

    result
  end

  @impl true
  def handle_info(:check_timeouts, state) do
    timeout_ms = Application.get_env(:pipeline, :producer_timeout_ms, @producer_timeout_ms)
    now = System.system_time(:millisecond)

    # Find files that have timed out
    {timed_out, still_active} =
      Enum.split_with(state.in_progress_files, fn {_file_id, file_info} ->
        timestamp = if is_map(file_info), do: file_info.enqueued_at, else: file_info
        now - timestamp > timeout_ms
      end)

    # Construct a new in_progress map
    new_in_progress = Map.new(still_active)

    new_state = %{state | in_progress_files: new_in_progress}

    if length(timed_out) > 0 do
      timed_out_ids = Enum.map(timed_out, fn {file_id, _} -> file_id end)

      Logger.warning(
        "Producer timeout: Files #{inspect(timed_out_ids)} exceeded #{timeout_ms}ms processing limit."
      )
    end

    # Reschedule the next timeout check
    schedule_timeout_check(timeout_ms)

    # Try to dispatch more files if there's demand
    if state.demand > 0 do
      dispatch_files(new_state)
    else
      {:noreply, [], new_state}
    end
  end

  @impl true
  def handle_info(msg, state) do
    Logger.warning("FileQueueProducer received unexpected message: #{inspect(msg)}")
    {:noreply, [], state}
  end

  # Helper function to dispatch files
  defp dispatch_files(state) do
    # Determine how many files we can dispatch
    available_slots = 5 - map_size(state.in_progress_files)
    to_dispatch = min(state.demand, available_slots)

    Logger.debug(
      "Calculating dispatch: demand=#{state.demand}, available_slots=#{available_slots}, to_dispatch=#{to_dispatch}"
    )

    # Don't try to dispatch if no slots available
    if to_dispatch <= 0 do
      {:noreply, [], state}
    else
      # Dequeue files up to the limit
      {events_to_dispatch, remaining_events} =
        dequeue_files(state.events, to_dispatch, state.in_progress_files, [])

      # Update state
      new_state = %{
        state
        | events: remaining_events,
          demand: state.demand - length(events_to_dispatch),
          in_progress_files: state.in_progress_files
      }

      if length(events_to_dispatch) > 0 do
        Logger.info("Dispatched #{length(events_to_dispatch)} files")
        Logger.debug("Dispatched files: #{inspect(Enum.map(events_to_dispatch, & &1.id))}")
      end

      {:noreply, events_to_dispatch, new_state}
    end
  end

  # Helper to dequeue files
  defp dequeue_files(queue, 0, in_progress, events) do
    # No more to dequeue
    {events, queue, in_progress}
  end

  defp dequeue_files([], _count, in_progress, events) do
    # Empty queue
    {events, [], in_progress}
  end

  defp dequeue_files([item | rest], count, in_progress, events) do
    # Add timestamp to in_progress map
    new_in_progress = Map.put(in_progress, item.id, System.system_time(:millisecond))

    Logger.debug("Dequeuing file #{item.id}: #{item.path}")

    # Continue dequeuing
    dequeue_files(rest, count - 1, new_in_progress, [item | events])
  end

  # Schedule periodic queue check
  defp schedule_queue_check(interval) do
    Process.send_after(self(), :check_queue, interval)
  end

  # Schedule periodic timeout check
  defp schedule_timeout_check(interval) do
    # Check more frequently than the timeout itself
    Process.send_after(self(), :check_timeouts, div(interval, 4))
  end
end
