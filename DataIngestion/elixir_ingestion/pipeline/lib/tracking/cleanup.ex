defmodule Pipeline.Tracking.CleanupManager do
  @moduledoc """
  Manages cleanup of completed pipeline items.

  This module is responsible for:
  - Collecting completed items for cleanup
  - Moving completed items to long-term storage
  - Terminating tracker processes for completed items
  - Periodic cleanup of stale tracking data
  """
  use GenServer
  require Logger

  # Cleanup interval (30 seconds)
  @cleanup_interval 30_000

  # Batch size for cleanup operations
  @cleanup_batch_size 50

  # Archive TTL for completed items (30 days in seconds)
  @archive_ttl 2_592_000

  # Client API

  @doc """
  Starts the cleanup manager.

  ## Parameters
    * _opts - Ignored options

  ## Returns
    * {:ok, pid} - PID of the started manager
    * {:error, reason} - If starting the manager failed
  """
  def start_link(_opts) do
    GenServer.start_link(__MODULE__, [], name: __MODULE__)
  end

  @doc """
  Notify that an item has been completed and ready for cleanup.

  ## Parameters
    * item_id - ID of the completed item

  ## Returns
    * :ok
  """
  def item_completed(item_id) do
    GenServer.cast(__MODULE__, {:item_completed, item_id})
  end

  @doc """
  Get the count of pending cleanup items.

  ## Returns
    * Integer count of pending cleanup items
  """
  def pending_cleanup_count do
    GenServer.call(__MODULE__, :pending_cleanup_count)
  end

  @doc """
  Trigger immediate cleanup of pending items.

  ## Returns
    * :ok
  """
  def trigger_cleanup do
    GenServer.cast(__MODULE__, :trigger_cleanup)
  end

  # Server Callbacks

  @impl true
  def init(_) do
    # Schedule periodic cleanup
    schedule_cleanup()

    {:ok, %{pending_cleanup: MapSet.new()}}
  end

  @impl true
  def handle_cast({:item_completed, item_id}, state) do
    # Add to pending cleanup
    new_state = %{state | pending_cleanup: MapSet.put(state.pending_cleanup, item_id)}

    # Schedule immediate cleanup if queue is getting large
    if MapSet.size(new_state.pending_cleanup) > 100 do
      Process.send(self(), :cleanup, [])
    end

    {:noreply, new_state}
  end

  @impl true
  def handle_cast(:trigger_cleanup, state) do
    Process.send(self(), :cleanup, [])
    {:noreply, state}
  end

  @impl true
  def handle_call(:pending_cleanup_count, _from, state) do
    {:reply, MapSet.size(state.pending_cleanup), state}
  end

  @impl true
  def handle_info(:cleanup, state) do
    # Process a batch of completed items
    {to_cleanup, remaining} = take_batch(state.pending_cleanup, @cleanup_batch_size)

    # Perform cleanup
    Enum.each(to_cleanup, &cleanup_item/1)

    # Schedule next cleanup
    schedule_cleanup()

    {:noreply, %{state | pending_cleanup: remaining}}
  end

  # Private Functions

  # Take a batch of items for cleanup
  defp take_batch(set, count) do
    {taken, remaining} = Enum.split(MapSet.to_list(set), count)
    {taken, MapSet.new(remaining)}
  end

  # Clean up a single item
  defp cleanup_item(item_id) do
    # Get final status to preserve in long-term storage if needed
    case Pipeline.Tracking.ItemTracker.get_status(item_id) do
      status when is_map(status) ->
        # Archive to permanent storage
        archive_completed_item(item_id, status)

        # Terminate the tracker process
        case Pipeline.Tracking.Registry.whereis(item_id) do
          pid when is_pid(pid) ->
            DynamicSupervisor.terminate_child(
              Pipeline.Tracking.TrackerSupervisor,
              pid
            )

            Logger.info("Cleaned up completed item #{item_id}")

          nil ->
            Logger.debug("Tracker for item #{item_id} already terminated")
        end

      other ->
        # Log that getting the tracker PID failed unexpectedly
        Logger.warning("Failed to get tracker PID for item #{item_id}, received: #{inspect(other)}")
    end
  end

  # Archive completed item to long-term storage
  defp archive_completed_item(item_id, status) do
    # Move to long-term storage with different TTL
    archive_key = "pipeline:archive:#{item_id}"
    binary_data = :erlang.term_to_binary(status)

    # Set archive data with longer TTL
    case ConnectionHandler.Client.set(archive_key, binary_data) do
      {:ok, _} ->
        ConnectionHandler.Client.expire(archive_key, @archive_ttl)

        # Add to archive index
        ConnectionHandler.Client.sadd("pipeline:archives", item_id)

        # Remove from active tracking
        ConnectionHandler.Client.del("pipeline:item:#{item_id}")
        :ok

      {:error, reason} ->
        Logger.error("Failed to archive item #{item_id}: #{inspect(reason)}")
        :error
    end
  end

  # Schedule the next cleanup operation
  defp schedule_cleanup do
    Process.send_after(self(), :cleanup, @cleanup_interval)
  end
end
