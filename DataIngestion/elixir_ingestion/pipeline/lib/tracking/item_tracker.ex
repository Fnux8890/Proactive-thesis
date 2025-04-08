defmodule Pipeline.Tracking.ItemTracker do
  @moduledoc """
  GenServer for tracking individual pipeline items as they progress through stages.

  This module provides a process-per-item approach to track the progress and status
  of each item flowing through the pipeline, with persistence to Redis for
  fault tolerance.
  """
  use GenServer
  require Logger
  alias Pipeline.Tracking.Status

  # How often to checkpoint state to Redis (5 seconds)
  @checkpoint_interval 5_000

  # Default TTL for item tracking data in Redis (24 hours)
  @default_ttl 86_400

  # Client API

  @doc """
  Starts a new item tracker process.

  ## Parameters
    * item_id - Unique identifier for the pipeline item
    * metadata - Optional initial metadata

  ## Returns
    * {:ok, pid} - PID of the started tracker
    * {:error, reason} - If starting the tracker failed
  """
  def start_link(item_id, metadata \\ %{}) do
    GenServer.start_link(__MODULE__, {item_id, metadata}, name: via_tuple(item_id))
  end

  @doc """
  Helper to create via tuple for Registry lookup.

  ## Parameters
    * item_id - Unique identifier for the pipeline item

  ## Returns
    * Registry via tuple
  """
  def via_tuple(item_id) do
    {:via, Registry, {Pipeline.Tracking.Registry, item_id}}
  end

  @doc """
  Registers a stage for the item.

  ## Parameters
    * item_id - ID of the item
    * stage_id - ID of the stage to register
    * metadata - Optional metadata for the stage

  ## Returns
    * :ok - If the stage was registered
    * {:error, reason} - If registration failed
  """
  def register_stage(item_id, stage_id, metadata \\ %{}) do
    GenServer.call(via_tuple(item_id), {:register_stage, stage_id, metadata})
  end

  @doc """
  Updates the status of a stage.

  ## Parameters
    * item_id - ID of the item
    * stage_id - ID of the stage to update
    * status_update - Map containing status updates

  ## Returns
    * :ok - If the stage was updated
    * {:error, reason} - If update failed
  """
  def update_stage(item_id, stage_id, status_update) do
    GenServer.call(via_tuple(item_id), {:update_stage, stage_id, status_update})
  end

  @doc """
  Marks a stage as processing.

  ## Parameters
    * item_id - ID of the item
    * stage_id - ID of the stage
    * metadata - Optional additional metadata

  ## Returns
    * :ok - If the stage was updated
    * {:error, reason} - If update failed
  """
  def mark_stage_processing(item_id, stage_id, metadata \\ %{}) do
    GenServer.call(via_tuple(item_id), {:mark_stage_processing, stage_id, metadata})
  end

  @doc """
  Marks a stage as completed.

  ## Parameters
    * item_id - ID of the item
    * stage_id - ID of the stage
    * metadata - Optional additional metadata

  ## Returns
    * :ok - If the stage was updated
    * {:error, reason} - If update failed
  """
  def mark_stage_completed(item_id, stage_id, metadata \\ %{}) do
    GenServer.call(via_tuple(item_id), {:mark_stage_completed, stage_id, metadata})
  end

  @doc """
  Marks a stage as failed.

  ## Parameters
    * item_id - ID of the item
    * stage_id - ID of the stage
    * error - Error description or reason
    * metadata - Optional additional metadata

  ## Returns
    * :ok - If the stage was updated
    * {:error, reason} - If update failed
  """
  def mark_stage_failed(item_id, stage_id, error, metadata \\ %{}) do
    GenServer.call(via_tuple(item_id), {:mark_stage_failed, stage_id, error, metadata})
  end

  @doc """
  Gets the current status of an item.

  ## Parameters
    * item_id - ID of the item

  ## Returns
    * item_status - Current status map
    * {:error, reason} - If retrieval failed
  """
  def get_status(item_id) do
    GenServer.call(via_tuple(item_id), :get_status)
  end

  @doc """
  Marks an item as completed.

  ## Parameters
    * item_id - ID of the item
    * metadata - Optional additional metadata

  ## Returns
    * :ok - If the item was updated
    * {:error, reason} - If update failed
  """
  def mark_completed(item_id, metadata \\ %{}) do
    GenServer.call(via_tuple(item_id), {:mark_completed, metadata})
  end

  @doc """
  Marks an item as failed.

  ## Parameters
    * item_id - ID of the item
    * error - Error description
    * metadata - Optional additional metadata

  ## Returns
    * :ok - If the item was updated
    * {:error, reason} - If update failed
  """
  def mark_failed(item_id, error, metadata \\ %{}) do
    GenServer.call(via_tuple(item_id), {:mark_failed, error, metadata})
  end

  # Server Callbacks

  @impl true
  def init({item_id, metadata}) do
    # Try to recover state from Redis if this is a restart
    state =
      case ConnectionHandler.Client.get("pipeline:item:#{item_id}") do
        {:ok, nil} ->
          # No existing state, create new
          Status.new_item(item_id, metadata)

        {:ok, data} ->
          # Existing state found, deserialize
          try do
            :erlang.binary_to_term(data)
          rescue
            _ -> Status.new_item(item_id, metadata)
          end

        {:error, _} ->
          # Error retrieving from Redis, create new
          Status.new_item(item_id, metadata)
      end

    # Schedule periodic checkpoint to Redis
    schedule_checkpoint()

    {:ok, state}
  end

  @impl true
  def handle_call({:register_stage, stage_id, metadata}, _from, state) do
    new_state = Status.register_stage(state, stage_id, metadata)

    # Persist to Redis
    checkpoint_to_redis(new_state)

    {:reply, :ok, new_state}
  end

  @impl true
  def handle_call({:update_stage, stage_id, status_update}, _from, state) do
    new_state = Status.update_stage(state, stage_id, status_update)

    # Persist to Redis
    checkpoint_to_redis(new_state)

    # If this update completed all stages, notify
    check_completion(new_state)

    {:reply, :ok, new_state}
  end

  @impl true
  def handle_call({:mark_stage_processing, stage_id, metadata}, _from, state) do
    new_state = Status.mark_stage_processing(state, stage_id, metadata)

    checkpoint_to_redis(new_state)

    {:reply, :ok, new_state}
  end

  @impl true
  def handle_call({:mark_stage_completed, stage_id, metadata}, _from, state) do
    new_state = Status.mark_stage_completed(state, stage_id, metadata)

    checkpoint_to_redis(new_state)

    # Check if all stages are now complete
    check_completion(new_state)

    {:reply, :ok, new_state}
  end

  @impl true
  def handle_call({:mark_stage_failed, stage_id, error, metadata}, _from, state) do
    new_state = Status.mark_stage_failed(state, stage_id, error, metadata)

    checkpoint_to_redis(new_state)

    {:reply, :ok, new_state}
  end

  @impl true
  def handle_call(:get_status, _from, state) do
    {:reply, state, state}
  end

  @impl true
  def handle_call({:mark_completed, metadata}, _from, state) do
    new_state = Status.mark_item_completed(state, metadata)

    # Persist final state to Redis
    checkpoint_to_redis(new_state)

    # Notify cleanup manager
    notify_completion(new_state)

    {:reply, :ok, new_state}
  end

  @impl true
  def handle_call({:mark_failed, error, metadata}, _from, state) do
    new_state = Status.mark_item_failed(state, error, metadata)

    # Persist to Redis
    checkpoint_to_redis(new_state)

    {:reply, :ok, new_state}
  end

  @impl true
  def handle_info(:checkpoint, state) do
    checkpoint_to_redis(state)
    schedule_checkpoint()
    {:noreply, state}
  end

  # Private Functions

  # Check if the item is now complete and notify if needed
  defp check_completion(state) do
    if Status.all_stages_completed?(state) && state.status != :completed do
      # All stages completed, mark the entire item as completed
      new_state = Status.mark_item_completed(state)

      # Notify completion
      notify_completion(new_state)

      new_state
    else
      state
    end
  end

  # Helper to notify cleanup manager
  defp notify_completion(state) do
    case Application.get_env(:pipeline, :cleanup_manager, Pipeline.Tracking.CleanupManager) do
      nil -> :ok # No manager configured
      manager ->
        case GenServer.cast(manager, {:item_complete, state.id}) do
          :ok -> :ok
          _ -> Logger.warning("Failed to notify cleanup manager about completed item #{state.id}")
        end
    end
  end

  # Save state checkpoint to Redis
  defp checkpoint_to_redis(state) do
    item_key = "pipeline:item:#{state.id}"

    # Serialize state and save to Redis
    binary_data = :erlang.term_to_binary(state)

    # Set with expiration
    case ConnectionHandler.Client.set(item_key, binary_data) do
      {:ok, _} ->
        # Set TTL
        ConnectionHandler.Client.expire(item_key, @default_ttl)
        :ok

      {:error, reason} ->
        Logger.error("Failed to checkpoint item #{state.id} to Redis: #{inspect(reason)}")
        :error
    end
  end

  # Schedule periodic checkpoint
  defp schedule_checkpoint do
    Process.send_after(self(), :checkpoint, @checkpoint_interval)
  end
end
