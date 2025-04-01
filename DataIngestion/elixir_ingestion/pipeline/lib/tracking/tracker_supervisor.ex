defmodule Pipeline.Tracking.TrackerSupervisor do
  @moduledoc """
  Dynamic supervisor for item tracker processes.

  This supervisor manages the lifecycle of individual item tracker processes,
  allowing them to be started and stopped dynamically as pipeline items
  are added and completed.
  """
  use DynamicSupervisor

  @doc """
  Starts the tracker supervisor.

  ## Parameters
    * _opts - Ignored options

  ## Returns
    * {:ok, pid} - PID of the started supervisor
    * {:error, reason} - If starting the supervisor failed
  """
  def start_link(_opts) do
    DynamicSupervisor.start_link(__MODULE__, [], name: __MODULE__)
  end

  @doc """
  Start a tracker for a specific pipeline item.

  ## Parameters
    * item_id - ID of the item to track
    * metadata - Optional initial metadata for the item

  ## Returns
    * {:ok, pid} - PID of the started tracker
    * {:error, reason} - If starting the tracker failed
  """
  def start_tracker(item_id, metadata \\ %{}) do
    DynamicSupervisor.start_child(
      __MODULE__,
      {Pipeline.Tracking.ItemTracker, {item_id, metadata}}
    )
  end

  @doc """
  Stop a tracker for a specific pipeline item.

  ## Parameters
    * item_id - ID of the item to stop tracking

  ## Returns
    * :ok - If the tracker was stopped
    * {:error, :not_found} - If no tracker was found
  """
  def stop_tracker(item_id) do
    case Pipeline.Tracking.Registry.whereis(item_id) do
      nil ->
        {:error, :not_found}

      pid ->
        DynamicSupervisor.terminate_child(__MODULE__, pid)
    end
  end

  @doc """
  Get a count of active trackers.

  ## Returns
    * Map with count information
  """
  def count_children do
    DynamicSupervisor.count_children(__MODULE__)
  end

  @impl true
  def init(_init_arg) do
    DynamicSupervisor.init(
      strategy: :one_for_one,
      max_restarts: 10,
      max_seconds: 60
    )
  end
end
