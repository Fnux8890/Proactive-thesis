defmodule Pipeline.Tracking.Registry do
  @moduledoc """
  Registry interface for tracking pipeline items.

  This module provides a wrapper around Elixir's Registry to manage the
  tracking processes for pipeline items, with helper functions to start,
  locate, and list trackers.
  """

  @doc """
  Child spec for the Registry.

  ## Parameters
    * _opts - Ignored options

  ## Returns
    * Registry child specification
  """
  def child_spec(_opts) do
    Registry.child_spec(
      keys: :unique,
      name: __MODULE__,
      partitions: System.schedulers_online()
    )
  end

  @doc """
  Start a new tracker for a pipeline item.

  Creates a new tracker process if one doesn't already exist for the given item ID.

  ## Parameters
    * item_id - ID of the item to track
    * metadata - Optional initial metadata for the item

  ## Returns
    * {:ok, pid} - PID of the started or existing tracker
    * {:error, reason} - If starting the tracker failed
  """
  def start_item_tracker(item_id, metadata \\ %{}) do
    case Registry.lookup(__MODULE__, item_id) do
      [] ->
        # No existing tracker, start a new one
        DynamicSupervisor.start_child(
          Pipeline.Tracking.TrackerSupervisor,
          {Pipeline.Tracking.ItemTracker, {item_id, metadata}}
        )

      [{pid, _}] ->
        # Tracker already exists
        {:ok, pid}
    end
  end

  @doc """
  Find the PID of an existing tracker.

  ## Parameters
    * item_id - ID of the item

  ## Returns
    * pid - PID of the tracker if found
    * nil - If no tracker exists for the item
  """
  def whereis(item_id) do
    case Registry.lookup(__MODULE__, item_id) do
      [{pid, _}] -> pid
      [] -> nil
    end
  end

  @doc """
  Get a list of all active tracked items.

  ## Returns
    * List of {item_id, pid} tuples for all active trackers
  """
  def list_active_items do
    Registry.select(__MODULE__, [{{:"$1", :"$2", :_}, [], [{{:"$1", :"$2"}}]}])
  end

  @doc """
  Get a count of active tracked items.

  ## Returns
    * Integer count of active trackers
  """
  def count_active_items do
    Registry.count(__MODULE__)
  end

  @doc """
  Check if a tracker exists for an item.

  ## Parameters
    * item_id - ID of the item

  ## Returns
    * true - If a tracker exists for the item
    * false - If no tracker exists
  """
  def tracker_exists?(item_id) do
    case Registry.lookup(__MODULE__, item_id) do
      [] -> false
      _ -> true
    end
  end

  @doc """
  Get the status of an item if it's being tracked.

  ## Parameters
    * item_id - ID of the item

  ## Returns
    * {:ok, status} - Status of the item
    * {:error, :not_found} - If no tracker exists
  """
  def get_item_status(item_id) do
    if tracker_exists?(item_id) do
      {:ok, Pipeline.Tracking.ItemTracker.get_status(item_id)}
    else
      {:error, :not_found}
    end
  end

  @doc """
  Find items in a specific status.

  ## Parameters
    * status - Status to filter by (:pending, :processing, :completed, or :failed)

  ## Returns
    * List of {item_id, status} tuples for items in the specified status
  """
  def find_items_by_status(status) do
    list_active_items()
    |> Enum.map(fn {item_id, _pid} ->
      {item_id, Pipeline.Tracking.ItemTracker.get_status(item_id)}
    end)
    |> Enum.filter(fn {_item_id, item_status} -> item_status.status == status end)
  end
end
