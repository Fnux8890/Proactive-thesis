defmodule Pipeline.Tracking.Supervisor do
  @moduledoc """
  Main supervisor for the pipeline tracking system.

  This supervisor manages all components of the tracking system:
  - Registry for tracking processes
  - Dynamic supervisor for tracker processes
  - Cleanup manager for handling completed items
  - Metrics collection
  """
  use Supervisor
  require Logger

  @doc """
  Starts the tracking supervisor.

  ## Parameters
    * _opts - Ignored options

  ## Returns
    * {:ok, pid} - PID of the started supervisor
    * {:error, reason} - If starting the supervisor failed
  """
  def start_link(_opts) do
    Supervisor.start_link(__MODULE__, [], name: __MODULE__)
  end

  @doc """
  Start tracking a new pipeline item.

  ## Parameters
    * item_id - Unique identifier for the item
    * metadata - Optional initial metadata

  ## Returns
    * {:ok, pid} - PID of the tracker process
    * {:error, reason} - If starting the tracker failed
  """
  def start_tracking(item_id, metadata \\ %{}) do
    Pipeline.Tracking.Registry.start_item_tracker(item_id, metadata)
  end

  @doc """
  Get status information about all active tracked items.

  ## Returns
    * Map with tracking statistics
  """
  def get_tracking_stats do
    active_count = Pipeline.Tracking.Registry.count_active_items()
    pending_cleanup = Pipeline.Tracking.CleanupManager.pending_cleanup_count()

    # Get counts by status
    pending_count = length(Pipeline.Tracking.Registry.find_items_by_status(:pending))
    processing_count = length(Pipeline.Tracking.Registry.find_items_by_status(:processing))
    completed_count = length(Pipeline.Tracking.Registry.find_items_by_status(:completed))
    failed_count = length(Pipeline.Tracking.Registry.find_items_by_status(:failed))

    %{
      active_items: active_count,
      pending_cleanup: pending_cleanup,
      by_status: %{
        pending: pending_count,
        processing: processing_count,
        completed: completed_count,
        failed: failed_count
      }
    }
  end

  @impl true
  def init(_init_arg) do
    # Removed Metrics.init() call here

    children = [
      # Start the Metrics GenServer first
      Pipeline.Tracking.Metrics,

      # Registry for tracking processes
      Pipeline.Tracking.Registry,

      # Dynamic supervisor for tracker processes
      Pipeline.Tracking.TrackerSupervisor,

      # Cleanup manager for completed items
      Pipeline.Tracking.CleanupManager
    ]

    Logger.info("Starting Pipeline Tracking Supervisor")
    Supervisor.init(children, strategy: :one_for_one)
  end
end
