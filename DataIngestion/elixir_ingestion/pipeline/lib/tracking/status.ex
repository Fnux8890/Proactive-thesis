defmodule Pipeline.Tracking.Status do
  @moduledoc """
  Defines the data structures and state transitions for tracking pipeline items.

  This module provides type definitions and helper functions for managing the state
  of items as they flow through the pipeline stages.
  """

  @type stage_id :: String.t()
  @type item_id :: String.t()
  @type timestamp :: integer()

  @type status :: :pending | :processing | :completed | :failed | :skipped

  @type stage_status :: %{
          id: stage_id,
          status: status,
          started_at: timestamp | nil,
          completed_at: timestamp | nil,
          attempts: integer(),
          error: String.t() | nil,
          metadata: map()
        }

  @type item_status :: %{
          id: item_id,
          stages: %{optional(stage_id) => stage_status},
          created_at: timestamp,
          updated_at: timestamp,
          completed_at: timestamp | nil,
          status: status,
          metadata: map()
        }

  @doc """
  Creates a new item status with the given ID.

  ## Parameters
    * id - Unique identifier for the pipeline item
    * metadata - Optional metadata for the item

  ## Returns
    * New item status map
  """
  def new_item(id, metadata \\ %{}) do
    now = System.system_time(:millisecond)

    %{
      id: id,
      stages: %{},
      created_at: now,
      updated_at: now,
      completed_at: nil,
      status: :pending,
      metadata: metadata
    }
  end

  @doc """
  Registers a stage for an item.

  ## Parameters
    * item_status - Current item status
    * stage_id - ID of the stage to register
    * metadata - Optional metadata for the stage

  ## Returns
    * Updated item status
  """
  def register_stage(item_status, stage_id, metadata \\ %{}) do
    now = System.system_time(:millisecond)

    new_stage = %{
      id: stage_id,
      status: :pending,
      started_at: nil,
      completed_at: nil,
      attempts: 0,
      error: nil,
      metadata: metadata
    }

    %{item_status | stages: Map.put(item_status.stages, stage_id, new_stage), updated_at: now}
  end

  @doc """
  Updates a stage status.

  ## Parameters
    * item_status - Current item status
    * stage_id - ID of the stage to update
    * status_update - Map containing status updates

  ## Returns
    * Updated item status
  """
  def update_stage(item_status, stage_id, status_update) do
    now = System.system_time(:millisecond)

    # Get current stage status or create a new one
    stage_status =
      Map.get(item_status.stages, stage_id, %{
        id: stage_id,
        status: :pending,
        started_at: nil,
        completed_at: nil,
        attempts: 0,
        error: nil,
        metadata: %{}
      })

    # Update the stage status
    new_stage_status = Map.merge(stage_status, status_update)

    # Update item status
    new_item_status = %{
      item_status
      | stages: Map.put(item_status.stages, stage_id, new_stage_status),
        updated_at: now
    }

    # Update overall status
    update_overall_status(new_item_status)
  end

  @doc """
  Marks a stage as processing.

  ## Parameters
    * item_status - Current item status
    * stage_id - ID of the stage
    * metadata - Optional additional metadata

  ## Returns
    * Updated item status
  """
  def mark_stage_processing(item_status, stage_id, metadata \\ %{}) do
    now = System.system_time(:millisecond)

    stage_status = Map.get(item_status.stages, stage_id)

    stage_update = %{
      status: :processing,
      started_at: now,
      attempts: (stage_status[:attempts] || 0) + 1,
      metadata: Map.merge(stage_status[:metadata] || %{}, metadata)
    }

    update_stage(item_status, stage_id, stage_update)
  end

  @doc """
  Marks a stage as completed.

  ## Parameters
    * item_status - Current item status
    * stage_id - ID of the stage
    * metadata - Optional additional metadata

  ## Returns
    * Updated item status
  """
  def mark_stage_completed(item_status, stage_id, metadata \\ %{}) do
    now = System.system_time(:millisecond)

    stage_update = %{
      status: :completed,
      completed_at: now,
      metadata: Map.merge(item_status.stages[stage_id][:metadata] || %{}, metadata)
    }

    update_stage(item_status, stage_id, stage_update)
  end

  @doc """
  Marks a stage as failed.

  ## Parameters
    * item_status - Current item status
    * stage_id - ID of the stage
    * error - Error description or reason
    * metadata - Optional additional metadata

  ## Returns
    * Updated item status
  """
  def mark_stage_failed(item_status, stage_id, error, metadata \\ %{}) do
    stage_update = %{
      status: :failed,
      error: error,
      metadata: Map.merge(item_status.stages[stage_id][:metadata] || %{}, metadata)
    }

    update_stage(item_status, stage_id, stage_update)
  end

  @doc """
  Marks an item as completed.

  ## Parameters
    * item_status - Current item status
    * metadata - Optional additional metadata

  ## Returns
    * Updated item status
  """
  def mark_item_completed(item_status, metadata \\ %{}) do
    now = System.system_time(:millisecond)

    %{
      item_status
      | status: :completed,
        completed_at: now,
        updated_at: now,
        metadata: Map.merge(item_status.metadata, metadata)
    }
  end

  @doc """
  Marks an item as failed.

  ## Parameters
    * item_status - Current item status
    * error - Error description
    * metadata - Optional additional metadata

  ## Returns
    * Updated item status
  """
  def mark_item_failed(item_status, error, metadata \\ %{}) do
    now = System.system_time(:millisecond)

    %{
      item_status
      | status: :failed,
        updated_at: now,
        metadata:
          Map.merge(
            item_status.metadata,
            Map.merge(%{error: error}, metadata)
          )
    }
  end

  @doc """
  Updates the overall status of an item based on its stage statuses.

  ## Parameters
    * item_status - Current item status

  ## Returns
    * Updated item status with recalculated overall status
  """
  def update_overall_status(item_status) do
    # If already completed or failed, don't change
    if item_status.status in [:completed, :failed] do
      item_status
    else
      stage_statuses = Map.values(item_status.stages)

      cond do
        # If any stage failed, mark item as failed
        Enum.any?(stage_statuses, fn s -> s.status == :failed end) ->
          %{item_status | status: :failed}

        # If all stages completed, mark item as completed
        Enum.all?(stage_statuses, fn s -> s.status == :completed end) &&
            length(stage_statuses) > 0 ->
          now = System.system_time(:millisecond)
          %{item_status | status: :completed, completed_at: now}

        # If any stage is processing, mark item as processing
        Enum.any?(stage_statuses, fn s -> s.status == :processing end) ->
          %{item_status | status: :processing}

        # Otherwise, remain pending
        true ->
          %{item_status | status: :pending}
      end
    end
  end

  @doc """
  Checks if all stages for an item are completed.

  ## Parameters
    * item_status - Current item status

  ## Returns
    * true if all stages are completed, false otherwise
  """
  def all_stages_completed?(item_status) do
    stage_statuses = Map.values(item_status.stages)
    length(stage_statuses) > 0 && Enum.all?(stage_statuses, fn s -> s.status == :completed end)
  end

  @doc """
  Gets a list of failed stages for an item.

  ## Parameters
    * item_status - Current item status

  ## Returns
    * List of {stage_id, stage_status} tuples for failed stages
  """
  def get_failed_stages(item_status) do
    Enum.filter(item_status.stages, fn {_id, status} -> status.status == :failed end)
  end

  @doc """
  Gets the pipeline progress as a percentage.

  ## Parameters
    * item_status - Current item status

  ## Returns
    * Float between 0.0 and 1.0 representing completion percentage
  """
  def get_progress(item_status) do
    stage_statuses = Map.values(item_status.stages)
    completed_count = Enum.count(stage_statuses, fn s -> s.status == :completed end)

    if length(stage_statuses) > 0 do
      completed_count / length(stage_statuses)
    else
      0.0
    end
  end
end
