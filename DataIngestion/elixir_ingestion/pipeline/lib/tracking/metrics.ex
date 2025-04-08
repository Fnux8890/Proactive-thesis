defmodule Pipeline.Tracking.Metrics do
  @moduledoc """
  Collects and reports metrics for pipeline performance.

  This module provides functions to:
  - Track processing times for pipeline stages
  - Count items processed through the pipeline
  - Generate performance reports
  - Monitor pipeline throughput and latency
  """
  use GenServer
  require Logger

  # ETS table name for metrics storage
  @metrics_table :pipeline_metrics

  # Metrics reporting interval (10 seconds)
  @reporting_interval 10_000

  # --- Client API ---

  @doc """
  Starts the Metrics GenServer.
  """
  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  # --- Server Callbacks ---

  @impl true
  def init(_opts) do
    # Create metrics table if it doesn't exist
    if :ets.info(@metrics_table) == :undefined do
      :ets.new(@metrics_table, [
        :named_table,
        :set,
        :public,
        read_concurrency: true,
        write_concurrency: true
      ])
    end

    # Initialize counters
    reset_counter(:items_received)
    reset_counter(:items_processed)
    reset_counter(:items_failed)
    reset_counter(:items_skipped)

    # Schedule metrics reporting
    schedule_reporting()

    {:ok, %{}}
  end

  # Handle periodic metrics reporting
  @impl true
  def handle_info(:report_metrics, state) do
    # Get current rates
    rates = get_processing_rates()

    # Log metrics
    Logger.info(
      "Pipeline metrics: received=#{rates.received}, processed=#{rates.processed}, " <>
        "failed=#{rates.failed}, skipped=#{rates.skipped}"
    )

    # Schedule next reporting
    schedule_reporting()

    {:noreply, state}
  end

  # --- Public Functions (that interact via ETS, no GenServer calls needed) ---

  @doc """
  Records the start of processing for a stage.

  ## Parameters
    * item_id - ID of the item being processed
    * stage_id - ID of the stage

  ## Returns
    * :ok
  """
  def record_stage_start(item_id, stage_id) do
    key = "#{item_id}:#{stage_id}:start"
    now = System.monotonic_time(:millisecond)
    :ets.insert(@metrics_table, {key, now})
    :ok
  end

  @doc """
  Records the completion of processing for a stage.

  ## Parameters
    * item_id - ID of the item being processed
    * stage_id - ID of the stage
    * status - Result status (:completed, :failed, or :skipped)

  ## Returns
    * :ok
  """
  def record_stage_end(item_id, stage_id, status \\ :completed) do
    start_key = "#{item_id}:#{stage_id}:start"
    now = System.monotonic_time(:millisecond)

    # Calculate duration if we have a start time
    case :ets.lookup(@metrics_table, start_key) do
      [{_, start_time}] ->
        duration = now - start_time
        # Record duration for this stage
        record_duration(stage_id, duration)
        # Clean up start time entry
        :ets.delete(@metrics_table, start_key)

      [] ->
        # No start time found, just record an event
        Logger.warning("No start time found for #{item_id} at stage #{stage_id}")
    end

    # Increment appropriate counter based on status
    case status do
      :completed -> increment_counter(:items_processed)
      :failed -> increment_counter(:items_failed)
      :skipped -> increment_counter(:items_skipped)
      _ -> :ok
    end

    :ok
  end

  @doc """
  Records that a new item was received by the pipeline.

  ## Parameters
    * _item_id - ID of the item (optional)

  ## Returns
    * :ok
  """
  def record_item_received(_item_id \\ nil) do
    increment_counter(:items_received)
    :ok
  end

  @doc """
  Get the current processing rates for the pipeline.

  ## Returns
    * Map with rate information
  """
  def get_processing_rates do
    %{
      received: get_counter(:items_received),
      processed: get_counter(:items_processed),
      failed: get_counter(:items_failed),
      skipped: get_counter(:items_skipped)
    }
  end

  @doc """
  Get the average processing time for a specific stage.

  ## Parameters
    * stage_id - ID of the stage

  ## Returns
    * Average processing time in milliseconds, or nil if no data
  """
  def get_avg_processing_time(stage_id) do
    key = "#{stage_id}:duration"

    case :ets.lookup(@metrics_table, key) do
      [{_, {total, count}}] when count > 0 ->
        total / count

      _ ->
        nil
    end
  end

  @doc """
  Get processing statistics for all stages.

  ## Returns
    * Map of stage IDs to their processing statistics
  """
  def get_stage_statistics do
    # Match object pattern: {KeyPattern, ValuePattern}
    # Key: "<stage_id>:duration"
    # Value: {<total_duration>, <count>}
    match_pattern = {"$1:duration", {"$2", "$3"}}

    # Fetch all matching entries: [[stage_id_str, total_duration, count]]
    all_stats = :ets.match_object(@metrics_table, match_pattern)

    # Process the matched statistics
    all_stats
    |> Enum.map(fn [stage_id_str, total, count] ->
      # Calculate average, handling count=0
      avg = if count > 0, do: total / count, else: 0.0
      # Return map for this stage
      {stage_id_str,
       %{
         avg_duration_ms: avg,
         total_items: count,
         total_duration_ms: total
       }}
    end)
    |> Enum.into(%{}) # Convert list of {stage_id, stats_map} into a map
  end

  @doc """
  Reset all metrics.

  ## Returns
    * :ok
  """
  def reset_all do
    :ets.delete_all_objects(@metrics_table)
    reset_counter(:items_received)
    reset_counter(:items_processed)
    reset_counter(:items_failed)
    reset_counter(:items_skipped)
    :ok
  end

  # Private Functions

  # Record a duration for a specific stage
  defp record_duration(stage_id, duration) do
    key = "#{stage_id}:duration"

    # Update or initialize duration statistics
    case :ets.lookup(@metrics_table, key) do
      [{_, {total, count}}] ->
        :ets.insert(@metrics_table, {key, {total + duration, count + 1}})

      [] ->
        :ets.insert(@metrics_table, {key, {duration, 1}})
    end
  end

  # Increment a counter
  defp increment_counter(counter) do
    key = "counter:#{counter}"

    case :ets.lookup(@metrics_table, key) do
      [{_, value}] ->
        :ets.insert(@metrics_table, {key, value + 1})

      [] ->
        :ets.insert(@metrics_table, {key, 1})
    end
  end

  # Reset a counter to zero
  defp reset_counter(counter) do
    :ets.insert(@metrics_table, {"counter:#{counter}", 0})
  end

  # Get the current value of a counter
  defp get_counter(counter) do
    key = "counter:#{counter}"

    case :ets.lookup(@metrics_table, key) do
      [{_, value}] -> value
      [] -> 0
    end
  end

  # Schedule periodic metrics reporting
  defp schedule_reporting do
    Process.send_after(self(), :report_metrics, @reporting_interval)
  end
end
