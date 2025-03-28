defmodule Pipeline.FaultHandling.Recovery do
  @moduledoc """
  Handles recovery of failed events in the pipeline.

  This module provides functionality to:
  - Track failed events that exceeded retry limits
  - Schedule background recovery attempts
  - Manage dead-letter queues for unrecoverable failures
  """

  require Logger

  # 1 minute
  @default_recovery_interval 60_000
  @default_max_recovery_attempts 3
  # 1 week
  @default_dead_letter_ttl 604_800

  @doc """
  Initialize a new recovery manager with optional configuration.

  ## Options
    * :redis_client - Module implementing Redis client. Defaults to ConnectionHandler.Client
    * :recovery_interval - Interval between recovery attempts in ms. Default: 60 seconds
    * :max_recovery_attempts - Maximum number of recovery attempts. Default: 3
    * :dead_letter_ttl - TTL for dead letter queue items in seconds. Default: 1 week
  """
  def new(opts \\ []) do
    %{
      redis_client: Keyword.get(opts, :redis_client, ConnectionHandler.Client),
      recovery_interval: Keyword.get(opts, :recovery_interval, @default_recovery_interval),
      max_recovery_attempts:
        Keyword.get(opts, :max_recovery_attempts, @default_max_recovery_attempts),
      dead_letter_ttl: Keyword.get(opts, :dead_letter_ttl, @default_dead_letter_ttl)
    }
  end

  @doc """
  Track a failed event in Redis for later recovery.

  ## Parameters
    * stage_id - Identifier of the stage where the failure occurred
    * event - The event that failed processing
    * error - The error that caused the failure
    * context - Additional context about the failure
    * config - Optional recovery configuration

  ## Returns
    * :ok on success
    * {:error, reason} on failure
  """
  def track_failed_event(stage_id, event, error, context, config \\ nil) do
    config = config || new()

    # Create unique ID for this failure
    failure_id =
      "#{stage_id}_#{:erlang.system_time(:millisecond)}_#{:erlang.unique_integer([:positive])}"

    # Store the failure details
    failure_data = %{
      stage_id: stage_id,
      event: event,
      error: inspect(error),
      context: context,
      timestamp: DateTime.utc_now() |> DateTime.to_iso8601(),
      recovery_attempts: 0
    }

    serialized_data = :erlang.term_to_binary(failure_data)

    # Store the failed event in Redis
    with {:ok, _} <- config.redis_client.set("failure:#{failure_id}", serialized_data),
         {:ok, _} <- config.redis_client.expire("failure:#{failure_id}", config.dead_letter_ttl),
         {:ok, _} <- config.redis_client.sadd("failures:#{stage_id}", failure_id),
         {:ok, _} <- config.redis_client.sadd("failures:pending", failure_id) do
      Logger.info("Tracked failed event #{failure_id} for stage #{stage_id}")
      :ok
    else
      {:error, reason} ->
        Logger.error("Failed to track failed event: #{inspect(reason)}")
        {:error, reason}
    end
  end

  @doc """
  Schedule recovery of failed events.

  This function triggers the recovery process to run at the specified interval.

  ## Parameters
    * config - Optional recovery configuration

  ## Returns
    * :ok on success
  """
  def schedule_recovery(config \\ nil) do
    config = config || new()

    # Schedule recovery process
    Process.send_after(self(), :recover_failed_events, config.recovery_interval)
    :ok
  end

  @doc """
  Process pending recovery items.

  This function is typically called by the scheduler and processes a batch of
  pending recovery items.

  ## Parameters
    * max_items - Maximum number of items to process in this batch
    * config - Optional recovery configuration

  ## Returns
    * {:ok, processed} - Number of items processed
    * {:error, reason} - If an error occurred during processing
  """
  def process_pending_recoveries(max_items \\ 10, config \\ nil) do
    config = config || new()

    # Get pending recovery items
    case config.redis_client.smembers("failures:pending") do
      {:ok, []} ->
        {:ok, 0}

      {:ok, failure_ids} ->
        # Process up to max_items
        failure_ids = Enum.take(failure_ids, max_items)

        # Process each failure
        processed =
          failure_ids
          |> Enum.map(fn failure_id -> process_recovery_item(failure_id, config) end)
          |> Enum.count(fn result -> result == :ok end)

        {:ok, processed}

      {:error, reason} ->
        Logger.error("Failed to get pending recovery items: #{inspect(reason)}")
        {:error, reason}
    end
  end

  @doc """
  Process a single recovery item.

  ## Parameters
    * failure_id - ID of the failure to process
    * config - Optional recovery configuration

  ## Returns
    * :ok if processed successfully
    * {:error, reason} if processing failed
  """
  def process_recovery_item(failure_id, config \\ nil) do
    config = config || new()

    # Get failure data
    case config.redis_client.get("failure:#{failure_id}") do
      {:ok, nil} ->
        # Failure no longer exists, remove from pending
        {:ok, _} = config.redis_client.srem("failures:pending", failure_id)
        :ok

      {:ok, data} ->
        failure_data = :erlang.binary_to_term(data)

        # Check if we've exceeded max recovery attempts
        if failure_data.recovery_attempts >= config.max_recovery_attempts do
          # Move to dead letter queue
          move_to_dead_letter_queue(failure_id, failure_data, config)
        else
          # Attempt recovery
          attempt_recovery(failure_id, failure_data, config)
        end

      {:error, reason} ->
        Logger.error("Failed to get failure data for #{failure_id}: #{inspect(reason)}")
        {:error, reason}
    end
  end

  @doc """
  Get all items in the dead letter queue for a stage.

  ## Parameters
    * stage_id - ID of the stage to get dead letter items for, or :all for all stages
    * config - Optional recovery configuration

  ## Returns
    * {:ok, items} - List of dead letter items
    * {:error, reason} - If an error occurred
  """
  def get_dead_letter_items(stage_id \\ :all, config \\ nil) do
    config = config || new()

    # Get dead letter queue key
    dead_letter_key = if stage_id == :all, do: "deadletter", else: "deadletter:#{stage_id}"

    case config.redis_client.smembers(dead_letter_key) do
      {:ok, item_ids} ->
        # Get data for each item
        items =
          Enum.map(item_ids, fn item_id ->
            case config.redis_client.get("deadletter:#{item_id}") do
              {:ok, data} when is_binary(data) ->
                {:ok, :erlang.binary_to_term(data)}

              _ ->
                nil
            end
          end)
          |> Enum.filter(fn
            {:ok, _} -> true
            _ -> false
          end)
          |> Enum.map(fn {:ok, data} -> data end)

        {:ok, items}

      {:error, reason} ->
        {:error, reason}
    end
  end

  # Move a failed item to the dead letter queue
  defp move_to_dead_letter_queue(failure_id, failure_data, config) do
    Logger.warn(
      "Moving failed event #{failure_id} to dead letter queue after #{failure_data.recovery_attempts} attempts"
    )

    # Update dead letter data with final timestamp
    dead_letter_data =
      Map.put(failure_data, :moved_to_dlq_at, DateTime.utc_now() |> DateTime.to_iso8601())

    serialized_data = :erlang.term_to_binary(dead_letter_data)

    # Store in dead letter queue
    with {:ok, _} <- config.redis_client.set("deadletter:#{failure_id}", serialized_data),
         {:ok, _} <-
           config.redis_client.expire("deadletter:#{failure_id}", config.dead_letter_ttl),
         {:ok, _} <- config.redis_client.sadd("deadletter", failure_id),
         {:ok, _} <- config.redis_client.sadd("deadletter:#{failure_data.stage_id}", failure_id),
         # Remove from pending recovery
         {:ok, _} <- config.redis_client.srem("failures:pending", failure_id) do
      :ok
    else
      {:error, reason} ->
        Logger.error("Failed to move item to dead letter queue: #{inspect(reason)}")
        {:error, reason}
    end
  end

  # Attempt to recover a failed item
  defp attempt_recovery(failure_id, failure_data, config) do
    Logger.info(
      "Attempting recovery for failed event #{failure_id}, attempt #{failure_data.recovery_attempts + 1}"
    )

    # Update recovery attempts
    updated_data = Map.update!(failure_data, :recovery_attempts, &(&1 + 1))
    serialized_data = :erlang.term_to_binary(updated_data)

    # Store updated data
    {:ok, _} = config.redis_client.set("failure:#{failure_id}", serialized_data)

    # Here we would typically send the event back to the stage for reprocessing
    # This is just a placeholder - actual implementation would depend on how
    # events are sent to stages in the specific pipeline architecture

    # For this implementation, we'll just reschedule it for later recovery
    # In a real implementation, we would send it back to the stage
    schedule_next_recovery(failure_id, config)

    :ok
  end

  # Schedule the next recovery attempt
  defp schedule_next_recovery(failure_id, config) do
    # Re-add to pending set after a delay
    # In a real implementation, this would be handled by the stage itself
    Process.send_after(
      self(),
      {:reschedule_recovery, failure_id},
      config.recovery_interval
    )

    :ok
  end

  @doc """
  Handler for recovery-related messages.

  This function can be used to handle recovery messages in GenServer implementations.

  ## Parameters
    * message - The message to handle
    * state - Current state
    * config - Optional recovery configuration

  ## Returns
    * {:noreply, new_state} - Updated state
  """
  def handle_info(message, state, config \\ nil) do
    config = config || new()

    case message do
      :recover_failed_events ->
        # Process a batch of pending recoveries
        {:ok, _} = process_pending_recoveries(10, config)

        # Reschedule for next interval
        schedule_recovery(config)
        {:noreply, state}

      {:reschedule_recovery, failure_id} ->
        # Re-add to pending set
        {:ok, _} = config.redis_client.sadd("failures:pending", failure_id)
        {:noreply, state}

      _ ->
        {:noreply, state}
    end
  end
end
