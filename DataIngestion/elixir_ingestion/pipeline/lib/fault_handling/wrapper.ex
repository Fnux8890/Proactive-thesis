require Logger

# Define behaviours outside the main Wrapper module to avoid @moduledoc warnings

defmodule Pipeline.FaultHandling.Wrapper.StageWrapper do
  @moduledoc """
  Base behaviour for all pipeline stage wrappers.
  """

  @callback process_events(events :: list(), state :: term()) ::
              {:ok, results :: list(), new_state :: term()}
              | {:error, reason :: term(), events :: list(), state :: term()}

  @callback handle_stage_error(error :: term(), events :: list(), state :: term()) ::
              {:retry, events :: list(), state :: term()}
              | {:skip, events :: list(), state :: term()}
              | {:fail, events :: list(), state :: term()}

  @callback get_stage_id() :: String.t()

  @callback serialize_event(event :: term()) :: binary()

  @callback deserialize_event(binary :: binary()) :: term()
end

defmodule Pipeline.FaultHandling.Wrapper.ProducerWrapper do
  @moduledoc """
  Producer stage wrapper behaviour.
  """

  @behaviour Pipeline.FaultHandling.Wrapper.StageWrapper

  @callback handle_demand(demand :: non_neg_integer(), state :: term()) ::
              {:noreply, events :: list(), new_state :: term()}

  # Dummy implementations to satisfy compiler - Implement in concrete stage
  def get_stage_id, do: raise "Not implemented: get_stage_id/0"
  def serialize_event(_event), do: raise "Not implemented: serialize_event/1"
  def deserialize_event(_binary), do: raise "Not implemented: deserialize_event/1"
  def process_events(_events, _state), do: raise "Not implemented: process_events/2"
  def handle_stage_error(_error, _events, _state), do: raise "Not implemented: handle_stage_error/3"
end

defmodule Pipeline.FaultHandling.Wrapper.ProcessorWrapper do
  @moduledoc """
  Processor (producer-consumer) stage wrapper behaviour.
  """

  @behaviour Pipeline.FaultHandling.Wrapper.StageWrapper

  @callback handle_events(events :: list(), from :: term(), state :: term()) ::
              {:noreply, events :: list(), new_state :: term()}

  # Dummy implementations to satisfy compiler - Implement in concrete stage
  def get_stage_id, do: raise "Not implemented: get_stage_id/0"
  def serialize_event(_event), do: raise "Not implemented: serialize_event/1"
  def deserialize_event(_binary), do: raise "Not implemented: deserialize_event/1"
  def process_events(_events, _state), do: raise "Not implemented: process_events/2"
  def handle_stage_error(_error, _events, _state), do: raise "Not implemented: handle_stage_error/3"
end

defmodule Pipeline.FaultHandling.Wrapper.ConsumerWrapper do
  @moduledoc """
  Consumer stage wrapper behaviour.
  """

  @behaviour Pipeline.FaultHandling.Wrapper.StageWrapper

  @callback handle_events(events :: list(), from :: term(), state :: term()) ::
              {:noreply, [], new_state :: term()}

  # Dummy implementations to satisfy compiler - Implement in concrete stage
  def get_stage_id, do: raise "Not implemented: get_stage_id/0"
  def serialize_event(_event), do: raise "Not implemented: serialize_event/1"
  def deserialize_event(_binary), do: raise "Not implemented: deserialize_event/1"
  def process_events(_events, _state), do: raise "Not implemented: process_events/2"
  def handle_stage_error(_error, _events, _state), do: raise "Not implemented: handle_stage_error/3"
end


# Main Wrapper module
defmodule Pipeline.FaultHandling.Wrapper do
  @moduledoc """
  Defines behaviours and injects fault handling for pipeline stages.

  This module provides a standardized way to handle errors across different
  pipeline stages while integrating with Redis checkpointing.
  """

  # Nested module definitions moved outside

  @doc """
  Injects standard GenStage implementation with error handling capabilities.

  Options:
    * :stage_type - Required. Either :producer, :producer_consumer, or :consumer
    * :redis_client - Module implementing Redis client functionality. Defaults to ConnectionHandler.Client
    * :retry_policy - Module implementing retry policy. Defaults to Pipeline.FaultHandling.Retry
    * :checkpoint_ttl - TTL for checkpoint data in Redis. Defaults to 86400 (1 day)
  """
  defmacro __using__(opts) do
    quote location: :keep, bind_quoted: [opts: opts] do
      use GenStage
      require Logger

      @stage_type Keyword.fetch!(opts, :stage_type)
      @redis_client Keyword.get(opts, :redis_client, ConnectionHandler.Client)
      @retry_policy Keyword.get(opts, :retry_policy, Pipeline.FaultHandling.Retry)
      @checkpoint_ttl Keyword.get(opts, :checkpoint_ttl, 86400)

      @before_compile Pipeline.FaultHandling.Wrapper

      # Store batch processing state
      defp init_processing_state(state) do
        # Handle both map and keyword list inputs
        state_map =
          case state do
            # If it's already a map, use it
            %{} = map -> map
            # If it's a keyword list, convert to map
            list when is_list(list) -> Enum.into(list, %{})
            # Otherwise, use an empty map
            _ -> %{}
          end

        Map.put(state_map, :_processing, %{
          current_batch: nil,
          retry_count: 0,
          failed_events: []
        })
      end

      # Wrap the init callback for all stage types
      def init(args) do
        case @stage_type do
          :producer ->
            {:producer, init_processing_state(args)}

          :producer_consumer ->
            producer_opts = Keyword.get(unquote(opts), :producer_opts, [])
            {:producer_consumer, init_processing_state(args), producer_opts}

          :consumer ->
            consumer_opts = Keyword.get(unquote(opts), :consumer_opts, [])
            {:consumer, init_processing_state(args), consumer_opts}
        end
      end

      # Track batch ID for checkpointing
      defp generate_batch_id() do
        "#{get_stage_id()}_#{:erlang.system_time(:millisecond)}_#{:erlang.unique_integer([:positive])}"
      end

      # Save checkpoint to Redis
      defp save_checkpoint(batch_id, events) do
        serialized_events = Enum.map(events, &serialize_event/1)

        {:ok, _} =
          @redis_client.set("checkpoint:#{batch_id}", :erlang.term_to_binary(serialized_events))

        {:ok, _} = @redis_client.expire("checkpoint:#{batch_id}", @checkpoint_ttl)
        :ok
      end

      # Retrieve checkpoint from Redis
      defp retrieve_checkpoint(batch_id) do
        case @redis_client.get("checkpoint:#{batch_id}") do
          {:ok, nil} ->
            {:error, :no_checkpoint}

          {:ok, data} ->
            events =
              data
              |> :erlang.binary_to_term()
              |> Enum.map(&deserialize_event/1)

            {:ok, events}

          {:error, reason} ->
            {:error, reason}
        end
      end

      # Clear checkpoint from Redis after successful processing
      defp clear_checkpoint(batch_id) do
        {:ok, _} = @redis_client.del("checkpoint:#{batch_id}")
        :ok
      end

      # Emit telemetry for stage events
      defp emit_telemetry(event, measurements \\ %{}, metadata \\ %{}) do
        :telemetry.execute(
          [:pipeline, :fault_handling, event],
          measurements,
          Map.merge(%{stage_id: get_stage_id(), stage_type: @stage_type}, metadata)
        )
      end

      # Generic error handler for all errors in the stage
      defp handle_error(error, events, state) do
        Logger.error("Error in stage #{get_stage_id()}: #{inspect(error)}")

        emit_telemetry(:error, %{count: 1}, %{
          error: error,
          events_count: length(events)
        })

        start_time = System.monotonic_time()

        result = handle_stage_error(error, events, state)

        emit_telemetry(
          :handle_error,
          %{
            duration: System.monotonic_time() - start_time
          },
          %{result: elem(result, 0)}
        )

        result
      end
    end
  end

  @doc false
  defmacro __before_compile__(_env) do
    quote do
      # ===== Producer implementations =====
      if @stage_type == :producer do
        # Wrap handle_demand to include error handling and checkpointing
        def handle_demand(demand, state) do
          batch_id = generate_batch_id()

          try do
            case process_events(demand, state) do
              {:ok, events, new_state} ->
                # Save checkpoint before sending events downstream
                save_checkpoint(batch_id, events)

                # Update processing state
                processing = Map.put(state._processing, :current_batch, batch_id)
                new_state = Map.put(new_state, :_processing, processing)

                emit_telemetry(:produced, %{count: length(events)})

                # Success - return events to downstream consumers
                {:noreply, events, new_state}

              {:error, reason, events, error_state} ->
                # Handle error based on stage-specific logic
                case handle_error(reason, events, error_state) do
                  {:retry, retry_events, retry_state} ->
                    # Apply retry policy
                    retry_count = state._processing.retry_count + 1
                    backoff = @retry_policy.calculate_backoff(retry_count)

                    # Schedule retry after backoff
                    Process.send_after(self(), {:retry_demand, demand}, backoff)

                    # Update retry count in state
                    processing = Map.put(state._processing, :retry_count, retry_count)
                    new_state = Map.put(retry_state, :_processing, processing)

                    {:noreply, [], new_state}

                  {:skip, skip_events, skip_state} ->
                    Logger.warning(
                      "Skipping #{length(skip_events)} events in stage #{get_stage_id()} due to error: #{inspect(reason)}"
                    )

                    emit_telemetry(:skipped, %{count: length(skip_events)})
                    {:noreply, [], skip_state}

                  {:fail, fail_events, fail_state} ->
                    Logger.error(
                      "Failing #{length(fail_events)} events in stage #{get_stage_id()} due to error: #{inspect(reason)}"
                    )

                    emit_telemetry(:failed, %{count: length(fail_events)})
                    # Decide if we need to stop the stage or just drop events
                    # For now, just drop the events and continue
                    {:noreply, [], fail_state}
                end
            end
          rescue
            e ->
              stacktrace = __STACKTRACE__

              Logger.error(
                "Unhandled exception in handle_demand for stage #{get_stage_id()}: #{inspect(e)}"
              )

              Logger.debug(inspect(stacktrace))
              emit_telemetry(:crash, %{count: 1}, %{error: e, stacktrace: stacktrace})
              # Let GenStage handle the crash, attempt recovery if supervised
              reraise e, stacktrace
          end
        end

        # Retry demand handler
        def handle_info({:retry_demand, demand}, state) do
          handle_demand(demand, state)
        end
      end

      # ===== Processor (Producer-Consumer) implementations =====
      if @stage_type == :producer_consumer do
        def handle_events(events, from, state) do
          batch_id = generate_batch_id()

          try do
            # Save checkpoint before processing
            save_checkpoint(batch_id, events)

            # Process events
            case process_events(events, state) do
              {:ok, results, new_state} ->
                # Clear checkpoint on success
                clear_checkpoint(batch_id)

                emit_telemetry(:processed, %{
                  count: length(events),
                  results_count: length(results)
                })

                {:noreply, results, new_state}

              {:error, reason, error_events, error_state} ->
                # Handle error
                case handle_error(reason, error_events, error_state) do
                  {:retry, retry_events, retry_state} ->
                    retry_count = state._processing.retry_count + 1
                    backoff = @retry_policy.calculate_backoff(retry_count)
                    Process.send_after(self(), {:retry_events, retry_events, from}, backoff)
                    processing = Map.put(state._processing, :retry_count, retry_count)
                    new_state = Map.put(retry_state, :_processing, processing)
                    {:noreply, [], new_state}

                  {:skip, skip_events, skip_state} ->
                    Logger.warning("Skipping #{length(skip_events)} events due to error")
                    emit_telemetry(:skipped, %{count: length(skip_events)})
                    {:noreply, [], skip_state}

                  {:fail, fail_events, fail_state} ->
                    Logger.error("Failing #{length(fail_events)} events due to error")
                    emit_telemetry(:failed, %{count: length(fail_events)})
                    {:noreply, [], fail_state}
                end
            end
          rescue
            e ->
              stacktrace = __STACKTRACE__
              Logger.error("Unhandled exception in handle_events for stage #{get_stage_id()}")
              Logger.debug(inspect(e))
              Logger.debug(inspect(stacktrace))
              emit_telemetry(:crash, %{count: 1}, %{error: e, stacktrace: stacktrace})
              reraise e, stacktrace
          end
        end

        # Retry events handler
        def handle_info({:retry_events, events, from}, state) do
          handle_events(events, from, state)
        end
      end

      # ===== Consumer implementations =====
      if @stage_type == :consumer do
        def handle_events(events, from, state) do
          batch_id = generate_batch_id()

          try do
            save_checkpoint(batch_id, events)

            case process_events(events, state) do
              {:ok, _results, new_state} ->
                clear_checkpoint(batch_id)
                emit_telemetry(:consumed, %{count: length(events)})
                {:noreply, [], new_state}

              {:error, reason, error_events, error_state} ->
                case handle_error(reason, error_events, error_state) do
                  {:retry, retry_events, retry_state} ->
                    retry_count = state._processing.retry_count + 1
                    backoff = @retry_policy.calculate_backoff(retry_count)
                    Process.send_after(self(), {:retry_events, retry_events, from}, backoff)
                    processing = Map.put(state._processing, :retry_count, retry_count)
                    new_state = Map.put(retry_state, :_processing, processing)
                    {:noreply, [], new_state}

                  {:skip, skip_events, skip_state} ->
                    Logger.warning("Skipping #{length(skip_events)} events due to error")
                    emit_telemetry(:skipped, %{count: length(skip_events)})
                    {:noreply, [], skip_state}

                  {:fail, fail_events, fail_state} ->
                    Logger.error("Failing #{length(fail_events)} events due to error")
                    emit_telemetry(:failed, %{count: length(fail_events)})
                    {:noreply, [], fail_state}
                end
            end
          rescue
            e ->
              stacktrace = __STACKTRACE__
              Logger.error("Unhandled exception in handle_events for stage #{get_stage_id()}")
              Logger.debug(inspect(e))
              Logger.debug(inspect(stacktrace))
              emit_telemetry(:crash, %{count: 1}, %{error: e, stacktrace: stacktrace})
              reraise e, stacktrace
          end
        end

        # Retry events handler
        def handle_info({:retry_events, events, from}, state) do
          handle_events(events, from, state)
        end
      end
    end
  end
end
