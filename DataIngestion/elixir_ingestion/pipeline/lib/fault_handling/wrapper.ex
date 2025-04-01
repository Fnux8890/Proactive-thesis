defmodule Pipeline.FaultHandling.Wrapper do
  @moduledoc """
  Defines behaviours and implements fault handling for pipeline stages.

  This module provides a standardized way to handle errors across different
  pipeline stages while integrating with Redis checkpointing.
  """

  @doc """
  Base behaviour for all pipeline stage wrappers.
  """
  defmodule StageWrapper do
    @moduledoc """
    Base behaviour for all pipeline stages that need fault handling.

    Defines common callbacks required for all stage types.
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

  @doc """
  Producer stage wrapper behaviour.
  """
  defmodule ProducerWrapper do
    @moduledoc """
    Behaviour for producer stages with fault handling capabilities.
    """

    @behaviour Pipeline.FaultHandling.Wrapper.StageWrapper

    @callback handle_demand(demand :: non_neg_integer(), state :: term()) ::
                {:noreply, events :: list(), new_state :: term()}
  end

  @doc """
  Processor (producer-consumer) stage wrapper behaviour.
  """
  defmodule ProcessorWrapper do
    @moduledoc """
    Behaviour for processor (producer-consumer) stages with fault handling capabilities.
    """

    @behaviour Pipeline.FaultHandling.Wrapper.StageWrapper

    @callback handle_events(events :: list(), from :: term(), state :: term()) ::
                {:noreply, events :: list(), new_state :: term()}
  end

  @doc """
  Consumer stage wrapper behaviour.
  """
  defmodule ConsumerWrapper do
    @moduledoc """
    Behaviour for consumer stages with fault handling capabilities.
    """

    @behaviour Pipeline.FaultHandling.Wrapper.StageWrapper

    @callback handle_events(events :: list(), from :: term(), state :: term()) ::
                {:noreply, [], new_state :: term()}
  end

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

                  {:skip, _, skip_state} ->
                    # Skip processing and move on
                    emit_telemetry(:skipped, %{count: length(events)})
                    {:noreply, [], skip_state}

                  {:fail, _, fail_state} ->
                    # Stop processing due to critical failure
                    {:stop, reason, fail_state}
                end
            end
          rescue
            error ->
              # Handle unexpected errors
              Logger.error("Unexpected error in stage #{get_stage_id()}: #{inspect(error)}")
              {:stop, error, state}
          end
        end

        # Handle retry messages
        def handle_info({:retry_demand, demand}, state) do
          # Reset retry count for new attempt
          processing = Map.put(state._processing, :retry_count, 0)
          new_state = Map.put(state, :_processing, processing)

          # Retry the demand
          handle_demand(demand, new_state)
        end
      end

      # ===== Producer-Consumer implementations =====
      if @stage_type == :producer_consumer do
        # Wrap handle_events to include error handling and checkpointing
        def handle_events(events, from, state) do
          batch_id = generate_batch_id()

          try do
            # Process events with error handling
            case process_events(events, state) do
              {:ok, results, new_state} ->
                # Save checkpoint before sending results downstream
                save_checkpoint(batch_id, results)

                # Update processing state
                processing = Map.put(state._processing, :current_batch, batch_id)
                new_state = Map.put(new_state, :_processing, processing)

                emit_telemetry(:processed, %{
                  input_count: length(events),
                  output_count: length(results)
                })

                # Success - forward processed results downstream
                {:noreply, results, new_state}

              {:error, reason, events, error_state} ->
                # Handle error based on stage-specific logic
                case handle_error(reason, events, error_state) do
                  {:retry, retry_events, retry_state} ->
                    # Apply retry policy
                    retry_count = state._processing.retry_count + 1
                    backoff = @retry_policy.calculate_backoff(retry_count)

                    # Schedule retry after backoff
                    Process.send_after(self(), {:retry_events, retry_events, from}, backoff)

                    # Update retry count in state
                    processing = Map.put(state._processing, :retry_count, retry_count)
                    new_state = Map.put(retry_state, :_processing, processing)

                    {:noreply, [], new_state}

                  {:skip, _, skip_state} ->
                    # Skip processing and move on
                    emit_telemetry(:skipped, %{count: length(events)})
                    {:noreply, [], skip_state}

                  {:fail, _, fail_state} ->
                    # Stop processing due to critical failure
                    {:stop, reason, fail_state}
                end
            end
          rescue
            error ->
              # Handle unexpected errors
              Logger.error("Unexpected error in stage #{get_stage_id()}: #{inspect(error)}")
              stack = System.stacktrace()
              Logger.error("Stack trace: #{inspect(stack)}")
              {:stop, error, state}
          end
        end

        # Handle retry messages for producer-consumer
        def handle_info({:retry_events, events, from}, state) do
          # Reset retry count for new attempt
          processing = Map.put(state._processing, :retry_count, 0)
          new_state = Map.put(state, :_processing, processing)

          # Retry processing the events
          handle_events(events, from, new_state)
        end
      end

      # ===== Consumer implementations =====
      if @stage_type == :consumer do
        # Wrap handle_events to include error handling and checkpointing
        def handle_events(events, from, state) do
          try do
            # Process events with error handling
            case process_events(events, state) do
              {:ok, _results, new_state} ->
                emit_telemetry(:consumed, %{count: length(events)})

                # Success - consumer doesn't forward events
                {:noreply, [], new_state}

              {:error, reason, events, error_state} ->
                # Handle error based on stage-specific logic
                case handle_error(reason, events, error_state) do
                  {:retry, retry_events, retry_state} ->
                    # Apply retry policy
                    retry_count = state._processing.retry_count + 1
                    backoff = @retry_policy.calculate_backoff(retry_count)

                    # Schedule retry after backoff
                    Process.send_after(self(), {:retry_events, retry_events, from}, backoff)

                    # Update retry count in state
                    processing = Map.put(state._processing, :retry_count, retry_count)
                    new_state = Map.put(retry_state, :_processing, processing)

                    {:noreply, [], new_state}

                  {:skip, _, skip_state} ->
                    # Skip processing and move on
                    emit_telemetry(:skipped, %{count: length(events)})
                    {:noreply, [], skip_state}

                  {:fail, _, fail_state} ->
                    # Stop processing due to critical failure
                    {:stop, reason, fail_state}
                end
            end
          rescue
            error ->
              # Handle unexpected errors
              Logger.error("Unexpected error in stage #{get_stage_id()}: #{inspect(error)}")
              {:stop, error, state}
          end
        end

        # Handle retry messages for consumer
        def handle_info({:retry_events, events, from}, state) do
          # Reset retry count for new attempt
          processing = Map.put(state._processing, :retry_count, 0)
          new_state = Map.put(state, :_processing, processing)

          # Retry processing the events
          handle_events(events, from, new_state)
        end
      end

      # Common implementation for all stage types
      def handle_info(message, state) do
        Logger.debug("#{__MODULE__} received unhandled message: #{inspect(message)}")
        {:noreply, [], state}
      end
    end
  end
end
