defmodule IngestionService.Pipeline.DynamicPipeline do
  use GenServer
  require Logger

  @moduledoc """
  A GenServer that enables dynamic configuration and management of data processing pipelines.
  This module allows creating custom pipelines with specified stages and connections
  for different data processing needs.

  Features:
  1. Dynamic creation of processing pipelines with specified stages
  2. Validation of stage dependencies and compatibility
  3. Connection of stages in the correct order
  4. Runtime updates to existing pipeline configurations
  """

  # Client API

  def start_link(opts) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Creates a new pipeline with the specified configuration.

  ## Parameters

  * `pipeline_id` - Unique identifier for this pipeline
  * `stages` - List of stages to include in the pipeline, in order
  * `options` - Configuration options for the pipeline

  ## Example

      iex> DynamicPipeline.create_pipeline("csv_timeseries", [
      ...>   :file_watcher,
      ...>   :producer,
      ...>   :csv_processor,
      ...>   :schema_inference,
      ...>   :data_profiler,
      ...>   :time_series_processor,
      ...>   :writer
      ...> ], %{parallelism: 2})
  """
  def create_pipeline(pipeline_id, stages, options \\ %{}) do
    GenServer.call(__MODULE__, {:create_pipeline, pipeline_id, stages, options})
  end

  @doc """
  Starts a previously created pipeline.
  """
  def start_pipeline(pipeline_id) do
    GenServer.call(__MODULE__, {:start_pipeline, pipeline_id})
  end

  @doc """
  Stops a running pipeline.
  """
  def stop_pipeline(pipeline_id) do
    GenServer.call(__MODULE__, {:stop_pipeline, pipeline_id})
  end

  @doc """
  Updates configuration of an existing pipeline.
  The pipeline must be stopped before updating.
  """
  def update_pipeline(pipeline_id, stages, options \\ %{}) do
    GenServer.call(__MODULE__, {:update_pipeline, pipeline_id, stages, options})
  end

  @doc """
  Lists all registered pipelines and their status.
  """
  def list_pipelines do
    GenServer.call(__MODULE__, :list_pipelines)
  end

  @doc """
  Gets detailed information about a specific pipeline.
  """
  def get_pipeline(pipeline_id) do
    GenServer.call(__MODULE__, {:get_pipeline, pipeline_id})
  end

  @doc """
  Deletes a pipeline configuration.
  The pipeline must be stopped before deleting.
  """
  def delete_pipeline(pipeline_id) do
    GenServer.call(__MODULE__, {:delete_pipeline, pipeline_id})
  end

  # Server callbacks

  @impl true
  def init(_opts) do
    Logger.info("Starting dynamic pipeline manager")

    # Initialize state
    state = %{
      # Map of pipeline_id -> pipeline_config
      pipelines: %{},
      # Map of pipeline_id -> supervisor_pid
      running: %{}
    }

    {:ok, state}
  end

  @impl true
  def handle_call({:create_pipeline, pipeline_id, stages, options}, _from, state) do
    if Map.has_key?(state.pipelines, pipeline_id) do
      # Pipeline already exists
      {:reply, {:error, :already_exists}, state}
    else
      # Validate the pipeline configuration
      case validate_pipeline_config(stages) do
        :ok ->
          # Create pipeline configuration
          pipeline_config = %{
            id: pipeline_id,
            stages: stages,
            options: options,
            created_at: DateTime.utc_now(),
            updated_at: DateTime.utc_now(),
            status: :stopped
          }

          # Add to state
          new_pipelines = Map.put(state.pipelines, pipeline_id, pipeline_config)
          new_state = %{state | pipelines: new_pipelines}

          {:reply, {:ok, pipeline_config}, new_state}

        {:error, reason} ->
          # Configuration invalid
          {:reply, {:error, reason}, state}
      end
    end
  end

  @impl true
  def handle_call({:start_pipeline, pipeline_id}, _from, state) do
    case Map.fetch(state.pipelines, pipeline_id) do
      {:ok, pipeline_config} ->
        if Map.has_key?(state.running, pipeline_id) do
          # Already running
          {:reply, {:error, :already_running}, state}
        else
          # Start the pipeline
          case start_pipeline_supervisor(pipeline_config) do
            {:ok, supervisor_pid} ->
              # Mark as running
              running = Map.put(state.running, pipeline_id, supervisor_pid)
              pipelines = Map.update!(state.pipelines, pipeline_id, &%{&1 | status: :running})
              new_state = %{state | running: running, pipelines: pipelines}

              {:reply, {:ok, supervisor_pid}, new_state}

            {:error, reason} ->
              # Failed to start
              {:reply, {:error, reason}, state}
          end
        end

      :error ->
        # Pipeline not found
        {:reply, {:error, :not_found}, state}
    end
  end

  @impl true
  def handle_call({:stop_pipeline, pipeline_id}, _from, state) do
    case Map.fetch(state.running, pipeline_id) do
      {:ok, supervisor_pid} ->
        # Stop the supervisor
        case stop_pipeline_supervisor(supervisor_pid) do
          :ok ->
            # Update state
            running = Map.delete(state.running, pipeline_id)
            pipelines = Map.update!(state.pipelines, pipeline_id, &%{&1 | status: :stopped})
            new_state = %{state | running: running, pipelines: pipelines}

            {:reply, :ok, new_state}

          {:error, reason} ->
            # Failed to stop
            {:reply, {:error, reason}, state}
        end

      :error ->
        # Not running
        {:reply, {:error, :not_running}, state}
    end
  end

  @impl true
  def handle_call({:update_pipeline, pipeline_id, stages, options}, _from, state) do
    case Map.fetch(state.pipelines, pipeline_id) do
      {:ok, pipeline_config} ->
        if Map.has_key?(state.running, pipeline_id) do
          # Can't update while running
          {:reply, {:error, :running}, state}
        else
          # Validate the new configuration
          case validate_pipeline_config(stages) do
            :ok ->
              # Update pipeline configuration
              updated_config = %{
                pipeline_config
                | stages: stages,
                  options: Map.merge(pipeline_config.options, options),
                  updated_at: DateTime.utc_now()
              }

              # Update state
              pipelines = Map.put(state.pipelines, pipeline_id, updated_config)
              new_state = %{state | pipelines: pipelines}

              {:reply, {:ok, updated_config}, new_state}

            {:error, reason} ->
              # Configuration invalid
              {:reply, {:error, reason}, state}
          end
        end

      :error ->
        # Pipeline not found
        {:reply, {:error, :not_found}, state}
    end
  end

  @impl true
  def handle_call(:list_pipelines, _from, state) do
    # Prepare summary list
    pipeline_list =
      Enum.map(state.pipelines, fn {id, config} ->
        %{
          id: id,
          status: config.status,
          stages: length(config.stages),
          created_at: config.created_at
        }
      end)

    {:reply, pipeline_list, state}
  end

  @impl true
  def handle_call({:get_pipeline, pipeline_id}, _from, state) do
    case Map.fetch(state.pipelines, pipeline_id) do
      {:ok, pipeline_config} ->
        {:reply, {:ok, pipeline_config}, state}

      :error ->
        {:reply, {:error, :not_found}, state}
    end
  end

  @impl true
  def handle_call({:delete_pipeline, pipeline_id}, _from, state) do
    if Map.has_key?(state.running, pipeline_id) do
      # Can't delete while running
      {:reply, {:error, :running}, state}
    else
      case Map.fetch(state.pipelines, pipeline_id) do
        {:ok, _pipeline_config} ->
          # Remove from state
          pipelines = Map.delete(state.pipelines, pipeline_id)
          new_state = %{state | pipelines: pipelines}

          {:reply, :ok, new_state}

        :error ->
          # Pipeline not found
          {:reply, {:error, :not_found}, state}
      end
    end
  end

  # Handle process monitoring for supervisors
  @impl true
  def handle_info({:DOWN, _ref, :process, pid, reason}, state) do
    # Find which pipeline this supervisor belongs to
    case Enum.find(state.running, fn {_, supervisor_pid} -> supervisor_pid == pid end) do
      {pipeline_id, _} ->
        # Supervisor crashed, update state
        Logger.warn("Pipeline #{pipeline_id} supervisor terminated: #{inspect(reason)}")

        running = Map.delete(state.running, pipeline_id)
        pipelines = Map.update!(state.pipelines, pipeline_id, &%{&1 | status: :crashed})
        new_state = %{state | running: running, pipelines: pipelines}

        {:noreply, new_state}

      nil ->
        # Unknown supervisor
        {:noreply, state}
    end
  end

  # Private functions

  # Validates pipeline configuration for correctness
  defp validate_pipeline_config(stages) do
    cond do
      Enum.empty?(stages) ->
        {:error, "Pipeline must contain at least one stage"}

      not validate_stage_dependencies(stages) ->
        {:error, "Invalid stage dependencies"}

      not validate_unique_stages(stages) ->
        {:error, "Duplicate stages not allowed"}

      true ->
        :ok
    end
  end

  # Check that stage dependencies are satisfied
  defp validate_stage_dependencies(stages) do
    # Define stage dependencies
    dependencies = %{
      # Each key depends on having at least one of the values in its list
      schema_inference: [:processor, :csv_processor, :json_processor, :excel_processor],
      data_profiler: [:schema_inference, :validator],
      time_series_processor: [:data_profiler],
      validator: [:processor, :schema_inference],
      transformer: [:validator],
      writer: [:transformer, :validator, :processor],
      processor: [:producer],
      csv_processor: [:producer],
      json_processor: [:producer],
      excel_processor: [:producer],
      producer: [:file_watcher]
    }

    # Check that each stage's dependencies are satisfied
    Enum.all?(stages, fn stage ->
      case Map.fetch(dependencies, stage) do
        {:ok, deps} ->
          # This stage has dependencies, check if any are satisfied
          Enum.any?(deps, &Enum.member?(stages, &1))

        :error ->
          # No dependencies defined for this stage (like file_watcher)
          true
      end
    end)
  end

  # Check that stages appear at most once
  defp validate_unique_stages(stages) do
    length(Enum.uniq(stages)) == length(stages)
  end

  # Start a supervisor for a pipeline
  defp start_pipeline_supervisor(pipeline_config) do
    # Create supervisor spec
    supervisor_spec = create_supervisor_spec(pipeline_config)

    # Start the supervisor
    case DynamicSupervisor.start_child(IngestionService.DynamicSupervisor, supervisor_spec) do
      {:ok, pid} ->
        # Monitor the supervisor
        Process.monitor(pid)
        {:ok, pid}

      {:error, reason} ->
        Logger.error("Failed to start pipeline supervisor: #{inspect(reason)}")
        {:error, reason}
    end
  end

  # Create a supervisor specification for a pipeline
  defp create_supervisor_spec(pipeline_config) do
    # Map stage atoms to actual module names
    modules = %{
      file_watcher: IngestionService.Pipeline.FileWatcher,
      producer: IngestionService.Pipeline.Producer,
      processor: IngestionService.Pipeline.Processor,
      csv_processor: IngestionService.Pipeline.Processor,
      json_processor: IngestionService.Pipeline.Processor,
      excel_processor: IngestionService.Pipeline.Processor,
      schema_inference: IngestionService.Pipeline.SchemaInference,
      data_profiler: IngestionService.Pipeline.DataProfiler,
      time_series_processor: IngestionService.Pipeline.TimeSeriesProcessor,
      validator: IngestionService.Pipeline.Validator,
      transformer: IngestionService.Pipeline.Transformer,
      writer: IngestionService.Pipeline.Writer
    }

    # Build child specs for each stage
    children =
      Enum.map(pipeline_config.stages, fn stage ->
        module = Map.fetch!(modules, stage)

        # Add stage-specific options
        stage_opts =
          case stage do
            :csv_processor -> [id: :"#{pipeline_config.id}_csv_processor", type: :csv]
            :json_processor -> [id: :"#{pipeline_config.id}_json_processor", type: :json]
            :excel_processor -> [id: :"#{pipeline_config.id}_excel_processor", type: :excel]
            _ -> [id: :"#{pipeline_config.id}_#{stage}"]
          end

        # Create the child spec
        %{
          id: :"#{pipeline_config.id}_#{stage}",
          start: {module, :start_link, [stage_opts]},
          restart: :permanent
        }
      end)

    # Create the supervisor spec
    %{
      id: :"#{pipeline_config.id}_supervisor",
      start: {Supervisor, :start_link, [children, [strategy: :one_for_one]]},
      type: :supervisor,
      restart: :permanent
    }
  end

  # Stop a pipeline supervisor
  defp stop_pipeline_supervisor(supervisor_pid) do
    case Process.alive?(supervisor_pid) do
      true ->
        # Terminate the supervisor
        Supervisor.stop(supervisor_pid)
        :ok

      false ->
        # Already terminated
        :ok
    end
  rescue
    e ->
      Logger.error("Error stopping pipeline supervisor: #{inspect(e)}")
      {:error, :stop_failed}
  end
end
