defmodule Pipeline.Supervisor do
  @moduledoc """
  Main supervisor for the pipeline components.

  This supervisor handles all the pipeline stages in the proper order:
  FileWatcher → Producer → Processors → SchemaInference → DataProfiler →
  TimeSeriesProcessor → Validator → MetaDataEnricher → Transformer → Writer
  """
  use Supervisor
  require Logger

  def start_link(init_arg) do
    Supervisor.start_link(__MODULE__, init_arg, name: __MODULE__)
  end

  @impl true
  def init(_init_arg) do
    Logger.info("Starting Pipeline Supervisor")

    children = [
      # Supervisors for connection handling, Redis pool, etc.
      ConnectionHandler.Supervisor,
      # Start Tracking system BEFORE Dispatcher
      Pipeline.Tracking.Supervisor,
      # Producer/Dispatcher system
      Producer.Dispatcher,
      # File Watching system
      FileWatcher.Supervisor,
      # Processor system (handles parsing)
      Processor.Supervisor,
      # Schema Inference system (consumes parsed data)
      {SchemaInference.Server, name: SchemaInference.Server}
    ]

    # Use :one_for_one strategy for pipeline components so that
    # a failure in one component doesn't affect others unnecessarily
    Supervisor.init(children, strategy: :one_for_one)
  end
end
