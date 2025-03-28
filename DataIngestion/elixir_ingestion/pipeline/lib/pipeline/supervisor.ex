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
      # Start with the FileWatcher supervisor
      # This will manage all file watcher related processes
      {FileWatcher.Supervisor, []},

      # Additional pipeline components will be added here
      # as they are implemented
      # {Pipeline.Processor.Supervisor, []},
      # ...

      # Add our producers supervisor
      Producer.Supervisor,

      # Add tracking supervisor if not already present
      Pipeline.Tracking.Supervisor
    ]

    # Use :one_for_one strategy for pipeline components so that
    # a failure in one component doesn't affect others unnecessarily
    Supervisor.init(children, strategy: :one_for_one)
  end
end
