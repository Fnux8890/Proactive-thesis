defmodule IngestionService.Pipeline.Supervisor do
  @moduledoc """
  Supervisor for the ingestion pipeline.

  This module supervises all stages of the ingestion pipeline, ensuring they are started
  and restarted appropriately.
  """

  use Supervisor
  require Logger

  @doc """
  Starts the ingestion pipeline supervisor.

  ## Parameters

  * `opts` - Options to pass to the supervisor

  ## Returns

  * `{:ok, pid}` - The PID of the started supervisor
  * `{:error, reason}` - If there was an error starting the supervisor
  """
  def start_link(opts \\ []) do
    Supervisor.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @impl true
  def init(_opts) do
    Logger.info("Starting ingestion pipeline supervisor")

    children = [
      # File Watcher monitors directories for new files
      {IngestionService.Pipeline.FileWatcher, [name: IngestionService.Pipeline.FileWatcher]},

      # Producer generates events from detected files
      {IngestionService.Pipeline.Producer,
       [
         name: IngestionService.Pipeline.Producer,
         subscribe_to: [{IngestionService.Pipeline.FileWatcher, max_demand: 10}]
       ]},

      # Processor processes CSV files
      {IngestionService.Pipeline.Processor.CSV,
       [
         name: IngestionService.Pipeline.Processor.CSV,
         subscribe_to: [{IngestionService.Pipeline.Producer, max_demand: 5}]
       ]},

      # Processor processes CSV files with enhanced features
      {IngestionService.Pipeline.Processor.CSVEnhanced,
       [
         name: IngestionService.Pipeline.Processor.CSVEnhanced,
         subscribe_to: [{IngestionService.Pipeline.Producer, max_demand: 5}]
       ]},

      # Processor processes JSON files
      {IngestionService.Pipeline.Processor.JSON,
       [
         name: IngestionService.Pipeline.Processor.JSON,
         subscribe_to: [{IngestionService.Pipeline.Producer, max_demand: 5}]
       ]},

      # Schema inference detects and enforces schema
      {IngestionService.Pipeline.SchemaInference,
       [
         name: IngestionService.Pipeline.SchemaInference,
         subscribe_to: [
           {IngestionService.Pipeline.Processor.CSV, max_demand: 5},
           {IngestionService.Pipeline.Processor.CSVEnhanced, max_demand: 5},
           {IngestionService.Pipeline.Processor.JSON, max_demand: 5}
         ]
       ]},

      # Data Profiler assesses data quality
      {IngestionService.Pipeline.DataProfiler,
       [
         name: IngestionService.Pipeline.DataProfiler,
         subscribe_to: [{IngestionService.Pipeline.SchemaInference, max_demand: 5}]
       ]},

      # Time Series Processor handles time series data
      {IngestionService.Pipeline.TimeSeriesProcessor,
       [
         name: IngestionService.Pipeline.TimeSeriesProcessor,
         subscribe_to: [{IngestionService.Pipeline.DataProfiler, max_demand: 5}]
       ]},

      # Validator validates data against rules
      {IngestionService.Pipeline.Validator,
       [
         name: IngestionService.Pipeline.Validator,
         subscribe_to: [{IngestionService.Pipeline.TimeSeriesProcessor, max_demand: 5}]
       ]},

      # MetadataEnricher enriches data with metadata
      {IngestionService.Pipeline.MetadataEnricher,
       [
         name: IngestionService.Pipeline.MetadataEnricher,
         subscribe_to: [{IngestionService.Pipeline.Validator, max_demand: 5}]
       ]},

      # Transformer transforms data before writing
      {IngestionService.Pipeline.Transformer,
       [
         name: IngestionService.Pipeline.Transformer,
         subscribe_to: [{IngestionService.Pipeline.MetadataEnricher, max_demand: 5}]
       ]},

      # Writer writes data to the database
      {IngestionService.Pipeline.Writer,
       [
         name: IngestionService.Pipeline.Writer,
         subscribe_to: [{IngestionService.Pipeline.Transformer, max_demand: 5}]
       ]}
    ]

    Supervisor.init(children, strategy: :one_for_one)
  end
end
