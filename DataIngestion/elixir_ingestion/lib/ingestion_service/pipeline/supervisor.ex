defmodule IngestionService.Pipeline.Supervisor do
  use Supervisor

  def start_link(init_arg) do
    Supervisor.start_link(__MODULE__, init_arg, name: __MODULE__)
  end

  @impl true
  def init(_init_arg) do
    children = [
      # Start the File Watcher for detecting new data files
      {IngestionService.Pipeline.FileWatcher, []},
      
      # Start the Producer to generate events from file changes
      {IngestionService.Pipeline.Producer, []},
      
      # Start multiple Processors (consumers/producers)
      {IngestionService.Pipeline.Processor, id: :processor_csv, type: :csv},
      {IngestionService.Pipeline.Processor, id: :processor_json, type: :json},
      {IngestionService.Pipeline.Processor, id: :processor_excel, type: :excel},
      
      # Start the Validator (consumer/producer)
      {IngestionService.Pipeline.Validator, []},
      
      # Start the Transformer (consumer/producer)
      {IngestionService.Pipeline.Transformer, []},
      
      # Start the Writer to save processed data to TimescaleDB
      {IngestionService.Pipeline.Writer, []}
    ]

    # Each stage subscribes to its upstream stage(s) during initialization
    Supervisor.init(children, strategy: :one_for_one)
  end
end 