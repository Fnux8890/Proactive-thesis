defmodule Pipeline.Application do
  @moduledoc """
  Main application for the data ingestion pipeline.
  """

  use Application
  require Logger

  def start(_type, _args) do
    # Only start the application's supervision tree if configured to do so
    # This allows tests to control startup manually
    start_supervision = Application.get_env(:pipeline, :start_supervision, true)

    if start_supervision do
      do_start()
    else
      Logger.info("Skipping application startup (start_supervision is false)")
      {:ok, self()}
    end
  end

  defp do_start do
    # Configure and start supervision tree
    children = [
      # Connection handling for Redis (required by both FileWatcher.Server and Producer.Dispatcher)
      {ConnectionHandler.PoolSupervisor, []},

      # Start the Processor Supervisor FIRST (needed by Dispatcher)
      {Processor.Supervisor, [name: Processor.Supervisor]},

      # Start the Producer Dispatcher (depends on Processor.Supervisor and ConnectionHandler)
      {Producer.Dispatcher, [name: Producer.Dispatcher]},

      # Start the FileWatcher Server (depends on Producer.Dispatcher and ConnectionHandler)
      {FileWatcher.Server,
       [
         name: FileWatcher.Server,
         watch_paths: Application.get_env(:pipeline, :watch_dir, ["/app/data"]),
         poll_interval: Application.get_env(:pipeline, :file_watcher_poll_interval, 5000)
       ]}
    ]

    # Start the supervision tree
    Logger.info("Starting Data Ingestion Pipeline with Redis-based processing")

    opts = [strategy: :one_for_one, name: Pipeline.Supervisor]
    Supervisor.start_link(children, opts)
  end
end
