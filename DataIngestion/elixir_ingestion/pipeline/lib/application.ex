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
      # Connection handling for Redis
      {ConnectionHandler.PoolSupervisor, []},

      # File Watcher Supervisor (manages the Server and StateStore)
      {FileWatcher.Supervisor, []},

      # File Queue Producer
      {Producer.FileQueueProducer, []},

      # File Watcher Connector - subscribes to Server and enqueues to Producer
      {FileWatcher.FileWatcherConnector, []}
    ]

    # Start the supervision tree
    Logger.info("Starting Data Ingestion Pipeline")

    opts = [strategy: :one_for_one, name: Pipeline.Supervisor]
    Supervisor.start_link(children, opts)
  end
end
