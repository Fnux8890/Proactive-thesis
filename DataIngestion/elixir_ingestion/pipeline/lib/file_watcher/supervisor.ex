defmodule FileWatcher.Supervisor do
  use Supervisor
  require Logger

  def start_link(args) do
    Supervisor.start_link(__MODULE__, args, name: __MODULE__)
  end

  @impl true
  def init(_args) do
    Logger.info("Starting FileWatcher Supervisor")

    children = [
      # Redis-based state store for file watcher
      {FileWatcher.StateStore, []},

      # Main file watcher server that tracks files
      {FileWatcher.Server, [
        watch_dir: Application.get_env(:pipeline, :watch_dir, "data")
      ]}

      # Additional file watcher components will go here
    ]

    Supervisor.init(children, strategy: :one_for_one)
  end
end
