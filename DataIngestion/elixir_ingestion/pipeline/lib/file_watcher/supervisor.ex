defmodule FileWatcher.Supervisor do
  use Supervisor
  require Logger

  def start_link(args) do
    Supervisor.start_link(__MODULE__, args, name: __MODULE__)
  end

  @impl true
  def init(_args) do
    # Ensure previous logs are kept if they existed
    Logger.info("[FileWatcher.Supervisor] Initializing...")
    Logger.info("Starting FileWatcher Supervisor") # Keep original log if present

    children = [
      # Redis-based state store for file watcher
      {FileWatcher.StateStore, []},

      # Main file watcher server that tracks files
      {FileWatcher.Server, [
        # Pass :watch_paths key, ensuring it's a list
        watch_paths:
          case Application.get_env(:pipeline, :watch_dir, "/app/data") do
            paths when is_list(paths) -> paths
            path when is_binary(path) -> [path]
            # Default to /app/data if config is invalid or missing
            _ -> ["/app/data"]
          end
      ]}

      # Additional file watcher components will go here
    ]

    Supervisor.init(children, strategy: :one_for_one)
  end
end
