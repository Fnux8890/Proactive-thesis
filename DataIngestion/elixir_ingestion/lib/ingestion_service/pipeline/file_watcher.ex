defmodule IngestionService.Pipeline.FileWatcher do
  use GenServer
  require Logger
  
  @moduledoc """
  A file system watcher that monitors data directories for changes.
  When new files are detected, it notifies the Producer to start ingestion.
  """
  
  # Client API
  
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end
  
  # Server callbacks
  
  @impl true
  def init(_opts) do
    # Determine data source path from environment variables
    data_path = System.get_env("DATA_SOURCE_PATH") || "/app/data"
    Logger.info("Starting file watcher for #{data_path}")
    
    # Start the file system monitor
    {:ok, watcher_pid} = FileSystem.start_link(dirs: [data_path])
    FileSystem.subscribe(watcher_pid)
    
    # Process existing files on startup
    process_existing_files(data_path)
    
    {:ok, %{watcher_pid: watcher_pid, data_path: data_path, processed_files: %{}}}
  end
  
  @impl true
  def handle_info({:file_event, _watcher_pid, {path, events}}, state) do
    # Only process new or modified files
    if :modified in events or :created in events do
      # Get file metadata
      case File.stat(path) do
        {:ok, %{type: :regular, size: size}} ->
          # Check if we've already processed this file with this size
          case Map.get(state.processed_files, path) do
            ^size -> 
              # Already processed this exact file
              {:noreply, state}
            _ -> 
              # New or modified file
              Logger.info("Detected new or modified file: #{path}")
              
              # Only process if it matches our expected file types
              if valid_file_type?(path) do
                # Notify the producer
                IngestionService.Pipeline.Producer.notify_file_change(path)
                
                # Add to processed files
                processed_files = Map.put(state.processed_files, path, size)
                {:noreply, %{state | processed_files: processed_files}}
              else
                {:noreply, state}
              end
          end
        _ -> 
          # Not a regular file or error in accessing it
          {:noreply, state}
      end
    else
      # Other file events (deleted, renamed, etc.)
      {:noreply, state}
    end
  end
  
  @impl true
  def handle_info({:file_event, _watcher_pid, :stop}, state) do
    # File system watcher stopped
    Logger.warning("File system watcher stopped")
    {:noreply, state}
  end
  
  @impl true
  def handle_info(msg, state) do
    Logger.debug("Unhandled message in FileWatcher: #{inspect(msg)}")
    {:noreply, state}
  end
  
  # Process existing files when starting up
  defp process_existing_files(data_path) do
    Logger.info("Processing existing files in #{data_path}")
    
    # Find all files recursively
    data_path
    |> Path.join("**/*")
    |> Path.wildcard()
    |> Enum.filter(&(File.regular?(&1) and valid_file_type?(&1)))
    |> Enum.each(fn file_path ->
      Logger.debug("Queueing existing file: #{file_path}")
      IngestionService.Pipeline.Producer.notify_file_change(file_path)
    end)
  end
  
  # Check if the file has a valid type we want to process
  defp valid_file_type?(path) do
    case Path.extname(path) |> String.downcase() do
      ext when ext in [".csv", ".json", ".xlsx", ".xls"] -> true
      _ -> false
    end
  end
end 