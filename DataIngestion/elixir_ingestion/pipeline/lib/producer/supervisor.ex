defmodule Producer.Supervisor do
  @moduledoc """
  Supervisor for file producer components.

  This supervisor:
  - Manages the FileQueueProducer
  - Manages the FileWatcherConnector
  - Provides helper functions to interact with them
  """
  use Supervisor
  require Logger

  # Client API

  @doc """
  Starts the producer supervisor.

  ## Parameters
    * opts - Options passed to the supervisor

  ## Returns
    * {:ok, pid} - PID of the started supervisor
    * {:error, reason} - If starting the supervisor failed
  """
  def start_link(opts \\ []) do
    Supervisor.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Gets the current queue status from the FileQueueProducer.

  ## Returns
    * Map with queue statistics
  """
  def queue_status do
    Producer.FileQueueProducer.queue_status()
  end

  @doc """
  Gets statistics from the FileWatcherConnector.

  ## Returns
    * Map with connector statistics
  """
  def connector_stats do
    Producer.FileWatcherConnector.get_stats()
  end

  @doc """
  Manually enqueues a file to the FileQueueProducer.

  ## Parameters
    * file_path - Path to the file to be processed
    * metadata - Optional metadata for the file

  ## Returns
    * {:ok, file_id} - ID assigned to the file
    * {:error, reason} - If enqueuing failed
  """
  def enqueue_file(file_path, metadata \\ %{}) do
    Producer.FileQueueProducer.enqueue_file(file_path, metadata)
  end

  @doc """
  Triggers an immediate check for new files in the FileWatcherConnector.

  ## Returns
    * :ok - Check was triggered
  """
  def check_for_files do
    Producer.FileWatcherConnector.check_for_files()
  end

  # Supervisor callbacks

  @impl true
  def init(_opts) do
    # Define child processes
    children = [
      # Start FileQueueProducer
      {Producer.FileQueueProducer,
       [
         name: Producer.FileQueueProducer,
         queue_check_interval: 1_000
       ]},

      # Start FileWatcherConnector (after FileQueueProducer)
      {Producer.FileWatcherConnector,
       [
         name: Producer.FileWatcherConnector,
         poll_interval: 3_000,
         producer: Producer.FileQueueProducer
       ]}
    ]

    # Use a one_for_one strategy: if a child process fails, only that
    # process is restarted
    Supervisor.init(children, strategy: :one_for_one)
  end
end
