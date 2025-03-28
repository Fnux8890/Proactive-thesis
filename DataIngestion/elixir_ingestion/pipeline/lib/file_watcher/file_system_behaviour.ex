defmodule FileWatcher.FileSystemBehaviour do
  @moduledoc """
  Behaviour for the underlying file system watcher library.

  This module defines the contract for file system operations used by the
  FileWatcher.Server, allowing for dependency injection and simpler testing.
  """

  @doc """
  Starts a new file system watcher process.

  ## Parameters
    * opts - Options for configuring the watcher

  ## Returns
    * `{:ok, pid}` - If the process started successfully
    * `{:error, reason}` - If the process failed to start
  """
  @callback start_link(opts :: Keyword.t()) :: GenServer.on_start()

  @doc """
  Subscribes a process to file system events.

  ## Parameters
    * pid - The process to receive file system events

  ## Returns
    * `:ok` - If the subscription was successful
    * `{:error, reason}` - If the subscription failed
  """
  @callback subscribe(pid :: pid()) :: :ok | {:error, any()}

  @doc """
  Stops a file system watcher process.

  ## Parameters
    * pid - The file system watcher process

  ## Returns
    * `:ok` - If the process was stopped successfully
    * `{:error, reason}` - If the process could not be stopped
  """
  @callback stop(pid :: pid()) :: :ok | {:error, any()}
end
