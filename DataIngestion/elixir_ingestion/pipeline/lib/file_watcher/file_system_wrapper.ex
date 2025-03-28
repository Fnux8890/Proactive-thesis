defmodule FileWatcher.FileSystemWrapper do
  @moduledoc """
  Wrapper module for the FileSystem library that implements the FileSystemBehaviour.

  This module provides a thin wrapper around the external FileSystem library,
  implementing our own behaviour interface for better testability and dependency injection.
  """

  @behaviour FileWatcher.FileSystemBehaviour

  @doc """
  Starts a new file system watcher process.

  Delegates to FileSystem.start_link/1.

  ## Parameters
    * opts - Options for configuring the watcher

  ## Returns
    * `{:ok, pid}` - If the process started successfully
    * `{:error, reason}` - If the process failed to start
  """
  @impl FileWatcher.FileSystemBehaviour
  def start_link(opts) do
    FileSystem.start_link(opts)
  end

  @doc """
  Subscribes a process to file system events.

  Delegates to FileSystem.subscribe/1.

  ## Parameters
    * pid - The process to receive file system events

  ## Returns
    * `:ok` - If the subscription was successful
    * `{:error, reason}` - If the subscription failed
  """
  @impl FileWatcher.FileSystemBehaviour
  def subscribe(pid) do
    FileSystem.subscribe(pid)
  end

  @doc """
  Stops a file system watcher process.

  Delegates to FileSystem.stop/1.

  ## Parameters
    * pid - The file system watcher process

  ## Returns
    * `:ok` - If the process was stopped successfully
    * `{:error, reason}` - If the process could not be stopped
  """
  @impl FileWatcher.FileSystemBehaviour
  def stop(pid) do
    FileSystem.stop(pid)
  end
end
