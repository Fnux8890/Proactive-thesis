defmodule FileWatcher.ServerBehaviour do
  @moduledoc """
  Behaviour module for FileWatcher.Server.
  Defines callbacks for file watching server implementation.
  """

  @doc """
  Retrieves all files currently being watched by the server.
  Returns `{:ok, files}` where `files` is a map of file paths to file metadata.
  """
  @callback get_files() :: {:ok, map()} | {:error, term()}

  @doc """
  Retrieves the content of a specific file path.
  Returns `{:ok, content}` if file exists and is readable, or an error tuple.
  """
  @callback get_file_content(String.t()) :: {:ok, binary()} | {:error, term()}

  @doc """
  Subscribes the given process to file system events.
  The server will monitor the subscribed process and automatically remove it on termination.
  """
  @callback subscribe(pid()) :: :ok | {:error, term()}

  @doc """
  Unsubscribes the given process from file system events.
  """
  @callback unsubscribe(pid()) :: :ok | {:error, term()}

  @doc """
  Saves the current state of the server to persistent storage.
  """
  @callback save_state() :: :ok | {:error, term()}
end
