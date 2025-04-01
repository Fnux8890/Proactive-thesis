defmodule FileWatcher.StateStoreBehaviour do
  @moduledoc """
  Behaviour for the FileWatcher state storage.

  This module defines the interface for storing and retrieving file state information,
  which allows for dependency injection and testing with mocks.
  """

  @doc """
  Loads the current state from storage.

  ## Returns

  Returns one of:
    * `{:ok, files}` - The state was loaded successfully, where `files` is a map of file paths to file metadata.
    * `{:error, reason}` - An error occurred while loading the state.
  """
  @callback load_state() :: {:ok, map()} | {:error, term()}

  @doc """
  Saves the current state to storage.

  ## Parameters

    * `files` - A map of file paths to file metadata to save.

  ## Returns

  Returns one of:
    * `:ok` - The state was saved successfully.
    * `{:error, reason}` - An error occurred while saving the state.
  """
  @callback save_state(files :: map()) :: :ok | {:error, term()}
end
